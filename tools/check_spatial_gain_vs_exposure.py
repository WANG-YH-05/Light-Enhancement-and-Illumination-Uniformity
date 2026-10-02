"""Check whether a frozen no-AGM model fixes broad facial lighting, not just exposure.

The identity/clean target is an approximate common-light target for MEAD, not
physical ground truth for transferring lighting between different people.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from tools.evaluate_reference_transfer_grid import load_groups, select_groups, rgb_float
from tools.make_reference_image_comparisons import load_mask, tensor_mask, tensor_rgb
from rrnet.config import load_config, model_kwargs
from rrnet.person_mask import suppress_boundary_gain
from rrnet.reference_mask import face_attention_from_person_mask
from rrnet.reference_relative_model import ReferenceRelativeRRNet


def luma(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    return float(value[mask].mean())


def broad_error(image: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    image_y = cv2.GaussianBlur(luma(image), (0, 0), 9.0)
    target_y = cv2.GaussianBlur(luma(target), (0, 0), 9.0)
    image_y /= max(masked_mean(image_y, mask), 1e-5)
    target_y /= max(masked_mean(target_y, mask), 1e-5)
    return masked_mean(np.abs(image_y - target_y), mask)


def panel(image: np.ndarray, label: str) -> np.ndarray:
    out = cv2.cvtColor(np.uint8(np.clip(image * 255 + 0.5, 0, 255)), cv2.COLOR_RGB2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, label, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def gain_panel(gain: np.ndarray, label: str, size: tuple[int, int]) -> np.ndarray:
    colored = cv2.applyColorMap(np.uint8(np.clip((gain - 0.4) / 2.6 * 255, 0, 255)), cv2.COLORMAP_TURBO)
    colored = cv2.resize(colored, (size[1], size[0]))
    cv2.rectangle(colored, (0, 0), (colored.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(colored, label, (9, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return colored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--people', type=int, default=5)
    parser.add_argument('--categories', nargs='+', default=[
        'warm_side_light', 'top_light_shadow', 'window_backlight', 'warm_overexposure'])
    args = parser.parse_args()

    root, output_dir = Path(args.dataset_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups, _, _, _ = load_groups(root, {'val', 'test'}, set())
    indices = select_groups(groups, args.people, 1, 20260926)
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ReferenceRelativeRRNet(**model_kwargs(cfg, args.config)).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    rows: list[dict[str, float | str]] = []
    preview_rows: list[np.ndarray] = []

    with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                dtype=torch.float16, enabled=device.type == 'cuda'):
        for position, index in enumerate(indices):
            source_group = groups[index]
            identity = source_group['identity']
            reference_group = groups[indices[(position + 1) % len(indices)]]
            reference_row = reference_group['identity']
            clean = rgb_float(root / identity['clean_frame'])
            person_mask = load_mask(root / identity['relight_mask'])
            face_mask = face_attention_from_person_mask(person_mask)
            binary_face = face_mask[..., 0] > 0.5
            if not binary_face.any():
                raise ValueError(f'Empty face mask: {identity["clean_frame"]}')
            blend_mask = suppress_boundary_gain(person_mask, fade_pixels=8)
            reference = rgb_float(root / reference_row['clean_frame'])
            ref_person_mask = load_mask(root / reference_row['relight_mask'])
            ref_face_mask = face_attention_from_person_mask(ref_person_mask)
            reference_theta = model.encode_reference(
                tensor_rgb(np.uint8(reference * 255 + 0.5), device),
                tensor_mask(ref_face_mask, device))['theta']

            for category in args.categories:
                source = rgb_float(root / source_group[category]['bad_light_frame'])
                source_tensor = tensor_rgb(np.uint8(source * 255 + 0.5), device)
                source_theta = model.estimate_light(
                    source_tensor, tensor_mask(face_mask, device))['theta']
                source_depth = model.depth(source_tensor)
                predicted = model.transfer(source_tensor, source_depth,
                                           source_theta, reference_theta,
                                           tensor_mask(blend_mask, device))
                output = predicted['output'][0].float().permute(1, 2, 0).cpu().numpy()
                gain = predicted['transfer_gain'][0, 0].float().cpu().numpy()
                source_y, target_y = luma(source), luma(clean)
                optimal_scalar = float(np.sum(source_y[binary_face] * target_y[binary_face]) /
                                       max(np.sum(source_y[binary_face] ** 2), 1e-8))
                optimal_scalar = float(np.clip(optimal_scalar, model.min_transfer_gain,
                                               model.max_transfer_gain))
                scalar_gain = np.minimum(optimal_scalar,
                                         model.output_headroom / np.maximum(source.max(axis=2), 1e-4))
                scalar_output = source * (1 - blend_mask) + np.clip(source * scalar_gain[..., None], 0, 1) * blend_mask
                ideal_gain = np.clip(target_y / np.maximum(source_y, 0.025),
                                     model.min_transfer_gain, model.max_transfer_gain)
                valid = binary_face & (source_y > 0.04) & (target_y > 0.04)
                log_gain = np.log(np.maximum(gain[valid], 1e-5))
                log_ideal = np.log(np.maximum(ideal_gain[valid], 1e-5))
                correlation = float(np.corrcoef(log_gain, log_ideal)[0, 1]) if (
                    valid.sum() > 10 and log_gain.std() > 1e-5 and log_ideal.std() > 1e-5) else float('nan')
                model_error = broad_error(output, clean, binary_face)
                scalar_error = broad_error(scalar_output, clean, binary_face)
                row = {
                    'person': identity['person_id'],
                    'category': category,
                    'reference_person': reference_row['person_id'],
                    'source_frame': source_group[category]['bad_light_frame'],
                    'target_frame': identity['clean_frame'],
                    'model_broad_error': model_error,
                    'optimal_scalar_broad_error': scalar_error,
                    'model_vs_scalar_improvement_pct': 100 * (scalar_error - model_error) / max(scalar_error, 1e-6),
                    'gain_direction_correlation': correlation,
                    'predicted_gain_std_face': float(gain[binary_face].std()),
                    'ideal_gain_std_face': float(ideal_gain[binary_face].std()),
                    'predicted_gain_p05_face': float(np.quantile(gain[binary_face], 0.05)),
                    'predicted_gain_p95_face': float(np.quantile(gain[binary_face], 0.95)),
                    'ideal_gain_p05_face': float(np.quantile(ideal_gain[binary_face], 0.05)),
                    'ideal_gain_p95_face': float(np.quantile(ideal_gain[binary_face], 0.95)),
                }
                rows.append(row)
                if position == 0:
                    size = source.shape[:2]
                    preview_rows.append(np.hstack([
                        panel(source, f'{category} INPUT'),
                        panel(scalar_output, 'OPTIMAL SCALAR'),
                        panel(output, f'MODEL {model_error:.3f}'),
                        panel(clean, 'CLEAN TARGET'),
                        gain_panel(gain, 'PREDICTED GAIN', size),
                        gain_panel(ideal_gain, 'APPROX IDEAL GAIN', size),
                    ]))

    with (output_dir / 'per_case.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for category in args.categories:
        subset = [r for r in rows if r['category'] == category]
        summary[category] = {
            'samples': len(subset),
            'model_beats_optimal_scalar': sum(r['model_broad_error'] < r['optimal_scalar_broad_error'] for r in subset),
            'mean_model_broad_error': float(np.mean([r['model_broad_error'] for r in subset])),
            'mean_optimal_scalar_broad_error': float(np.mean([r['optimal_scalar_broad_error'] for r in subset])),
            'mean_gain_direction_correlation': float(np.nanmean([r['gain_direction_correlation'] for r in subset])),
            'mean_predicted_gain_std_face': float(np.mean([r['predicted_gain_std_face'] for r in subset])),
            'mean_ideal_gain_std_face': float(np.mean([r['ideal_gain_std_face'] for r in subset])),
        }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    cv2.imwrite(str(output_dir / 'preview_first_person.jpg'), np.vstack(preview_rows),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {output_dir}')


if __name__ == '__main__':
    main()
