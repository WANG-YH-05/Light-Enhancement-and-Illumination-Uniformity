"""Fixed-reference, exact synthetic relighting check on MEAD validation people."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

MODEL_ROOT = Path(__file__).resolve().parents[1]
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from rrnet.config import load_config, model_kwargs
from rrnet.person_mask import suppress_boundary_gain
from rrnet.reference_mask import face_attention_from_person_mask
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from tools.evaluate_reference_transfer_grid import load_groups, make_target, rgb_float
from tools.make_reference_image_comparisons import load_mask, tensor_mask, tensor_rgb


def luma(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def broad_error(image: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    out_y = cv2.GaussianBlur(luma(image), (0, 0), 9.0)
    tgt_y = cv2.GaussianBlur(luma(target), (0, 0), 9.0)
    out_y /= max(float(out_y[mask].mean()), 1e-5)
    tgt_y /= max(float(tgt_y[mask].mean()), 1e-5)
    return float(np.abs(out_y[mask] - tgt_y[mask]).mean())


def image_tile(rgb: np.ndarray, header: str, footer: str, width: int = 224) -> Image.Image:
    image = Image.fromarray(np.uint8(np.clip(rgb * 255 + 0.5, 0, 255)))
    image.thumbnail((width, width), Image.Resampling.LANCZOS)
    tile = Image.new('RGB', (width, width + 52), '#111318')
    tile.paste(image, ((width-image.width)//2, 30+(width-image.height)//2))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, width, 28), fill='black')
    draw.text((6, 6), header[:30], fill='white')
    draw.text((6, width + 32), footer[:38], fill='#E0E3E8')
    return tile


def map_tile(value: np.ndarray, header: str, width: int = 224) -> Image.Image:
    colored = cv2.applyColorMap(np.uint8(np.clip((value - 0.4) / 2.6 * 255, 0, 255)),
                                cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return image_tile(colored, header, 'blue < 0.4, red > 3.0', width)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reference-person', default='MEAD_front_11')
    parser.add_argument('--source-people', nargs='+', default=['MEAD_front_14', 'MEAD_front_37'])
    parser.add_argument('--reference-categories', nargs='+', default=['identity', 'warm_side_light'])
    parser.add_argument('--source-categories', nargs='+', default=[
        'warm_side_light', 'top_light_shadow', 'window_backlight', 'warm_overexposure'])
    args = parser.parse_args()

    root, out_dir = Path(args.dataset_root), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups, profiles, clip_configs, progress = load_groups(root, {'val'}, set())
    by_person: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        person = group['identity']['person_id']
        by_person.setdefault(person, []).append(index)
    if args.reference_person not in by_person:
        raise ValueError(f'Reference person is not in validation split: {args.reference_person}')
    for person in args.source_people:
        if person not in by_person:
            raise ValueError(f'Source person is not in validation split: {person}')
    ref_index = by_person[args.reference_person][0]
    ref_group = groups[ref_index]
    ref_identity = ref_group['identity']
    ref_mask = load_mask(root / ref_identity['relight_mask'])
    ref_attention = face_attention_from_person_mask(ref_mask)

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fp16 = device.type == 'cuda'
    model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    ref_clean = rgb_float(root / ref_identity['clean_frame'])
    identity_ref_tensor = tensor_rgb(np.uint8(ref_clean * 255 + 0.5), device)
    ref_mask_tensor = tensor_mask(ref_attention, device)

    result_rows: list[dict[str, object]] = []
    for ref_category in args.reference_categories:
        ref_row = ref_group[ref_category]
        ref_image = rgb_float(root / ref_row['bad_light_frame'])
        with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                     dtype=torch.float16, enabled=fp16):
            ref_theta = model.encode_reference(
                tensor_rgb(np.uint8(ref_image * 255 + 0.5), device), ref_mask_tensor)['theta']

        tiles_by_source: list[Image.Image] = []
        for source_person in args.source_people:
            source_index = by_person[source_person][0]
            source_group = groups[source_index]
            identity = source_group['identity']
            clean = rgb_float(root / identity['clean_frame'])
            person_mask = load_mask(root / identity['relight_mask'])
            face_mask = face_attention_from_person_mask(person_mask)
            face = face_mask[..., 0] > 0.5
            output_mask = suppress_boundary_gain(person_mask, fade_pixels=8)

            for category in args.source_categories:
                source = rgb_float(root / source_group[category]['bad_light_frame'])
                target, _, _ = make_target(root, source_group, ref_group, ref_index,
                                           ref_category, profiles, clip_configs, progress)
                source_tensor = tensor_rgb(np.uint8(source * 255 + 0.5), device)
                with torch.inference_mode(), torch.autocast(device_type=device.type,
                                                             dtype=torch.float16, enabled=fp16):
                    theta = model.estimate_light(source_tensor, tensor_mask(face_mask, device))['theta']
                    depth = model.depth(source_tensor)
                    prediction = model.transfer(source_tensor, depth, theta, ref_theta,
                                                tensor_mask(output_mask, device))
                output = prediction['output'][0].float().permute(1, 2, 0).cpu().numpy()
                gain = prediction['transfer_gain'][0, 0].float().cpu().numpy()
                source_y, target_y = luma(source), luma(target)
                scalar = float(np.sum(source_y[face] * target_y[face]) /
                               max(float(np.sum(source_y[face] ** 2)), 1e-8))
                scalar = float(np.clip(scalar, model.min_transfer_gain, model.max_transfer_gain))
                scalar_map = np.minimum(scalar, model.output_headroom /
                                        np.maximum(source.max(axis=2), 1e-4))
                scalar_output = source * (1 - output_mask) + np.clip(
                    source * scalar_map[..., None], 0, 1) * output_mask
                ideal_gain = np.clip(target_y / np.maximum(source_y, 0.025),
                                     model.min_transfer_gain, model.max_transfer_gain)
                valid = face & (source_y > 0.04) & (target_y > 0.04)
                lg, li = np.log(np.maximum(gain[valid], 1e-5)), np.log(np.maximum(ideal_gain[valid], 1e-5))
                corr = float(np.corrcoef(lg, li)[0, 1]) if lg.std() > 1e-5 and li.std() > 1e-5 else float('nan')
                model_error = broad_error(output, target, face)
                scalar_error = broad_error(scalar_output, target, face)
                row = {
                    'reference_person': args.reference_person,
                    'reference_category': ref_category,
                    'reference_image': ref_row['bad_light_frame'],
                    'source_person': source_person,
                    'source_category': category,
                    'source_image': source_group[category]['bad_light_frame'],
                    'source_clean_frame': identity['clean_frame'],
                    'target_construction': ('source clean with reference scenario parameters'
                                            if ref_category != 'identity'
                                            else 'source clean; identity reference'),
                    'broad_error_model': model_error,
                    'broad_error_best_global_exposure': scalar_error,
                    'model_improvement_vs_global_exposure_pct': 100 * (scalar_error-model_error) / max(scalar_error, 1e-6),
                    'gain_direction_correlation': corr,
                    'predicted_gain_std_face': float(gain[face].std()),
                    'target_gain_std_face': float(ideal_gain[face].std()),
                }
                result_rows.append(row)
                footer = f'{source_person} {category} err={model_error:.3f} vs exp={scalar_error:.3f}'
                tiles_by_source.extend([
                    image_tile(source, 'INPUT', footer),
                    image_tile(ref_image, f'REF {ref_category}', args.reference_person),
                    image_tile(target, 'PAIRED TARGET', 'source identity + reference light'),
                    image_tile(output, 'MODEL OUTPUT', f'gain corr={corr:.2f}'),
                    map_tile(gain, 'PREDICTED GAIN'),
                    map_tile(ideal_gain, 'TARGET GAIN'),
                ])

        columns = 6
        rows = [Image.new('RGB', (224 * columns, 276), '#111318')
                for _ in range((len(tiles_by_source) + columns - 1) // columns)]
        for tile_i, tile in enumerate(tiles_by_source):
            rows[tile_i // columns].paste(tile, ((tile_i % columns) * 224, 0))
        board = Image.new('RGB', (224 * columns, 276 * len(rows) + 36), '#0B0D12')
        ImageDraw.Draw(board).text((10, 8),
            f'VAL fixed reference: {args.reference_person} / {ref_category} | synthetic target uses exact reference-light parameters',
            fill='white')
        for row_i, row_image in enumerate(rows):
            board.paste(row_image, (0, 36 + 276 * row_i))
        board.save(out_dir / f'fixed_ref_{ref_category}.jpg', quality=94)

    fields = list(result_rows[0])
    with (out_dir / 'per_case.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)
    print(f'Validation identities only: reference={args.reference_person}; sources={args.source_people}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Results: {out_dir}')


if __name__ == '__main__':
    main()
