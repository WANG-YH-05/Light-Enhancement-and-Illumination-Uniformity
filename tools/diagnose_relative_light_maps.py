"""Visualize whether reference-relative RRNet uses spatial virtual lighting.

This is a read-only diagnostic: it never changes a checkpoint or model config.
For each source image, it saves the source light, reference light rendered on
the source geometry, effective transfer gain, and a numeric ambient fraction.
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rrnet.config import load_config, model_kwargs
from rrnet.lighting import unpack_lights
from rrnet.losses import luminance
from rrnet.reference_relative_model import ReferenceRelativeRRNet


def read_rgb(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    if value.shape != size:
        value = cv2.resize(value, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    return (value.astype(np.float32) / 255.0)[..., None]


def to_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def map_stats(value: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    samples = value[mask > 0.5]
    if samples.numel() == 0:
        raise ValueError("Diagnostic mask has no foreground pixels.")
    mean = float(samples.mean().item())
    std = float(samples.std(unbiased=False).item())
    return {"mean": mean, "std": std, "cv": std / max(abs(mean), 1.0e-6),
            "p05": float(torch.quantile(samples, 0.05).item()),
            "p95": float(torch.quantile(samples, 0.95).item())}


def colorize(value: np.ndarray, lower: float, upper: float, title: str) -> np.ndarray:
    clipped = np.clip((value - lower) / max(upper - lower, 1.0e-6), 0.0, 1.0)
    bgr = cv2.applyColorMap(np.uint8(clipped * 255.0), cv2.COLORMAP_TURBO)
    cv2.putText(bgr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr


def rgb_panel(rgb: np.ndarray, title: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return bgr


def add_metrics(panel: np.ndarray, lines: list[str]) -> np.ndarray:
    result = panel.copy()
    y = 54
    for line in lines:
        cv2.putText(result, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (255, 255, 255), 2, cv2.LINE_AA)
        y += 22
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--source-root", required=True,
                        help="Clip folder containing degraded_frame/<category>/000000.png.")
    parser.add_argument("--mask", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--frame", default="000000.png")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = args.precision == "fp16" and device.type == "cuda"
    config = load_config(args.config)
    model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    reference_rgb = read_rgb(Path(args.reference))
    reference = to_tensor(reference_rgb, device)
    source_root = Path(args.source_root)
    rows: list[dict[str, float | str]] = []
    rendered_rows: list[np.ndarray] = []

    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
        for category in args.categories:
            source_path = source_root / "degraded_frame" / category / args.frame
            source_rgb = read_rgb(source_path)
            source = to_tensor(source_rgb, device)
            mask_rgb = read_mask(Path(args.mask), source_rgb.shape[:2])
            mask = torch.from_numpy(mask_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

            prediction = model(source, reference, mask, mask, mask)
            source_light = prediction["source_illumination"][:, 0]
            reference_light = prediction["reference_illumination_on_source"][:, 0]
            gain = prediction["transfer_gain"][:, 0]
            output = prediction["output"][0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            theta = prediction["source_theta"]
            ambient_rgb = unpack_lights(theta, model.base.lprm.num_lights)["ambient"]
            ambient = luminance(ambient_rgb[:, :, None, None])[:, 0]
            source_stats = map_stats(source_light, mask[:, 0])
            ref_stats = map_stats(reference_light, mask[:, 0])
            gain_stats = map_stats(gain, mask[:, 0])
            # This is a signed diagnostic ratio, not a physical percentage:
            # virtual-light contributions can be negative because their range
            # is only softly regularized during training.
            ambient_to_total_mean = float((ambient / source_light.masked_select(mask[:, 0] > 0.5).mean().clamp_min(1.0e-6)).item())
            min_clip_fraction = float(((gain <= model.min_transfer_gain + 1.0e-4) & (mask[:, 0] > 0.5)).sum().item() /
                                      (mask[:, 0] > 0.5).sum().item())
            max_clip_fraction = float(((gain >= model.max_transfer_gain - 1.0e-4) & (mask[:, 0] > 0.5)).sum().item() /
                                      (mask[:, 0] > 0.5).sum().item())

            row: dict[str, float | str] = {
                "category": category,
                "ambient_luma": float(ambient.item()),
                "ambient_to_source_light_mean": ambient_to_total_mean,
                "source_light_mean": source_stats["mean"], "source_light_std": source_stats["std"], "source_light_cv": source_stats["cv"],
                "reference_light_mean": ref_stats["mean"], "reference_light_std": ref_stats["std"], "reference_light_cv": ref_stats["cv"],
                "gain_mean": gain_stats["mean"], "gain_std": gain_stats["std"], "gain_cv": gain_stats["cv"],
                "gain_p05": gain_stats["p05"], "gain_p95": gain_stats["p95"],
                "gain_at_min_fraction": min_clip_fraction, "gain_at_max_fraction": max_clip_fraction,
            }
            rows.append(row)
            source_map = source_light[0].float().cpu().numpy()
            ref_map = reference_light[0].float().cpu().numpy()
            gain_map = gain[0].float().cpu().numpy()
            rendered_rows.append(np.hstack((
                rgb_panel(source_rgb, f"INPUT: {category}"),
                add_metrics(colorize(source_map, 0.0, 2.0, "SOURCE LIGHT"), [
                    f"mean={source_stats['mean']:.3f} std={source_stats['std']:.3f}",
                    f"ambient/total mean={ambient_to_total_mean:.1%}",
                ]),
                add_metrics(colorize(ref_map, 0.0, 2.0, "REF LIGHT ON SOURCE"), [
                    f"mean={ref_stats['mean']:.3f} std={ref_stats['std']:.3f}",
                    f"CV={ref_stats['cv']:.3f}",
                ]),
                add_metrics(colorize(gain_map, 0.4, 5.0, "EFFECTIVE GAIN"), [
                    f"mean={gain_stats['mean']:.3f} std={gain_stats['std']:.3f}",
                    f"CV={gain_stats['cv']:.3f}  P05-P95={gain_stats['p05']:.2f}-{gain_stats['p95']:.2f}",
                ]),
                rgb_panel(np.uint8(np.rint(output * 255.0)), "MODEL OUTPUT"),
            )))

    board = np.vstack(rendered_rows)
    board_path = output_dir / "relative_light_diagnostic.jpg"
    cv2.imwrite(str(board_path), board, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"Saved board: {board_path}")


if __name__ == "__main__":
    main()
