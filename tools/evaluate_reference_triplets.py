"""Render and measure reference-conditioned RRNet triplets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rrnet.config import load_config, model_kwargs
from rrnet.losses import luminance
from rrnet.reference_data import MEADReferenceTriplets
from rrnet.reference_model import ReferenceRRNet
from rrnet.reference_relative_model import ReferenceRelativeRRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--split", default="train")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    return parser.parse_args()


def image_from_tensor(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().float().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(array * 255.0).astype(np.uint8), "RGB")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    dataset = MEADReferenceTriplets(
        data_config["root"], args.split,
        metadata_file=data_config.get("metadata_file", "metadata.csv"),
        manifest_file=data_config.get("manifest_file", "generation_manifest.json"),
        seed=int(data_config.get("sampling_seed", 20260906)),
        variants_per_source=int(data_config.get("reference_variants_per_source", 4)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    relative = config.get("task", "").lower() == "reference_relative"
    model_class = ReferenceRelativeRRNet if relative else ReferenceRRNet
    model = model_class(**model_kwargs(config, args.config)).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    rows: list[tuple[dict, torch.Tensor]] = []
    metrics: list[dict[str, float | str | int]] = []
    use_fp16 = args.precision == "fp16" and device.type == "cuda"
    with torch.inference_mode():
        for index in range(min(args.samples, len(dataset))):
            item = dataset[index]
            source = item["input"].unsqueeze(0).to(device)
            reference = item["reference"].unsqueeze(0).to(device)
            target = item["target"].unsqueeze(0).to(device)
            relight = item["relight_mask"].unsqueeze(0).to(device)
            reference_mask = item["reference_mask"].unsqueeze(0).to(device)
            source_light_mask = item["source_light_mask"].unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_fp16):
                if relative:
                    prediction = model(
                        source, reference, relight, reference_mask,
                        source_light_mask)
                else:
                    prediction = model(source, reference, relight, reference_mask)
            output = prediction["output"]
            denominator = (relight.sum() * 3.0).clamp_min(1.0)
            mae = ((output - target).abs() * relight).sum() / denominator
            chroma = lambda rgb: rgb.clamp_min(0.0) / rgb.clamp_min(0.0).sum(
                dim=1, keepdim=True).clamp_min(1.0e-3)
            chroma_mae = ((chroma(output) - chroma(target)).abs() * relight).sum() / denominator
            mask_denominator = relight.sum().clamp_min(1.0)
            metrics.append({
                "index": index,
                "source_category": str(item["source_category"]),
                "target_category": str(item["target_category"]),
                "mae": float(mae),
                "chroma_mae": float(chroma_mae),
                "target_luma": float((luminance(target) * relight).sum() / mask_denominator),
                "output_luma": float((luminance(output) * relight).sum() / mask_denominator),
            })
            rows.append((item, output[0]))

    tile = 256
    label_height = 28
    canvas = Image.new("RGB", (tile * 4, (tile + label_height) * len(rows)), "black")
    draw = ImageDraw.Draw(canvas)
    for row_index, ((item, output), metric) in enumerate(zip(rows, metrics)):
        y = row_index * (tile + label_height)
        panels = (
            ("SOURCE", item["input"]),
            (f"REF {item['target_category']}", item["reference"]),
            ("TARGET", item["target"]),
            (f"OUTPUT mae={metric['mae']:.3f}", output),
        )
        for column, (label, tensor) in enumerate(panels):
            image = image_from_tensor(tensor).resize((tile, tile), Image.Resampling.LANCZOS)
            x = column * tile
            canvas.paste(image, (x, y + label_height))
            draw.text((x + 5, y + 6), label, fill=(255, 255, 0))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "mean_mae": float(np.mean([row["mae"] for row in metrics])),
        "mean_chroma_mae": float(np.mean([row["chroma_mae"] for row in metrics])),
        "samples": metrics,
    }
    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_path.resolve())
    print(metrics_path.resolve())


if __name__ == "__main__":
    main()
