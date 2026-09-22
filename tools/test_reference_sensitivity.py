"""Measure whether changing only the reference changes RRNet output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from make_reference_image_comparisons import (
    Request,
    find_row,
    fit_square,
    font,
    infer,
    load_mask,
    load_rgb,
    load_rows,
    tensor_mask,
    tensor_rgb,
)
from rrnet.config import load_config, model_kwargs
from rrnet.reference_mask import face_attention_from_person_mask
from rrnet.reference_relative_model import ReferenceRelativeRRNet


REFERENCES = [
    ("NORMAL", Request("MEAD_front_31", 1, 0, "identity")),
    ("DARK / COOL", Request("MEAD_front_31", 1, 0, "underexposed_cool")),
    ("WARM / OVEREXPOSED", Request("MEAD_front_31", 1, 0, "warm_overexposure")),
]

INPUTS = [
    Request("MEAD_front_32", 2, 10, "underexposed_cool"),
    Request("MEAD_front_8", 3, 20, "warm_overexposure"),
    Request("MEAD_front_31", 4, 30, "window_backlight"),
    Request("MEAD_front_32", 5, 40, "top_light_shadow"),
]


def luma(image: np.ndarray, mask: np.ndarray) -> float:
    rgb = image.astype(np.float32) / 255.0
    value = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    weight = np.clip(mask[..., 0], 0.0, 1.0)
    return float((value * weight).sum() / max(float(weight.sum()), 1.0))


def label_tile(canvas: Image.Image, image: np.ndarray, x: int, y: int,
               size: int, title: str, subtitle: str) -> None:
    canvas.paste(fit_square(image, size), (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + size, y + 38), fill="#000000")
    draw.text((x + 9, y + 7), title, font=font(19, True), fill="white")
    draw.text((x + 7, y + size + 5), subtitle, font=font(15), fill="#D2D7E0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(root)
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.precision == "fp16" and device.type == "cuda"
    model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)

    references = []
    for name, request in REFERENCES:
        row = find_row(rows, request)
        image = load_rgb(root / row["bad_light_frame"])
        person_mask = load_mask(root / row["relight_mask"])
        face_mask = face_attention_from_person_mask(person_mask)
        with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            theta = model.encode_reference(
                tensor_rgb(image, device), tensor_mask(face_mask, device))["theta"].detach()
        references.append((name, image, theta, luma(image, face_mask)))

    results = []
    metrics = []
    for request in INPUTS:
        row = find_row(rows, request)
        source = load_rgb(root / row["bad_light_frame"])
        person_mask = load_mask(root / row["relight_mask"])
        face_mask = face_attention_from_person_mask(person_mask)
        source_luma = luma(source, face_mask)
        outputs = []
        for ref_name, _, theta, ref_luma in references:
            output = infer(model, source, theta, person_mask, device, use_fp16)
            output_luma = luma(output, face_mask)
            outputs.append(output)
            metrics.append({
                "input_person": row["person_id"],
                "input_variant": row["variant"],
                "input_luma": f"{source_luma:.6f}",
                "reference": ref_name,
                "reference_luma": f"{ref_luma:.6f}",
                "output_luma": f"{output_luma:.6f}",
                "output_over_input": f"{output_luma / max(source_luma, 1e-6):.6f}",
            })
        results.append((row, source, source_luma, outputs))

    tile, gap = 255, 24
    left, top = 30, 105
    width = left * 2 + tile * 4 + gap * 3
    height = top + (tile + 38) * 5 + 25
    canvas = Image.new("RGB", (width, height), "#0B0D12")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 18), "REFERENCE SENSITIVITY TEST", font=font(31, True), fill="white")
    draw.text((left, 61), "Inputs stay fixed; only the reference changes", font=font(19), fill="#F6D365")
    label_tile(canvas, np.zeros((512, 512, 3), dtype=np.uint8), left, top,
               tile, "FIXED INPUTS", "Same row = same source image")
    for column, (name, image, _, ref_luma) in enumerate(references, start=1):
        x = left + column * (tile + gap)
        label_tile(canvas, image, x, top, tile, f"REFERENCE {column}",
                   f"{name} | face Y={ref_luma:.3f}")

    for row_index, (row, source, source_luma, outputs) in enumerate(results, start=1):
        y = top + row_index * (tile + 38)
        label_tile(canvas, source, left, y, tile, f"INPUT {row_index}",
                   f"{row['variant']} | face Y={source_luma:.3f}")
        for column, output in enumerate(outputs, start=1):
            x = left + column * (tile + gap)
            output_luma = float(metrics[(row_index - 1) * len(references) + column - 1]["output_luma"])
            label_tile(canvas, output, x, y, tile, f"OUTPUT R{column}",
                       f"face Y={output_luma:.3f}")

    board_path = output_dir / "reference_sensitivity_matrix.jpg"
    canvas.save(board_path, quality=96)
    csv_path = output_dir / "reference_sensitivity_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    print(f"Saved: {board_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
