"""Test one fixed reference on visually verified, distinct MEAD identities."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from make_reference_image_comparisons import (
    infer,
    load_mask,
    load_rgb,
    tensor_mask,
    tensor_rgb,
)
from rrnet.config import load_config, model_kwargs
from rrnet.reference_mask import face_attention_from_person_mask
from rrnet.reference_relative_model import ReferenceRelativeRRNet


# These identities were checked from a labelled clean-frame contact sheet.
# MEAD_front_31 and MEAD_front_32 are deliberately excluded because their
# actual images show the same person despite the different processed IDs.
SAMPLES = (
    ("MEAD_front_8", "underexposed_cool"),
    ("MEAD_front_1", "window_backlight"),
    ("MEAD_front_13", "warm_overexposure"),
    ("MEAD_front_17", "top_light_shadow"),
    ("MEAD_front_27", "warm_side_light"),
    ("MEAD_front_37", "mixed_office"),
)
REFERENCE_PERSON = "MEAD_front_0"


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_square(image: np.ndarray, size: int) -> Image.Image:
    result = Image.new("RGB", (size, size), "#111318")
    value = Image.fromarray(image)
    value.thumbnail((size, size), Image.Resampling.LANCZOS)
    result.paste(value, ((size - value.width) // 2, (size - value.height) // 2))
    return result


def luma(image: np.ndarray, mask: np.ndarray) -> float:
    rgb = image.astype(np.float32) / 255.0
    value = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    weight = np.clip(mask[..., 0], 0.0, 1.0)
    return float((value * weight).sum() / max(float(weight.sum()), 1.0))


def masked_mae(output: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    error = np.abs(output.astype(np.float32) - target.astype(np.float32)) / 255.0
    weight = np.clip(mask, 0.0, 1.0)
    return float((error * weight).sum() / max(float(weight.sum()) * 3.0, 1.0))


def label_tile(canvas: Image.Image, image: np.ndarray, x: int, y: int,
               size: int, heading: str, detail: str) -> None:
    canvas.paste(fit_square(image, size), (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((x, y, x + size, y + 34), fill="#000000")
    draw.text((x + 8, y + 7), heading, font=get_font(17, True), fill="white")
    draw.text((x + 5, y + size + 5), detail, font=get_font(14), fill="#D7DCE5")


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
    with (root / "metadata.csv").open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    def first_row(person: str, variant: str) -> dict[str, str]:
        matches = [row for row in rows
                   if row["person_id"] == person and row["variant"] == variant]
        if not matches:
            raise LookupError(f"Missing {person=} {variant=}")
        return min(matches, key=lambda row: (row["clip_id"], int(row["frame_id"])))

    reference_row = first_row(REFERENCE_PERSON, "identity")
    reference = load_rgb(root / reference_row["clean_frame"])
    reference_person_mask = load_mask(root / reference_row["relight_mask"])

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.precision == "fp16" and device.type == "cuda"
    model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    reference_mask = tensor_mask(
        face_attention_from_person_mask(reference_person_mask), device)
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        reference_theta = model.encode_reference(
            tensor_rgb(reference, device), reference_mask)["theta"].detach()

    results: list[dict[str, object]] = []
    for person, variant in SAMPLES:
        row = first_row(person, variant)
        source = load_rgb(root / row["bad_light_frame"])
        target = load_rgb(root / row["clean_frame"])
        person_mask = load_mask(root / row["relight_mask"])
        face_mask = face_attention_from_person_mask(person_mask)
        output = infer(model, source, reference_theta, person_mask, device, use_fp16)
        results.append({
            "person": person,
            "variant": variant,
            "source": source,
            "target": target,
            "output": output,
            "input_luma": luma(source, face_mask),
            "target_luma": luma(target, face_mask),
            "output_luma": luma(output, face_mask),
            "mae": masked_mae(output, target, person_mask),
        })

    tile, gap, top = 230, 24, 105
    left = 30
    width = left * 2 + tile * 4 + gap * 3
    row_height = tile + 36
    height = top + row_height * len(results) + 25
    canvas = Image.new("RGB", (width, height), "#0B0D12")
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 18), "ONE FIXED REFERENCE - SIX DISTINCT PEOPLE",
              font=get_font(29, True), fill="white")
    draw.text((left, 60),
              "Different person + different input light; target is that person's clean frame",
              font=get_font(17), fill="#F6D365")
    label_tile(canvas, reference, left, top, tile,
               "FIXED REFERENCE", f"{REFERENCE_PERSON} | clean light")

    x_input = left + tile + gap
    x_target = x_input + tile + gap
    x_output = x_target + tile + gap
    for index, item in enumerate(results):
        y = top + index * row_height
        person = str(item["person"])
        variant = str(item["variant"])
        if index > 0:
            # The reference tile occupies only the first row; leave the first
            # column dark so every comparison row remains aligned.
            draw.text((left + 8, y + 88), person,
                      font=get_font(19, True), fill="#F6D365")
            draw.text((left + 8, y + 119), variant,
                      font=get_font(14), fill="#D7DCE5")
        label_tile(canvas, item["source"], x_input, y, tile, "INPUT",
                   f"{person} | {variant} | Y={item['input_luma']:.3f}")
        label_tile(canvas, item["target"], x_target, y, tile, "CORRECT TARGET",
                   f"same identity, clean light | Y={item['target_luma']:.3f}")
        label_tile(canvas, item["output"], x_output, y, tile, "MODEL OUTPUT",
                   f"Y={item['output_luma']:.3f} | MAE={item['mae']:.3f}")

    board_path = output_dir / "same_reference_six_inputs.jpg"
    canvas.save(board_path, quality=96)
    csv_path = output_dir / "same_reference_six_inputs_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "person", "variant", "input_luma", "target_luma", "output_luma", "mae"))
        writer.writeheader()
        for item in results:
            writer.writerow({key: item[key] for key in writer.fieldnames})
    print(f"Saved: {board_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
