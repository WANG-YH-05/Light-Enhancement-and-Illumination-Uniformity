"""Evaluate reference-conditioned relighting on exact synthetic triplets.

For every source-light/target-light category pair, this script:

1. reads a held-out source person under the selected source light;
2. reads a different held-out person under the selected reference light;
3. applies the reference clip's exact generation parameters and temporal phase
   to the source person's clean frame to construct the paired target; and
4. compares the model output with that same-identity target.

The output contains per-sample metrics, a category-by-category summary, metric
heatmaps, and contact sheets for visual inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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
from rrnet.reference_data import source_chroma_with_target_luminance
from rrnet.reference_mask import face_attention_from_person_mask
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from tools.build_mead_rrnet_dataset_v2 import apply_scenario
from tools.make_reference_image_comparisons import (
    load_mask,
    load_rgb,
    tensor_mask,
    tensor_rgb,
)


DEFAULT_CATEGORIES = (
    "identity",
    "underexposed_cool",
    "window_backlight",
    "warm_side_light",
    "top_light_shadow",
    "warm_overexposure",
    "mixed_office",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=("val", "test"))
    parser.add_argument("--exclude-person", action="append", default=[])
    parser.add_argument("--people", type=int, default=0,
                        help="Maximum identities to evaluate; 0 uses all selected identities.")
    parser.add_argument("--samples-per-person", type=int, default=1)
    parser.add_argument("--source-categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--target-categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--boundary-fade", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def rgb_float(path: Path) -> np.ndarray:
    return load_rgb(path).astype(np.float32) / 255.0


def weighted_mean(value: np.ndarray, weight: np.ndarray) -> float:
    return float((value * weight).sum() / max(float(weight.sum()), 1.0e-12))


def psnr(output: np.ndarray, target: np.ndarray,
         mask: np.ndarray | None = None) -> float:
    error = (output - target) ** 2
    if mask is None:
        mse = float(error.mean())
    else:
        mse = float(
            (error * mask[..., None]).sum()
            / max(float(mask.sum()) * 3.0, 1.0e-12)
        )
    return 10.0 * math.log10(1.0 / max(mse, 1.0e-12))


def ssim_map(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Standard RGB SSIM map with an 11x11 Gaussian window."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    channels = []
    for channel in range(3):
        x, y = output[..., channel], target[..., channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5,
                                borderType=cv2.BORDER_REFLECT)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5,
                                borderType=cv2.BORDER_REFLECT)
        var_x = cv2.GaussianBlur(x * x, (11, 11), 1.5,
                                 borderType=cv2.BORDER_REFLECT) - mu_x * mu_x
        var_y = cv2.GaussianBlur(y * y, (11, 11), 1.5,
                                 borderType=cv2.BORDER_REFLECT) - mu_y * mu_y
        covariance = cv2.GaussianBlur(
            x * y, (11, 11), 1.5,
            borderType=cv2.BORDER_REFLECT) - mu_x * mu_y
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)
        denominator = ((mu_x * mu_x + mu_y * mu_y + c1)
                       * (var_x + var_y + c2))
        channels.append(numerator / (denominator + 1.0e-12))
    return np.mean(np.stack(channels, axis=-1), axis=-1)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to CIELAB using the D65 white point."""
    value = np.clip(rgb, 0.0, 1.0)
    linear = np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )
    x = (0.4124564 * linear[..., 0] + 0.3575761 * linear[..., 1]
         + 0.1804375 * linear[..., 2]) / 0.95047
    y = (0.2126729 * linear[..., 0] + 0.7151522 * linear[..., 1]
         + 0.0721750 * linear[..., 2])
    z = (0.0193339 * linear[..., 0] + 0.1191920 * linear[..., 1]
         + 0.9503041 * linear[..., 2]) / 1.08883
    xyz = np.stack((x, y, z), axis=-1)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta ** 3,
        np.cbrt(xyz),
        xyz / (3.0 * delta * delta) + 4.0 / 29.0,
    )
    return np.stack((
        116.0 * transformed[..., 1] - 16.0,
        500.0 * (transformed[..., 0] - transformed[..., 1]),
        200.0 * (transformed[..., 1] - transformed[..., 2]),
    ), axis=-1)


def lpips_tensor(rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        .to(device=device, dtype=torch.float32) * 2.0 - 1.0
    )


def load_groups(root: Path, splits: set[str], excluded: set[str]) -> tuple[
        list[dict[str, dict[str, str]]], dict[str, dict[str, object]],
        dict[str, dict[str, object]], dict[int, float]]:
    with (root / "metadata.csv").open(
            newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["split"] in splits and row["person_id"] not in excluded]
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["clean_frame"]][row["degradation_id"]] = row
    groups = [grouped[key] for key in sorted(grouped)]
    if not groups:
        raise ValueError("No dataset rows matched the requested splits/people.")
    required = set(DEFAULT_CATEGORIES)
    incomplete = [group["identity"]["clean_frame"] for group in groups
                  if not required.issubset(group)]
    if incomplete:
        raise ValueError(f"Incomplete degradation group: {incomplete[0]}")

    manifest = json.loads(
        (root / "generation_manifest.json").read_text(encoding="utf-8"))
    profiles = manifest["people"]
    clip_configs = {
        clip_key: {str(item["category"]): dict(item) for item in configs}
        for clip_key, configs in manifest["clips"].items()
    }
    clips: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        row = group["identity"]
        key = f'{row["split"]}/{row["person_id"]}/{row["clip_id"]}'
        clips[key].append(index)
    progress: dict[int, float] = {}
    for indices in clips.values():
        indices.sort(key=lambda index: groups[index]["identity"]["frame_id"])
        denominator = max(len(indices) - 1, 1)
        for position, index in enumerate(indices):
            progress[index] = position / denominator
    return groups, profiles, clip_configs, progress


def select_groups(groups: list[dict[str, dict[str, str]]], people: int,
                  samples_per_person: int, seed: int) -> list[int]:
    by_person: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        by_person[group["identity"]["person_id"]].append(index)
    identities = sorted(by_person)
    if people:
        identities = identities[:people]
    if len(identities) < 2:
        raise ValueError("Evaluation requires at least two different identities.")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for person in identities:
        candidates = by_person[person]
        count = min(samples_per_person, len(candidates))
        positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
        # A deterministic offset avoids selecting only the first clip/frame.
        offset = int(rng.integers(0, max(len(candidates), 1)))
        selected.extend(candidates[(int(position) + offset) % len(candidates)]
                        for position in positions)
    return selected


def make_target(root: Path, source_group: dict[str, dict[str, str]],
                reference_group: dict[str, dict[str, str]],
                reference_index: int, target_category: str,
                profiles: dict[str, dict[str, object]],
                clip_configs: dict[str, dict[str, dict[str, object]]],
                progress: dict[int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_identity = source_group["identity"]
    reference_identity = reference_group["identity"]
    clean = rgb_float(root / source_identity["clean_frame"])
    person_mask = load_mask(root / source_identity["relight_mask"])[..., 0]
    skin_mask = load_mask(root / source_identity["skin_mask"])[..., 0]
    if target_category == "identity":
        return clean, person_mask, skin_mask
    clip_key = (f'{reference_identity["split"]}/'
                f'{reference_identity["person_id"]}/'
                f'{reference_identity["clip_id"]}')
    config = dict(clip_configs[clip_key][target_category])
    config["profile"] = profiles[reference_identity["person_id"]]
    lit = apply_scenario(
        clean, person_mask[..., None], config, progress[reference_index])
    target = source_chroma_with_target_luminance(clean, lit)
    return target, person_mask, skin_mask


def image_tile(rgb: np.ndarray, heading: str, detail: str,
               size: int = 220) -> Image.Image:
    array = np.uint8(np.clip(rgb * 255.0 + 0.5, 0.0, 255.0))
    value = Image.fromarray(array)
    value.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 54), "#101218")
    tile.paste(value, ((size - value.width) // 2, 34 + (size - value.height) // 2))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size, 34), fill="#000000")
    draw.text((7, 7), heading, font=font(16, True), fill="white")
    draw.text((5, size + 38), detail, font=font(12), fill="#D7DCE5")
    return tile


def heat_color(value: float, low: float, high: float,
               higher_is_better: bool) -> tuple[int, int, int]:
    ratio = float(np.clip((value - low) / max(high - low, 1.0e-12), 0.0, 1.0))
    if not higher_is_better:
        ratio = 1.0 - ratio
    return (round(205 - 120 * ratio), round(65 + 120 * ratio), round(55 + 65 * ratio))


def write_heatmaps(path: Path, summary: list[dict[str, object]],
                   source_categories: list[str], target_categories: list[str]) -> None:
    metrics = (
        ("person_psnr", "Person PSNR", True),
        ("person_ssim", "Person SSIM", True),
        ("lpips", "LPIPS Alex", False),
        ("skin_delta_e76", "Skin DeltaE76", False),
        ("gain_log_mae", "Gain log-MAE", False),
        ("gain_mean_error", "Mean gain error", False),
    )
    lookup = {(str(row["source_category"]), str(row["target_category"])): row
              for row in summary}
    cell_w, cell_h, label_w, label_h = 145, 54, 190, 70
    panel_w = label_w + cell_w * len(target_categories)
    panel_h = label_h + cell_h * len(source_categories)
    canvas = Image.new("RGB", (panel_w * 2 + 30, panel_h * 3 + 60), "#0B0D12")
    draw = ImageDraw.Draw(canvas)
    for metric_index, (key, title, higher) in enumerate(metrics):
        panel_x = (metric_index % 2) * (panel_w + 30)
        panel_y = (metric_index // 2) * (panel_h + 20)
        values = [float(row[key]) for row in summary]
        low, high = min(values), max(values)
        draw.text((panel_x + 4, panel_y + 4), title,
                  font=font(19, True), fill="white")
        for column, category in enumerate(target_categories):
            draw.text((panel_x + label_w + column * cell_w + 5, panel_y + 34),
                      category[:18], font=font(11), fill="#F6D365")
        for row_index, source_category in enumerate(source_categories):
            y = panel_y + label_h + row_index * cell_h
            draw.text((panel_x + 4, y + 17), source_category[:23],
                      font=font(12), fill="#D7DCE5")
            for column, target_category in enumerate(target_categories):
                value = float(lookup[(source_category, target_category)][key])
                x = panel_x + label_w + column * cell_w
                draw.rectangle((x, y, x + cell_w - 3, y + cell_h - 3),
                               fill=heat_color(value, low, high, higher))
                draw.text((x + 9, y + 16), f"{value:.4f}",
                          font=font(13, True), fill="white")
    canvas.save(path, quality=96)


def main() -> None:
    args = parse_args()
    if args.samples_per_person < 1:
        raise ValueError("--samples-per-person must be positive.")
    root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_categories = list(args.source_categories)
    target_categories = list(args.target_categories)
    unknown = (set(source_categories) | set(target_categories)) - set(DEFAULT_CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown categories: {sorted(unknown)}")

    try:
        import lpips
    except ImportError as error:
        raise RuntimeError(
            "LPIPS evaluation requires `pip install lpips scipy`."
        ) from error

    groups, profiles, clip_configs, progress = load_groups(
        root, set(args.splits), set(args.exclude_person))
    source_indices = select_groups(
        groups, args.people, args.samples_per_person, args.seed)
    people_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        people_to_indices[group["identity"]["person_id"]].append(index)
    reference_people = sorted(people_to_indices)

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.precision == "fp16" and device.type == "cuda"
    model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state)
    perceptual = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    results: list[dict[str, object]] = []
    examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    total = len(source_indices) * len(source_categories) * len(target_categories)
    completed = 0
    for source_position, source_index in enumerate(source_indices):
        source_group = groups[source_index]
        source_identity = source_group["identity"]
        source_person = source_identity["person_id"]
        source_clean = rgb_float(root / source_identity["clean_frame"])
        person_mask_3d = load_mask(root / source_identity["relight_mask"])
        person_mask = person_mask_3d[..., 0].astype(np.float32)
        face_mask = face_attention_from_person_mask(person_mask_3d)
        output_mask = suppress_boundary_gain(
            person_mask_3d, fade_pixels=args.boundary_fade)

        for source_category in source_categories:
            source = rgb_float(root / source_group[source_category]["bad_light_frame"])
            source_tensor = tensor_rgb(
                np.uint8(np.clip(source * 255.0 + 0.5, 0.0, 255.0)), device)
            source_face_tensor = tensor_mask(face_mask, device)
            with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                source_theta = model.estimate_light(
                    source_tensor, source_face_tensor)["theta"]
                source_depth = model.depth(source_tensor)

            for target_position, target_category in enumerate(target_categories):
                # Rotate reference identities across source/target cells while
                # guaranteeing that source and reference people differ.
                ref_person_position = (
                    source_position + target_position + 1) % len(reference_people)
                reference_person = reference_people[ref_person_position]
                if reference_person == source_person:
                    reference_person = reference_people[
                        (ref_person_position + 1) % len(reference_people)]
                candidates = people_to_indices[reference_person]
                reference_index = candidates[
                    (source_position + target_position) % len(candidates)]
                reference_group = groups[reference_index]
                reference_row = reference_group[target_category]
                reference = rgb_float(root / reference_row["bad_light_frame"])
                reference_identity = reference_group["identity"]
                reference_person_mask = load_mask(
                    root / reference_identity["relight_mask"])
                reference_face_mask = face_attention_from_person_mask(
                    reference_person_mask)
                target, _, skin_mask = make_target(
                    root, source_group, reference_group, reference_index,
                    target_category, profiles, clip_configs, progress)

                reference_tensor = tensor_rgb(
                    np.uint8(np.clip(reference * 255.0 + 0.5, 0.0, 255.0)), device)
                with torch.inference_mode(), torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                    reference_theta = model.encode_reference(
                        reference_tensor,
                        tensor_mask(reference_face_mask, device))["theta"]
                    prediction = model.transfer(
                        source_tensor, source_depth, source_theta,
                        reference_theta, tensor_mask(output_mask, device))
                    output_tensor = prediction["output"]
                    predicted_gain = prediction["transfer_gain"][:, 0]
                output = output_tensor[0].permute(1, 2, 0).float().cpu().numpy()
                output = np.clip(output, 0.0, 1.0)

                ssim = ssim_map(output, target)
                output_lab, target_lab = rgb_to_lab(output), rgb_to_lab(target)
                lab_difference = np.abs(output_lab - target_lab)
                delta_e = np.sqrt(((output_lab - target_lab) ** 2).sum(axis=-1))
                source_luma = (0.2126 * source[..., 0] + 0.7152 * source[..., 1]
                               + 0.0722 * source[..., 2])
                target_luma = (0.2126 * target[..., 0] + 0.7152 * target[..., 1]
                               + 0.0722 * target[..., 2])
                target_gain = np.clip(
                    target_luma / np.maximum(source_luma, 0.025),
                    model.min_transfer_gain, model.max_transfer_gain)
                gain = predicted_gain[0].float().cpu().numpy()
                gain_weight = face_mask[..., 0].astype(np.float32)
                gain_log_error = np.abs(
                    np.log(np.maximum(gain, 1.0e-6))
                    - np.log(np.maximum(target_gain, 1.0e-6)))
                with torch.inference_mode():
                    lpips_value = float(perceptual(
                        lpips_tensor(output, device),
                        lpips_tensor(target, device)).item())

                row: dict[str, object] = {
                    "source_person": source_person,
                    "reference_person": reference_person,
                    "source_category": source_category,
                    "target_category": target_category,
                    "source_frame": source_identity["clean_frame"],
                    "reference_frame": reference_row["bad_light_frame"],
                    "full_psnr": psnr(output, target),
                    "full_ssim": float(ssim.mean()),
                    "person_psnr": psnr(output, target, person_mask),
                    "person_ssim": weighted_mean(ssim, person_mask),
                    "lpips": lpips_value,
                    "skin_delta_l": weighted_mean(lab_difference[..., 0], skin_mask),
                    "skin_delta_a": weighted_mean(lab_difference[..., 1], skin_mask),
                    "skin_delta_b": weighted_mean(lab_difference[..., 2], skin_mask),
                    "skin_delta_e76": weighted_mean(delta_e, skin_mask),
                    "predicted_gain_mean": weighted_mean(gain, gain_weight),
                    "target_gain_mean": weighted_mean(target_gain, gain_weight),
                    "gain_mean_error": abs(
                        weighted_mean(gain, gain_weight)
                        - weighted_mean(target_gain, gain_weight)),
                    "gain_log_mae": weighted_mean(gain_log_error, gain_weight),
                }
                results.append(row)
                if len(examples[source_category]) < len(target_categories):
                    examples[source_category].append({
                        **row,
                        "source": source,
                        "reference": reference,
                        "target": target,
                        "output": output,
                    })
                completed += 1
                if completed % 25 == 0 or completed == total:
                    print(f"Evaluated {completed}/{total}", flush=True)

    per_sample_path = output_dir / "per_sample_metrics.csv"
    with per_sample_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    metric_keys = (
        "full_psnr", "full_ssim", "person_psnr", "person_ssim", "lpips",
        "skin_delta_l", "skin_delta_a", "skin_delta_b", "skin_delta_e76",
        "predicted_gain_mean", "target_gain_mean", "gain_mean_error",
        "gain_log_mae",
    )
    summary: list[dict[str, object]] = []
    for source_category in source_categories:
        for target_category in target_categories:
            subset = [row for row in results
                      if row["source_category"] == source_category
                      and row["target_category"] == target_category]
            aggregate: dict[str, object] = {
                "source_category": source_category,
                "target_category": target_category,
                "count": len(subset),
            }
            aggregate.update({
                key: float(np.mean([float(row[key]) for row in subset]))
                for key in metric_keys
            })
            summary.append(aggregate)
    summary_path = output_dir / "category_matrix_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    write_heatmaps(
        output_dir / "category_matrix_heatmaps.jpg", summary,
        source_categories, target_categories)
    for source_category, items in examples.items():
        rows = []
        for item in items:
            rows.append(Image.new("RGB", (220 * 4 + 18, 274), "#0B0D12"))
            for column, (key, heading) in enumerate((
                    ("source", "INPUT"), ("reference", "REFERENCE"),
                    ("target", "PAIRED TARGET"), ("output", "OUTPUT"))):
                detail = (
                    f'{item["source_person"]} | {item["source_category"]}'
                    if key == "source" else
                    f'{item["reference_person"]} | {item["target_category"]}'
                    if key == "reference" else
                    f'PSNR={float(item["person_psnr"]):.2f} '
                    f'DE={float(item["skin_delta_e76"]):.2f}'
                    if key == "output" else
                    f'source identity under {item["target_category"]}'
                )
                rows[-1].paste(image_tile(
                    item[key], heading, detail), (column * 226, 0))
        sheet = Image.new("RGB", (rows[0].width, len(rows) * rows[0].height + 55),
                          "#0B0D12")
        draw = ImageDraw.Draw(sheet)
        draw.text((12, 13), f"SOURCE LIGHT: {source_category}",
                  font=font(24, True), fill="#F6D365")
        for index, row_image in enumerate(rows):
            sheet.paste(row_image, (0, 55 + index * row_image.height))
        sheet.save(output_dir / f"examples_{source_category}.jpg", quality=96)

    overall = {
        key: float(np.mean([float(row[key]) for row in results]))
        for key in metric_keys
    }
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "config": str(Path(args.config).resolve()),
        "splits": args.splits,
        "excluded_people": args.exclude_person,
        "source_identities": sorted({str(row["source_person"]) for row in results}),
        "source_categories": source_categories,
        "target_categories": target_categories,
        "sample_count": len(results),
        "overall": overall,
    }
    summary_json_path = output_dir / "evaluation_summary.json"
    summary_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {per_sample_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {output_dir / 'category_matrix_heatmaps.jpg'}")
    print(f"Saved: {summary_json_path}")


if __name__ == "__main__":
    main()
