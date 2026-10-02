"""Held-out visual/metric audit for the known-light RGB AGM pilot."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rrnet.agm_physical_stage import masked_mean
from rrnet.config import load_config
from rrnet.losses import luminance
from rrnet.lprm import resize_shorter_side
from rrnet.model import RRNet
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from train_agm_physical import load_frozen_encoder, make_dataset, physical_batch


def scalar_baseline(source: torch.Tensor, target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    source_y, target_y = luminance(source), luminance(target)
    scalar = ((source_y * target_y * mask).sum((1, 2, 3), keepdim=True)
              / (source_y.square() * mask).sum((1, 2, 3), keepdim=True).clamp_min(1e-6))
    return (source * scalar).clamp(0.0, 1.0)


def masked_corr(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask[0, 0] > 0.65
    left = a[0, 0][valid].float()
    right = b[0, 0][valid].float()
    if left.numel() < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    return float((left * right).mean() / (
        left.square().mean() * right.square().mean()).sqrt().clamp_min(1e-8))


def pil_image(tensor: torch.Tensor) -> Image.Image:
    pixels = (tensor.detach().float().clamp(0.0, 1.0)
              .permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(pixels)


def save_grid(path: Path, rows: list[dict[str, Image.Image]],
              names: list[str]) -> None:
    cell = 256
    gap = 4
    title_height = 34
    canvas = Image.new("RGB", (len(names) * (cell + gap) + gap,
                               len(rows) * (cell + title_height + gap) + gap),
                       (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for row_index, images in enumerate(rows):
        for column, name in enumerate(names):
            x = gap + column * (cell + gap)
            y = gap + row_index * (cell + title_height + gap)
            draw.text((x + 4, y + 7), f"{name} | light {row_index}", fill="white")
            canvas.paste(images[name].resize((cell, cell)), (x, y + title_height))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_agm_physical_pilot.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--groups-per-person", type=int, default=8)
    parser.add_argument("--steps", type=int, nargs="+", default=[500, 1000, 2000])
    args = parser.parse_args()
    if args.groups_per_person < 1:
        raise ValueError("groups-per-person must be positive")
    config = load_config(args.config)
    run = Path(args.run)
    selected_steps = sorted(set(args.steps))
    final_step = selected_steps[-1]
    checkpoints = {
        step: run / f"agm_step_{step:07d}.pt" for step in selected_steps
    }
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RRNet(**config["model"]).to(device).eval()
    encoder_checkpoint = config["training"]["encoder_checkpoint"]
    load_frozen_encoder(model, encoder_checkpoint)
    dataset = make_dataset(config, "val")
    person_indices: dict[str, list[int]] = defaultdict(list)
    for sample_index, person in enumerate(dataset.people):
        person_indices[person].append(sample_index * dataset.variants_per_source)
    selected: dict[str, list[int]] = {}
    for person, indices in person_indices.items():
        chosen = np.linspace(0, len(indices) - 1,
                             min(len(indices), args.groups_per_person)).round().astype(int)
        selected[person] = [indices[int(index)] for index in chosen]
    totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    visual_rows: dict[str, dict[str, list[dict[str, Image.Image]]]] = defaultdict(dict)
    oracle_rows: dict[str, dict[str, list[dict[str, Image.Image]]]] = defaultdict(dict)
    oracle_totals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for person, indices in selected.items():
        for index_number, index in enumerate(indices):
            item = dataset[index]
            mode = item["target_lighting_mode"][0]
            batch = default_collate([item])
            source, clean, target, mask, skin, depth, target_theta, identity = (
                physical_batch(model, batch, device))
            scalar = scalar_baseline(source, target, mask)
            with torch.inference_mode():
                source_theta = batch["source_theta"].to(device).flatten(0, 1)
                source_light, _ = model.renderer.illumination(depth, source_theta)
                target_light, _ = model.renderer.illumination(depth, target_theta)
                effective_source_light = 1.0 + mask * (source_light - 1.0)
                effective_source_light = torch.where(
                    identity[:, None, None, None],
                    torch.ones_like(effective_source_light), effective_source_light)
                oracle_base = (source / effective_source_light.clamp_min(1e-4)).clamp(0, 1)
                oracle_render = model.renderer.blend_relight(
                    oracle_base * target_light, source, mask).clamp(0, 1)
            prepared = ReferenceRelativeRRNet.prepare_light_input(source, mask)
            resized = resize_shorter_side(prepared, model.lprm.shorter_side)
            model_outputs = {}
            model_bases = {}
            for step, path in checkpoints.items():
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                model.agm.load_state_dict(checkpoint["agm"])
                if checkpoint.get("encoder") is not None:
                    model.lprm.encoder.load_state_dict(checkpoint["encoder"])
                with torch.inference_mode(), torch.autocast(
                        device_type=device.type, dtype=torch.float16,
                        enabled=device.type == "cuda"):
                    features, _ = model.lprm.encoder(resized)
                    albedo = model.agm(source, features)["albedo"]
                    raw = model.renderer(albedo, depth, target_theta)["raw_output"]
                    output = model.renderer.blend_relight(raw, source, mask).clamp(0, 1)
                model_outputs[step] = output.float()
                if step == final_step:
                    model_bases[step] = albedo.float()
                for row in range(source.shape[0]):
                    category = "identity" if bool(identity[row]) else "relit"
                    key = f"{person}/{mode}/{category}/step{step}"
                    roi = mask[row:row + 1]
                    model_error = float(masked_mean(
                        (output[row:row + 1] - target[row:row + 1]).abs(), roi))
                    scalar_error = float(masked_mean(
                        (scalar[row:row + 1] - target[row:row + 1]).abs(), roi))
                    totals[key]["model_mae"].append(model_error)
                    totals[key]["oracle_scalar_mae"].append(scalar_error)
                    totals[key]["advantage"].append(scalar_error - model_error)
                    correction = luminance(albedo[row:row + 1] - source[row:row + 1])
                    ideal = luminance(clean[row:row + 1] - source[row:row + 1])
                    totals[key]["local_correction_corr"].append(
                        masked_corr(correction, ideal, roi))
                    if bool(identity[row]):
                        totals[key]["identity_base_mae"].append(float(masked_mean(
                            (albedo[row:row + 1] - source[row:row + 1]).abs(), roi)))
                    valid_skin = (skin[row:row + 1] * roi *
                                  (source[row:row + 1].amax(1, keepdim=True) < 0.98))
                    src_colour = source[row:row + 1] / source[row:row + 1].sum(
                        1, keepdim=True).clamp_min(.02)
                    alb_colour = albedo[row:row + 1] / albedo[row:row + 1].sum(
                        1, keepdim=True).clamp_min(.02)
                    totals[key]["skin_chroma_mae"].append(float(masked_mean(
                        (src_colour - alb_colour).abs(), valid_skin)))
                    if step == final_step:
                        oracle_key = f"{person}/{mode}/{category}"
                        oracle_totals[oracle_key]["oracle_base_mae"].append(
                            float(masked_mean(
                                (oracle_base[row:row + 1] - clean[row:row + 1]).abs(), roi)))
                        oracle_totals[oracle_key]["agm_base_mae"].append(
                            float(masked_mean(
                                (albedo[row:row + 1] - clean[row:row + 1]).abs(), roi)))
                        oracle_totals[oracle_key]["oracle_render_mae"].append(
                            float(masked_mean(
                                (oracle_render[row:row + 1] - target[row:row + 1]).abs(), roi)))
                        oracle_totals[oracle_key]["agm_render_mae"].append(model_error)
                        saturated = ((source[row:row + 1].amax(1, keepdim=True) >= .9985)
                                     & (roi > .65)).float()
                        oracle_totals[oracle_key]["saturated_fraction"].append(
                            float(saturated.sum() / (roi > .65).sum().clamp_min(1)))
            if mode not in visual_rows[person]:
                visual_rows[person][mode] = []
                oracle_rows[person][mode] = []
                for row in range(source.shape[0]):
                    steps_row = {
                        "input": pil_image(source[row]),
                        "target": pil_image(target[row]),
                        "oracle scalar": pil_image(scalar[row]),
                    }
                    steps_row.update({
                        f"AGM {step}": pil_image(model_outputs[step][row])
                        for step in selected_steps})
                    visual_rows[person][mode].append(steps_row)
                    oracle_rows[person][mode].append({
                        "input": pil_image(source[row]),
                        "clean": pil_image(clean[row]),
                        "oracle base": pil_image(oracle_base[row]),
                        "AGM base": pil_image(model_bases[final_step][row]),
                        "target": pil_image(target[row]),
                        "oracle render": pil_image(oracle_render[row]),
                        "AGM output": pil_image(model_outputs[final_step][row]),
                    })
    destination = run / "inspection_known_light_oracle"
    destination.mkdir(parents=True, exist_ok=True)
    names = ["input", "target", "oracle scalar"] + [
        f"AGM {step}" for step in selected_steps]
    for person, modes in visual_rows.items():
        for mode, rows in modes.items():
            save_grid(destination / f"{person}_{mode}_steps.png", rows, names)
            save_grid(destination / f"{person}_{mode}_oracle.png",
                      oracle_rows[person][mode],
                      ["input", "clean", "oracle base", "AGM base", "target",
                       "oracle render", "AGM output"])
    report = {
        "note": "Val identities only. Oracle scalar uses target and is not deployable.",
        "groups_per_person": args.groups_per_person,
        "steps": selected_steps,
        "by_person_category_step": {
            key: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            | {"samples": len(metrics["model_mae"])}
            for key, metrics in totals.items()
        },
        "known_light_oracle": {
            key: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            | {"samples": len(metrics["oracle_base_mae"])}
            for key, metrics in oracle_totals.items()
        },
    }
    (destination / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved inspection: {destination.resolve()}")


if __name__ == "__main__":
    main()
