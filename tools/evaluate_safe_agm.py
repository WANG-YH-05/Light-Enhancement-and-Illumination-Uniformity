"""Compare safe AGM with an oracle scalar fitted to each clean target."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Permit `python tools/evaluate_safe_agm.py` from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rrnet.data import load_mask, load_rgb
from rrnet.losses import luminance
from rrnet.safe_agm import SafeAlbedoModel


def correlation(left: torch.Tensor, right: torch.Tensor,
                mask: torch.Tensor) -> float:
    valid = mask > 0.65
    left = left[valid]
    right = right[valid]
    if left.numel() < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = (left.square().mean() * right.square().mean()).sqrt()
    return float((left * right).mean() / denominator.clamp_min(1.0e-8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe_agm_stage1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples-per-category", type=int, default=100)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import yaml
    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeAlbedoModel(**config["model"]).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    root = Path(args.dataset_root)
    with (root / "metadata.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]
    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    selected = []
    for category_rows in by_variant.values():
        count = min(args.samples_per_category, len(category_rows))
        chosen = np.linspace(0, len(category_rows) - 1, count).round().astype(int)
        selected.extend(category_rows[int(index)] for index in chosen)

    totals: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    by_person: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for row in selected:
            source = load_rgb(root / row["bad_light_frame"]).unsqueeze(0).to(device)
            target = load_rgb(root / row["clean_frame"]).unsqueeze(0).to(device)
            mask = load_mask(root / row["relight_mask"]).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp):
                output = model(source, mask)
            source_y, target_y = luminance(source), luminance(target)
            model_y = luminance(output["albedo"])
            weight = mask
            scalar = ((source_y * target_y * weight).sum()
                      / (source_y.square() * weight).sum().clamp_min(1.0e-6))
            global_output = (source * scalar).clamp(0.0, 1.0)
            denominator = (weight.sum() * 3.0).clamp_min(1.0)
            model_mae = float(((output["albedo"] - target).abs() * weight).sum()
                              / denominator)
            global_mae = float(((global_output - target).abs() * weight).sum()
                               / denominator)
            model_luma_mae = float(
                ((model_y - target_y).abs() * weight).sum()
                / weight.sum().clamp_min(1.0))
            global_luma_mae = float(
                ((luminance(global_output) - target_y).abs() * weight).sum()
                / weight.sum().clamp_min(1.0))
            ideal_gain = ((target_y + 1.0e-3).log()
                          - (source_y + 1.0e-3).log()).clamp(
                              -model.max_log_gain, model.max_log_gain)
            gain_corr = correlation(output["log_gain"], ideal_gain, mask)
            for bucket in (totals[row["variant"]], by_person[row["person_id"]]):
                bucket["model_mae"].append(model_mae)
                bucket["global_mae"].append(global_mae)
                bucket["advantage"].append(global_mae - model_mae)
                bucket["model_luma_mae"].append(model_luma_mae)
                bucket["global_luma_mae"].append(global_luma_mae)
                bucket["luma_advantage"].append(global_luma_mae - model_luma_mae)
                bucket["gain_correlation"].append(gain_corr)

    report = {
        category: {key: float(np.mean(values)) for key, values in metrics.items()}
        | {"samples": len(metrics["model_mae"])}
        for category, metrics in totals.items()
    }
    report["overall"] = {
        key: float(np.mean([
            value for metrics in totals.values() for value in metrics[key]
        ]))
        for key in ("model_mae", "global_mae", "advantage", "gain_correlation",
                    "model_luma_mae", "global_luma_mae", "luma_advantage")
    } | {"samples": sum(len(metrics["model_mae"]) for metrics in totals.values())}
    report["by_person"] = {
        person: {key: float(np.mean(values)) for key, values in metrics.items()}
        | {"samples": len(metrics["model_mae"])}
        for person, metrics in by_person.items()
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {destination.resolve()}")


if __name__ == "__main__":
    main()
