"""Estimate theta mean/std for equation (2) from aligned training pairs.

The RRNet paper says the statistics come from "optimal parameters" but does
not publish how those parameters are obtained. This utility documents and
implements the reproduction assumption used for the aligned MEAD pairs:
directly optimize renderer parameters to reconstruct each clean target.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rrnet.config import load_config, model_kwargs
from rrnet.data import MEADPairs
from rrnet.depth import DepthAnythingV2Small
from rrnet.lighting import default_parameter_statistics
from rrnet.losses import RRNetLoss
from rrnet.renderer import RenderingModule


def stratified_indices(rows: list[dict[str, str]], probabilities: dict[str, float],
                       count: int, seed: int) -> list[int]:
    """Select an approximately exact category-balanced subset without replacement."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["degradation_id"]].append(index)
    if set(groups) != set(probabilities):
        raise ValueError(
            f"Sampling categories mismatch; dataset={sorted(groups)}, "
            f"config={sorted(probabilities)}"
        )
    total_probability = sum(float(value) for value in probabilities.values())
    normalized = {key: float(value) / total_probability
                  for key, value in probabilities.items()}
    exact = {key: count * value for key, value in normalized.items()}
    allocation = {key: int(np.floor(value)) for key, value in exact.items()}
    remaining = count - sum(allocation.values())
    for key in sorted(exact, key=lambda name: exact[name] - allocation[name], reverse=True)[:remaining]:
        allocation[key] += 1
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for key, amount in allocation.items():
        if amount > len(groups[key]):
            raise ValueError(f"Requested {amount} {key} samples but only {len(groups[key])} exist")
        selected.extend(rng.choice(groups[key], size=amount, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_mead.yaml")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimization-steps", type=int, default=100)
    parser.add_argument("--output", default="checkpoints/mead_theta_stats.npz")
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    config = load_config(args.config)
    kwargs = model_kwargs(config, args.config)
    if not kwargs["depth_vendor_root"] or not kwargs["depth_checkpoint"]:
        raise ValueError("Depth Anything V2 paths are required")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = DepthAnythingV2Small(
        kwargs["depth_vendor_root"], kwargs["depth_checkpoint"],
        kwargs["depth_input_size"], kwargs["depth_invert"]
    ).to(device)
    renderer = RenderingModule(
        kwargs["num_lights"], kwargs["sigma1"], kwargs["sigma2"], clamp_output=False
    ).to(device)
    criterion = RRNetLoss(**config["loss"]).to(device)
    metadata_file = config["data"].get("metadata_file", "metadata.csv")
    use_masks = bool(config["data"].get("use_masks", False))
    dataset = MEADPairs(config["data"]["root"], "train", metadata_file,
                        load_masks=use_masks)
    count = min(args.samples, len(dataset))
    sampling = config["data"].get("train_sampling")
    if sampling:
        indices = stratified_indices(dataset.rows, sampling, count, args.seed)
    else:
        indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=0)
    initial, _ = default_parameter_statistics(kwargs["num_lights"])
    optimized = []
    for batch in tqdm(loader, desc="Fitting optimal theta", file=sys.stdout):
        source = batch["input"].to(device)
        target = batch["target"].to(device)
        relight_mask = batch.get("relight_mask")
        skin_mask = batch.get("skin_mask")
        if relight_mask is not None:
            relight_mask = relight_mask.to(device)
        if skin_mask is not None:
            skin_mask = skin_mask.to(device)
        with torch.no_grad():
            depth = depth_model(source)
        theta = torch.nn.Parameter(initial.to(device).repeat(source.shape[0], 1))
        optimizer = torch.optim.Adam([theta], lr=0.03)
        for _ in range(args.optimization_steps):
            optimizer.zero_grad(set_to_none=True)
            result = renderer(source, depth, theta)
            if relight_mask is not None:
                result["loss_output"] = renderer.blend_relight(
                    result["raw_output"], source, relight_mask
                )
                result["output"] = result["loss_output"]
                result["depth"] = depth
                result["theta"] = theta
                loss = criterion(result, target, source, relight_mask, skin_mask)["total"]
            else:
                pixel = F.l1_loss(result["output"], target)
                reg = criterion.lighting_regularization(theta)
                loss = pixel + criterion.lambda_reg * reg
            loss.backward()
            optimizer.step()
        optimized.append(theta.detach().cpu())
    values = torch.cat(optimized).numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, mean=values.mean(axis=0), std=values.std(axis=0).clip(1e-6))
    category_counts: dict[str, int] = defaultdict(int)
    for index in indices:
        category_counts[dataset.rows[index]["degradation_id"]] += 1
    summary = {
        "config": str(Path(args.config).resolve()),
        "metadata": str((Path(config["data"]["root"]) / metadata_file).resolve()),
        "samples": len(indices),
        "batch_size": args.batch_size,
        "optimization_steps": args.optimization_steps,
        "seed": args.seed,
        "mask_aware": use_masks,
        "category_counts": dict(sorted(category_counts.items())),
        "output": str(output.resolve()),
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved {values.shape[0]} fitted parameter vectors to {output}")


if __name__ == "__main__":
    main()
