"""Fit absolute illumination theta statistics for relative-light RRNet.

Unlike calibrate_lighting_stats.py (bad-light -> clean correction), this script
fits clean * illumination(theta) -> bad-light.  The resulting statistics are
therefore suitable for estimating Lin and Lref before ratio transfer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from rrnet.config import load_config, model_kwargs
from rrnet.data import MEADPairs
from rrnet.depth import DepthAnythingV2Small
from rrnet.lighting import default_parameter_statistics
from rrnet.losses import RRNetLoss, luminance
from rrnet.renderer import RenderingModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_mead_reference_relative.yaml")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--optimization-steps", type=int, default=100)
    parser.add_argument("--output", default="checkpoints/mead_relative_theta_stats_1000.npz")
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()

    config = load_config(args.config)
    kwargs = model_kwargs(config, args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = DepthAnythingV2Small(
        kwargs["depth_vendor_root"], kwargs["depth_checkpoint"],
        kwargs["depth_input_size"], kwargs["depth_invert"],
    ).to(device).eval()
    renderer = RenderingModule(
        kwargs["num_lights"], kwargs["sigma1"], kwargs["sigma2"],
        clamp_output=False,
    ).to(device)
    regularizer = RRNetLoss(
        num_lights=kwargs["num_lights"],
        lambda_reg=float(config["loss"].get("lambda_reg", 0.01)),
        lambda_ambient=float(config["loss"].get("lambda_ambient", 1.0)),
        ambient_max=float(config["loss"].get("ambient_max", 2.0)),
    ).to(device)
    dataset = MEADPairs(
        config["data"]["root"], "train",
        config["data"].get("metadata_file", "metadata.csv"),
        load_masks=True,
    )
    count = min(args.samples, len(dataset))
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=count, replace=False).tolist()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    initial, _ = default_parameter_statistics(kwargs["num_lights"])
    optimized: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="Fitting absolute illumination theta"):
        lit = batch["input"].to(device)
        clean = batch["target"].to(device)
        mask = batch["relight_mask"].to(device)
        with torch.no_grad():
            depth = depth_model(clean)
        theta = torch.nn.Parameter(initial.to(device).repeat(clean.shape[0], 1))
        optimizer = torch.optim.Adam([theta], lr=0.03)
        for _ in range(args.optimization_steps):
            optimizer.zero_grad(set_to_none=True)
            illumination, _ = renderer.illumination(depth, theta)
            illumination = luminance(illumination)
            prediction = luminance(clean) * illumination
            reconstruction = regularizer.masked_mean(
                (prediction - luminance(lit)).abs(), mask)
            penalty = regularizer.lighting_regularization(theta)
            loss = reconstruction + regularizer.lambda_reg * penalty
            loss.backward()
            optimizer.step()
        optimized.append(theta.detach().cpu())

    values = torch.cat(optimized).numpy()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        mean=values.mean(axis=0),
        std=values.std(axis=0).clip(1.0e-6),
    )
    summary = {
        "purpose": "absolute illumination: clean * L(theta) -> degraded",
        "config": str(Path(args.config).resolve()),
        "samples": len(indices),
        "batch_size": args.batch_size,
        "optimization_steps": args.optimization_steps,
        "seed": args.seed,
        "output": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {values.shape[0]} absolute-light theta vectors to {output.resolve()}")


if __name__ == "__main__":
    main()
