"""One-batch integrity check for the real relative-light training pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rrnet.config import load_config, model_kwargs
from rrnet.reference_data import MEADReferenceTriplets
from rrnet.reference_relative_loss import ReferenceRelativeLoss
from rrnet.reference_relative_model import ReferenceRelativeRRNet


def chromaticity(image: torch.Tensor) -> torch.Tensor:
    positive = image.clamp_min(0.0)
    return positive / positive.sum(dim=1, keepdim=True).clamp_min(1.0e-4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_mead_reference_relative.yaml")
    parser.add_argument("--statistics", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2 because LPRM uses BatchNorm1d")

    config = load_config(args.config)
    if config.get("task", "").lower() != "reference_relative":
        raise ValueError("Expected task: reference_relative")
    kwargs = model_kwargs(config, args.config)
    if args.statistics:
        kwargs["statistics_path"] = str(Path(args.statistics).resolve())
    statistics = Path(kwargs["statistics_path"])
    if not statistics.is_file():
        raise FileNotFoundError(
            f"Absolute-light statistics do not exist: {statistics}\n"
            "Run calibrate_relative_lighting_stats.py first."
        )
    statistics_metadata = statistics.with_suffix(".json")
    if statistics_metadata.is_file():
        metadata = json.loads(statistics_metadata.read_text(encoding="utf-8"))
        purpose = str(metadata.get("purpose", "")).lower()
        if "absolute illumination" not in purpose:
            raise ValueError(
                f"Statistics are not marked as absolute illumination: {statistics_metadata}")

    dataset = MEADReferenceTriplets(
        config["data"]["root"], "train",
        metadata_file=config["data"].get("metadata_file", "metadata.csv"),
        manifest_file=config["data"].get("manifest_file", "generation_manifest.json"),
        seed=int(config["data"].get("sampling_seed", 20260908)),
        variants_per_source=2,
        dynamic_epoch=False,
    )
    batch = next(iter(DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)))
    required = {
        "input", "reference", "source_clean", "reference_clean", "target",
        "relight_mask", "skin_mask", "source_light_mask", "reference_mask",
    }
    missing = required - set(batch)
    if missing:
        raise KeyError(f"Dataset is missing fields: {sorted(missing)}")
    if any(source == reference for source, reference in zip(
            batch["source_person_id"], batch["reference_person_id"])):
        raise AssertionError("A triplet selected the same source/reference identity")
    if not torch.equal(batch["input"][0], batch["input"][1]):
        raise AssertionError(
            "Adjacent variants must keep the exact same source input")
    if batch["target_category"][0] == batch["target_category"][1]:
        raise AssertionError(
            "Adjacent variants must request different reference lights")
    if torch.equal(batch["target"][0], batch["target"][1]):
        raise AssertionError(
            "Different reference lights unexpectedly produced identical targets")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReferenceRelativeRRNet(**kwargs).to(device).train()
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state)
    if model.agm is not None:
        raise AssertionError("Relative-light mode must not instantiate AGM")
    criterion = ReferenceRelativeLoss(**config["loss"]).to(device)
    tensors = {
        key: batch[key].to(device)
        for key in required
    }
    prediction = model(
        tensors["input"], tensors["reference"], tensors["relight_mask"],
        tensors["reference_mask"], tensors["source_light_mask"],
    )
    losses = criterion(
        prediction, tensors["target"], tensors["input"],
        tensors["relight_mask"], tensors["skin_mask"],
        source_clean=tensors["source_clean"],
        reference=tensors["reference"],
        reference_clean=tensors["reference_clean"],
        source_light_mask=tensors["source_light_mask"],
        reference_mask=tensors["reference_mask"],
    )
    if not all(torch.isfinite(value) for value in losses.values()):
        raise FloatingPointError("A loss or diagnostic metric is not finite")
    losses["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters()
                 if parameter.requires_grad and parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("Model gradients are missing or non-finite")
    if any(parameter.grad is not None for parameter in model.depth.parameters()):
        raise AssertionError("Frozen depth model unexpectedly received gradients")

    mask = tensors["relight_mask"].to(prediction["output"].dtype).clamp(0.0, 1.0)
    expected_composite = tensors["input"] + mask * (
        prediction["raw_output"].clamp(0.0, 1.0) - tensors["input"])
    composite_error = (prediction["output"] - expected_composite).abs().max().item()
    if composite_error > 1.0e-6:
        raise AssertionError(
            f"Output does not follow the exact soft-mask composite: "
            f"max={composite_error:.6g}")
    pure_background = mask <= 1.0e-6
    background_error = (prediction["output"] - tensors["input"]).abs()[
        pure_background.expand_as(prediction["output"])]
    max_background_error = (float(background_error.max())
                            if background_error.numel() else 0.0)
    if max_background_error > 1.0e-6:
        raise AssertionError(
            f"Pixels where the relight mask is zero changed: "
            f"max={max_background_error:.6g}")

    source_chroma = chromaticity(tensors["input"])
    output_chroma = chromaticity(prediction["output"])
    valid = tensors["input"].sum(dim=1, keepdim=True) > 0.04
    chroma_error = (source_chroma - output_chroma).abs()[valid.expand_as(source_chroma)]
    max_chroma_error = float(chroma_error.max()) if chroma_error.numel() else 0.0
    if max_chroma_error > 2.0e-4:
        raise AssertionError(
            f"Chromaticity changed unexpectedly: max error={max_chroma_error:.6g}")

    print(f"PASS device={device} batch={args.batch_size} samples={len(dataset)}")
    print(
        f"total={float(losses['total']):.6f} "
        f"Lin={float(losses['illum_source']):.6f} "
        f"Lref={float(losses['illum_reference']):.6f} "
        f"Ltgt={float(losses['illum_target']):.6f} "
        f"Lgain={float(losses['gain_log']):.6f} "
        f"gain={float(losses['gain_face_mean']):.4f}/"
        f"{float(losses['gain_target_mean']):.4f} "
        f"theta_gap={float(losses['theta_gap']):.6f}"
    )
    print(f"max_chromaticity_error={max_chroma_error:.8f}")
    print(f"max_composite_error={composite_error:.8f}")
    print(f"max_background_error={max_background_error:.8f}")


if __name__ == "__main__":
    main()
