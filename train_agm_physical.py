"""Train the reconstructed RRNet RGB AGM through known-light rendering.

This is an isolated pilot: the encoder and depth network stay frozen, exact
synthetic theta is supplied, and no existing reference model is overwritten.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from rrnet.agm_physical_data import MEADPhysicalAGMGroups
from rrnet.agm_physical_stage import PhysicalAGMLoss, oracle_scalar_mae
from rrnet.config import load_config
from rrnet.lprm import resize_shorter_side
from rrnet.model import RRNet
from rrnet.reference_relative_model import ReferenceRelativeRRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_agm_physical_pilot.yaml")
    parser.add_argument("--encoder-checkpoint", help="Override pretrained encoder path")
    parser.add_argument("--init-agm-from", help="Load AGM head weights only; start a new run")
    parser.add_argument("--resume", help="Resume AGM, optimizer and step from pilot checkpoint")
    parser.add_argument("--steps", type=int, help="Override total steps, useful for smoke tests")
    parser.add_argument("--validation-batches", type=int)
    parser.add_argument("--output-dir", help="Override output root; a timestamped run is created")
    return parser.parse_args()


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_frozen_encoder(model: RRNet, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    for prefix in ("base.lprm.encoder.", "lprm.encoder.", "encoder."):
        selected = {key[len(prefix):]: value for key, value in state.items()
                    if key.startswith(prefix)}
        if selected:
            model.lprm.encoder.load_state_dict(selected, strict=True)
            return
    raise ValueError(f"No compatible LPRM encoder in {checkpoint_path}")


def make_dataset(config: dict, split: str) -> MEADPhysicalAGMGroups:
    data = config["data"]
    return MEADPhysicalAGMGroups(
        data["root"], split, metadata_file=data.get("metadata_file", "metadata.csv"),
        seed=int(data.get("sampling_seed", 20260926)) + (split == "val"),
        variants_per_source=int(data.get("variants_per_source", 1)),
        num_lights=int(config["model"]["num_lights"]),
        dynamic_epoch=split == "train",
        target_lighting_weights=data.get(
            "target_lighting_weights", {"dark": 0.5, "normal": 0.5, "bright": 0.0}),
        group_size=int(data.get("group_size", 4)),
    )


def physical_batch(model: RRNet, batch: dict, device: torch.device
                   ) -> tuple[torch.Tensor, ...]:
    # Each group contains one person/frame under multiple source lights and
    # one shared target light. Group member zero is an exact identity anchor.
    grouped_clean = batch["source_clean"].to(device, non_blocking=True)
    grouped_mask = batch["relight_mask"].to(device, non_blocking=True)
    grouped_skin = batch["skin_mask"].to(device, non_blocking=True)
    source_theta = batch["source_theta"].to(device, non_blocking=True)
    target_theta = batch["target_theta"].to(device, non_blocking=True)
    count, group = grouped_clean.shape[:2]
    clean_single = grouped_clean[:, 0]
    with torch.no_grad():
        depth_single = model.depth(clean_single)
        depth = depth_single[:, None].expand(-1, group, -1, -1, -1).reshape(
            count * group, *depth_single.shape[1:])
        clean = grouped_clean.flatten(0, 1)
        mask = grouped_mask.flatten(0, 1)
        skin = grouped_skin.flatten(0, 1)
        source_light, _ = model.renderer.illumination(
            depth, source_theta.flatten(0, 1))
        target_light, _ = model.renderer.illumination(
            depth, target_theta.flatten(0, 1))
        source = model.renderer.blend_relight(
            clean * source_light, clean, mask).clamp(0.0, 0.999)
        target = model.renderer.blend_relight(
            clean * target_light, clean, mask).clamp(0.0, 0.999)
        identity_flags = torch.zeros(count, group, dtype=torch.bool, device=device)
        identity_flags[:, 0] = True
        source = torch.where(identity_flags.reshape(-1, 1, 1, 1), clean, source)
    return source, clean, target, mask, skin, depth, target_theta.flatten(0, 1), identity_flags.flatten()


def forward_loss(model: RRNet, criterion: PhysicalAGMLoss, batch: dict,
                 device: torch.device, use_amp: bool,
                 train_encoder: bool = False) -> dict[str, torch.Tensor]:
    source, clean, target, mask, skin, depth, target_theta, identity = physical_batch(
        model, batch, device)
    with (nullcontext() if train_encoder else torch.no_grad()), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_amp):
        # The selected reference-relative checkpoint trained this encoder on
        # masked grayscale, not raw RGB. Match that input distribution.
        prepared = ReferenceRelativeRRNet.prepare_light_input(source, mask)
        resized = resize_shorter_side(prepared, model.lprm.shorter_side)
        features, _ = model.lprm.encoder(resized)
    with torch.autocast(device_type=device.type, dtype=torch.float16,
                        enabled=use_amp):
        agm = model.agm(source, features)
        rendered = model.renderer(agm["albedo"], depth, target_theta)
        output = model.renderer.blend_relight(
            rendered["raw_output"], source, mask).clamp(0.0, 1.0)
        losses = criterion(
            source=source, clean=clean, target=target, albedo=agm["albedo"],
            output=output, mask=mask, skin_mask=skin, identity_flags=identity)
    losses["oracle_scalar_mae"] = oracle_scalar_mae(source, target, mask)
    losses["scalar_advantage"] = losses["oracle_scalar_mae"] - losses["mae"]
    return losses


@torch.no_grad()
def validate(model: RRNet, criterion: PhysicalAGMLoss, loader: DataLoader,
             device: torch.device, use_amp: bool, batches: int,
             train_encoder: bool = False) -> dict[str, float]:
    model.agm.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        losses = forward_loss(model, criterion, batch, device, use_amp,
                              train_encoder=train_encoder)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        if count >= batches:
            break
    model.agm.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.resume and args.init_agm_from:
        raise ValueError("Use --resume or --init-agm-from, not both")
    config = load_config(args.config)
    seed = int(config["training"].get("seed", 20260926))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    steps = args.steps or int(config["training"]["iterations"])
    if steps < 1:
        raise ValueError("steps must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Physical AGM pilot requires CUDA for Depth Anything V2")
    encoder_checkpoint = args.encoder_checkpoint or config["training"]["encoder_checkpoint"]
    init_agm_checkpoint = args.init_agm_from or config["training"].get("init_agm_checkpoint")
    train_encoder = bool(config["training"].get("train_encoder", False))
    model = RRNet(**config["model"]).to(device)
    if model.agm is None:
        raise ValueError("Physical AGM pilot requires model.use_agm: true")
    load_frozen_encoder(model, encoder_checkpoint)
    model.agm.output.weight.data.zero_()
    model.agm.output.bias.data.zero_()
    model.requires_grad_(False)
    model.agm.requires_grad_(True)
    model.lprm.encoder.requires_grad_(train_encoder)
    model.depth.eval()
    model.lprm.encoder.eval()
    model.agm.train()
    criterion = PhysicalAGMLoss(**config["loss"]).to(device)
    train_set = make_dataset(config, "train")
    val_set = make_dataset(config, "val")
    max_val_groups = int(config["training"].get("validation_groups", 48))
    if max_val_groups < len(val_set):
        indices = torch.linspace(0, len(val_set) - 1,
                                 steps=max_val_groups).round().long().tolist()
        val_dataset = Subset(val_set, indices)
    else:
        val_dataset = val_set
    loader_args = {
        "batch_size": int(config["training"].get("batch_size", 1)),
        "num_workers": int(config["training"].get("workers", 2)),
        "pin_memory": True,
    }
    order_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                              generator=order_generator, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    parameter_groups = [{
        "params": model.agm.parameters(),
        "lr": float(config["training"]["learning_rate"]),
    }]
    if train_encoder:
        parameter_groups.append({
            "params": model.lprm.encoder.parameters(),
            "lr": float(config["training"]["encoder_learning_rate"]),
        })
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(config["training"].get("weight_decay", 1.0e-4)))
    use_amp = bool(config["training"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    output_root = Path(args.output_dir or config["training"]["output_dir"])
    output_dir = (Path(args.resume).resolve().parent if args.resume else
                  output_root / f"run_{datetime.now():%Y%m%d_%H%M%S}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if Path(checkpoint["encoder_checkpoint"]).resolve() != Path(encoder_checkpoint).resolve():
            raise ValueError("Resume must use the same frozen encoder checkpoint")
        if bool(checkpoint.get("train_encoder", False)) != train_encoder:
            raise ValueError("Resume must keep training.train_encoder unchanged")
        if train_encoder:
            model.lprm.encoder.load_state_dict(checkpoint["encoder"])
        model.agm.load_state_dict(checkpoint["agm"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
    elif init_agm_checkpoint:
        checkpoint = torch.load(init_agm_checkpoint, map_location="cpu", weights_only=False)
        model.agm.load_state_dict(checkpoint["agm"], strict=True)
    (output_dir / "run_config.json").write_text(json.dumps({
        **config, "encoder_checkpoint_resolved": str(Path(encoder_checkpoint).resolve()),
        "steps_requested": steps,
        "init_agm_checkpoint_resolved": (
            str(Path(init_agm_checkpoint).resolve()) if init_agm_checkpoint else None),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log(log_path, f"Known-light RGB AGM pilot | steps={steps} batch={loader_args['batch_size']} "
        f"group={train_set.group_size} amp={use_amp} resume={start_step} "
        f"train_encoder={train_encoder} seed={seed} "
        f"init_agm={Path(init_agm_checkpoint).resolve() if init_agm_checkpoint else None} "
        f"encoder={Path(encoder_checkpoint).resolve()}")
    iterator = iter(train_loader)
    epoch = 0
    started = time.time()
    log_every = int(config["training"].get("log_every", 100))
    validate_every = int(config["training"].get("validate_every", 500))
    save_every = int(config["training"].get("save_every", 500))
    val_batches = args.validation_batches or int(config["training"].get("validation_batches", 48))
    metric_keys = ("total", "output", "base", "spatial", "chroma",
                   "background", "identity", "mae", "oracle_scalar_mae",
                   "scalar_advantage")
    running = {key: 0.0 for key in metric_keys}
    count = 0
    for step in range(start_step, steps):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            train_set.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        losses = forward_loss(model, criterion, batch, device, use_amp,
                              train_encoder=train_encoder)
        if not bool(torch.isfinite(losses["total"]).item()):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}")
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        current = step + 1
        for key in metric_keys:
            running[key] += float(losses[key].detach())
        count += 1
        if current % log_every == 0 or current == steps:
            metrics = {key: running[key] / count for key in metric_keys}
            elapsed = time.time() - started
            eta = elapsed / max(current - start_step, 1) * (steps - current)
            with (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": current, "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"], **metrics}) + "\n")
            log(log_path, f"TRAIN {current}/{steps} total={metrics['total']:.5f} "
                f"out={metrics['output']:.5f} base={metrics['base']:.5f} "
                f"spatial={metrics['spatial']:.5f} chroma={metrics['chroma']:.5f} "
                f"identity={metrics['identity']:.5f} "
                f"oracle_adv={metrics['scalar_advantage']:+.5f} ETA={eta/60:.1f}m")
            running = {key: 0.0 for key in metric_keys}
            count = 0
        if current % validate_every == 0 or current == steps:
            metrics = validate(model, criterion, val_loader, device, use_amp,
                               val_batches, train_encoder=train_encoder)
            with (output_dir / "val_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": current, **metrics}) + "\n")
            log(log_path, f"VALID {current}/{steps} total={metrics['total']:.5f} "
                f"out={metrics['output']:.5f} base={metrics['base']:.5f} "
                f"spatial={metrics['spatial']:.5f} chroma={metrics['chroma']:.5f} "
                f"identity={metrics['identity']:.5f} "
                f"oracle_adv={metrics['scalar_advantage']:+.5f}")
        if current % save_every == 0 or current == steps:
            path = output_dir / f"agm_step_{current:07d}.pt"
            torch.save({"step": current, "agm": model.agm.state_dict(),
                        "encoder": (model.lprm.encoder.state_dict()
                                    if train_encoder else None),
                        "train_encoder": train_encoder,
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "encoder_checkpoint": str(Path(encoder_checkpoint).resolve())}, path)
            log(log_path, f"CHECKPOINT saved: {path.name}")
    log(log_path, "Known-light RGB AGM pilot completed")


if __name__ == "__main__":
    main()
