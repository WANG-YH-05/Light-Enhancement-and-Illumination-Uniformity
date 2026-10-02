"""Stage-1 training for spatial source-light removal with the safe AGM."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from rrnet.agm_data import MEADAGMGroups
from rrnet.config import load_config
from rrnet.safe_agm import SafeAlbedoModel
from rrnet.safe_agm_loss import SafeAGMLoss


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/safe_agm_stage1.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--init-from", help="Load full AGM weights into a new run.")
    parser.add_argument("--init-encoder-from")
    return parser.parse_args()


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def load_encoder(model: SafeAlbedoModel, checkpoint_path: str) -> tuple[int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    prefixes = ("base.lprm.encoder.", "lprm.encoder.", "encoder.")
    mapped = {}
    for key, value in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                mapped[key[len(prefix):]] = value
                break
    incompatible = model.encoder.load_state_dict(mapped, strict=False)
    return len(incompatible.missing_keys), len(incompatible.unexpected_keys)


def flatten_group(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    values = []
    for key in ("input", "target", "relight_mask", "skin_mask"):
        tensor = batch[key]
        if tensor.ndim != 5:
            raise ValueError(f"Expected grouped {key}, got {tuple(tensor.shape)}")
        values.append(tensor.flatten(0, 1).to(device, non_blocking=True))
    return tuple(values)


def identity_flags(batch: dict, device: torch.device) -> torch.Tensor | None:
    flags = batch.get("identity_flags")
    return None if flags is None else flags.flatten(0, 1).to(device)


@torch.no_grad()
def validate(model: SafeAlbedoModel, loader: DataLoader, criterion: SafeAGMLoss,
             device: torch.device, group_size: int, batches: int,
             use_amp: bool) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        source, target, relight, skin = flatten_group(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            prediction = model(source, relight)
            losses = criterion(prediction, target, source, relight, skin,
                               group_size=group_size,
                               identity_flags=identity_flags(batch, device))
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        if count >= batches:
            break
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    cli = args()
    if sum(bool(value) for value in (
            cli.resume, cli.init_from, cli.init_encoder_from)) > 1:
        raise ValueError("Use only one initialization option")
    config = load_config(cli.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafeAlbedoModel(**config["model"]).to(device)
    criterion = SafeAGMLoss(**config["loss"]).to(device)
    group_size = int(config["data"].get("group_size", 4))
    train_set = MEADAGMGroups(
        config["data"]["root"], "train",
        metadata_file=config["data"].get("metadata_file", "metadata.csv"),
        group_size=group_size,
        seed=int(config["data"].get("sampling_seed", 20260926)),
        dynamic_epoch=True,
        include_identity=bool(config["data"].get("include_identity", True)),
        anchor_identity=bool(config["data"].get("anchor_identity", False)),
    )
    val_set = MEADAGMGroups(
        config["data"]["root"], "val",
        metadata_file=config["data"].get("metadata_file", "metadata.csv"),
        group_size=group_size,
        seed=int(config["data"].get("sampling_seed", 20260926)) + 1,
        dynamic_epoch=False,
        include_identity=bool(config["data"].get("include_identity", True)),
        anchor_identity=bool(config["data"].get("anchor_identity", False)),
    )
    loader_kwargs = dict(
        batch_size=int(config["training"]["batch_size"]),
        num_workers=int(config["training"]["workers"]),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    # The metadata is sorted by person and clip.  Validating on the first N
    # entries would measure only one narrow segment of the validation split.
    max_val_groups = int(config["training"].get("validation_groups", 0))
    if max_val_groups and max_val_groups < len(val_set):
        indices = torch.linspace(0, len(val_set) - 1,
                                 steps=max_val_groups).round().long().tolist()
        val_dataset = Subset(val_set, indices)
    else:
        val_dataset = val_set
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False,
                            **loader_kwargs)

    encoder_lr = float(config["training"].get(
        "encoder_learning_rate", config["training"]["learning_rate"]))
    decoder_lr = float(config["training"]["learning_rate"])
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.agm.parameters(), "lr": decoder_lr},
    ], weight_decay=float(config["training"].get("weight_decay", 1.0e-4)))
    steps = int(config["training"]["iterations"])
    min_lr_ratio = float(config["training"].get("min_learning_rate", 1.0e-6)) / decoder_lr
    warmup = int(config["training"].get("warmup_steps", 0))

    def schedule(step: int) -> float:
        if warmup and step < warmup:
            return 0.1 + 0.9 * step / warmup
        progress = (step - warmup) / max(steps - warmup, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_amp = bool(config["training"].get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_root = Path(config["training"]["output_dir"])
    if cli.resume:
        output_dir = Path(cli.resume).resolve().parent
    else:
        output_dir = output_root / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    start_step = 0
    if cli.resume:
        checkpoint = torch.load(cli.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"])
        log(log_path, f"Resumed {Path(cli.resume).resolve()} at step {start_step}")
    elif cli.init_encoder_from:
        missing, unexpected = load_encoder(model, cli.init_encoder_from)
        log(log_path, f"Initialized encoder from {Path(cli.init_encoder_from).resolve()} "
            f"| missing={missing} unexpected={unexpected}; safe AGM remains zero-initialized")
    elif cli.init_from:
        checkpoint = torch.load(cli.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("model", checkpoint))
        log(log_path, f"Initialized full safe AGM weights from "
            f"{Path(cli.init_from).resolve()}; optimizer and schedule are new")

    (output_dir / "run_config.json").write_text(json.dumps(
        {"run_directory": str(output_dir.resolve()), **config},
        ensure_ascii=False, indent=2), encoding="utf-8")
    log(log_path, f"Safe AGM stage 1 | device={device} steps={steps} "
        f"batch={loader_kwargs['batch_size']} group={group_size} AMP={use_amp}")

    keys = ("total", "albedo", "gain", "gradient", "consistency", "chroma",
            "smooth", "background", "identity", "model_mae", "global_mae",
            "beats_global", "model_luma_mae", "global_luma_mae",
            "luma_advantage")
    running = {key: 0.0 for key in keys}
    running_count = 0
    iterator = iter(train_loader)
    epoch = 0
    started = time.time()
    log_every = int(config["training"].get("log_every", 100))
    model.train()
    for step in range(start_step, steps):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            train_set.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        source, target, relight, skin = flatten_group(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=use_amp):
            prediction = model(source, relight)
            losses = criterion(prediction, target, source, relight, skin,
                               group_size=group_size,
                               identity_flags=identity_flags(batch, device))
        scaler.scale(losses["total"]).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        for key in keys:
            running[key] += float(losses[key].detach())
        running_count += 1
        current = step + 1
        if current % log_every == 0 or current == steps:
            values = {key: running[key] / running_count for key in keys}
            elapsed = time.time() - started
            eta = elapsed / max(current - start_step, 1) * (steps - current)
            record = {"step": current, "epoch": epoch,
                      "learning_rate": optimizer.param_groups[1]["lr"], **values}
            with (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            log(log_path, f"TRAIN {current}/{steps} total={values['total']:.5f} "
                f"A={values['albedo']:.5f} G={values['gain']:.5f} "
                f"grad={values['gradient']:.5f} cons={values['consistency']:.5f} "
                f"identity={values['identity']:.5f} "
                f"chroma={values['chroma']:.5f} | model/global="
                f"{values['model_mae']:.5f}/{values['global_mae']:.5f} "
                f"adv={values['beats_global']:+.5f} "
                f"Yadv={values['luma_advantage']:+.5f} | ETA={eta/60:.1f}m")
            running = {key: 0.0 for key in keys}
            running_count = 0

        if current % int(config["training"]["validate_every"]) == 0:
            metrics = validate(
                model, val_loader, criterion, device, group_size,
                int(config["training"]["validation_batches"]), use_amp)
            with (output_dir / "val_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": current, **metrics}) + "\n")
            log(log_path, f"VALID {current}/{steps} total={metrics['total']:.5f} "
                f"A={metrics['albedo']:.5f} G={metrics['gain']:.5f} "
                f"grad={metrics['gradient']:.5f} cons={metrics['consistency']:.5f} "
                f"model/global={metrics['model_mae']:.5f}/{metrics['global_mae']:.5f} "
                f"adv={metrics['beats_global']:+.5f} "
                f"Yadv={metrics['luma_advantage']:+.5f}")

        if current % int(config["training"]["save_every"]) == 0 or current == steps:
            path = output_dir / f"safe_agm_step_{current:07d}.pt"
            torch.save({"step": current, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict()}, path)
            log(log_path, f"CHECKPOINT saved: {path.name}")

    log(log_path, "Safe AGM stage-1 training completed")


if __name__ == "__main__":
    main()
