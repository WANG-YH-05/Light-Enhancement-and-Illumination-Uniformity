"""Iteration-based RRNet training matching the paper's 300k-step protocol."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from rrnet.config import load_config, model_kwargs
from rrnet.data import MEADPairs, category_sample_weights
from rrnet.losses import RRNetLoss
from rrnet.model import RRNet
from rrnet.physical_reference_data import MEADPhysicalReferencePairs
from rrnet.reference_data import MEADReferenceTriplets
from rrnet.reference_model import ReferenceRRNet
from rrnet.reference_relative_loss import ReferenceRelativeLoss
from rrnet.reference_relative_model import ReferenceRelativeRRNet
from rrnet.reference_theta_model import ReferenceThetaRRNet
from rrnet.validation_sampling import balanced_validation_indices


def format_duration(seconds: float) -> str:
    """Format a duration compactly for terminal and text logs."""
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def append_log(path: Path, message: str, terminal_message: str | None = None) -> None:
    """Persist a detailed event and print a terminal-friendly version."""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    terminal_line = terminal_message if terminal_message is not None else message
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {terminal_line}", flush=True)


def train_terminal_message(record: dict[str, float], current: int, steps: int,
                           elapsed_seconds: float, eta_seconds: float) -> str:
    """Keep training status readable in narrow VS Code terminals."""
    percent = 100.0 * current / max(steps, 1)
    summary = (f"TRAIN {current}/{steps} {percent:.1f}% | total={record['total']:.5f} | "
               f"{record['steps_per_second']:.2f} step/s | ETA={format_duration(eta_seconds)}")
    details = (f"rel={record['relight']:.5f} skin={record['skin']:.5f} "
               f"dark={record['dark']:.5f} over={record['overexposure_log']:.5f} "
               f"chroma={record['chroma']:.5f}")
    tail = (f"roi={record['roi']:.5f} reg={record['regularization']:.5f} "
            f"hi={record['highlight']:.5f} bg={record['background']:.5f} "
            f"elapsed={format_duration(elapsed_seconds)}")
    if "illum_source" in record:
        tail += (f"\n      Lin={record['illum_source']:.5f} "
                 f"Lref={record['illum_reference']:.5f} "
                 f"Ltgt={record['illum_target']:.5f} "
                 f"Lgain={record['gain_log']:.5f} "
                 f"gain={record['gain_face_mean']:.3f}/"
                 f"{record['gain_target_mean']:.3f} "
                 f"Ldark={record['dark_log_lift']:.5f} "
                 f"Lcons={record['consistency']:.5f} "
                 f"Lrefc={record['reference_contrast']:.5f} "
                 f"Lgrad={record['gain_gradient']:.5f} "
                 f"Lexp={record['illum_exposure']:.4f} "
                 f"Lshape={record['illum_shape']:.4f}/"
                 f"{record['illum_shape_gradient']:.4f} "
                 f"Lvar={record['illum_variance']:.4f} "
                 f"Lambient={record['ambient_ratio']:.4f} "
                 f"LshapeCons={record['shape_consistency']:.4f} "
                 f"theta={record['theta_source']:.4f}/"
                 f"{record['theta_reference']:.4f} "
                 f"theta_gap={record['theta_gap']:.4f}")
    if shutil.get_terminal_size(fallback=(120, 24)).columns < 140:
        return f"{summary}\n      {details}\n      {tail}"
    return f"{summary} | {details} | {tail}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rrnet_mead.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--init-from", help="Load model weights only into a new run.")
    return parser.parse_args()


def model_prediction(model, batch: dict, device: torch.device,
                     reference_task: bool,
                     relative_task: bool = False,
                     physical_task: bool = False) -> tuple[
                         torch.Tensor, torch.Tensor, torch.Tensor | None,
                         torch.Tensor | None, dict[str, torch.Tensor], dict]:
    if physical_task:
        raw_source_clean = batch["source_clean"]
        physical_group_size = (int(raw_source_clean.shape[1])
                               if raw_source_clean.ndim == 5 else 1)

        def physical_tensor(key: str) -> torch.Tensor:
            value = batch[key]
            if physical_group_size > 1:
                if value.ndim < 2 or value.shape[1] != physical_group_size:
                    raise ValueError(
                        f"Physical grouped field {key!r} does not share group "
                        f"size {physical_group_size}: shape={tuple(value.shape)}")
                value = value.flatten(0, 1)
            return value.to(device, non_blocking=True)

        source_clean = physical_tensor("source_clean")
        reference_clean = physical_tensor("reference_clean")
        relight_mask = physical_tensor("relight_mask")
        skin_mask = physical_tensor("skin_mask")
        source_light_mask = physical_tensor("source_light_mask")
        reference_mask = physical_tensor("reference_mask")
        reference_relight_mask = physical_tensor("reference_relight_mask")
        source_theta_target = physical_tensor("source_theta_target")
        reference_theta_target = physical_tensor("reference_theta_target")
        uniform_group = physical_tensor("uniform_group").bool()
        with torch.no_grad():
            source_depth = model.depth(source_clean)
            reference_depth = model.depth(reference_clean)
            source_light, _ = model.illumination_from_theta(
                source_depth, source_theta_target)
            reference_on_source, _ = model.illumination_from_theta(
                source_depth, reference_theta_target)
            reference_light, _ = model.illumination_from_theta(
                reference_depth, reference_theta_target)
            source = model.renderer.blend_relight(
                source_clean * source_light, source_clean, relight_mask
            ).clamp(0.0, 0.999)
            target = model.renderer.blend_relight(
                source_clean * reference_on_source, source_clean, relight_mask
            ).clamp(0.0, 0.999)
            reference = model.renderer.blend_relight(
                reference_clean * reference_light,
                reference_clean,
                reference_relight_mask,
            ).clamp(0.0, 0.999)
        if physical_group_size > 1:
            prediction = model.forward_physical_grouped(
                source, reference, physical_group_size, uniform_group,
                relight_mask, reference_mask, source_light_mask,
                source_depth_override=source_depth,
                reference_depth_override=reference_depth)
        else:
            prediction = model(
                source, reference, relight_mask, reference_mask,
                source_light_mask,
                source_depth_override=source_depth,
                reference_depth_override=reference_depth)
        denormalizer = model.base.lprm.denormalize
        source_theta_normalized_target = (
            source_theta_target - denormalizer.mean
        ) / denormalizer.std
        reference_theta_normalized_target = (
            reference_theta_target - denormalizer.mean
        ) / denormalizer.std
        loss_extras = {
            "source_clean": source_clean,
            "reference": reference,
            "reference_clean": reference_clean,
            "source_light_mask": source_light_mask,
            "reference_mask": reference_mask,
            "group_size": physical_group_size,
            "uniform_group_mask": uniform_group,
            "source_theta_normalized_target": source_theta_normalized_target,
            "reference_theta_normalized_target": reference_theta_normalized_target,
            "source_theta_target": source_theta_target,
            "reference_theta_target": reference_theta_target,
        }
        return source, target, relight_mask, skin_mask, prediction, loss_extras

    raw_source = batch["input"]
    group_size = int(raw_source.shape[1]) if raw_source.ndim == 5 else 1

    def tensor(key: str) -> torch.Tensor:
        value = batch[key]
        if group_size > 1:
            if value.ndim < 2 or value.shape[1] != group_size:
                raise ValueError(
                    f"Grouped batch field {key!r} does not share group size "
                    f"{group_size}: shape={tuple(value.shape)}"
                )
            value = value.flatten(0, 1)
        return value.to(device, non_blocking=True)

    source = tensor("input")
    target = tensor("target")
    relight_mask = batch.get("relight_mask")
    skin_mask = batch.get("skin_mask")
    if relight_mask is not None:
        relight_mask = tensor("relight_mask")
    if skin_mask is not None:
        skin_mask = tensor("skin_mask")
    if reference_task:
        reference = tensor("reference")
        reference_mask = batch.get("reference_mask")
        if reference_mask is not None:
            reference_mask = tensor("reference_mask")
        if relative_task:
            source_light_mask = tensor("source_light_mask")
            if group_size > 1:
                prediction = model.forward_grouped(
                    source, reference, group_size, relight_mask,
                    reference_mask, source_light_mask)
            else:
                prediction = model(
                    source, reference, relight_mask, reference_mask,
                    source_light_mask)
            loss_extras = {
                "source_clean": tensor("source_clean"),
                "reference": reference,
                "reference_clean": tensor("reference_clean"),
                "source_light_mask": source_light_mask,
                "reference_mask": reference_mask,
                "group_size": group_size,
            }
        else:
            prediction = model(source, reference, relight_mask, reference_mask)
            loss_extras = {}
    else:
        prediction = model(source, relight_mask=relight_mask)
        loss_extras = {}
    return source, target, relight_mask, skin_mask, prediction, loss_extras


def set_dataset_epoch(dataset, epoch: int) -> None:
    """Update an epoch-aware dataset, including one wrapped by Subset."""
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    setter = getattr(current, "set_epoch", None)
    if setter is not None:
        setter(epoch)


@torch.no_grad()
def validate(model, loader: DataLoader, criterion: RRNetLoss,
             device: torch.device, max_batches: int,
             reference_task: bool = False,
             relative_task: bool = False,
             physical_task: bool = False) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        source, target, relight_mask, skin_mask, prediction, loss_extras = model_prediction(
            model, batch, device, reference_task, relative_task, physical_task)
        losses = criterion(
            prediction, target, source, relight_mask, skin_mask, **loss_extras)
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
        if count >= max_batches:
            break
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.resume and args.init_from:
        raise ValueError("Use --resume or --init-from, not both.")
    config = load_config(args.config)
    if 'seed' in config['training']:
        torch.manual_seed(int(config['training']['seed']))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config['training']['seed']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = config.get("task", "canonical").lower()
    physical_task = task == "reference_relative_physical"
    relative_task = task in {"reference_relative", "reference_relative_physical"}
    reference_task = task in {
        "reference", "reference_theta", "reference_relative",
        "reference_relative_physical",
    }
    if relative_task:
        model = ReferenceRelativeRRNet(**model_kwargs(config, args.config)).to(device)
    elif task == "reference_theta":
        model = ReferenceThetaRRNet(**model_kwargs(config, args.config)).to(device)
    elif task == "reference":
        model = ReferenceRRNet(**model_kwargs(config, args.config)).to(device)
    else:
        model = RRNet(**model_kwargs(config, args.config)).to(device)
    criterion = (ReferenceRelativeLoss(**config["loss"]).to(device)
                 if relative_task else RRNetLoss(**config["loss"]).to(device))
    metadata_file = config["data"].get("metadata_file", "metadata.csv")
    use_masks = bool(config["data"].get("use_masks", False))
    if physical_task:
        train_set = MEADPhysicalReferencePairs(
            config["data"]["root"], "train", metadata_file=metadata_file,
            seed=int(config["data"].get("sampling_seed", 20260924)),
            variants_per_source=int(
                config["data"].get("reference_variants_per_source", 4)),
            num_lights=int(config["model"].get("num_lights", 9)),
            dynamic_epoch=bool(
                config["data"].get("reference_dynamic_epoch", True)),
            reference_lighting_weights=config["data"].get(
                "reference_lighting_weights"),
            group_size=int(config["data"].get("physical_group_size", 1)),
            reference_sensitivity_fraction=float(config["data"].get(
                "reference_sensitivity_fraction", 0.0)),
        )
        val_set = MEADPhysicalReferencePairs(
            config["data"]["root"], "val", metadata_file=metadata_file,
            seed=int(config["data"].get("sampling_seed", 20260924)) + 1,
            variants_per_source=1,
            num_lights=int(config["model"].get("num_lights", 9)),
            dynamic_epoch=False,
            reference_lighting_weights=config["data"].get(
                "reference_lighting_weights"),
            group_size=int(config["data"].get(
                "validation_physical_group_size",
                config["data"].get("physical_group_size", 1))),
            reference_sensitivity_fraction=float(config["data"].get(
                "validation_reference_sensitivity_fraction", 0.0)),
        )
    elif reference_task:
        train_set = MEADReferenceTriplets(
            config["data"]["root"], "train", metadata_file=metadata_file,
            manifest_file=config["data"].get("manifest_file", "generation_manifest.json"),
            seed=int(config["data"].get("sampling_seed", 20260906)),
            variants_per_source=int(config["data"].get("reference_variants_per_source", 4)),
            grouped_source_count=int(config["data"].get("grouped_source_count", 1)),
            dynamic_epoch=bool(config["data"].get("reference_dynamic_epoch", True)),
            source_category_weights=config["data"].get("source_category_weights"),
            target_category_weights=config["data"].get("target_category_weights"),
        )
        val_set = MEADReferenceTriplets(
            config["data"]["root"], "val", metadata_file=metadata_file,
            manifest_file=config["data"].get("manifest_file", "generation_manifest.json"),
            seed=int(config["data"].get("sampling_seed", 20260906)) + 1,
            variants_per_source=1,
            grouped_source_count=int(
                config["data"].get(
                    "validation_grouped_source_count",
                    config["data"].get("grouped_source_count", 1),
                )
            ),
            dynamic_epoch=False,
        )
    else:
        train_set = MEADPairs(config["data"]["root"], "train", metadata_file,
                              load_masks=use_masks)
        val_set = MEADPairs(config["data"]["root"], "val", metadata_file,
                            load_masks=use_masks)
    max_train_samples = int(config["data"].get("max_train_samples", 0))
    max_val_samples = int(config["data"].get("max_val_samples", 0))
    if max_train_samples:
        train_set = Subset(train_set, range(min(max_train_samples, len(train_set))))
    if max_val_samples:
        val_set = Subset(val_set, range(min(max_val_samples, len(val_set))))
    loader_args = dict(batch_size=config["training"]["batch_size"],
                       num_workers=config["training"]["workers"], pin_memory=True)
    sampling = None if reference_task or isinstance(train_set, Subset) else config["data"].get("train_sampling")
    if physical_task:
        sampling_description = (
            "exact cross-person physical groups with known theta_in/theta_ref"
            f" | group={config['data'].get('physical_group_size', 1)}"
            f" | reference-sensitivity={float(config['data'].get('reference_sensitivity_fraction', 0.0)):.0%}")
    elif reference_task:
        sampling_description = "dynamic cross-person reference triplets per epoch"
        grouped_source_count = int(config["data"].get("grouped_source_count", 1))
        if grouped_source_count > 1:
            sampling_description += (
                f" | {grouped_source_count} distinct input lights share one "
                "reference and target"
            )
        for label, weights in (
                ("source", config["data"].get("source_category_weights")),
                ("target", config["data"].get("target_category_weights"))):
            if weights:
                total_weight = sum(float(value) for value in weights.values())
                mix = ",".join(
                    f"{key}:{float(value) / total_weight:.1%}"
                    for key, value in weights.items())
                sampling_description += f" | {label}=({mix})"
    else:
        sampling_description = "uniform rows"
    if sampling:
        sample_weights = category_sample_weights(train_set.rows, sampling)
        sampling_generator = torch.Generator()
        sampling_generator.manual_seed(int(config["data"].get("sampling_seed", 20260906)))
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(train_set),
            replacement=True,
            generator=sampling_generator,
        )
        train_loader = DataLoader(train_set, sampler=train_sampler, shuffle=False,
                                  drop_last=True, **loader_args)
        probability_sum = sum(float(value) for value in sampling.values())
        normalized = {key: float(value) / probability_sum
                      for key, value in sampling.items()}
        sampling_description = ", ".join(
            f"{key}={value:.2%}" for key, value in normalized.items()
        )
    else:
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    if physical_task:
        validation_count = (int(config['training']['validation_batches'])
                            * int(config['training']['batch_size']))
        validation_indices = balanced_validation_indices(
            val_set.people, val_set.variants_per_source, validation_count)
        print('Validation identities:', sorted(set(
            val_set.people[i // val_set.variants_per_source]
            for i in validation_indices)), flush=True)
        val_set = Subset(val_set, validation_indices)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    scaler = GradScaler('cuda', enabled=bool(config["training"]["amp"] and device.type == "cuda"))
    steps = int(config["training"]["iterations"])
    scheduler_name = config["training"].get("scheduler", "none").lower()
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps,
            eta_min=float(config["training"].get("min_learning_rate", 0.0)),
        )
    elif scheduler_name == "warmup_cosine":
        warmup_steps = int(config["training"].get("warmup_steps", 0))
        warmup_start_factor = float(config["training"].get("warmup_start_factor", 0.1))
        eta_min = float(config["training"].get("min_learning_rate", 0.0))
        base_learning_rate = float(config["training"]["learning_rate"])
        if not 0 <= warmup_steps < steps:
            raise ValueError("warmup_steps must be non-negative and smaller than iterations")
        if not 0 < warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must be in (0, 1]")

        def warmup_cosine_factor(step: int) -> float:
            if step < warmup_steps:
                return warmup_start_factor + (1.0 - warmup_start_factor) * (step / max(warmup_steps, 1))
            progress = (step - warmup_steps) / max(steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
            return (eta_min / base_learning_rate) + (1.0 - eta_min / base_learning_rate) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine_factor)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name!r}")
    output_root = Path(config["training"]["output_dir"])
    if args.resume:
        output_dir = Path(args.resume).resolve().parent
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root / f"run_{timestamp}"
        suffix = 1
        while output_dir.exists():
            output_dir = output_root / f"run_{timestamp}_{suffix:02d}"
            suffix += 1
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    if not run_config_path.exists():
        with run_config_path.open("w", encoding="utf-8") as handle:
            json.dump({"run_directory": str(output_dir.resolve()), **config},
                      handle, ensure_ascii=False, indent=2)
    text_log_path = output_dir / "train.log"
    append_log(text_log_path,
               f"Run directory: {output_dir.resolve()} | device: {device} | "
               f"target steps: {config['training']['iterations']} | "
               f"batch size: {config['training']['batch_size']} | "
               f"AMP: {scaler.is_enabled()}")
    append_log(text_log_path, f"Train sampling: {sampling_description}")
    append_log(text_log_path, f"Mask-aware training: {use_masks}")
    append_log(text_log_path, f"Training task: {task}")
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint["step"])
        with (output_dir / f"resume_config_step_{start_step:07d}.json").open("w", encoding="utf-8") as handle:
            json.dump({"run_directory": str(output_dir.resolve()),
                       "resume_checkpoint": str(Path(args.resume).resolve()),
                       **config}, handle, ensure_ascii=False, indent=2)
        append_log(text_log_path, f"Resumed checkpoint: {Path(args.resume).resolve()} at step {start_step}.")
    elif args.init_from:
        checkpoint = torch.load(args.init_from, map_location="cpu", weights_only=False)
        incoming = checkpoint["model"] if "model" in checkpoint else checkpoint
        if reference_task and not any(key.startswith("base.") for key in incoming):
            incoming = {f"base.{key}": value for key, value in incoming.items()}
        if physical_task:
            # Preserve the new config's parameter statistics.  The historical
            # checkpoint statistics were fitted to an ambiguous decomposition.
            incoming = {
                key: value for key, value in incoming.items()
                if not key.endswith("lprm.denormalize.mean")
                and not key.endswith("lprm.denormalize.std")
            }
            if bool(config["training"].get("reset_light_heads_on_init", True)):
                incoming = {
                    key: value for key, value in incoming.items()
                    if ".lprm.r0." not in key and ".lprm.r1." not in key
                }
        incompatible = model.load_state_dict(incoming, strict=False)
        append_log(
            text_log_path,
            f"Initialized model weights from: {Path(args.init_from).resolve()} | "
            f"new keys={len(incompatible.missing_keys)} ignored keys={len(incompatible.unexpected_keys)}",
        )

    loader_steps = max(len(train_loader), 1)
    train_epoch = start_step // loader_steps
    set_dataset_epoch(train_set, train_epoch)
    iterator = iter(train_loader)
    model.train()
    freeze_lprm_bn = bool(config['training'].get('freeze_lprm_bn', False))
    if freeze_lprm_bn:
        append_log(text_log_path,
                   'LPRM BatchNorm uses fixed running statistics during training; '
                   'affine weights and other model parameters remain trainable.')
    freeze_encoder_steps = int(
        config["training"].get("freeze_encoder_steps", 0))
    if freeze_encoder_steps < 0:
        raise ValueError("freeze_encoder_steps must be non-negative")
    lighting_encoder = (
        model.base.lprm.encoder if hasattr(model, "base")
        else model.lprm.encoder)
    if start_step < freeze_encoder_steps:
        lighting_encoder.requires_grad_(False)
        lighting_encoder.eval()
        append_log(
            text_log_path,
            f"Lighting encoder frozen until step {freeze_encoder_steps}; "
            "new regression heads train first.",
        )
    log_every = int(config["training"].get("log_every", 10))
    running = {
        "total": 0.0, "pixel": 0.0, "relight": 0.0, "skin": 0.0,
        "highlight": 0.0, "background": 0.0, "roi": 0.0,
        "dark": 0.0, "overexposure_log": 0.0,
        "chroma": 0.0, "regularization": 0.0,
    }
    if relative_task:
        running.update({
            "illum_source": 0.0,
            "illum_reference": 0.0,
            "illum_target": 0.0,
            "gain_log": 0.0,
            "gain_mean": 0.0,
            "gain_face_mean": 0.0,
            "gain_target_mean": 0.0,
            "dark_log_lift": 0.0,
            "consistency": 0.0,
            "reference_contrast": 0.0,
            "gain_gradient": 0.0,
            "illum_exposure": 0.0,
            "illum_shape": 0.0,
            "illum_shape_gradient": 0.0,
            "illum_variance": 0.0,
            "ambient_ratio": 0.0,
            "shape_consistency": 0.0,
            "theta_gap": 0.0,
            "theta_source": 0.0,
            "theta_reference": 0.0,
        })
    running_count = 0
    started_at = time.time()
    last_log_time = started_at
    for step in range(start_step, steps):
        if step == freeze_encoder_steps and freeze_encoder_steps > 0:
            lighting_encoder.requires_grad_(True)
            lighting_encoder.train()
            append_log(text_log_path, f"Lighting encoder unfrozen at step {step}.")
        try:
            batch = next(iterator)
        except StopIteration:
            train_epoch += 1
            set_dataset_epoch(train_set, train_epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        if freeze_lprm_bn:
            lprm = model.base.lprm if hasattr(model, 'base') else model.lprm
            for module in lprm.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        with autocast('cuda', enabled=scaler.is_enabled()):
            source, target, relight_mask, skin_mask, prediction, loss_extras = model_prediction(
                model, batch, device, reference_task, relative_task,
                physical_task)
            losses = criterion(
                prediction, target, source, relight_mask, skin_mask,
                **loss_extras)
        scaler.scale(losses["total"]).backward()
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_updated = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
        if scheduler is not None and optimizer_updated:
            scheduler.step()
        current = step + 1
        for key in running:
            running[key] += float(losses[key].detach())
        running_count += 1
        if current % log_every == 0 or current == steps:
            now = time.time()
            interval_seconds = now - last_log_time
            elapsed_seconds = now - started_at
            completed_this_run = max(current - start_step, 1)
            seconds_per_step = elapsed_seconds / completed_this_run
            eta_seconds = max(steps - current, 0) * seconds_per_step
            record = {
                "step": current,
                "epoch": round(current / loader_steps, 4),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": round(elapsed_seconds, 3),
                "interval_seconds": round(interval_seconds, 3),
                "steps_per_second": round(1.0 / max(seconds_per_step, 1e-9), 4),
                "eta_seconds": round(eta_seconds, 3),
                **{key: value / running_count for key, value in running.items()},
            }
            with (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            append_log(
                text_log_path,
                "TRAIN "
                f"step {current}/{steps} epoch={record['epoch']:.3f} | "
                f"total={record['total']:.6f} relight={record['relight']:.6f} "
                f"skin={record['skin']:.6f} highlight={record['highlight']:.6f} "
                f"background={record['background']:.6f} roi={record['roi']:.6f} "
                f"dark={record['dark']:.6f} "
                f"over={record['overexposure_log']:.6f} "
                f"chroma={record['chroma']:.6f} "
                f"reg={record['regularization']:.6f} | "
                + (f"Lin={record['illum_source']:.6f} "
                   f"Lref={record['illum_reference']:.6f} "
                   f"Ltgt={record['illum_target']:.6f} "
                   f"Lgain={record['gain_log']:.6f} "
                   f"gain={record['gain_face_mean']:.4f}/"
                   f"{record['gain_target_mean']:.4f} "
                   f"Ldark={record['dark_log_lift']:.6f} "
                   f"Lcons={record['consistency']:.6f} "
                   f"Lrefc={record['reference_contrast']:.6f} "
                   f"Lgrad={record['gain_gradient']:.6f} "
                   f"Lexp={record['illum_exposure']:.6f} "
                   f"Lshape={record['illum_shape']:.6f}/"
                   f"{record['illum_shape_gradient']:.6f} "
                   f"Lvar={record['illum_variance']:.6f} "
                   f"Lambient={record['ambient_ratio']:.6f} "
                   f"LshapeCons={record['shape_consistency']:.6f} "
                   f"Ltheta={record['theta_source']:.6f}/"
                   f"{record['theta_reference']:.6f} "
                   f"theta_gap={record['theta_gap']:.6f} | "
                   if relative_task else "")
                + f"lr={record['learning_rate']:.2e} | "
                f"speed={record['steps_per_second']:.3f} step/s | "
                f"elapsed={format_duration(elapsed_seconds)} | ETA={format_duration(eta_seconds)}",
                train_terminal_message(record, current, steps, elapsed_seconds, eta_seconds),
            )
            running = {key: 0.0 for key in running}
            running_count = 0
            last_log_time = now
        if current % int(config["training"]["validate_every"]) == 0:
            validation_started_at = time.time()
            metrics = validate(model, val_loader, criterion, device,
                               int(config["training"]["validation_batches"]),
                               reference_task, relative_task, physical_task)
            with (output_dir / "val_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": current, **metrics}) + "\n")
            append_log(
                text_log_path,
                "VALID "
                f"step {current}/{steps} | total={metrics['total']:.6f} "
                f"relight={metrics['relight']:.6f} skin={metrics['skin']:.6f} "
                f"highlight={metrics['highlight']:.6f} "
                f"background={metrics['background']:.6f} roi={metrics['roi']:.6f} "
                f"dark={metrics['dark']:.6f} "
                f"over={metrics['overexposure_log']:.6f} "
                f"chroma={metrics['chroma']:.6f} "
                f"reg={metrics['regularization']:.6f} | "
                + (f"Lin={metrics['illum_source']:.6f} "
                   f"Lref={metrics['illum_reference']:.6f} "
                   f"Ltgt={metrics['illum_target']:.6f} "
                   f"Lgain={metrics['gain_log']:.6f} "
                   f"gain={metrics['gain_face_mean']:.4f}/"
                   f"{metrics['gain_target_mean']:.4f} "
                   f"Ldark={metrics['dark_log_lift']:.6f} "
                   f"Lcons={metrics['consistency']:.6f} "
                   f"Lrefc={metrics['reference_contrast']:.6f} "
                   f"Lgrad={metrics['gain_gradient']:.6f} "
                   f"Lexp={metrics['illum_exposure']:.6f} "
                   f"Lshape={metrics['illum_shape']:.6f}/"
                   f"{metrics['illum_shape_gradient']:.6f} "
                   f"Lvar={metrics['illum_variance']:.6f} "
                   f"Lambient={metrics['ambient_ratio']:.6f} "
                   f"LshapeCons={metrics['shape_consistency']:.6f} "
                   f"theta_gap={metrics['theta_gap']:.6f} | "
                   if relative_task else "")
                + f"duration={format_duration(time.time() - validation_started_at)}",
            )
        if current % int(config["training"]["save_every"]) == 0 or current == steps:
            checkpoint_path = output_dir / f"rrnet_step_{current:07d}.pt"
            torch.save({"step": current, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "config": config}, checkpoint_path)
            append_log(text_log_path, f"CHECKPOINT saved: {checkpoint_path.name}")
    append_log(text_log_path, "Training completed.")


if __name__ == "__main__":
    main()
