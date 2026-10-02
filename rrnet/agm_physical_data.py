"""Single-person known-light groups for the isolated AGM pilot.

No different-person reference image is read. Target light parameters are
sampled independently; the common target is rendered from the same clean frame.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import load_mask, load_rgb
from .physical_reference_data import (
    REFERENCE_LIGHTING_MODES, _stable_seed, sample_diverse_source_thetas,
    sample_reference_theta,
)


class MEADPhysicalAGMGroups(Dataset):
    def __init__(self, root: str | Path, split: str, *,
                 metadata_file: str = "metadata.csv", seed: int = 20260926,
                 variants_per_source: int = 2, group_size: int = 4,
                 num_lights: int = 9, dynamic_epoch: bool = True,
                 target_lighting_weights: dict[str, float] | None = None) -> None:
        if variants_per_source < 1 or group_size < 2 or num_lights < 1:
            raise ValueError("variants_per_source >= 1, group_size >= 2, num_lights >= 1 required")
        self.root = Path(root)
        self.split = split
        self.seed = int(seed)
        self.variants_per_source = int(variants_per_source)
        self.group_size = int(group_size)
        self.num_lights = int(num_lights)
        self.dynamic_epoch = bool(dynamic_epoch)
        self.epoch = 0
        weights = target_lighting_weights or {
            "dark": 0.5, "normal": 0.5, "bright": 0.0}
        unknown = set(weights) - set(REFERENCE_LIGHTING_MODES)
        if unknown:
            raise ValueError(f"Unknown target lighting modes: {sorted(unknown)}")
        values = np.asarray([float(weights.get(mode, 0.0))
                             for mode in REFERENCE_LIGHTING_MODES])
        if np.any(values < 0) or not np.isfinite(values).all() or values.sum() <= 0:
            raise ValueError("target_lighting_weights must be finite, nonnegative and nonzero")
        self.probabilities = values / values.sum()
        with (self.root / metadata_file).open(newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.DictReader(handle)
                    if row["split"] == split and row["degradation_id"] == "identity"]
        unique = {row["clean_frame"]: row for row in rows}
        self.samples = [unique[key] for key in sorted(unique)]
        if not self.samples:
            raise ValueError(f"No identity clean frames for {split!r}")
        self.people = [row["person_id"] for row in self.samples]

    def __len__(self) -> int:
        return len(self.samples) * self.variants_per_source

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch) if self.dynamic_epoch else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | tuple[str, ...]]:
        sample_index = (index // self.variants_per_source) % len(self.samples)
        variant = index % self.variants_per_source
        row = self.samples[sample_index]
        rng = np.random.default_rng(_stable_seed(
            self.seed, self.split, self.epoch, sample_index, variant))
        target_mode = str(rng.choice(
            REFERENCE_LIGHTING_MODES, p=self.probabilities))
        source_thetas = sample_diverse_source_thetas(
            rng, self.num_lights, self.group_size)
        target_theta = sample_reference_theta(rng, self.num_lights, target_mode)
        clean = load_rgb(self.root / row["clean_frame"])
        relight = load_mask(self.root / row["relight_mask"])
        skin = load_mask(self.root / row["skin_mask"])
        return {
            "source_clean": clean.unsqueeze(0).expand(
                self.group_size, -1, -1, -1).contiguous(),
            "relight_mask": relight.unsqueeze(0).expand(
                self.group_size, -1, -1, -1).contiguous(),
            "skin_mask": skin.unsqueeze(0).expand(
                self.group_size, -1, -1, -1).contiguous(),
            "source_theta": torch.stack(source_thetas),
            "target_theta": target_theta.unsqueeze(0).expand(
                self.group_size, -1).contiguous(),
            "target_lighting_mode": (target_mode,) * self.group_size,
            "person_id": row["person_id"],
            "sample_id": row["sample_id"],
        }
