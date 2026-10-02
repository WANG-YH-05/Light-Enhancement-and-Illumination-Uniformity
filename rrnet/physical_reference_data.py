"""Exact cross-person reference pairs generated from known virtual lights.

The dataset only loads clean MEAD portraits, masks, and deterministic physical
light parameters.  Images are rendered on GPU inside train.py so Depth Anything
never runs in DataLoader workers.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import load_mask, load_rgb
from .lighting import FIELDS_PER_LIGHT, parameter_dim
from .reference_mask import face_attention_from_person_mask


REFERENCE_LIGHTING_MODES = ("dark", "normal", "bright")


def _stable_seed(*parts: object) -> int:
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def sample_physical_theta(rng: np.random.Generator,
                          num_lights: int = 9) -> torch.Tensor:
    """Sample valid, ordered virtual lights with a sparse spatial pattern.

    Light indices are anchored to a regular image-space grid.  This removes
    permutation ambiguity, making direct theta supervision meaningful.
    """
    if num_lights < 1:
        raise ValueError("num_lights must be positive")
    columns = max(1, int(np.ceil(np.sqrt(num_lights))))
    rows = int(np.ceil(num_lights / columns))
    uniform_case = bool(rng.random() < 0.20)
    active_count = (1 if uniform_case else
                    int(rng.integers(1, min(3, num_lights) + 1)))
    active = set(int(index) for index in rng.choice(
        num_lights, size=active_count, replace=False))
    lights: list[float] = []
    for index in range(num_lights):
        row, column = divmod(index, columns)
        base_x = (column + 0.5) / columns
        base_y = (row + 0.5) / rows
        px = float(np.clip(base_x + rng.normal(0.0, 0.055), 0.04, 0.96))
        py = float(np.clip(base_y + rng.normal(0.0, 0.055), 0.04, 0.96))
        pz = float(rng.uniform(0.08, 0.42))
        direction = np.asarray([
            0.65 * (0.5 - px) + rng.normal(0.0, 0.08),
            0.65 * (0.5 - py) + rng.normal(0.0, 0.08),
            rng.uniform(0.72, 1.0),
        ], dtype=np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
        if index in active:
            intensity = float(rng.uniform(
                0.003, 0.018) if uniform_case else
                rng.uniform(0.040, 0.115))
        else:
            intensity = float(rng.uniform(0.0, 0.004))
        # Keep physical-pair pretraining achromatic. Colour handling remains a
        # separate, bounded design choice and cannot leak a reference identity.
        lights.extend([intensity, intensity, intensity])
        lights.extend(float(value) for value in direction)
        lights.extend([px, py, pz, float(rng.uniform(0.55, 1.75))])
    ambient = float(rng.uniform(
        0.58, 0.88) if uniform_case else rng.uniform(0.16, 0.58))
    theta = torch.tensor(
        lights + [ambient, ambient, ambient], dtype=torch.float32)
    if theta.numel() != parameter_dim(num_lights):
        raise AssertionError(
            f"sampled {theta.numel()} parameters, expected {parameter_dim(num_lights)}")
    return theta


def sample_reference_theta(rng: np.random.Generator, num_lights: int = 9,
                           mode: str = "normal") -> torch.Tensor:
    """Sample one of the only three allowed reference-lighting regimes.

    A reference participant is deliberately not assigned side, back, top, or
    mixed lighting. Those remain source-only degradations to be corrected.
    """
    if mode not in REFERENCE_LIGHTING_MODES:
        raise ValueError(
            f"Unknown reference lighting mode {mode!r}; "
            f"expected one of {REFERENCE_LIGHTING_MODES}")
    if num_lights < 1:
        raise ValueError("num_lights must be positive")
    ambient_ranges = {
        "dark": (0.25, 0.42),
        "normal": (0.68, 0.88),
        "bright": (0.98, 1.12),
    }
    local_ranges = {
        "dark": (0.000, 0.006),
        "normal": (0.002, 0.012),
        "bright": (0.003, 0.016),
    }
    columns = max(1, int(np.ceil(np.sqrt(num_lights))))
    rows = int(np.ceil(num_lights / columns))
    local_low, local_high = local_ranges[mode]
    lights: list[float] = []
    for index in range(num_lights):
        row, column = divmod(index, columns)
        px = (column + 0.5) / columns
        py = (row + 0.5) / rows
        direction = np.asarray([
            rng.normal(0.0, 0.04), rng.normal(0.0, 0.04), 1.0,
        ], dtype=np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-6)
        intensity = float(rng.uniform(local_low, local_high))
        lights.extend([intensity, intensity, intensity])
        lights.extend(float(value) for value in direction)
        lights.extend([px, py, float(rng.uniform(0.16, 0.32)),
                       float(rng.uniform(0.8, 1.3))])
    ambient = float(rng.uniform(*ambient_ranges[mode]))
    return torch.tensor(lights + [ambient, ambient, ambient], dtype=torch.float32)


def _illumination_signature(theta: torch.Tensor, num_lights: int) -> torch.Tensor:
    """Keep only light-energy and ambient terms when comparing source lights."""
    lights = theta[:-3].reshape(num_lights, 10)[:, :3].flatten()
    return torch.cat((lights, theta[-3:]))


def sample_diverse_source_thetas(rng: np.random.Generator, num_lights: int,
                                 count: int) -> list[torch.Tensor]:
    """Produce visibly distinct input-light settings for one uniformity group."""
    samples: list[torch.Tensor] = []
    for _ in range(count):
        candidate = sample_physical_theta(rng, num_lights)
        # Do not let a group differ only in inactive light positions. The
        # threshold is measured on active RGB intensities plus ambient light.
        for _ in range(32):
            if not samples or min(
                float((_illumination_signature(candidate, num_lights)
                      - _illumination_signature(previous, num_lights)).abs().mean())
                for previous in samples
            ) >= 0.030:
                break
            candidate = sample_physical_theta(rng, num_lights)
        samples.append(candidate)
    return samples


class MEADPhysicalReferencePairs(Dataset):
    """Different-person physical reference pairs with exact theta labels."""

    def __init__(self, root: str | Path, split: str = "train", *,
                 metadata_file: str = "metadata.csv", seed: int = 20260924,
                 variants_per_source: int = 4, num_lights: int = 9,
                 dynamic_epoch: bool = True,
                 reference_lighting_weights: dict[str, float] | None = None,
                 group_size: int = 1,
                 reference_sensitivity_fraction: float = 0.0,
                 ) -> None:
        self.root = Path(root)
        self.split = split
        self.seed = int(seed)
        self.num_lights = int(num_lights)
        self.dynamic_epoch = bool(dynamic_epoch)
        self.epoch = 0
        if group_size < 1:
            raise ValueError("group_size must be at least one")
        if not 0.0 <= reference_sensitivity_fraction <= 1.0:
            raise ValueError("reference_sensitivity_fraction must be in [0, 1]")
        self.group_size = int(group_size)
        self.reference_sensitivity_fraction = float(reference_sensitivity_fraction)
        if reference_lighting_weights is None:
            reference_lighting_weights = {
                "dark": 1.0, "normal": 1.0, "bright": 1.0}
        unknown = sorted(set(reference_lighting_weights) - set(REFERENCE_LIGHTING_MODES))
        if unknown:
            raise ValueError(f"Unknown reference lighting modes: {unknown}")
        values = np.asarray(
            [float(reference_lighting_weights.get(mode, 0.0))
             for mode in REFERENCE_LIGHTING_MODES], dtype=np.float64)
        if np.any(values < 0.0) or not np.all(np.isfinite(values)) or values.sum() <= 0.0:
            raise ValueError("reference_lighting_weights must be finite, non-negative, and nonzero")
        self.reference_lighting_probabilities = values / values.sum()
        if variants_per_source < 1:
            raise ValueError("variants_per_source must be at least one")
        self.variants_per_source = int(variants_per_source)
        metadata = self.root / metadata_file
        with metadata.open(newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.DictReader(handle)
                    if row["split"] == split and row["degradation_id"] == "identity"]
        # metadata contains one identity row per clean frame.  Keep a defensive
        # de-duplication in case a regenerated manifest repeats a row.
        unique: dict[str, dict[str, str]] = {}
        for row in rows:
            unique.setdefault(row["clean_frame"], row)
        self.samples = [unique[key] for key in sorted(unique)]
        if not self.samples:
            raise ValueError(f"No identity rows for split={split!r} in {metadata}")
        self.people = [row["person_id"] for row in self.samples]
        if len(set(self.people)) < 2:
            raise ValueError("Physical reference pairs require at least two people")

    def __len__(self) -> int:
        return len(self.samples) * self.variants_per_source

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch) if self.dynamic_epoch else 0

    def _different_person(self, source_index: int,
                          rng: np.random.Generator) -> int:
        source_person = self.people[source_index]
        for _ in range(64):
            candidate = int(rng.integers(0, len(self.samples)))
            if self.people[candidate] != source_person:
                return candidate
        for offset in range(1, len(self.samples)):
            candidate = (source_index + offset) % len(self.samples)
            if self.people[candidate] != source_person:
                return candidate
        raise RuntimeError("Unable to select a different-person reference")

    def sample_parameters(self, index: int) -> dict:
        """Exact group sampler shared by image loading and train-only calibration."""
        source_index = (index // self.variants_per_source) % len(self.samples)
        variant = index % self.variants_per_source
        rng = np.random.default_rng(_stable_seed(
            self.seed, self.split, self.epoch, source_index, variant))
        reference_index = self._different_person(source_index, rng)
        mode = str(rng.choice(REFERENCE_LIGHTING_MODES,
                              p=self.reference_lighting_probabilities))
        sensitivity = False
        if self.group_size == 1:
            sources = [sample_physical_theta(rng, self.num_lights)]
            references = [sample_reference_theta(rng, self.num_lights, mode)]
            modes = [mode]
        else:
            sensitivity = bool(rng.random() < self.reference_sensitivity_fraction)
            if sensitivity:
                sources = [sample_physical_theta(rng, self.num_lights)] * self.group_size
                enabled = [m for m, p in zip(REFERENCE_LIGHTING_MODES,
                           self.reference_lighting_probabilities) if p > 0]
                if len(enabled) < 2:
                    sensitivity = False
                else:
                    modes = [enabled[i % len(enabled)] for i in range(self.group_size)]
                    rng.shuffle(modes)
                    references = [sample_reference_theta(rng, self.num_lights, m) for m in modes]
            if not sensitivity:
                sources = sample_diverse_source_thetas(rng, self.num_lights, self.group_size)
                references = [sample_reference_theta(rng, self.num_lights, mode)] * self.group_size
                modes = [mode] * self.group_size
        return dict(source_index=source_index, reference_index=reference_index,
                    source_theta_target=torch.stack(sources),
                    reference_theta_target=torch.stack(references),
                    reference_lighting_mode=tuple(modes), uniform_group=not sensitivity)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        parameters = self.sample_parameters(index)
        source_index = parameters['source_index']
        reference_index = parameters['reference_index']
        source_row = self.samples[source_index]
        reference_row = self.samples[reference_index]
        source_mask = load_mask(self.root / source_row["relight_mask"])
        reference_mask = load_mask(self.root / reference_row["relight_mask"])
        source_face = face_attention_from_person_mask(
            source_mask.squeeze(0).numpy())
        reference_face = face_attention_from_person_mask(
            reference_mask.squeeze(0).numpy())
        source_clean = load_rgb(self.root / source_row["clean_frame"])
        reference_clean = load_rgb(self.root / reference_row["clean_frame"])
        base = {
            "source_clean": source_clean,
            "reference_clean": reference_clean,
            "skin_mask": load_mask(self.root / source_row["skin_mask"]),
            "relight_mask": source_mask,
            "source_light_mask": torch.from_numpy(source_face).permute(2, 0, 1).contiguous(),
            "reference_mask": torch.from_numpy(reference_face).permute(2, 0, 1).contiguous(),
            "reference_relight_mask": reference_mask,
            "source_person_id": source_row["person_id"],
            "reference_person_id": reference_row["person_id"],
        }
        if self.group_size == 1:
            return {
                **base,
                "source_theta_target": parameters['source_theta_target'][0],
                "reference_theta_target": parameters['reference_theta_target'][0],
                "reference_lighting_mode": parameters['reference_lighting_mode'][0],
                "uniform_group": torch.tensor(True),
            }

        # Main groups: several deliberately different input lights must reach
        # one fixed reference light and therefore one identical target. A small
        # auxiliary portion reverses the axis (same input, changing reference)
        # so the network cannot safely ignore Lref.

        grouped: dict[str, torch.Tensor | str] = {}
        for key, value in base.items():
            if isinstance(value, torch.Tensor):
                grouped[key] = value.unsqueeze(0).expand(
                    self.group_size, *value.shape).contiguous()
            else:
                grouped[key] = value
        grouped.update({
            "source_theta_target": parameters['source_theta_target'],
            "reference_theta_target": parameters['reference_theta_target'],
            "reference_lighting_mode": parameters['reference_lighting_mode'],
            "uniform_group": torch.full((self.group_size,), parameters['uniform_group'],
                                        dtype=torch.bool),
        })
        return grouped
