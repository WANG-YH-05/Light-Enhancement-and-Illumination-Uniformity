"""Reference-conditioned MEAD triplets for learning a selected target light."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from tools.build_mead_rrnet_dataset_v2 import apply_scenario

from .data import load_mask, load_rgb
from .reference_mask import face_attention_from_person_mask


def _stable_seed(*parts: object) -> int:
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def _rgb_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2])


def source_chroma_with_target_luminance(source_clean: np.ndarray,
                                        lit_target: np.ndarray) -> np.ndarray:
    """Match target luminance by RGB gain, preserving source chromaticity.

    A single gain is applied to all three channels at every pixel.  The gain is
    capped before any channel clips, so neither the reference person's color nor
    clipping-induced hue shifts can enter the supervision target.
    """
    eps = np.float32(1.0e-4)
    source_luma = _luminance(source_clean)
    target_luma = _luminance(lit_target)
    gain = target_luma / np.maximum(source_luma, eps)
    max_channel = np.max(source_clean, axis=-1)
    max_gain = (np.float32(1.0) - eps) / np.maximum(max_channel, eps)
    gain = np.clip(gain, 0.0, max_gain)
    return (source_clean * gain[..., None]).astype(np.float32)


def _category_probabilities(
        categories: tuple[str, ...], weights: dict[str, float] | None,
        field_name: str) -> np.ndarray | None:
    """Validate optional category weights and return normalized probabilities."""
    if weights is None:
        return None
    unknown = sorted(set(weights) - set(categories))
    if unknown:
        raise ValueError(f"Unknown {field_name} categories: {unknown}")
    values = np.asarray(
        [float(weights.get(category, 0.0)) for category in categories],
        dtype=np.float64,
    )
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name} values must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError(f"{field_name} must contain positive probability mass")
    return values / total


class MEADReferenceTriplets(Dataset):
    """Create exact synthetic A->B relighting pairs with a different-person reference.

    Input A and reference B use already generated PNGs.  The target is generated
    on demand by applying the *same clip-level B lighting configuration and
    temporal phase* seen in the reference image to the source person's clean
    frame.  This prevents the network from learning category names as a proxy
    for the selected participant's actual lighting.
    """

    def __init__(self, root: str | Path, split: str = "train", *,
                 metadata_file: str = "metadata.csv",
                 manifest_file: str = "generation_manifest.json",
                 seed: int = 20260906,
                 variants_per_source: int = 4,
                 grouped_source_count: int = 1,
                 dynamic_epoch: bool = True,
                 source_category_weights: dict[str, float] | None = None,
                 target_category_weights: dict[str, float] | None = None) -> None:
        self.root = Path(root)
        self.seed = seed
        if variants_per_source < 1:
            raise ValueError("variants_per_source must be at least one.")
        self.variants_per_source = variants_per_source
        if grouped_source_count < 1:
            raise ValueError("grouped_source_count must be at least one.")
        self.grouped_source_count = int(grouped_source_count)
        self.dynamic_epoch = dynamic_epoch
        self.epoch = 0
        with (self.root / metadata_file).open(newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not rows:
            raise ValueError(f"No rows for split={split!r}")
        self.split = split
        grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in rows:
            grouped[row["clean_frame"]][row["degradation_id"]] = row
        self.samples = [grouped[key] for key in sorted(grouped)]
        self.categories = tuple(sorted({row["degradation_id"] for row in rows}))
        if "identity" in self.categories:
            self.categories = ("identity",) + tuple(
                name for name in self.categories if name != "identity")
        self.source_category_probabilities = _category_probabilities(
            self.categories, source_category_weights, "source_category_weights")
        self.target_category_probabilities = _category_probabilities(
            self.categories, target_category_weights, "target_category_weights")
        if self.grouped_source_count > len(self.categories):
            raise ValueError(
                "grouped_source_count cannot exceed the number of degradation "
                f"categories ({len(self.categories)})."
            )
        if (self.source_category_probabilities is not None
                and np.count_nonzero(self.source_category_probabilities) <
                self.grouped_source_count):
            raise ValueError(
                "source_category_weights must give positive probability to at "
                "least grouped_source_count categories."
            )
        expected = set(self.categories)
        incomplete = [next(iter(group.values()))["clean_frame"]
                      for group in self.samples if set(group) != expected]
        if incomplete:
            raise ValueError(f"Incomplete degradation groups, first={incomplete[0]}")

        manifest = json.loads((self.root / manifest_file).read_text(encoding="utf-8"))
        self.profiles = manifest["people"]
        self.clip_configs: dict[str, dict[str, dict[str, object]]] = {}
        for clip_key, configs in manifest["clips"].items():
            self.clip_configs[clip_key] = {
                str(config["category"]): dict(config) for config in configs
            }

        clips: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(self.samples):
            row = group["identity"]
            clips[self._clip_key(row)].append(index)
        self.progress: dict[int, float] = {}
        for indices in clips.values():
            indices.sort(key=lambda i: self.samples[i]["identity"]["frame_id"])
            denominator = max(len(indices) - 1, 1)
            for position, sample_index in enumerate(indices):
                self.progress[sample_index] = position / denominator

        self.people = [group["identity"]["person_id"] for group in self.samples]
        if len(set(self.people)) < 2:
            raise ValueError("Reference triplets require at least two people.")

    def _clip_key(self, row: dict[str, str]) -> str:
        return f'{self.split}/{row["person_id"]}/{row["clip_id"]}'

    def __len__(self) -> int:
        return len(self.samples) * self.variants_per_source

    def set_epoch(self, epoch: int) -> None:
        """Change deterministic triplet combinations for the next DataLoader epoch."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = int(epoch) if self.dynamic_epoch else 0

    def _reference_index(self, index: int, rng: np.random.Generator) -> int:
        source_person = self.people[index]
        for _ in range(64):
            candidate = int(rng.integers(0, len(self.samples)))
            if self.people[candidate] != source_person:
                return candidate
        # Deterministic fallback without constructing a quadratic person index.
        for offset in range(1, len(self.samples)):
            candidate = (index + offset) % len(self.samples)
            if self.people[candidate] != source_person:
                return candidate
        raise RuntimeError("Unable to select a different-person reference.")

    def _source_categories(self, rng: np.random.Generator) -> tuple[str, ...]:
        """Select distinct input lights for one shared reference/target group."""
        if self.grouped_source_count == 1:
            if self.source_category_probabilities is None:
                chosen = int(rng.integers(0, len(self.categories)))
                return (self.categories[chosen],)
            return (str(rng.choice(
                self.categories, p=self.source_category_probabilities)),)
        selected = rng.choice(
            self.categories,
            size=self.grouped_source_count,
            replace=False,
            p=self.source_category_probabilities,
        )
        return tuple(str(category) for category in selected)

    def __getitem__(
            self, index: int
    ) -> dict[str, torch.Tensor | str | tuple[str, ...]]:
        # Keep all deterministic reference variants for one source adjacent.
        # This also makes a small contiguous Subset a meaningful overfit test:
        # the model must map one identical source to several reference lights.
        source_index = (index // self.variants_per_source) % len(self.samples)
        epoch = self.epoch if self.dynamic_epoch else 0
        rng = np.random.default_rng(_stable_seed(
            self.seed, self.split, "reference", epoch, index))
        source_rng = np.random.default_rng(
            _stable_seed(self.seed, self.split, "source", epoch, source_index))
        target_rng = np.random.default_rng(
            _stable_seed(self.seed, self.split, "target", source_index))
        reference_index = self._reference_index(source_index, rng)
        # All reference variants of one source share the exact same input A.
        # Therefore a network cannot solve the task without looking at B.
        source_categories = self._source_categories(source_rng)
        variant_index = index % self.variants_per_source
        if self.target_category_probabilities is None:
            category_offset = int(target_rng.integers(0, len(self.categories)))
            target_category = self.categories[
                (category_offset + epoch * self.variants_per_source + variant_index)
                % len(self.categories)
            ]
        else:
            weighted_target_rng = np.random.default_rng(_stable_seed(
                self.seed, self.split, "weighted-target", epoch,
                source_index, variant_index))
            target_category = str(weighted_target_rng.choice(
                self.categories, p=self.target_category_probabilities))
        source_group = self.samples[source_index]
        reference_group = self.samples[reference_index]
        reference_row = reference_group[target_category]
        identity_row = source_group["identity"]
        reference_identity_row = reference_group["identity"]

        clean_path = self.root / identity_row["clean_frame"]
        relight_path = self.root / identity_row["relight_mask"]
        clean = _rgb_array(clean_path)
        if target_category == "identity":
            target = clean
        else:
            reference_identity = reference_identity_row
            clip_key = self._clip_key(reference_identity)
            config = dict(self.clip_configs[clip_key][target_category])
            config["profile"] = self.profiles[reference_identity["person_id"]]
            person_mask = np.asarray(
                Image.open(relight_path).convert("L"), dtype=np.float32
            ) / 255.0
            if person_mask.shape != clean.shape[:2]:
                from cv2 import INTER_LINEAR, resize
                person_mask = resize(person_mask, (clean.shape[1], clean.shape[0]),
                                     interpolation=INTER_LINEAR)
            lit_target = apply_scenario(
                clean, person_mask[..., None], config,
                self.progress[reference_index],
            )
            target = source_chroma_with_target_luminance(clean, lit_target)

        reference_person_mask = np.asarray(
            Image.open(self.root / reference_identity_row["relight_mask"]).convert("L"),
            dtype=np.float32,
        ) / 255.0
        reference_face_mask = face_attention_from_person_mask(reference_person_mask)
        source_person_mask = np.asarray(
            Image.open(relight_path).convert("L"), dtype=np.float32,
        ) / 255.0
        source_face_mask = face_attention_from_person_mask(source_person_mask)
        sample: dict[str, torch.Tensor | str | tuple[str, ...]] = {
            "input": torch.stack([
                load_rgb(self.root / source_group[category]["bad_light_frame"])
                for category in source_categories
            ]),
            "reference": load_rgb(self.root / reference_row["bad_light_frame"]),
            "source_clean": torch.from_numpy(np.ascontiguousarray(clean))
                              .permute(2, 0, 1).contiguous(),
            "reference_clean": load_rgb(
                self.root / reference_identity_row["clean_frame"]),
            "target": torch.from_numpy(np.ascontiguousarray(target))
                           .permute(2, 0, 1).contiguous(),
            "skin_mask": load_mask(self.root / identity_row["skin_mask"]),
            "relight_mask": load_mask(relight_path),
            "source_light_mask": torch.from_numpy(source_face_mask)
                                      .permute(2, 0, 1).contiguous(),
            "reference_mask": torch.from_numpy(reference_face_mask).permute(2, 0, 1).contiguous(),
            "reference_relight_mask": load_mask(
                self.root / reference_identity_row["relight_mask"]),
            "source_person_id": identity_row["person_id"],
            "reference_person_id": reference_row["person_id"],
            "source_category": source_categories,
            "target_category": target_category,
        }
        if self.grouped_source_count == 1:
            # Preserve the historical dataset interface for all old configs.
            sample["input"] = sample["input"][0]
            sample["source_category"] = source_categories[0]
            return sample

        # Only the input light varies inside a group. Repeating the common
        # tensors here keeps PyTorch's default collation and the old training
        # code path simple; train.py flattens [batch, group] before inference.
        for key in (
            "reference", "source_clean", "reference_clean", "target",
            "skin_mask", "relight_mask", "source_light_mask",
            "reference_mask", "reference_relight_mask",
        ):
            value = sample[key]
            assert isinstance(value, torch.Tensor)
            sample[key] = value.unsqueeze(0).expand(
                self.grouped_source_count, *value.shape).contiguous()
        return sample
