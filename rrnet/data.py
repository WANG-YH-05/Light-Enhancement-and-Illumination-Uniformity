"""Paired MEAD dataset adapter."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_mask(path: Path) -> torch.Tensor:
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).contiguous()


def category_sample_weights(rows: list[dict[str, str]],
                            probabilities: Mapping[str, float]) -> torch.Tensor:
    """Return per-row weights whose category masses match probabilities."""
    counts = Counter(row["degradation_id"] for row in rows)
    configured = set(probabilities)
    available = set(counts)
    if configured != available:
        missing = sorted(available - configured)
        unknown = sorted(configured - available)
        raise ValueError(f"Sampling categories mismatch; missing={missing}, unknown={unknown}")
    values = {name: float(value) for name, value in probabilities.items()}
    if any(value < 0.0 for value in values.values()):
        raise ValueError("Sampling probabilities must be non-negative")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("Sampling probabilities must have a positive sum")
    normalized = {name: value / total for name, value in values.items()}
    return torch.as_tensor(
        [normalized[row["degradation_id"]] / counts[row["degradation_id"]]
         for row in rows],
        dtype=torch.double,
    )


class MEADPairs(Dataset):
    def __init__(self, root: str | Path, split: str = "train",
                 metadata_file: str = "metadata.csv",
                 load_masks: bool = False) -> None:
        self.root = Path(root)
        self.load_masks = load_masks
        metadata = self.root / metadata_file
        if not metadata.exists():
            raise FileNotFoundError(metadata)
        with metadata.open(newline="", encoding="utf-8-sig") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows for split={split!r} in {metadata}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        input_path = self.root / row["bad_light_frame"]
        target_path = self.root / row["clean_frame"]
        sample: dict[str, torch.Tensor | str] = {
            "input": load_rgb(input_path),
            "target": load_rgb(target_path),
            "sample_id": row["sample_id"],
            "person_id": row["person_id"],
            "clip_id": row["clip_id"],
            "frame_id": row["frame_id"],
        }
        if self.load_masks:
            for column in ("skin_mask", "relight_mask"):
                value = row.get(column)
                if not value:
                    raise ValueError(f"Missing {column!r} for sample {row['sample_id']}")
                sample[column] = load_mask(self.root / value)
        return sample
