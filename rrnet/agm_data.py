"""Grouped MEAD samples for explicit source-light removal supervision."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import load_mask, load_rgb


class MEADAGMGroups(Dataset):
    """Return several degradations of one frame with one shared clean target."""

    def __init__(self, root: str | Path, split: str = "train", *,
                 metadata_file: str = "metadata.csv", group_size: int = 4,
                 seed: int = 20260926, dynamic_epoch: bool = True,
                 include_identity: bool = True,
                 anchor_identity: bool = False) -> None:
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        self.root = Path(root)
        self.group_size = int(group_size)
        self.seed = int(seed)
        self.dynamic_epoch = bool(dynamic_epoch)
        self.include_identity = bool(include_identity)
        self.anchor_identity = bool(anchor_identity)
        if self.anchor_identity and not self.include_identity:
            raise ValueError("anchor_identity requires include_identity")
        self.epoch = 0
        with (self.root / metadata_file).open(newline="", encoding="utf-8-sig") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if include_identity or row["variant"] != "identity":
                groups[(row["person_id"], row["clip_id"], row["frame_id"])].append(row)
        self.groups = [sorted(value, key=lambda row: row["variant"])
                       for value in groups.values()
                       if len({row["variant"] for row in value}) >= group_size
                       and (not self.anchor_identity or any(
                           row["variant"] == "identity" for row in value))]
        if not self.groups:
            raise ValueError(f"No AGM groups for split={split!r}")

    def set_epoch(self, epoch: int) -> None:
        if self.dynamic_epoch:
            self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | list[str]]:
        rows = self.groups[index]
        rng = np.random.default_rng(self.seed + self.epoch * len(self.groups) + index)
        if self.anchor_identity:
            identity_index = next(i for i, row in enumerate(rows)
                                  if row["variant"] == "identity")
            others = [i for i in range(len(rows)) if i != identity_index]
            chosen_indices = [identity_index, *rng.choice(
                others, size=self.group_size - 1, replace=False).tolist()]
            rng.shuffle(chosen_indices)
        else:
            chosen_indices = rng.choice(len(rows), size=self.group_size,
                                        replace=False)
        selected = [rows[int(value)] for value in chosen_indices]
        first = selected[0]
        clean = load_rgb(self.root / first["clean_frame"])
        relight = load_mask(self.root / first["relight_mask"])
        skin = load_mask(self.root / first["skin_mask"])
        return {
            "input": torch.stack([
                load_rgb(self.root / row["bad_light_frame"]) for row in selected
            ]),
            "target": clean.unsqueeze(0).expand(self.group_size, -1, -1, -1).clone(),
            "relight_mask": relight.unsqueeze(0).expand(
                self.group_size, -1, -1, -1).clone(),
            "skin_mask": skin.unsqueeze(0).expand(
                self.group_size, -1, -1, -1).clone(),
            "variants": [row["variant"] for row in selected],
            "identity_flags": torch.tensor(
                [row["variant"] == "identity" for row in selected],
                dtype=torch.bool),
            "sample_id": first["sample_id"],
        }
