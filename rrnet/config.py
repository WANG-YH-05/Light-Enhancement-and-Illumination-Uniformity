"""YAML configuration and construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str | None, base: Path) -> str | None:
    if not value:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def model_kwargs(config: dict[str, Any], config_path: str | Path) -> dict[str, Any]:
    values = dict(config["model"])
    base = Path(config_path).resolve().parent.parent
    for key in ("statistics_path", "depth_vendor_root", "depth_checkpoint"):
        values[key] = resolve_path(values.get(key), base)
    return values
