"""Virtual-light parameter packing and normalization."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


FIELDS_PER_LIGHT = 10  # c(3), d(3), p(3), s(1)


def parameter_dim(num_lights: int) -> int:
    return num_lights * FIELDS_PER_LIGHT + 3  # RGB ambient term


def default_parameter_statistics(num_lights: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable fallback only; the paper requires dataset-derived statistics."""
    lights = []
    columns = max(1, int(math.ceil(math.sqrt(num_lights))))
    for index in range(num_lights):
        row, column = divmod(index, columns)
        px = (column + 0.5) / columns
        py = (row + 0.5) / columns
        direction = torch.tensor([px - 0.5, py - 0.5, 1.0])
        direction = direction / direction.norm()
        lights.extend([0.045, 0.045, 0.045])
        lights.extend(direction.tolist())
        lights.extend([px, py, 0.25, 1.0])
    mean = torch.tensor(lights + [0.60, 0.60, 0.60], dtype=torch.float32)
    one_light_std = [0.15] * 3 + [0.35] * 3 + [0.35] * 3 + [0.50]
    std = torch.tensor(one_light_std * num_lights + [0.25] * 3, dtype=torch.float32)
    return mean, std


class ParameterDenormalizer(nn.Module):
    """Equation (2): theta*=sigma_hat*(theta0+theta')+mu_hat."""

    def __init__(self, num_lights: int, statistics_path: str | None = None) -> None:
        super().__init__()
        mean, std = default_parameter_statistics(num_lights)
        if statistics_path:
            values = np.load(Path(statistics_path))
            mean = torch.as_tensor(values["mean"], dtype=torch.float32)
            std = torch.as_tensor(values["std"], dtype=torch.float32)
        expected = parameter_dim(num_lights)
        if mean.numel() != expected or std.numel() != expected:
            raise ValueError(f"Expected {expected} parameter statistics values")
        self.register_buffer("mean", mean.reshape(1, -1))
        self.register_buffer("std", std.clamp_min(1e-6).reshape(1, -1))

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        return self.std * normalized + self.mean


def unpack_lights(theta: torch.Tensor, num_lights: int) -> dict[str, torch.Tensor]:
    batch = theta.shape[0]
    expected = parameter_dim(num_lights)
    if theta.shape[-1] != expected:
        raise ValueError(f"Expected theta[..., {expected}], got {tuple(theta.shape)}")
    lights = theta[:, :num_lights * FIELDS_PER_LIGHT].reshape(batch, num_lights, FIELDS_PER_LIGHT)
    return {
        "color": lights[..., 0:3],
        "direction": lights[..., 3:6],
        "position": lights[..., 6:9],
        "attenuation": lights[..., 9:10],
        "ambient": theta[:, -3:],
    }
