"""Virtual-light parameter packing and normalization."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


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
    """Default: paper Eq. (2). Opt-in smooth_physical is our constrained extension."""

    def __init__(self, num_lights: int, statistics_path: str | None = None,
                 parameterization: str = "affine") -> None:
        super().__init__()
        if parameterization not in ("affine", "smooth_physical"):
            raise ValueError(f"Unknown light parameterization: {parameterization}")
        if parameterization == "smooth_physical" and not statistics_path:
            raise ValueError("smooth_physical requires fitted training statistics")
        self.parameterization = parameterization
        self.num_lights = num_lights
        mean, std = default_parameter_statistics(num_lights)
        if statistics_path:
            values = np.load(Path(statistics_path))
            mean = torch.as_tensor(values["mean"], dtype=torch.float32)
            std = torch.as_tensor(values["std"], dtype=torch.float32)
        expected = parameter_dim(num_lights)
        if mean.numel() != expected or std.numel() != expected:
            raise ValueError(f"Expected {expected} parameter statistics values")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError("Statistics must be finite with non-negative std")
        if parameterization == "smooth_physical":
            fields = mean[:-3].reshape(num_lights, 10)
            if ((fields[:, :3] < 0).any() or (fields[:, 9] < 0).any()
                    or (mean[-3:] < 0).any() or (fields[:, 6:9] < 0).any()
                    or (fields[:, 6:9] > 1).any()
                    or (fields[:, 3:6].norm(dim=-1) < 1e-6).any()):
                raise ValueError("Smooth decoder statistics must have physically valid means")
        self.register_buffer("mean", mean.reshape(1, -1))
        self.register_buffer("std", std.clamp_min(1e-6).reshape(1, -1))

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        if self.parameterization == "affine":
            return self.std * normalized + self.mean
        # Smooth constraints before rendering: no negative-color clamp dead zone.
        # At zero latent, positive/position fields reproduce empirical means.
        mu = self.mean[:, :-3].reshape(1, self.num_lights, 10)
        std = self.std[:, :-3].reshape(1, self.num_lights, 10)
        z = normalized[:, :-3].reshape(-1, self.num_lights, 10)
        def positive(latent, mean, scale):
            tau = scale.clamp_min(1e-4)
            ratio = mean.clamp_min(1e-6) / tau
            inverse = ratio + torch.log(-torch.expm1(-ratio))
            return tau * F.softplus(inverse + latent * scale / tau)
        color = positive(z[..., :3], mu[..., :3], std[..., :3])
        direction = F.normalize(mu[..., 3:6] + std[..., 3:6] * z[..., 3:6], dim=-1)
        pos_mean = mu[..., 6:9].clamp(1e-4, 1 - 1e-4)
        position = torch.sigmoid(torch.logit(pos_mean) + z[..., 6:9]
                                 * std[..., 6:9] / (pos_mean * (1 - pos_mean)))
        attenuation = positive(z[..., 9:10], mu[..., 9:10], std[..., 9:10])
        ambient = positive(normalized[:, -3:], self.mean[:, -3:], self.std[:, -3:])
        return torch.cat((torch.cat((color, direction, position, attenuation), -1)
                          .flatten(1), ambient), 1)


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
