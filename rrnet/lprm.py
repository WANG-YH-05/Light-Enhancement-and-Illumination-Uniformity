"""Dual-branch Lighting Parameter Regression Module (paper Fig. 3a)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .lighting import ParameterDenormalizer, parameter_dim
from .repvit import RepViTM09Encoder


def resize_shorter_side(image: torch.Tensor, shorter_side: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    scale = shorter_side / min(height, width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return F.interpolate(image, size=(new_height, new_width), mode="bilinear", align_corners=False)


class LPRM(nn.Module):
    def __init__(self, num_lights: int = 9, shorter_side: int = 512,
                 coarse_factor: int = 4, statistics_path: str | None = None) -> None:
        super().__init__()
        self.num_lights = num_lights
        self.shorter_side = shorter_side
        self.coarse_factor = coarse_factor
        self.encoder = RepViTM09Encoder()  # E0 and E1 are literally the same object.
        dim = parameter_dim(num_lights)
        embedding_dim = self.encoder.embedding_dim
        self.r0 = nn.Sequential(nn.BatchNorm1d(embedding_dim), nn.Linear(embedding_dim, dim))
        self.r1 = nn.Linear(embedding_dim + dim, dim)
        self.denormalize = ParameterDenormalizer(num_lights, statistics_path)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        resized = resize_shorter_side(image, self.shorter_side)
        coarse_size = (
            max(1, math.ceil(resized.shape[-2] / self.coarse_factor)),
            max(1, math.ceil(resized.shape[-1] / self.coarse_factor)),
        )
        coarse = F.interpolate(resized, size=coarse_size, mode="bilinear", align_corners=False)
        _, coarse_embedding = self.encoder(coarse)
        theta0 = self.r0(coarse_embedding)
        full_features, full_embedding = self.encoder(resized)
        theta_offset = self.r1(torch.cat((full_embedding, theta0), dim=1))
        theta_normalized = theta0 + theta_offset
        theta = self.denormalize(theta_normalized)
        return {
            "theta": theta,
            "theta_normalized": theta_normalized,
            "theta0": theta0,
            "theta_offset": theta_offset,
            "features": full_features,
            "embedding": full_embedding,
            "resized_input": resized,
        }
