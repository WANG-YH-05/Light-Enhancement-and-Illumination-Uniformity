"""Optional Albedo Generation Module from RRNet section III-B."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class AlbedoGenerationModule(nn.Module):
    """One residual block plus bilinear U-Net-style decoding to a 3ch mask Z'."""

    def __init__(self, feature_channels: tuple[int, ...] = (48, 96, 192, 384),
                 decoder_channels: int = 64) -> None:
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_channels[-1], decoder_channels, 1),
            ResidualBlock(decoder_channels),
        )
        self.laterals = nn.ModuleList(
            nn.Conv2d(channels, decoder_channels, 1) for channels in reversed(feature_channels[:-1])
        )
        self.output = nn.Conv2d(decoder_channels, 3, 3, padding=1)

    def forward(self, image: torch.Tensor, features: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.bottleneck(features[-1])
        for skip, lateral in zip(reversed(features[:-1]), self.laterals):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + lateral(skip)
        z_prime = self.output(x)
        z = F.interpolate(z_prime, size=image.shape[-2:], mode="bilinear", align_corners=False)
        albedo = image - z
        return {"albedo": albedo, "z": z, "z_prime": z_prime}
