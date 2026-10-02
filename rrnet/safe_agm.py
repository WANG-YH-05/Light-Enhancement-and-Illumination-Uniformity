"""Chromaticity-preserving albedo recovery for the staged AGM training.

The paper's unconstrained three-channel ``A = I - Z`` form is kept in
``agm.py`` for reproduction experiments.  This module is the production-safe
variant: it predicts a bounded, one-channel log-gain field, so it can remove
spatial shading without inventing per-pixel colour changes.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .agm import ResidualBlock
from .lprm import resize_shorter_side
from .repvit import RepViTM09Encoder


class SafeAlbedoGenerationModule(nn.Module):
    """Decode encoder features into a bounded one-channel inverse-light map."""

    def __init__(self, feature_channels: tuple[int, ...] = (48, 96, 192, 384),
                 decoder_channels: int = 64, max_log_gain: float = 1.20,
                 output_headroom: float = 0.999) -> None:
        super().__init__()
        if max_log_gain <= 0.0:
            raise ValueError("max_log_gain must be positive")
        if not 0.0 < output_headroom <= 1.0:
            raise ValueError("output_headroom must be in (0, 1]")
        self.max_log_gain = float(max_log_gain)
        self.output_headroom = float(output_headroom)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feature_channels[-1], decoder_channels, 1),
            ResidualBlock(decoder_channels),
        )
        self.laterals = nn.ModuleList(
            nn.Conv2d(channels, decoder_channels, 1)
            for channels in reversed(feature_channels[:-1])
        )
        self.refine = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.GELU(),
        )
        self.output = nn.Conv2d(decoder_channels, 1, 3, padding=1)
        # Identity at initialization prevents the old random colour/exposure jump.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, image: torch.Tensor, features: list[torch.Tensor],
                relight_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.bottleneck(features[-1])
        for skip, lateral in zip(reversed(features[:-1]), self.laterals):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            x = x + lateral(skip)
        x = self.refine(x)
        raw_log_gain_low = self.output(x)
        log_gain_low = self.max_log_gain * torch.tanh(raw_log_gain_low)
        log_gain = F.interpolate(
            log_gain_low, size=image.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        if relight_mask is not None:
            mask = relight_mask.to(device=image.device, dtype=image.dtype)
            if mask.shape[-2:] != image.shape[-2:]:
                mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear",
                                     align_corners=False)
            log_gain = log_gain * mask.clamp(0.0, 1.0)
        requested_gain = log_gain.exp()
        # A shared RGB gain preserves chromaticity only while no channel clips.
        # Cap brightening at the largest safe value for the current pixel.
        safe_cap = self.output_headroom / image.amax(dim=1, keepdim=True).clamp_min(1.0e-4)
        gain = torch.where(
            requested_gain > 1.0,
            torch.minimum(requested_gain, safe_cap.clamp_min(1.0)),
            requested_gain,
        )
        log_gain = gain.clamp_min(1.0e-6).log()
        raw_albedo = image * gain
        albedo = raw_albedo.clamp(0.0, 1.0)
        return {
            "albedo": albedo,
            "raw_albedo": raw_albedo,
            "gain": gain,
            "requested_gain": requested_gain,
            "log_gain": log_gain,
            "log_gain_low": log_gain_low,
            "raw_log_gain_low": raw_log_gain_low,
            "output_headroom": self.output_headroom,
        }


class SafeAlbedoModel(nn.Module):
    """RepViT encoder plus the safe AGM; no renderer can cancel its output."""

    def __init__(self, shorter_side: int = 512, decoder_channels: int = 64,
                 max_log_gain: float = 1.20,
                 output_headroom: float = 0.999) -> None:
        super().__init__()
        self.shorter_side = int(shorter_side)
        self.encoder = RepViTM09Encoder()
        self.agm = SafeAlbedoGenerationModule(
            self.encoder.feature_channels, decoder_channels, max_log_gain,
            output_headroom)

    @property
    def max_log_gain(self) -> float:
        return self.agm.max_log_gain

    def forward(self, image: torch.Tensor,
                relight_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        resized = resize_shorter_side(image, self.shorter_side)
        features, _ = self.encoder(resized)
        return self.agm(image, features, relight_mask)
