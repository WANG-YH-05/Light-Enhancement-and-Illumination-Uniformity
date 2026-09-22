"""Self-contained RepViT-M0.9 encoder adapted from the official architecture.

RRNet states that its two LPRM branches share a RepViT encoder but does not
identify the RepViT variant. M0.9 is the smallest published RepViT model and is
therefore the documented default reproduction assumption.
"""

from __future__ import annotations

import torch
from torch import nn


def _make_divisible(value: float, divisor: int = 8) -> int:
    result = max(divisor, int(value + divisor / 2) // divisor * divisor)
    return result + divisor if result < 0.9 * value else result


class ConvBN(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 stride: int = 1, padding: int = 0, groups: int = 1,
                 bn_weight_init: float = 1.0) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        nn.init.constant_(self[1].weight, bn_weight_init)
        nn.init.zeros_(self[1].bias)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden = _make_divisible(channels * ratio)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


class RepVGGDepthwise(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv3 = ConvBN(channels, channels, 3, 1, 1, groups=channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv3(x) + self.conv1(x) + x)


class RepViTBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int,
                 use_se: bool) -> None:
        super().__init__()
        if stride == 2:
            self.token_mixer = nn.Sequential(
                ConvBN(in_channels, in_channels, 3, 2, 1, groups=in_channels),
                SqueezeExcite(in_channels) if use_se else nn.Identity(),
                ConvBN(in_channels, out_channels),
            )
        else:
            if in_channels != out_channels:
                raise ValueError("A stride-1 RepViT block must preserve channels")
            self.token_mixer = nn.Sequential(
                RepVGGDepthwise(in_channels),
                SqueezeExcite(in_channels) if use_se else nn.Identity(),
            )
        self.channel_mixer = Residual(nn.Sequential(
            ConvBN(out_channels, 2 * out_channels),
            nn.GELU(),
            ConvBN(2 * out_channels, out_channels, bn_weight_init=0.0),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel_mixer(self.token_mixer(x))


M09_STAGES = (
    ((48, 1, True), (48, 1, False), (48, 1, False)),
    ((96, 2, False), (96, 1, True), (96, 1, False), (96, 1, False)),
    ((192, 2, False), (192, 1, True), (192, 1, False), (192, 1, True),
     (192, 1, False), (192, 1, True), (192, 1, False), (192, 1, True),
     (192, 1, False), (192, 1, True), (192, 1, False), (192, 1, True),
     (192, 1, False), (192, 1, True), (192, 1, False), (192, 1, False)),
    ((384, 2, False), (384, 1, True), (384, 1, False)),
)


class RepViTM09Encoder(nn.Module):
    """RepViT-M0.9 feature encoder returning four scales and a pooled vector."""

    feature_channels = (48, 96, 192, 384)
    embedding_dim = 384

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBN(3, 24, 3, 2, 1),
            nn.GELU(),
            ConvBN(24, 48, 3, 2, 1),
        )
        stages: list[nn.Module] = []
        in_channels = 48
        for entries in M09_STAGES:
            blocks: list[nn.Module] = []
            for out_channels, stride, use_se in entries:
                blocks.append(RepViTBlock(in_channels, out_channels, stride, use_se))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        embedding = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return features, embedding
