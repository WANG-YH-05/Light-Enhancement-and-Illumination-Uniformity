"""Frozen Depth Anything V2 Small adapter and depth-to-normal conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def normalize_depth(depth: torch.Tensor, invert: bool = True) -> torch.Tensor:
    minimum = depth.amin(dim=(-2, -1), keepdim=True)
    maximum = depth.amax(dim=(-2, -1), keepdim=True)
    depth = (depth - minimum) / (maximum - minimum).clamp_min(1e-6)
    return 1.0 - depth if invert else depth


def depth_to_normals(depth: torch.Tensor) -> torch.Tensor:
    """Derive N from M using centered finite differences."""
    dzdx = F.pad((depth[..., 2:] - depth[..., :-2]) * 0.5, (1, 1, 0, 0), mode="replicate")
    dzdy = F.pad((depth[..., 2:, :] - depth[..., :-2, :]) * 0.5, (0, 0, 1, 1), mode="replicate")
    ones = torch.ones_like(depth)
    return F.normalize(torch.cat((-dzdx, -dzdy, ones), dim=1), dim=1, eps=1e-6)


class DepthAnythingV2Small(nn.Module):
    """Load the official Depth-Anything-V2 implementation from a local checkout."""

    def __init__(self, vendor_root: str, checkpoint: str, input_size: int = 518,
                 invert: bool = False) -> None:
        super().__init__()
        root = Path(vendor_root).resolve()
        checkpoint_path = Path(checkpoint).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Depth Anything V2 repository not found: {root}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Depth Anything V2 Small checkpoint not found: {checkpoint_path}")
        sys.path.insert(0, str(root))
        try:
            from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        finally:
            sys.path.pop(0)
        self.network = DepthAnythingV2(
            encoder="vits", features=64, out_channels=[48, 96, 192, 384]
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.network.load_state_dict(state)
        self.network.eval()
        self.network.requires_grad_(False)
        self.input_size = input_size
        self.invert = invert
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        original_size = image.shape[-2:]
        height, width = original_size
        # Official relative-depth preprocessing uses lower_bound resizing:
        # the shorter side reaches input_size and both sides are multiples of 14.
        scale = self.input_size / min(height, width)
        resized = (max(14, round(height * scale / 14) * 14), max(14, round(width * scale / 14) * 14))
        x = F.interpolate(image, size=resized, mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        depth = self.network(x)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        depth = F.interpolate(depth, size=original_size, mode="bilinear", align_corners=False)
        return normalize_depth(depth, self.invert)


class LuminanceDepthProxy(nn.Module):
    """Dependency-free smoke-test helper; forbidden for paper training."""

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        luminance = 0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] + 0.0722 * image[:, 2:3]
        return normalize_depth(F.avg_pool2d(luminance, 9, stride=1, padding=4), invert=False)
