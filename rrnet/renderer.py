"""Equation (3-4), simplified depth-aware Blinn-Phong rendering module."""

from __future__ import annotations

import torch
from torch import nn

from .depth import depth_to_normals
from .lighting import unpack_lights


class RenderingModule(nn.Module):
    def __init__(self, num_lights: int = 9, sigma1: float = 0.10,
                 sigma2: float = 0.05, clamp_output: bool = True) -> None:
        super().__init__()
        self.num_lights = num_lights
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.clamp_output = clamp_output

    def illumination(self, depth: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = unpack_lights(theta, self.num_lights)
        batch, _, height, width = depth.shape
        dtype, device = depth.dtype, depth.device
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, height, dtype=dtype, device=device),
            torch.linspace(0.0, 1.0, width, dtype=dtype, device=device), indexing="ij"
        )
        xx = xx.unsqueeze(0).expand(batch, -1, -1)
        yy = yy.unsqueeze(0).expand(batch, -1, -1)
        points = torch.stack((xx, yy, depth[:, 0]), dim=1)
        normals = depth_to_normals(depth)
        light_direction = params["direction"][:, :, :, None, None]
        normal_dot = (normals[:, None] * light_direction).sum(dim=2)
        diffuse = torch.maximum(normal_dot, torch.as_tensor(self.sigma2, dtype=dtype, device=device))
        delta = params["position"][:, :, :, None, None] - points[:, None]
        distance_squared = delta.square().sum(dim=2)
        denominator = params["attenuation"][:, :, 0, None, None] * distance_squared + self.sigma1
        contribution = params["color"][:, :, :, None, None] * (diffuse / denominator)[:, :, None]
        light = params["ambient"][:, :, None, None] + contribution.sum(dim=1)
        return light, normals

    def forward(self, image_or_albedo: torch.Tensor, depth: torch.Tensor,
                theta: torch.Tensor) -> dict[str, torch.Tensor]:
        light, normals = self.illumination(depth, theta)
        raw_output = image_or_albedo * light
        output = raw_output.clamp(0.0, 1.0) if self.clamp_output else raw_output
        return {"output": output, "raw_output": raw_output,
                "illumination": light, "normals": normals}

    def apply_illumination(self, image_or_albedo: torch.Tensor,
                           light: torch.Tensor) -> torch.Tensor:
        """Apply an already-computed illumination map to the current RGB frame."""
        output = image_or_albedo * light
        if self.clamp_output:
            output = output.clamp(0.0, 1.0)
        return output

    @staticmethod
    def blend_relight(enhanced: torch.Tensor, source: torch.Tensor,
                      relight_mask: torch.Tensor) -> torch.Tensor:
        """Keep the input outside a soft foreground relighting mask."""
        if relight_mask.ndim != 4 or relight_mask.shape[1] != 1:
            raise ValueError("relight_mask must have shape [B, 1, H, W]")
        if relight_mask.shape[0] != source.shape[0] or relight_mask.shape[-2:] != source.shape[-2:]:
            raise ValueError("relight_mask batch and spatial dimensions must match source")
        mask = relight_mask.to(device=source.device, dtype=source.dtype).clamp(0.0, 1.0)
        return source + mask * (enhanced - source)
