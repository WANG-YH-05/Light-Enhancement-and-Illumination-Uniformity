"""Top-level RRNet: LPRM + frozen depth + RM + optional AGM."""

from __future__ import annotations

import torch
from torch import nn

from .agm import AlbedoGenerationModule
from .depth import DepthAnythingV2Small, LuminanceDepthProxy
from .lprm import LPRM
from .renderer import RenderingModule


class RRNet(nn.Module):
    def __init__(self, *, num_lights: int = 9, shorter_side: int = 512,
                 coarse_factor: int = 4, statistics_path: str | None = None,
                 depth_vendor_root: str | None = None, depth_checkpoint: str | None = None,
                 depth_input_size: int = 518, depth_invert: bool = False,
                 allow_depth_proxy: bool = False, use_agm: bool = True,
                 agm_channels: int = 64, sigma1: float = 0.10,
                 sigma2: float = 0.05, clamp_output: bool = True,
                 enforce_physical_parameters: bool = False,
                 split_resolution_bn: bool = False,
                 light_parameterization: str = "affine") -> None:
        super().__init__()
        self.lprm = LPRM(num_lights, shorter_side, coarse_factor, statistics_path,
                         split_resolution_bn=split_resolution_bn,
                         light_parameterization=light_parameterization)
        if depth_vendor_root and depth_checkpoint:
            self.depth = DepthAnythingV2Small(
                depth_vendor_root, depth_checkpoint, depth_input_size, depth_invert
            )
        elif allow_depth_proxy:
            self.depth = LuminanceDepthProxy()
        else:
            raise ValueError(
                "Paper-faithful RRNet requires the official Depth Anything V2 Small "
                "repository and checkpoint. Set allow_depth_proxy only for smoke tests."
            )
        self.depth.eval()
        self.depth.requires_grad_(False)
        self.agm = AlbedoGenerationModule(
            self.lprm.encoder.feature_channels, agm_channels
        ) if use_agm else None
        self.renderer = RenderingModule(
            num_lights, sigma1, sigma2, clamp_output,
            enforce_physical_parameters=enforce_physical_parameters)

    def train(self, mode: bool = True) -> "RRNet":
        super().train(mode)
        self.depth.eval()  # paper: depth estimator remains frozen during training
        return self

    def estimate_lighting(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self.lprm(image)

    def render_with_theta(self, image: torch.Tensor, theta: torch.Tensor,
                          features: list[torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            depth = self.depth(image)
        result: dict[str, torch.Tensor] = {"depth": depth, "theta": theta}
        base = image
        if self.agm is not None:
            if features is None:
                features, _ = self.lprm.encoder(self.lprm(image)["resized_input"])
            agm_result = self.agm(image, features)
            base = agm_result["albedo"]
            result.update(agm_result)
        render_result = self.renderer(base, depth, theta)
        result.update(render_result)
        return result

    def forward(self, image: torch.Tensor, theta_override: torch.Tensor | None = None,
                relight_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        prediction = self.lprm(image)
        theta = prediction["theta"] if theta_override is None else theta_override
        result = self.render_with_theta(image, theta, prediction["features"])
        if relight_mask is not None:
            result["loss_output"] = self.renderer.blend_relight(
                result["raw_output"], image, relight_mask
            )
            result["output"] = self.renderer.blend_relight(
                result["output"], image, relight_mask
            )
        result.update({
            "theta0": prediction["theta0"],
            "theta_offset": prediction["theta_offset"],
        })
        return result
