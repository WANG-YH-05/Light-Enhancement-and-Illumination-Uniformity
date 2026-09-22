"""RRNet equations (6-9)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .lighting import unpack_lights


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 2:3]


class RRNetLoss(nn.Module):
    def __init__(self, num_lights: int = 9, sigma_c: float = 0.10,
                 sigma_l: float = 0.10, lambda_reg: float = 0.01,
                 lambda_ambient: float = 1.0, ambient_max: float = 2.0,
                 pixel_loss: str = "l1", lambda_skin: float = 0.50,
                 lambda_highlight: float = 0.20,
                 lambda_background: float = 1.0,
                 highlight_threshold: float = 0.95,
                 highlight_margin: float = 0.05,
                 lambda_dark: float = 0.0,
                 dark_lift_threshold: float = 0.03,
                 dark_source_max: float = 0.65,
                 dark_mode: str = "lift_only",
                 lambda_overexposure_log: float = 0.0,
                 overexposure_threshold: float = 0.03,
                 overexposure_log_epsilon: float = 1.0e-3,
                 lambda_chroma: float = 0.0) -> None:
        super().__init__()
        self.num_lights = num_lights
        self.sigma_c = sigma_c
        self.sigma_l = sigma_l
        self.lambda_reg = lambda_reg
        self.lambda_ambient = lambda_ambient
        self.ambient_max = ambient_max
        self.pixel_loss = pixel_loss
        self.lambda_skin = lambda_skin
        self.lambda_highlight = lambda_highlight
        self.lambda_background = lambda_background
        self.highlight_threshold = highlight_threshold
        self.highlight_margin = highlight_margin
        self.lambda_dark = lambda_dark
        self.dark_lift_threshold = dark_lift_threshold
        self.dark_source_max = dark_source_max
        if dark_mode not in {"lift_only", "symmetric"}:
            raise ValueError("dark_mode must be 'lift_only' or 'symmetric'")
        self.dark_mode = dark_mode
        if lambda_overexposure_log < 0.0:
            raise ValueError("lambda_overexposure_log must be non-negative")
        if overexposure_threshold < 0.0:
            raise ValueError("overexposure_threshold must be non-negative")
        if overexposure_log_epsilon <= 0.0:
            raise ValueError("overexposure_log_epsilon must be positive")
        self.lambda_overexposure_log = float(lambda_overexposure_log)
        self.overexposure_threshold = float(overexposure_threshold)
        self.overexposure_log_epsilon = float(overexposure_log_epsilon)
        self.lambda_chroma = lambda_chroma

    @staticmethod
    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=value.device, dtype=value.dtype).clamp(0.0, 1.0)
        channels = value.shape[1] if value.ndim >= 4 else 1
        denominator = (mask.sum() * channels).clamp_min(1.0)
        return (value * mask).sum() / denominator

    def reconstruction_error(self, output: torch.Tensor,
                             target: torch.Tensor) -> torch.Tensor:
        if self.pixel_loss == "l1":
            return (output - target).abs()
        if self.pixel_loss == "l2":
            return (output - target).square()
        raise ValueError(f"Unsupported pixel loss: {self.pixel_loss}")

    def lighting_regularization(self, theta: torch.Tensor) -> torch.Tensor:
        params = unpack_lights(theta, self.num_lights)
        constrained_color = params["color"].clamp_min(0.0)
        constrained_direction = F.normalize(params["direction"], dim=-1, eps=1e-6)
        constrained_position = params["position"].clamp(0.0, 1.0)
        constrained_attenuation = params["attenuation"].clamp_min(0.0)
        light_penalty = (
            (params["color"] - constrained_color).square().mean()
            + (params["direction"] - constrained_direction).square().mean()
            + (params["position"] - constrained_position).square().mean()
            + (params["attenuation"] - constrained_attenuation).square().mean()
        )
        constrained_ambient = params["ambient"].clamp(0.0, self.ambient_max)
        ambient_penalty = (params["ambient"] - constrained_ambient).square().mean()
        return light_penalty + self.lambda_ambient * ambient_penalty

    @staticmethod
    def chromaticity(rgb: torch.Tensor) -> torch.Tensor:
        return rgb.clamp_min(0.0) / rgb.clamp_min(0.0).sum(dim=1, keepdim=True).clamp_min(1.0e-3)

    def forward(self, prediction: dict[str, torch.Tensor], target: torch.Tensor,
                source: torch.Tensor | None = None,
                relight_mask: torch.Tensor | None = None,
                skin_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        output = prediction.get("loss_output", prediction["output"])
        depth, theta = prediction["depth"], prediction["theta"]
        error = self.reconstruction_error(output, target)
        roi_weight = torch.maximum(depth, torch.as_tensor(self.sigma_c, device=depth.device))
        roi_weight = roi_weight * torch.maximum(
            luminance(target), torch.as_tensor(self.sigma_l, device=target.device)
        )
        regularization = self.lighting_regularization(theta)
        if relight_mask is None:
            pixel = error.mean()
            roi = (roi_weight * (output - target)).square().mean()
            zero = output.new_zeros(())
            total = pixel + roi + self.lambda_reg * regularization
            return {
                "total": total, "pixel": pixel, "relight": pixel,
                "skin": zero, "highlight": zero, "background": zero,
                "dark": zero, "overexposure_log": zero, "chroma": zero,
                "roi": roi, "regularization": regularization,
            }

        relight = self.masked_mean(error, relight_mask)
        effective_skin = skin_mask if skin_mask is not None else relight_mask
        skin = self.masked_mean(error, effective_skin)
        roi_error = (roi_weight * (output - target)).square()
        roi = self.masked_mean(roi_error, relight_mask)
        output_luma = luminance(output)
        target_luma = luminance(target)
        source_luma = luminance(source) if source is not None else target_luma
        highlight_limit = torch.maximum(
            target_luma + self.highlight_margin,
            torch.as_tensor(self.highlight_threshold, device=output.device,
                            dtype=output.dtype),
        )
        highlight = self.masked_mean(
            torch.relu(output_luma - highlight_limit).square(), effective_skin
        )
        if source is None:
            raise ValueError("source is required when relight_mask is provided")
        if self.dark_mode == "lift_only":
            # Historical behaviour: prioritize recovery only when the desired
            # target is brighter than the source.
            lift = torch.relu(target_luma - source_luma - self.dark_lift_threshold)
            darkness = torch.relu(self.dark_source_max - source_luma) / max(self.dark_source_max, 1.0e-6)
            dark_weight = relight_mask * lift * darkness
        else:
            # Keep the historical dark-region lifting supervision intact, but
            # add the missing inverse case.  Reference transfer must also be
            # penalized when a brighter input should become darker because its
            # selected reference is underexposed or shadowed.
            lift = torch.relu(target_luma - source_luma - self.dark_lift_threshold)
            darkness = torch.relu(self.dark_source_max - source_luma) / max(self.dark_source_max, 1.0e-6)
            lift_weight = lift * darkness
            darken_weight = torch.relu(source_luma - target_luma - self.dark_lift_threshold)
            dark_weight = relight_mask * (lift_weight + darken_weight)
        dark = self.masked_mean((output_luma - target_luma).abs(), dark_weight)
        # A moderately overexposed face often never reaches the old 0.95
        # highlight threshold. Supervise the actual inverse-transfer case in
        # log luminance instead: whenever the paired target is darker than the
        # input, failure to compress exposure receives a perceptually stronger
        # penalty. This term is training-only and adds no inference work.
        darken_need = torch.relu(
            source_luma - target_luma - self.overexposure_threshold
        )
        overexposure_mask = relight_mask * darken_need
        overexposure_log = self.masked_mean(
            (
                (output_luma + self.overexposure_log_epsilon).log()
                - (target_luma + self.overexposure_log_epsilon).log()
            ).abs(),
            overexposure_mask,
        )
        chroma = self.masked_mean(
            (self.chromaticity(output) - self.chromaticity(target)).abs(),
            effective_skin,
        )
        background = self.masked_mean(
            (output - source).abs(), 1.0 - relight_mask
        )
        total = (
            relight
            + self.lambda_skin * skin
            + roi
            + self.lambda_highlight * highlight
            + self.lambda_background * background
            + self.lambda_dark * dark
            + self.lambda_overexposure_log * overexposure_log
            + self.lambda_chroma * chroma
            + self.lambda_reg * regularization
        )
        return {
            "total": total, "pixel": relight, "relight": relight,
            "skin": skin, "highlight": highlight, "background": background,
            "dark": dark, "overexposure_log": overexposure_log,
            "chroma": chroma,
            "roi": roi, "regularization": regularization,
        }
