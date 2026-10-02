"""Known-light pilot for the reconstructed RRNet RGB AGM.

The MEAD clean frame is a *synthetic base*, not measured intrinsic albedo.
Only the additional virtual illumination applied here has an exact inverse.
"""

from __future__ import annotations

import torch
from torch import nn

from .losses import luminance


def masked_mean(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if error.shape[1] != mask.shape[1]:
        mask = mask.expand_as(error)
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def gradient_error(prediction: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    horizontal = masked_mean(
        (prediction[..., :, 1:] - prediction[..., :, :-1]
         - target[..., :, 1:] + target[..., :, :-1]).abs(),
        torch.minimum(mask[..., :, 1:], mask[..., :, :-1]))
    vertical = masked_mean(
        (prediction[..., 1:, :] - prediction[..., :-1, :]
         - target[..., 1:, :] + target[..., :-1, :]).abs(),
        torch.minimum(mask[..., 1:, :], mask[..., :-1, :]))
    return 0.5 * (horizontal + vertical)


class PhysicalAGMLoss(nn.Module):
    """Supervise both the AGM base and the final re-rendered portrait."""

    def __init__(self, lambda_output: float = 1.0,
                 lambda_base: float = 1.0,
                 lambda_spatial: float = 0.5,
                 lambda_chroma: float = 0.1,
                 lambda_background: float = 0.5,
                 lambda_identity: float = 0.5) -> None:
        super().__init__()
        self.weights = {
            "output": float(lambda_output), "base": float(lambda_base),
            "spatial": float(lambda_spatial), "chroma": float(lambda_chroma),
            "background": float(lambda_background),
            "identity": float(lambda_identity),
        }

    def forward(self, *, source: torch.Tensor, clean: torch.Tensor,
                target: torch.Tensor, albedo: torch.Tensor,
                output: torch.Tensor, mask: torch.Tensor,
                skin_mask: torch.Tensor, identity_flags: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        mask = mask.clamp(0.0, 1.0)
        skin = (mask * skin_mask.clamp(0.0, 1.0)).clamp(0.0, 1.0)
        # One identical synthetic-light multiplier is applied to all RGB
        # channels. Chromaticity should therefore survive where no clipping
        # occurred; ignore saturated pixels for this specific constraint.
        valid_colour = (source.amax(dim=1, keepdim=True) < 0.98).to(source.dtype)
        source_chroma = source / source.sum(dim=1, keepdim=True).clamp_min(0.02)
        albedo_chroma = albedo / albedo.sum(dim=1, keepdim=True).clamp_min(0.02)
        source_correction = luminance(albedo - source)
        ideal_correction = luminance(clean - source)
        terms = {
            "output": masked_mean((output - target).abs(), mask),
            "base": masked_mean((albedo - clean).abs(), mask),
            "spatial": gradient_error(
                source_correction, ideal_correction, mask),
            "chroma": masked_mean(
                (albedo_chroma - source_chroma).abs(), skin * valid_colour),
            "background": masked_mean(
                (albedo - source).abs(), 1.0 - mask),
            "identity": masked_mean(
                (albedo - source).abs(),
                mask * identity_flags[:, None, None, None].to(mask.dtype)),
        }
        terms["total"] = sum(self.weights[key] * value
                             for key, value in terms.items())
        terms["mae"] = terms["output"].detach()
        terms["identity_base_mae"] = terms["identity"].detach()
        return terms


@torch.no_grad()
def oracle_scalar_mae(source: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """Target-informed scalar diagnostic; not an inference-time baseline."""
    source_y = luminance(source)
    target_y = luminance(target)
    numerator = (source_y * target_y * mask).sum(dim=(1, 2, 3), keepdim=True)
    denominator = (source_y.square() * mask).sum(
        dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-6)
    scalar = numerator / denominator
    return masked_mean(((source * scalar).clamp(0.0, 1.0) - target).abs(), mask)
