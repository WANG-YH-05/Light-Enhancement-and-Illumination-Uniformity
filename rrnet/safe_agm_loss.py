"""Direct supervision for the safe AGM source-light removal stage."""

from __future__ import annotations

import torch
from torch import nn

from .losses import luminance


class SafeAGMLoss(nn.Module):
    def __init__(self, *, max_log_gain: float = 1.20,
                 lambda_albedo: float = 1.0, lambda_gain: float = 1.0,
                 lambda_gradient: float = 0.5, lambda_consistency: float = 0.5,
                 lambda_chroma: float = 0.25, lambda_smooth: float = 0.05,
                 lambda_background: float = 1.0, edge_scale: float = 12.0,
                 epsilon: float = 1.0e-3,
                 lambda_identity: float = 0.0) -> None:
        super().__init__()
        self.max_log_gain = float(max_log_gain)
        self.lambda_albedo = float(lambda_albedo)
        self.lambda_gain = float(lambda_gain)
        self.lambda_gradient = float(lambda_gradient)
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_chroma = float(lambda_chroma)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_background = float(lambda_background)
        self.edge_scale = float(edge_scale)
        self.epsilon = float(epsilon)
        self.lambda_identity = float(lambda_identity)

    @staticmethod
    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(value).clamp(0.0, 1.0)
        channels = value.shape[1]
        return (value * mask).sum() / (mask.sum() * channels).clamp_min(1.0)

    @staticmethod
    def gradients(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value[..., :, 1:] - value[..., :, :-1], value[..., 1:, :] - value[..., :-1, :]

    @staticmethod
    def chromaticity(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp_min(0.0)
        return value / value.sum(dim=1, keepdim=True).clamp_min(1.0e-3)

    def forward(self, prediction: dict[str, torch.Tensor], target: torch.Tensor,
                source: torch.Tensor, relight_mask: torch.Tensor,
                skin_mask: torch.Tensor, *, group_size: int,
                identity_flags: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        albedo = prediction["albedo"]
        log_gain = prediction["log_gain"]
        source_y = luminance(source)
        target_y = luminance(target)
        target_log_gain = (
            (target_y + self.epsilon).log() - (source_y + self.epsilon).log()
        ).clamp(-self.max_log_gain, self.max_log_gain)
        # The shared RGB gain is capped before a bright channel can clip.
        # Supervise the achievable gain, so the loss does not push against
        # the model's own colour-preservation constraint.
        headroom = float(prediction.get("output_headroom", 0.999))
        safe_cap = headroom / source.amax(dim=1, keepdim=True).clamp_min(1.0e-4)
        target_log_gain = torch.where(
            target_log_gain > 0,
            torch.minimum(target_log_gain, safe_cap.clamp_min(1.0).log()),
            target_log_gain)

        # Stage 1 removes spatial illumination only.  A shared RGB gain cannot
        # and must not chase synthetic per-channel white-balance changes.
        albedo_y = luminance(albedo)
        albedo_loss = self.masked_mean((albedo_y - target_y).abs(), relight_mask)
        gain_loss = self.masked_mean((log_gain - target_log_gain).abs(), relight_mask)

        gx, gy = self.gradients(log_gain)
        tx, ty = self.gradients(target_log_gain)
        mx = relight_mask[..., :, 1:] * relight_mask[..., :, :-1]
        my = relight_mask[..., 1:, :] * relight_mask[..., :-1, :]
        gradient = self.masked_mean((gx - tx).abs(), mx) + self.masked_mean((gy - ty).abs(), my)

        target_gx, target_gy = self.gradients(target_y)
        smooth = (
            self.masked_mean(gx.abs() * torch.exp(-self.edge_scale * target_gx.abs()), mx)
            + self.masked_mean(gy.abs() * torch.exp(-self.edge_scale * target_gy.abs()), my)
        )
        chroma = self.masked_mean(
            (self.chromaticity(albedo) - self.chromaticity(target)).abs(), skin_mask)
        background = self.masked_mean((albedo - source).abs(), 1.0 - relight_mask)
        if identity_flags is not None:
            identity_weight = relight_mask * identity_flags.reshape(
                -1, 1, 1, 1).to(relight_mask.dtype)
            identity = self.masked_mean(log_gain.abs(), identity_weight)
        else:
            identity = albedo.new_zeros(())

        if group_size > 1:
            if albedo.shape[0] % group_size:
                raise ValueError("batch does not divide into complete AGM groups")
            grouped = albedo_y.reshape(-1, group_size, *albedo_y.shape[1:])
            grouped_mask = relight_mask.reshape(-1, group_size, *relight_mask.shape[1:])
            mean = grouped.mean(dim=1, keepdim=True)
            consistency = self.masked_mean(
                (grouped - mean).abs().flatten(0, 1), grouped_mask.flatten(0, 1))
        else:
            consistency = albedo.new_zeros(())

        total = (
            self.lambda_albedo * albedo_loss
            + self.lambda_gain * gain_loss
            + self.lambda_gradient * gradient
            + self.lambda_consistency * consistency
            + self.lambda_chroma * chroma
            + self.lambda_smooth * smooth
            + self.lambda_background * background
            + self.lambda_identity * identity
        )

        # Diagnostic: compare against the best single exposure multiplier.
        with torch.no_grad():
            weight = relight_mask
            numerator = (source_y * target_y * weight).sum(dim=(1, 2, 3), keepdim=True)
            denominator = (source_y.square() * weight).sum(
                dim=(1, 2, 3), keepdim=True).clamp_min(self.epsilon)
            scalar = numerator / denominator
            global_output = (source * scalar).clamp(0.0, 1.0)
            global_mae = self.masked_mean((global_output - target).abs(), relight_mask)
            model_mae = self.masked_mean((albedo - target).abs(), relight_mask)
            beats_global = (global_mae - model_mae)
            global_luma_mae = self.masked_mean(
                (luminance(global_output) - target_y).abs(), relight_mask)
            model_luma_mae = self.masked_mean(
                (albedo_y - target_y).abs(), relight_mask)

        return {
            "total": total,
            "albedo": albedo_loss,
            "gain": gain_loss,
            "gradient": gradient,
            "consistency": consistency,
            "chroma": chroma,
            "smooth": smooth,
            "background": background,
            "identity": identity,
            "model_mae": model_mae,
            "global_mae": global_mae,
            "beats_global": beats_global,
            "model_luma_mae": model_luma_mae,
            "global_luma_mae": global_luma_mae,
            "luma_advantage": global_luma_mae - model_luma_mae,
        }
