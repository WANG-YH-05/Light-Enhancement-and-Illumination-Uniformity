"""Losses for supervised input/reference illumination decomposition."""

from __future__ import annotations

import torch
from torch import nn

from .losses import RRNetLoss, luminance


class ReferenceRelativeLoss(nn.Module):
    """RRNet image losses plus explicit illumination-map supervision."""

    def __init__(self, *, lambda_source_illumination: float = 0.50,
                 lambda_reference_illumination: float = 0.50,
                 lambda_target_illumination: float = 1.00,
                 lambda_gain: float = 0.0,
                 lambda_dark_log_lift: float = 0.0,
                 illumination_target_floor: float = 0.04,
                 illumination_target_max: float = 4.00,
                 gain_target_floor: float = 0.025,
                 gain_target_min: float = 0.40,
                 gain_target_max: float = 3.00,
                 dark_log_lift_threshold: float = 0.03,
                 dark_log_source_max: float = 0.20,
                 dark_log_epsilon: float = 1.0e-3,
                 **rrnet_loss_kwargs) -> None:
        super().__init__()
        if illumination_target_floor <= 0.0:
            raise ValueError("illumination_target_floor must be positive")
        if illumination_target_max < 1.0:
            raise ValueError("illumination_target_max must be at least one")
        if lambda_gain < 0.0:
            raise ValueError("lambda_gain must be non-negative")
        if lambda_dark_log_lift < 0.0:
            raise ValueError("lambda_dark_log_lift must be non-negative")
        if gain_target_floor <= 0.0:
            raise ValueError("gain_target_floor must be positive")
        if not 0.0 < gain_target_min <= 1.0:
            raise ValueError("gain_target_min must be in (0, 1]")
        if gain_target_max < 1.0 or gain_target_min > gain_target_max:
            raise ValueError("gain_target_max must be >= 1 and >= gain_target_min")
        if dark_log_lift_threshold < 0.0:
            raise ValueError("dark_log_lift_threshold must be non-negative")
        if not 0.0 < dark_log_source_max <= 1.0:
            raise ValueError("dark_log_source_max must be in (0, 1]")
        if dark_log_epsilon <= 0.0:
            raise ValueError("dark_log_epsilon must be positive")
        self.image_loss = RRNetLoss(**rrnet_loss_kwargs)
        self.lambda_source_illumination = float(lambda_source_illumination)
        self.lambda_reference_illumination = float(lambda_reference_illumination)
        self.lambda_target_illumination = float(lambda_target_illumination)
        self.lambda_gain = float(lambda_gain)
        self.lambda_dark_log_lift = float(lambda_dark_log_lift)
        self.illumination_target_floor = float(illumination_target_floor)
        self.illumination_target_max = float(illumination_target_max)
        self.gain_target_floor = float(gain_target_floor)
        self.gain_target_min = float(gain_target_min)
        self.gain_target_max = float(gain_target_max)
        self.dark_log_lift_threshold = float(dark_log_lift_threshold)
        self.dark_log_source_max = float(dark_log_source_max)
        self.dark_log_epsilon = float(dark_log_epsilon)

    def illumination_target(self, lit: torch.Tensor,
                            clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clean_luma = luminance(clean)
        ratio = (luminance(lit) / clean_luma.clamp_min(
            self.illumination_target_floor)).clamp(0.0, self.illumination_target_max)
        valid = (clean_luma >= self.illumination_target_floor).to(clean.dtype)
        return ratio, valid

    def forward(self, prediction: dict[str, torch.Tensor], target: torch.Tensor,
                source: torch.Tensor, relight_mask: torch.Tensor,
                skin_mask: torch.Tensor | None, *, source_clean: torch.Tensor,
                reference: torch.Tensor, reference_clean: torch.Tensor,
                source_light_mask: torch.Tensor,
                reference_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        losses = self.image_loss(
            prediction, target, source, relight_mask, skin_mask)
        source_target, source_valid = self.illumination_target(source, source_clean)
        reference_target, reference_valid = self.illumination_target(
            reference, reference_clean)
        target_light, target_valid = self.illumination_target(target, source_clean)

        source_mask = source_light_mask * source_valid
        reference_valid_mask = reference_mask * reference_valid
        target_mask = source_light_mask * target_valid
        source_illumination = luminance(prediction["source_illumination"])
        reference_illumination = luminance(prediction["reference_illumination"])
        reference_on_source = luminance(
            prediction["reference_illumination_on_source"])
        source_light_loss = self.image_loss.masked_mean(
            (source_illumination - source_target).abs(), source_mask)
        reference_light_loss = self.image_loss.masked_mean(
            (reference_illumination - reference_target).abs(),
            reference_valid_mask)
        target_light_loss = self.image_loss.masked_mean(
            (reference_on_source - target_light).abs(), target_mask)

        # Directly supervise the quantity used by relative-light inference.
        # Log space treats reciprocal exposure errors symmetrically: predicting
        # 2x instead of 4x is penalized like predicting 1x instead of 2x.
        source_luma = luminance(source)
        target_luma = luminance(target)
        gain_valid = (source_luma >= self.gain_target_floor).to(source.dtype)
        gain_mask = source_light_mask * gain_valid
        target_gain = (target_luma / source_luma.clamp_min(
            self.gain_target_floor)).clamp(
                self.gain_target_min, self.gain_target_max)
        predicted_gain = luminance(prediction["transfer_gain"]).clamp(
            self.gain_target_min, self.gain_target_max)
        gain_log_loss = self.image_loss.masked_mean(
            (predicted_gain.log() - target_gain.log()).abs(), gain_mask)
        predicted_gain_mean = self.image_loss.masked_mean(
            predicted_gain, gain_mask)
        target_gain_mean = self.image_loss.masked_mean(target_gain, gain_mask)

        # Ordinary RGB L1 underweights deep shadows: an error of 0.03 near
        # black is numerically small even though it is visually important.
        # Apply an extra log-luminance loss only when a dark source should be
        # lifted toward a brighter reference target.  This changes training
        # supervision only; inference remains the same achromatic gain model.
        output = prediction.get("loss_output", prediction["output"])
        output_luma = luminance(output).clamp_min(0.0)
        lift_need = torch.relu(
            target_luma - source_luma - self.dark_log_lift_threshold)
        darkness = torch.relu(
            self.dark_log_source_max - source_luma
        ) / self.dark_log_source_max
        dark_lift_mask = source_light_mask * lift_need * darkness
        dark_log_lift = self.image_loss.masked_mean(
            (
                (output_luma + self.dark_log_epsilon).log()
                - (target_luma + self.dark_log_epsilon).log()
            ).abs(),
            dark_lift_mask,
        )

        # The base image loss regularizes only prediction["theta"]. Replace it
        # with a symmetric penalty on the independently estimated source and
        # reference light parameters.
        source_reg = self.image_loss.lighting_regularization(
            prediction["source_theta"])
        reference_reg = self.image_loss.lighting_regularization(
            prediction["reference_theta"])
        symmetric_reg = 0.5 * (source_reg + reference_reg)
        total = (
            losses["total"]
            - self.image_loss.lambda_reg * losses["regularization"]
            + self.image_loss.lambda_reg * symmetric_reg
            + self.lambda_source_illumination * source_light_loss
            + self.lambda_reference_illumination * reference_light_loss
            + self.lambda_target_illumination * target_light_loss
            + self.lambda_gain * gain_log_loss
            + self.lambda_dark_log_lift * dark_log_lift
        )
        theta_gap = (
            prediction["source_theta_normalized"]
            - prediction["reference_theta_normalized"]
        ).abs().mean()
        losses.update({
            "total": total,
            "regularization": symmetric_reg,
            "illum_source": source_light_loss,
            "illum_reference": reference_light_loss,
            "illum_target": target_light_loss,
            "gain_log": gain_log_loss,
            "gain_mean": prediction["transfer_gain"].mean(),
            "gain_face_mean": predicted_gain_mean,
            "gain_target_mean": target_gain_mean,
            "dark_log_lift": dark_log_lift,
            "theta_gap": theta_gap,
        })
        return losses
