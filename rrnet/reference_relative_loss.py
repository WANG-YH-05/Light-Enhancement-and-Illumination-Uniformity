"""Losses for supervised input/reference illumination decomposition."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .losses import RRNetLoss, luminance


class ReferenceRelativeLoss(nn.Module):
    """RRNet image losses plus explicit illumination-map supervision."""

    def __init__(self, *, lambda_source_illumination: float = 0.50,
                 lambda_reference_illumination: float = 0.50,
                 lambda_target_illumination: float = 1.00,
                 lambda_gain: float = 0.0,
                 lambda_consistency: float = 0.0,
                 lambda_reference_contrast: float = 0.0,
                 lambda_gain_gradient: float = 0.0,
                 lambda_illumination_exposure: float = 0.0,
                 lambda_illumination_shape: float = 0.0,
                 lambda_illumination_shape_gradient: float = 0.0,
                 lambda_spatial_variance: float = 0.0,
                 lambda_ambient_ratio: float = 0.0,
                 lambda_shape_consistency: float = 0.0,
                 lambda_theta_source: float = 0.0,
                 lambda_theta_reference: float = 0.0,
                 gain_gradient_scales: tuple[int, ...] | list[int] = (1, 2, 4),
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
        if lambda_consistency < 0.0:
            raise ValueError("lambda_consistency must be non-negative")
        if lambda_reference_contrast < 0.0:
            raise ValueError("lambda_reference_contrast must be non-negative")
        if lambda_gain_gradient < 0.0:
            raise ValueError("lambda_gain_gradient must be non-negative")
        decomposition_weights = {
            "lambda_illumination_exposure": lambda_illumination_exposure,
            "lambda_illumination_shape": lambda_illumination_shape,
            "lambda_illumination_shape_gradient": lambda_illumination_shape_gradient,
            "lambda_spatial_variance": lambda_spatial_variance,
            "lambda_ambient_ratio": lambda_ambient_ratio,
            "lambda_shape_consistency": lambda_shape_consistency,
        }
        if any(float(value) < 0.0 for value in decomposition_weights.values()):
            raise ValueError("illumination-decomposition weights must be non-negative")
        if lambda_theta_source < 0.0 or lambda_theta_reference < 0.0:
            raise ValueError("theta supervision weights must be non-negative")
        if not gain_gradient_scales or any(
                int(scale) < 1 for scale in gain_gradient_scales):
            raise ValueError("gain_gradient_scales must contain positive integers")
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
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_reference_contrast = float(lambda_reference_contrast)
        self.lambda_gain_gradient = float(lambda_gain_gradient)
        self.lambda_illumination_exposure = float(lambda_illumination_exposure)
        self.lambda_illumination_shape = float(lambda_illumination_shape)
        self.lambda_illumination_shape_gradient = float(
            lambda_illumination_shape_gradient)
        self.lambda_spatial_variance = float(lambda_spatial_variance)
        self.lambda_ambient_ratio = float(lambda_ambient_ratio)
        self.lambda_shape_consistency = float(lambda_shape_consistency)
        self.lambda_theta_source = float(lambda_theta_source)
        self.lambda_theta_reference = float(lambda_theta_reference)
        self.gain_gradient_scales = tuple(
            sorted(set(int(scale) for scale in gain_gradient_scales)))
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

    def gain_gradient_loss(self, predicted_log_gain: torch.Tensor,
                           target_log_gain: torch.Tensor,
                           mask: torch.Tensor) -> torch.Tensor:
        """Match local illumination changes at several spatial resolutions."""
        total = predicted_log_gain.new_zeros(())
        for scale in self.gain_gradient_scales:
            if scale > 1:
                predicted = F.avg_pool2d(
                    predicted_log_gain, scale, stride=scale)
                target = F.avg_pool2d(target_log_gain, scale, stride=scale)
                valid = F.avg_pool2d(mask, scale, stride=scale)
            else:
                predicted, target, valid = (
                    predicted_log_gain, target_log_gain, mask)
            error_x = (
                (predicted[:, :, :, 1:] - predicted[:, :, :, :-1])
                - (target[:, :, :, 1:] - target[:, :, :, :-1])
            ).abs()
            error_y = (
                (predicted[:, :, 1:, :] - predicted[:, :, :-1, :])
                - (target[:, :, 1:, :] - target[:, :, :-1, :])
            ).abs()
            mask_x = torch.minimum(valid[:, :, :, 1:], valid[:, :, :, :-1])
            mask_y = torch.minimum(valid[:, :, 1:, :], valid[:, :, :-1, :])
            total = total + 0.5 * (
                self.image_loss.masked_mean(error_x, mask_x)
                + self.image_loss.masked_mean(error_y, mask_y)
            )
        return total / len(self.gain_gradient_scales)

    @staticmethod
    def masked_sample_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return one masked mean per image without mixing batch members."""
        mask = mask.to(value).clamp(0.0, 1.0)
        return ((value * mask).sum(dim=(1, 2, 3), keepdim=True)
                / mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0))

    def normalized_illumination(self, value: torch.Tensor,
                                mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        exposure = self.masked_sample_mean(value, mask).clamp_min(1.0e-4)
        return value / exposure, exposure

    def masked_sample_std(self, value: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        mean = self.masked_sample_mean(value, mask)
        variance = self.masked_sample_mean((value - mean).square(), mask)
        return (variance + 1.0e-8).sqrt()

    def illumination_decomposition_loss(
            self, predicted: torch.Tensor, target: torch.Tensor,
            mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Separate global exposure from normalized spatial light shape."""
        predicted_shape, predicted_exposure = self.normalized_illumination(
            predicted, mask)
        target_shape, target_exposure = self.normalized_illumination(target, mask)
        exposure = (
            predicted_exposure.log() - target_exposure.log()).abs().mean()
        shape = self.image_loss.masked_mean(
            (predicted_shape - target_shape).abs(), mask)
        shape_gradient = self.gain_gradient_loss(
            predicted_shape.clamp_min(1.0e-4).log(),
            target_shape.clamp_min(1.0e-4).log(), mask)
        spatial_variance = (
            self.masked_sample_std(predicted_shape, mask)
            - self.masked_sample_std(target_shape, mask)
        ).abs().mean()
        return (exposure, shape, shape_gradient, spatial_variance,
                predicted_shape, target_shape, predicted_exposure,
                target_exposure)

    def forward(self, prediction: dict[str, torch.Tensor], target: torch.Tensor,
                source: torch.Tensor, relight_mask: torch.Tensor,
                skin_mask: torch.Tensor | None, *, source_clean: torch.Tensor,
                reference: torch.Tensor, reference_clean: torch.Tensor,
                source_light_mask: torch.Tensor,
                reference_mask: torch.Tensor,
                group_size: int = 1,
                uniform_group_mask: torch.Tensor | None = None,
                source_theta_normalized_target: torch.Tensor | None = None,
                reference_theta_normalized_target: torch.Tensor | None = None,
                source_theta_target: torch.Tensor | None = None,
                reference_theta_target: torch.Tensor | None = None,
                ) -> dict[str, torch.Tensor]:
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

        source_decomposition = self.illumination_decomposition_loss(
            source_illumination, source_target, source_mask)
        reference_decomposition = self.illumination_decomposition_loss(
            reference_illumination, reference_target, reference_valid_mask)
        target_decomposition = self.illumination_decomposition_loss(
            reference_on_source, target_light, target_mask)
        decompositions = (
            source_decomposition, reference_decomposition, target_decomposition)
        illumination_exposure = sum(item[0] for item in decompositions) / 3.0
        illumination_shape = sum(item[1] for item in decompositions) / 3.0
        illumination_shape_gradient = sum(
            item[2] for item in decompositions) / 3.0
        spatial_variance = sum(item[3] for item in decompositions) / 3.0

        # Match the *target-dependent* ambient share. Uniform lights may use a
        # large ambient term; directional samples may not explain their spatial
        # pattern by ambient light alone.
        ambient_ratio = source_illumination.new_zeros(())
        if source_theta_target is not None and reference_theta_target is not None:
            predicted_source_ambient = luminance(
                prediction["source_theta"][:, -3:, None, None])
            predicted_reference_ambient = luminance(
                prediction["reference_theta"][:, -3:, None, None])
            target_source_ambient = luminance(
                source_theta_target[:, -3:, None, None])
            target_reference_ambient = luminance(
                reference_theta_target[:, -3:, None, None])
            predicted_source_ratio = (
                predicted_source_ambient / source_decomposition[6])
            predicted_reference_ratio = (
                predicted_reference_ambient / reference_decomposition[6])
            target_source_ratio = target_source_ambient / source_decomposition[7]
            target_reference_ratio = (
                target_reference_ambient / reference_decomposition[7])
            ambient_ratio = 0.5 * (
                (predicted_source_ratio - target_source_ratio).abs().mean()
                + (predicted_reference_ratio
                   - target_reference_ratio).abs().mean())

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
        gain_gradient = self.gain_gradient_loss(
            predicted_gain.log(), target_gain.log(), gain_mask)
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

        # Every member of a grouped sample is the same aligned source frame
        # under a different input light, with one exact shared reference and
        # target. Penalizing disagreement makes input-light invariance explicit
        # instead of hoping it emerges from independent reconstruction pairs.
        consistency = output.new_zeros(())
        shape_consistency = output.new_zeros(())
        reference_contrast = output.new_zeros(())
        if group_size > 1:
            if output.shape[0] % group_size:
                raise ValueError(
                    f"Batch of {output.shape[0]} cannot be divided into "
                    f"groups of {group_size}."
                )
            grouped_output = luminance(output).reshape(
                -1, group_size, 1, *output.shape[-2:])
            grouped_mask = source_light_mask.reshape(
                -1, group_size, 1, *source_light_mask.shape[-2:])
            if uniform_group_mask is not None:
                if uniform_group_mask.numel() != output.shape[0]:
                    raise ValueError("uniform_group_mask must match batch size")
                group_flags = uniform_group_mask.reshape(-1, group_size)
                if not torch.all(group_flags == group_flags[:, :1]):
                    raise ValueError("uniform_group_mask must be constant within a group")
                grouped_mask = grouped_mask * group_flags[:, :1, None, None, None].to(
                    grouped_mask.dtype)
            group_mean = grouped_output.mean(dim=1, keepdim=True)
            consistency_error = (grouped_output - group_mean).abs().flatten(0, 1)
            consistency_mask = grouped_mask.flatten(0, 1)
            consistency = self.image_loss.masked_mean(
                consistency_error, consistency_mask)
            output_shape, _ = self.normalized_illumination(
                luminance(output), source_light_mask)
            grouped_shape = output_shape.reshape(
                -1, group_size, 1, *output.shape[-2:])
            shape_mean = grouped_shape.mean(dim=1, keepdim=True)
            shape_error = (grouped_shape - shape_mean).abs().flatten(0, 1)
            shape_consistency = self.image_loss.masked_mean(
                shape_error, consistency_mask)
            if uniform_group_mask is not None:
                sensitivity_mask = (
                    (~group_flags[:, :1])[:, :, None, None, None].to(
                        grouped_mask.dtype)
                    * source_light_mask.reshape(
                        -1, group_size, 1, *source_light_mask.shape[-2:])
                )
                grouped_target = target_luma.reshape(
                    -1, group_size, 1, *target_luma.shape[-2:])
                output_delta = grouped_output - grouped_output.mean(
                    dim=1, keepdim=True)
                target_delta = grouped_target - grouped_target.mean(
                    dim=1, keepdim=True)
                reference_contrast = self.image_loss.masked_mean(
                    (output_delta - target_delta).abs().flatten(0, 1),
                    sensitivity_mask.flatten(0, 1))

        # The base image loss regularizes only prediction["theta"]. Replace it
        # with a symmetric penalty on the independently estimated source and
        # reference light parameters.
        source_reg = self.image_loss.lighting_regularization(
            prediction["source_theta"])
        reference_reg = self.image_loss.lighting_regularization(
            prediction["reference_theta"])
        symmetric_reg = 0.5 * (source_reg + reference_reg)
        theta_source_loss = output.new_zeros(())
        theta_reference_loss = output.new_zeros(())
        if source_theta_normalized_target is not None:
            theta_source_loss = (
                prediction["source_theta_normalized"]
                - source_theta_normalized_target
            ).abs().mean()
        if reference_theta_normalized_target is not None:
            theta_reference_loss = (
                prediction["reference_theta_normalized"]
                - reference_theta_normalized_target
            ).abs().mean()
        total = (
            losses["total"]
            - self.image_loss.lambda_reg * losses["regularization"]
            + self.image_loss.lambda_reg * symmetric_reg
            + self.lambda_source_illumination * source_light_loss
            + self.lambda_reference_illumination * reference_light_loss
            + self.lambda_target_illumination * target_light_loss
            + self.lambda_gain * gain_log_loss
            + self.lambda_consistency * consistency
            + self.lambda_reference_contrast * reference_contrast
            + self.lambda_gain_gradient * gain_gradient
            + self.lambda_illumination_exposure * illumination_exposure
            + self.lambda_illumination_shape * illumination_shape
            + self.lambda_illumination_shape_gradient * illumination_shape_gradient
            + self.lambda_spatial_variance * spatial_variance
            + self.lambda_ambient_ratio * ambient_ratio
            + self.lambda_shape_consistency * shape_consistency
            + self.lambda_dark_log_lift * dark_log_lift
            + self.lambda_theta_source * theta_source_loss
            + self.lambda_theta_reference * theta_reference_loss
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
            "consistency": consistency,
            "reference_contrast": reference_contrast,
            "gain_gradient": gain_gradient,
            "illum_exposure": illumination_exposure,
            "illum_shape": illumination_shape,
            "illum_shape_gradient": illumination_shape_gradient,
            "illum_variance": spatial_variance,
            "ambient_ratio": ambient_ratio,
            "shape_consistency": shape_consistency,
            "gain_mean": prediction["transfer_gain"].mean(),
            "gain_face_mean": predicted_gain_mean,
            "gain_target_mean": target_gain_mean,
            "dark_log_lift": dark_log_lift,
            "theta_gap": theta_gap,
            "theta_source": theta_source_loss,
            "theta_reference": theta_reference_loss,
        })
        return losses
