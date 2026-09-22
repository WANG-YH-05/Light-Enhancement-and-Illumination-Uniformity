"""RRNet reference relighting through explicit relative illumination transfer."""

from __future__ import annotations

import torch
from torch import nn

from .losses import luminance
from .model import RRNet


class ReferenceRelativeRRNet(nn.Module):
    """Estimate input/reference light separately and transfer their ratio.

    Both images share the same LPRM.  The reference light is re-rendered on the
    source depth before division, so identity, texture, and geometry always come
    from the source participant.  No dense image residual is used.
    """

    def __init__(self, *, illumination_floor: float = 0.05,
                 min_transfer_gain: float = 0.40,
                 max_transfer_gain: float = 3.00,
                 achromatic_illumination: bool = True,
                 preserve_chromaticity: bool = True,
                 output_headroom: float = 0.999,
                 **rrnet_kwargs) -> None:
        super().__init__()
        if illumination_floor <= 0.0:
            raise ValueError("illumination_floor must be positive")
        if not 0.0 < min_transfer_gain <= 1.0:
            raise ValueError("min_transfer_gain must be in (0, 1]")
        if max_transfer_gain < 1.0:
            raise ValueError("max_transfer_gain must be at least one")
        if min_transfer_gain > max_transfer_gain:
            raise ValueError("min_transfer_gain cannot exceed max_transfer_gain")
        if not 0.0 < output_headroom <= 1.0:
            raise ValueError("output_headroom must be in (0, 1]")
        self.base = RRNet(**rrnet_kwargs)
        if self.base.agm is not None:
            raise ValueError("ReferenceRelativeRRNet currently requires use_agm: false")
        self.illumination_floor = float(illumination_floor)
        self.min_transfer_gain = float(min_transfer_gain)
        self.max_transfer_gain = float(max_transfer_gain)
        self.achromatic_illumination = bool(achromatic_illumination)
        self.preserve_chromaticity = bool(preserve_chromaticity)
        self.output_headroom = float(output_headroom)

    @property
    def agm(self):
        return self.base.agm

    @property
    def depth(self):
        return self.base.depth

    @property
    def renderer(self):
        return self.base.renderer

    def train(self, mode: bool = True) -> "ReferenceRelativeRRNet":
        super().train(mode)
        self.base.depth.eval()
        return self

    @staticmethod
    def prepare_light_input(image: torch.Tensor,
                            mask: torch.Tensor | None) -> torch.Tensor:
        """Remove identity colour and ignored background before light encoding."""
        gray = luminance(image)
        if mask is None:
            return gray.expand(-1, 3, -1, -1)
        mask = mask.to(device=gray.device, dtype=gray.dtype).clamp(0.0, 1.0)
        valid = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (gray * mask).sum(dim=(2, 3), keepdim=True) / valid
        prepared = gray * mask + mean * (1.0 - mask)
        return prepared.expand(-1, 3, -1, -1)

    def estimate_light(self, image: torch.Tensor,
                       mask: torch.Tensor | None = None) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        return self.base.lprm(self.prepare_light_input(image, mask))

    def encode_reference(self, reference: torch.Tensor,
                         reference_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        prediction = self.estimate_light(reference, reference_mask)
        return {
            "theta": prediction["theta"],
            "theta_normalized": prediction["theta_normalized"],
        }

    def illumination_from_theta(self, depth: torch.Tensor,
                                theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        illumination, normals = self.renderer.illumination(depth, theta)
        if self.achromatic_illumination:
            illumination = luminance(illumination).expand(-1, 3, -1, -1)
        return illumination, normals

    def transfer(self, source: torch.Tensor, source_depth: torch.Tensor,
                 source_theta: torch.Tensor,
                 reference_theta: torch.Tensor,
                 relight_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        source_light, normals = self.illumination_from_theta(source_depth, source_theta)
        reference_light, _ = self.illumination_from_theta(source_depth, reference_theta)
        transfer_gain = self.compute_transfer_gain(
            source, source_light, reference_light)
        raw_output = source * transfer_gain
        output = (raw_output.clamp(0.0, 1.0)
                  if self.renderer.clamp_output else raw_output)
        result = {
            "output": output,
            "raw_output": raw_output,
            "depth": source_depth,
            "theta": reference_theta,
            "source_theta": source_theta,
            "reference_theta": reference_theta,
            "source_illumination": source_light,
            "reference_illumination_on_source": reference_light,
            "transfer_gain": transfer_gain,
            "normals": normals,
        }
        if relight_mask is not None:
            result["loss_output"] = self.renderer.blend_relight(
                raw_output, source, relight_mask)
            result["output"] = self.renderer.blend_relight(
                output, source, relight_mask)
        return result

    def compute_transfer_gain(self, source: torch.Tensor,
                              source_light: torch.Tensor,
                              reference_light: torch.Tensor) -> torch.Tensor:
        """Return a bounded shared RGB gain without channel clipping.

        A common gain preserves per-pixel chromaticity only while no individual
        channel clips.  The optional headroom cap enforces that condition.
        """
        denominator = source_light.clamp_min(self.illumination_floor)
        gain = (reference_light.clamp_min(0.0) / denominator).clamp(
            self.min_transfer_gain, self.max_transfer_gain)
        if self.preserve_chromaticity:
            source_peak = source.clamp_min(0.0).amax(
                dim=1, keepdim=True).clamp_min(1.0e-4)
            chroma_safe_gain = self.output_headroom / source_peak
            gain = torch.minimum(gain, chroma_safe_gain)
        return gain

    def forward(self, source: torch.Tensor, reference: torch.Tensor,
                relight_mask: torch.Tensor | None = None,
                reference_mask: torch.Tensor | None = None,
                source_light_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        source_prediction = self.estimate_light(source, source_light_mask)
        reference_prediction = self.estimate_light(reference, reference_mask)
        with torch.no_grad():
            source_depth = self.depth(source)
            reference_depth = self.depth(reference)
        result = self.transfer(
            source,
            source_depth,
            source_prediction["theta"],
            reference_prediction["theta"],
            relight_mask,
        )
        reference_light, _ = self.illumination_from_theta(
            reference_depth, reference_prediction["theta"])
        result.update({
            "reference_depth": reference_depth,
            "reference_illumination": reference_light,
            "source_theta_normalized": source_prediction["theta_normalized"],
            "reference_theta_normalized": reference_prediction["theta_normalized"],
            "source_theta0": source_prediction["theta0"],
            "source_theta_offset": source_prediction["theta_offset"],
            "reference_theta0": reference_prediction["theta0"],
            "reference_theta_offset": reference_prediction["theta_offset"],
        })
        return result
