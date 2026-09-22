"""Reference-conditioned RRNet using only compact lighting parameters.

The reference participant controls the virtual-light vector theta.  No dense
image residual is predicted, so source texture remains on the renderer path.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .losses import luminance
from .model import RRNet


class ReferenceThetaConditioner(nn.Module):
    """Fuse source and reference lighting evidence into a theta correction."""

    def __init__(self, embedding_dim: int, theta_dim: int,
                 hidden_channels: int = 256,
                 theta_delta_scale: float = 2.0) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if theta_delta_scale <= 0.0:
            raise ValueError("theta_delta_scale must be positive")
        self.theta_delta_scale = float(theta_delta_scale)
        input_dim = 2 * embedding_dim + 2 * theta_dim + 18
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, theta_dim),
        )
        # Training starts from the ordinary source-only RRNet prediction.  The
        # reference correction is learned only when triplet supervision needs it.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, source_embedding: torch.Tensor,
                source_theta_normalized: torch.Tensor,
                reference_embedding: torch.Tensor,
                reference_theta_normalized: torch.Tensor,
                reference_stats: torch.Tensor) -> torch.Tensor:
        fused = torch.cat((
            source_embedding,
            reference_embedding,
            source_theta_normalized,
            reference_theta_normalized,
            reference_stats,
        ), dim=1)
        return self.theta_delta_scale * torch.tanh(self.net(fused))


class ReferenceThetaRRNet(nn.Module):
    """RRNet extension that transfers reference lighting through theta only.

    Reference RGB is converted to luminance before encoding, and rendered
    illumination is achromatic by default.  This prevents the selected
    participant's skin colour from being copied to the source participant.
    """

    def __init__(self, *, reference_fusion_channels: int = 256,
                 reference_theta_delta_scale: float = 2.0,
                 achromatic_illumination: bool = True,
                 **rrnet_kwargs) -> None:
        super().__init__()
        self.base = RRNet(**rrnet_kwargs)
        self.achromatic_illumination = bool(achromatic_illumination)
        embedding_dim = self.base.lprm.encoder.embedding_dim
        theta_dim = self.base.lprm.denormalize.mean.shape[1]
        self.conditioner = ReferenceThetaConditioner(
            embedding_dim,
            theta_dim,
            hidden_channels=reference_fusion_channels,
            theta_delta_scale=reference_theta_delta_scale,
        )

    @property
    def agm(self):
        return self.base.agm

    @property
    def depth(self):
        return self.base.depth

    @property
    def renderer(self):
        return self.base.renderer

    def train(self, mode: bool = True) -> "ReferenceThetaRRNet":
        super().train(mode)
        self.base.depth.eval()
        return self

    @staticmethod
    def _prepare_reference(reference: torch.Tensor,
                           reference_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        gray = luminance(reference)
        if reference_mask is None:
            mask = torch.ones_like(gray)
        else:
            mask = reference_mask.to(device=gray.device, dtype=gray.dtype)
            mask = mask.clamp(0.0, 1.0)
        valid = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        mean = (gray * mask).sum(dim=(2, 3), keepdim=True) / valid
        # Filling the ignored region with the face mean avoids teaching the
        # reference encoder that a black masked background is part of the light.
        prepared_gray = gray * mask + mean * (1.0 - mask)
        return prepared_gray.expand(-1, 3, -1, -1), mask

    @staticmethod
    def _reference_stats(reference_gray: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        weighted = reference_gray * mask
        pooled_mask = F.adaptive_avg_pool2d(mask, (4, 4)).clamp_min(1.0e-4)
        grid = F.adaptive_avg_pool2d(weighted, (4, 4)) / pooled_mask
        valid = mask.sum(dim=(2, 3)).clamp_min(1.0)
        mean = weighted.sum(dim=(2, 3)) / valid
        variance = (((reference_gray - mean[:, :, None, None]) * mask).square()
                    .sum(dim=(2, 3)) / valid)
        return torch.cat((grid.flatten(1), mean, variance.sqrt()), dim=1)

    def encode_reference(self, reference: torch.Tensor,
                         reference_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        prepared, mask = self._prepare_reference(reference, reference_mask)
        prediction = self.base.lprm(prepared)
        return {
            "embedding": prediction["embedding"],
            "theta_normalized": prediction["theta_normalized"],
            "stats": self._reference_stats(prepared[:, 0:1], mask),
        }

    def estimate_conditioned_lighting(
            self, source: torch.Tensor,
            reference_code: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        source_prediction = self.base.lprm(source)
        theta_delta = self.conditioner(
            source_prediction["embedding"],
            source_prediction["theta_normalized"],
            reference_code["embedding"],
            reference_code["theta_normalized"],
            reference_code["stats"],
        )
        theta_normalized = source_prediction["theta_normalized"] + theta_delta
        theta = self.base.lprm.denormalize(theta_normalized)
        return {
            **source_prediction,
            "theta": theta,
            "conditioned_theta_normalized": theta_normalized,
            "reference_theta_delta": theta_delta,
        }

    def illumination_from_theta(self, depth: torch.Tensor,
                                theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        illumination, normals = self.renderer.illumination(depth, theta)
        if self.achromatic_illumination:
            illumination = luminance(illumination).expand(-1, 3, -1, -1)
        return illumination, normals

    def render_conditioned(self, source: torch.Tensor,
                           lighting: dict[str, torch.Tensor | list[torch.Tensor]],
                           relight_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            depth = self.depth(source)
        render_base = source
        result: dict[str, torch.Tensor] = {
            "depth": depth,
            "theta": lighting["theta"],
            "theta0": lighting["theta0"],
            "theta_offset": lighting["theta_offset"],
            "reference_theta_delta": lighting["reference_theta_delta"],
        }
        if self.agm is not None:
            agm_result = self.agm(source, lighting["features"])
            render_base = agm_result["albedo"]
            result.update(agm_result)
        illumination, normals = self.illumination_from_theta(
            depth, lighting["theta"])
        raw_output = render_base * illumination
        output = (raw_output.clamp(0.0, 1.0)
                  if self.renderer.clamp_output else raw_output)
        result.update({
            "raw_output": raw_output,
            "output": output,
            "illumination": illumination,
            "normals": normals,
        })
        if relight_mask is not None:
            result["loss_output"] = self.renderer.blend_relight(
                raw_output, source, relight_mask)
            result["output"] = self.renderer.blend_relight(
                output, source, relight_mask)
        return result

    def forward(self, source: torch.Tensor, reference: torch.Tensor,
                relight_mask: torch.Tensor | None = None,
                reference_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        reference_code = self.encode_reference(reference, reference_mask)
        lighting = self.estimate_conditioned_lighting(source, reference_code)
        return self.render_conditioned(source, lighting, relight_mask)
