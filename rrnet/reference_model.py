"""Reference-conditioned luminance relighting built on the RRNet canonicalizer."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .losses import luminance
from .lprm import resize_shorter_side
from .model import RRNet


class LightnessResidualDecoder(nn.Module):
    """Small quarter-resolution decoder that can lift shadows additively."""

    def __init__(self, code_dim: int = 32, channels: int = 32,
                 factor: int = 4, max_lift: float = 0.35,
                 max_darken: float = 0.35) -> None:
        super().__init__()
        if factor < 1:
            raise ValueError("Residual factor must be positive.")
        self.factor = factor
        if max_lift <= 0.0 or max_darken <= 0.0:
            raise ValueError("Luminance residual limits must be positive.")
        self.max_lift = max_lift
        self.max_darken = max_darken
        input_channels = code_dim + 5  # source/base Y, depth, x, y
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, source: torch.Tensor, canonical: torch.Tensor,
                depth: torch.Tensor, reference_code: torch.Tensor) -> torch.Tensor:
        height, width = source.shape[-2:]
        small_size = (max(1, (height + self.factor - 1) // self.factor),
                      max(1, (width + self.factor - 1) // self.factor))
        source_y = F.interpolate(luminance(source), small_size, mode="bilinear",
                                 align_corners=False)
        canonical_y = F.interpolate(luminance(canonical), small_size, mode="bilinear",
                                    align_corners=False)
        small_depth = F.interpolate(depth, small_size, mode="bilinear",
                                    align_corners=False)
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, small_size[0], device=source.device,
                           dtype=source.dtype),
            torch.linspace(-1.0, 1.0, small_size[1], device=source.device,
                           dtype=source.dtype), indexing="ij")
        coords = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(
            source.shape[0], -1, -1, -1)
        code = reference_code[:, :, None, None].expand(
            -1, -1, small_size[0], small_size[1])
        features = torch.cat((source_y, canonical_y, small_depth, coords, code), dim=1)
        unit_delta = torch.tanh(self.net(features))
        delta = torch.where(
            unit_delta >= 0.0,
            self.max_lift * unit_delta,
            self.max_darken * unit_delta,
        )
        return F.interpolate(delta, (height, width), mode="bilinear", align_corners=False)


class ReferenceRRNet(nn.Module):
    """Canonicalize a source, then apply luminance-only lighting from a reference."""

    def __init__(self, *, reference_code_dim: int = 32,
                 residual_channels: int = 32, residual_factor: int = 4,
                 max_luma_residual: float = 0.35,
                 max_luma_lift: float | None = None,
                 max_luma_darken: float | None = None,
                 safe_spatial_color: bool = True,
                 max_global_wb_shift: float = 0.10,
                 shadow_chroma_floor: float = 0.04,
                 **rrnet_kwargs) -> None:
        super().__init__()
        if not 0.0 <= max_global_wb_shift < 1.0:
            raise ValueError("max_global_wb_shift must be in [0, 1).")
        if shadow_chroma_floor <= 0.0:
            raise ValueError("shadow_chroma_floor must be positive.")
        self.base = RRNet(**rrnet_kwargs)
        self.safe_spatial_color = safe_spatial_color
        self.max_global_wb_shift = max_global_wb_shift
        self.shadow_chroma_floor = shadow_chroma_floor
        embedding_dim = self.base.lprm.encoder.embedding_dim
        self.reference_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, reference_code_dim),
        )
        # A masked 4x4 luminance grid plus its mean/std gives the decoder a
        # direct, identity-light summary of global and spatial illumination.
        self.reference_stats_head = nn.Sequential(
            nn.Linear(18, 64),
            nn.GELU(),
            nn.Linear(64, reference_code_dim),
        )
        self.decoder = LightnessResidualDecoder(
            reference_code_dim, residual_channels, residual_factor,
            max_luma_residual if max_luma_lift is None else max_luma_lift,
            max_luma_residual if max_luma_darken is None else max_luma_darken)

    @property
    def agm(self):
        return self.base.agm

    @property
    def depth(self):
        return self.base.depth

    @property
    def renderer(self):
        return self.base.renderer

    def train(self, mode: bool = True) -> "ReferenceRRNet":
        super().train(mode)
        self.base.depth.eval()
        return self

    def encode_reference(self, reference: torch.Tensor,
                         reference_mask: torch.Tensor | None = None) -> torch.Tensor:
        gray = luminance(reference)
        if reference_mask is None:
            mask = torch.ones_like(gray)
        else:
            mask = reference_mask.to(device=gray.device, dtype=gray.dtype).clamp(0.0, 1.0)
        masked_gray = gray * mask
        reference_gray = masked_gray.expand(-1, 3, -1, -1)
        shorter_side = max(32, self.base.lprm.shorter_side // self.base.lprm.coarse_factor)
        small = resize_shorter_side(reference_gray, shorter_side)
        _, embedding = self.base.lprm.encoder(small)
        pooled_mask = F.adaptive_avg_pool2d(mask, (4, 4)).clamp_min(1.0e-4)
        grid = F.adaptive_avg_pool2d(masked_gray, (4, 4)) / pooled_mask
        valid = mask.sum(dim=(2, 3)).clamp_min(1.0)
        mean = masked_gray.sum(dim=(2, 3)) / valid
        variance = (((gray - mean[:, :, None, None]) * mask).square().sum(dim=(2, 3))
                    / valid)
        stats = torch.cat((grid.flatten(1), mean, variance.sqrt()), dim=1)
        return torch.tanh(self.reference_head(embedding) + self.reference_stats_head(stats))

    def sanitize_illumination(self, illumination: torch.Tensor) -> torch.Tensor:
        """Keep spatial light achromatic and allow only bounded global WB."""
        if not self.safe_spatial_color:
            return illumination
        spatial_luma = luminance(illumination).clamp_min(0.0)
        global_rgb = illumination.mean(dim=(2, 3), keepdim=True).clamp_min(1.0e-4)
        global_luma = luminance(global_rgb).clamp_min(1.0e-4)
        white_balance = global_rgb / global_luma
        limit = self.max_global_wb_shift
        white_balance = white_balance.clamp(1.0 - limit, 1.0 + limit)
        white_balance = white_balance / luminance(white_balance).clamp_min(1.0e-4)
        return spatial_luma * white_balance

    def safe_base_prediction(self, source: torch.Tensor) -> dict[str, torch.Tensor]:
        prediction = self.base(source)
        raw_illumination = prediction["illumination"]
        safe_illumination = self.sanitize_illumination(raw_illumination)
        render_base = prediction.get("albedo", source)
        raw_output = render_base * safe_illumination
        prediction["unsafe_illumination"] = raw_illumination
        prediction["illumination"] = safe_illumination
        prediction["raw_output"] = raw_output
        prediction["output"] = raw_output.clamp(0.0, 1.0)
        return prediction

    def apply_reference_code(self, source: torch.Tensor,
                             base_prediction: dict[str, torch.Tensor],
                             reference_code: torch.Tensor,
                             relight_mask: torch.Tensor | None = None,
                             luma_residual_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        canonical = base_prediction["output"]
        delta_y = (self.decoder(source, canonical, base_prediction["depth"], reference_code)
                   if luma_residual_override is None else luma_residual_override)
        return self.apply_luma_residual(
            source, base_prediction, delta_y, reference_code, relight_mask)

    def apply_luma_residual(self, source: torch.Tensor,
                            base_prediction: dict[str, torch.Tensor],
                            delta_y: torch.Tensor,
                            reference_code: torch.Tensor,
                            relight_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        canonical = base_prediction["output"]
        # Apply a shared RGB gain rather than adding reference RGB values.  This
        # changes brightness while preserving the canonical person's per-pixel
        # chromaticity (and therefore cannot copy the reference person's skin).
        canonical_luma = luminance(canonical).clamp_min(0.0)
        canonical_y = canonical_luma.clamp_min(1.0e-4)
        target_y = (canonical_luma + delta_y).clamp_min(0.0)
        gain = target_y / canonical_y
        # Cap the shared gain before any individual RGB channel clips.  Without
        # this cap, channel-wise output clamping could introduce a hue shift.
        max_channel = canonical.clamp_min(0.0).amax(dim=1, keepdim=True).clamp_min(1.0e-4)
        gain = torch.minimum(gain, 1.0 / max_channel)
        multiplicative = canonical * gain
        # In nearly black regions multiplication cannot recover visibility.
        # Blend in a neutral additive fill only for positive residuals and only
        # below the chroma-reliable luminance floor.
        shadow_weight = (1.0 - canonical_luma / self.shadow_chroma_floor).clamp(0.0, 1.0)
        shadow_weight = shadow_weight * (delta_y > 0.0).to(delta_y.dtype)
        additive_lift = torch.minimum(
            delta_y.clamp_min(0.0), 1.0 - canonical.amax(dim=1, keepdim=True))
        additive = canonical + additive_lift
        raw_output = multiplicative * (1.0 - shadow_weight) + additive * shadow_weight
        output = raw_output.clamp(0.0, 1.0)
        result = dict(base_prediction)
        result.update({
            "canonical_output": canonical,
            "raw_output": raw_output,
            "output": output,
            "luma_residual": delta_y,
            "reference_code": reference_code,
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
        base_prediction = self.safe_base_prediction(source)
        code = self.encode_reference(reference, reference_mask)
        return self.apply_reference_code(source, base_prediction, code, relight_mask)
