"""Dual-branch Lighting Parameter Regression Module (paper Fig. 3a)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .lighting import ParameterDenormalizer, parameter_dim
from .repvit import RepViTM09Encoder


class ResolutionBatchNorm2d(nn.BatchNorm2d):
    """Shared affine weights with separate coarse/full running statistics."""

    def __init__(self, original: nn.BatchNorm2d) -> None:
        super().__init__(original.num_features, original.eps, original.momentum,
                         original.affine, original.track_running_stats)
        self.register_buffer('coarse_running_mean', original.running_mean.clone())
        self.register_buffer('coarse_running_var', original.running_var.clone())
        self.register_buffer('coarse_num_batches_tracked', original.num_batches_tracked.clone())
        self.load_state_dict(original.state_dict())
        self.coarse = False

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Upgrade historical checkpoints without changing their full-resolution
        # statistics. Coarse statistics initially copy the historical ones.
        for name in ('running_mean', 'running_var', 'num_batches_tracked'):
            if prefix + 'coarse_' + name not in state_dict and prefix + name in state_dict:
                state_dict[prefix + 'coarse_' + name] = state_dict[prefix + name].clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.coarse:
            return super().forward(x)
        factor = self.momentum
        if self.training:
            self.coarse_num_batches_tracked.add_(1)
            if factor is None:
                factor = 1.0 / float(self.coarse_num_batches_tracked)
        return F.batch_norm(x, self.coarse_running_mean, self.coarse_running_var,
                            self.weight, self.bias, self.training,
                            0.0 if factor is None else factor, self.eps)


def split_resolution_statistics(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, ResolutionBatchNorm2d(child))
        else:
            split_resolution_statistics(child)


def resize_shorter_side(image: torch.Tensor, shorter_side: int) -> torch.Tensor:
    height, width = image.shape[-2:]
    scale = shorter_side / min(height, width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    return F.interpolate(image, size=(new_height, new_width), mode="bilinear", align_corners=False)


class LPRM(nn.Module):
    def __init__(self, num_lights: int = 9, shorter_side: int = 512,
                 coarse_factor: int = 4, statistics_path: str | None = None,
                 split_resolution_bn: bool = False,
                 light_parameterization: str = "affine") -> None:
        super().__init__()
        self.num_lights = num_lights
        self.shorter_side = shorter_side
        self.coarse_factor = coarse_factor
        self.encoder = RepViTM09Encoder()  # E0 and E1 are literally the same object.
        self.split_resolution_bn = bool(split_resolution_bn)
        if self.split_resolution_bn:
            split_resolution_statistics(self.encoder)
        dim = parameter_dim(num_lights)
        embedding_dim = self.encoder.embedding_dim
        self.r0 = nn.Sequential(nn.BatchNorm1d(embedding_dim), nn.Linear(embedding_dim, dim))
        self.r1 = nn.Linear(embedding_dim + dim, dim)
        self.denormalize = ParameterDenormalizer(num_lights, statistics_path, light_parameterization)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        resized = resize_shorter_side(image, self.shorter_side)
        coarse_size = (
            max(1, math.ceil(resized.shape[-2] / self.coarse_factor)),
            max(1, math.ceil(resized.shape[-1] / self.coarse_factor)),
        )
        coarse = F.interpolate(resized, size=coarse_size, mode="bilinear", align_corners=False)
        if self.split_resolution_bn:
            for module in self.encoder.modules():
                if isinstance(module, ResolutionBatchNorm2d):
                    module.coarse = True
        try:
            _, coarse_embedding = self.encoder(coarse)
        finally:
            if self.split_resolution_bn:
                for module in self.encoder.modules():
                    if isinstance(module, ResolutionBatchNorm2d):
                        module.coarse = False
        theta0 = self.r0(coarse_embedding)
        full_features, full_embedding = self.encoder(resized)
        theta_offset = self.r1(torch.cat((full_embedding, theta0), dim=1))
        theta_latent = theta0 + theta_offset
        theta = self.denormalize(theta_latent)
        theta_normalized = (theta_latent if self.denormalize.parameterization == "affine"
                            else (theta - self.denormalize.mean) / self.denormalize.std)
        return {
            "theta": theta,
            "theta_latent": theta_latent,
            "theta_normalized": theta_normalized,
            "theta0": theta0,
            "theta_offset": theta_offset,
            "features": full_features,
            "embedding": full_embedding,
            "resized_input": resized,
        }
