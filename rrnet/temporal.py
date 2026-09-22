"""Equation (5), parameter-space temporal smoothing."""

from __future__ import annotations

import torch


class LightingEMA:
    def __init__(self, beta: float = 0.95) -> None:
        if not 0.8 <= beta <= 0.99:
            raise ValueError("RRNet paper requires beta in [0.8, 0.99]")
        self.beta = beta
        self.state: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = None

    def update(self, theta: torch.Tensor) -> torch.Tensor:
        value = theta.detach()
        if self.state is None or self.state.shape != value.shape:
            self.state = value.clone()
        else:
            self.state.mul_(self.beta).add_(value, alpha=1.0 - self.beta)
        return self.state


class ResidualEMA:
    """Motion-adaptive smoothing for a dense luminance-residual map.

    Static regions use the requested EMA beta.  Where the source luminance
    changes, the previous residual is trusted less to avoid motion trails.
    """

    def __init__(self, beta: float = 0.85, motion_threshold: float = 0.05) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("residual beta must be in [0, 1).")
        if motion_threshold <= 0.0:
            raise ValueError("motion_threshold must be positive.")
        self.beta = beta
        self.motion_threshold = motion_threshold
        self.state: torch.Tensor | None = None
        self.previous_guide: torch.Tensor | None = None

    def reset(self) -> None:
        self.state = None
        self.previous_guide = None

    def update(self, residual: torch.Tensor, guide_luma: torch.Tensor) -> torch.Tensor:
        value = residual.detach()
        guide = guide_luma.detach()
        if (self.state is None or self.previous_guide is None
                or self.state.shape != value.shape
                or self.previous_guide.shape != guide.shape):
            self.state = value.clone()
        else:
            motion = (guide - self.previous_guide).abs()
            local_beta = self.beta * torch.exp(-motion / self.motion_threshold)
            self.state = local_beta * self.state + (1.0 - local_beta) * value
        self.previous_guide = guide.clone()
        return self.state
