"""Small convolutional encoder-decoder baseline (no spectral layers)."""

from __future__ import annotations

import torch
import torch.nn as nn

DEFAULT_WIDTH = 32


class CNN2d(nn.Module):
    """Same-resolution conv stack: occupancy + (x, y) grid -> scalar field V."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        width: int = DEFAULT_WIDTH,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
