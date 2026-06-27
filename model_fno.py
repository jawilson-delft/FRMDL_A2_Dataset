"""Standard FNO2d (Li et al., 2021 style) for eikonal value-function regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default FNO2d capacity (baseline λ=100 runs).
DEFAULT_WIDTH = 32
DEFAULT_MODES = 12
DEFAULT_N_LAYERS = 4

# neuraloperator / FNO benchmark convention: modes = half training resolution (64 -> 32).
REFERENCE_MODES_HALF_RES = 32

# Reduced capacity: half width, one fewer spectral layer, ~33% fewer Fourier modes.
REDUCED_WIDTH = DEFAULT_WIDTH // 2
REDUCED_MODES = 8
REDUCED_N_LAYERS = DEFAULT_N_LAYERS - 1


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SpectralConv2d(nn.Module):
    """2D Fourier layer: spectral convolution over low Fourier modes."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def compl_mul2d(
        self, input_ft: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, height)
        m2 = min(self.modes2, width // 2 + 1)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(height, width))


class FNOBlock2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, kernel_size=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral(x)
        x2 = self.w(x)
        x = x1 + x2
        x = x + self.mlp(x)
        return F.gelu(x)


class FNO2d(nn.Module):
    """FNO2d with lifting → Fourier layers → projection.

    Input channels default to occupancy + (x, y) grid coordinates.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        width: int = 32,
        modes: int = 12,
        n_layers: int = 4,
    ):
        super().__init__()
        self.width = width
        self.modes = modes
        self.n_layers = n_layers

        self.lifting = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=1),
        )
        self.blocks = nn.ModuleList(
            [FNOBlock2d(width, modes, modes) for _ in range(n_layers)]
        )
        self.projection = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width * 2, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        return self.projection(x)


def build_grid(resolution: int, device: torch.device, batch: int) -> torch.Tensor:
    """Return (B, 2, H, W) coordinate channels on [0, 1]."""
    coords = torch.linspace(0.0, 1.0, resolution, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)
    return grid


def prepare_input(occupancy: torch.Tensor, use_grid: bool = True) -> torch.Tensor:
    """occupancy: (B, 1, H, W) binary map."""
    if not use_grid:
        return occupancy
    b, _, h, w = occupancy.shape
    grid = build_grid(h, occupancy.device, b)
    return torch.cat([occupancy, grid], dim=1)


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean relative L2: ||pred-target||_2 / ||target||_2 per sample."""
    diff = (pred - target).reshape(pred.shape[0], -1)
    tgt = target.reshape(target.shape[0], -1)
    num = torch.linalg.vector_norm(diff, dim=1)
    den = torch.linalg.vector_norm(tgt, dim=1).clamp_min(eps)
    return torch.mean(num / den)


def relative_l2_masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-sample relative L2 over masked pixels; empty masks contribute zero."""
    if mask.dtype != torch.bool:
        mask = mask.bool()
    m = mask.unsqueeze(1)
    diff = (pred - target) * m
    tgt = target * m
    diff_flat = diff.reshape(pred.shape[0], -1)
    tgt_flat = tgt.reshape(target.shape[0], -1)
    num = torch.linalg.vector_norm(diff_flat, dim=1)
    den = torch.linalg.vector_norm(tgt_flat, dim=1)
    valid = mask.reshape(mask.shape[0], -1).any(dim=1)
    if not valid.any():
        return pred.sum() * 0.0
    per_sample = num[valid] / den[valid].clamp_min(eps)
    return torch.mean(per_sample)


def combined_kink_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lambda_kink: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, whole_domain, near_kink) losses."""
    whole = relative_l2_loss(pred, target, eps=eps)
    kink = relative_l2_masked_loss(pred, target, mask, eps=eps)
    total = whole + lambda_kink * kink
    return total, whole, kink


@torch.no_grad()
def relative_l2_error(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return relative_l2_loss(pred, target, eps=eps)
