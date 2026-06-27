"""Near-kink region masks and masked relative-L2 utilities."""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import torch

from geometry import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    THETA_VALUES,
    SampleGeometry,
    notch_tip_world,
)

DOMAIN_DIAGONAL = float(np.hypot(DOMAIN_MAX - DOMAIN_MIN, DOMAIN_MAX - DOMAIN_MIN))
DEFAULT_KINK_RADIUS_FRAC = 0.05
DEFAULT_LAMBDA_KINK = 100.0
FLAT_WALL_THETA_THRESHOLD = 179.9

_COORD_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def get_coord_grids(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    if resolution not in _COORD_CACHE:
        xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
        ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
        _COORD_CACHE[resolution] = np.meshgrid(xs, ys)
    return _COORD_CACHE[resolution]


def geom_from_npz_row(data: np.lib.npyio.NpzFile, idx: int) -> SampleGeometry:
    vertices = tuple(tuple(v) for v in json.loads(str(data["vertices_json"][idx])))
    return SampleGeometry(
        theta_deg=float(data["theta_deg"][idx]),
        center=(float(data["center"][idx, 0]), float(data["center"][idx, 1])),
        rotation_deg=float(data["rotation_deg"][idx]),
        scale=float(data["scale"][idx]),
        goal=(float(data["goal"][idx, 0]), float(data["goal"][idx, 1])),
        vertices=vertices,
        sample_id=int(data["sample_id"][idx]),
    )


def get_kink_mask(
    theta_deg: float,
    center: tuple[float, float],
    rotation_deg: float,
    scale: float,
    resolution: int,
    radius_frac: float = DEFAULT_KINK_RADIUS_FRAC,
) -> np.ndarray:
    """Boolean (H, W) mask: pixels within radius_frac * domain diagonal of notch tip.

    Returns all-False for flat-wall control (theta >= 179.9°, no notch).
    """
    if theta_deg >= FLAT_WALL_THETA_THRESHOLD:
        return np.zeros((resolution, resolution), dtype=bool)

    geom = SampleGeometry(
        theta_deg=theta_deg,
        center=center,
        rotation_deg=rotation_deg,
        scale=scale,
        goal=(0.0, 0.0),
        vertices=(),
        sample_id=0,
    )
    tip = notch_tip_world(geom)
    xx, yy = get_coord_grids(resolution)
    radius = radius_frac * DOMAIN_DIAGONAL
    return np.hypot(xx - tip[0], yy - tip[1]) <= radius


def get_kink_mask_from_npz(
    data: np.lib.npyio.NpzFile,
    idx: int,
    resolution: int,
    radius_frac: float = DEFAULT_KINK_RADIUS_FRAC,
) -> np.ndarray:
    return get_kink_mask(
        float(data["theta_deg"][idx]),
        (float(data["center"][idx, 0]), float(data["center"][idx, 1])),
        float(data["rotation_deg"][idx]),
        float(data["scale"][idx]),
        resolution,
        radius_frac=radius_frac,
    )


def precompute_kink_masks(
    data: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    resolution: int,
    radius_frac: float = DEFAULT_KINK_RADIUS_FRAC,
) -> np.ndarray:
    """Return (N, H, W) bool masks for the given sample indices."""
    masks = np.zeros((len(indices), resolution, resolution), dtype=bool)
    for i, idx in enumerate(indices):
        masks[i] = get_kink_mask_from_npz(data, int(idx), resolution, radius_frac)
    return masks


def mean_kink_pixel_fraction(masks: np.ndarray, occupancy: np.ndarray) -> float:
    """Mean fraction of free-space pixels inside the kink mask."""
    fracs: list[float] = []
    for i in range(len(masks)):
        free = occupancy[i] > 0.5
        n_free = int(free.sum())
        if n_free == 0:
            continue
        fracs.append(float((masks[i] & free).sum()) / n_free)
    return float(np.mean(fracs)) if fracs else 0.0


def relative_l2_masked_numpy(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    if not np.any(mask):
        return 0.0
    diff = (pred - target)[mask]
    tgt = target[mask]
    return float(np.linalg.norm(diff) / max(np.linalg.norm(tgt), eps))
