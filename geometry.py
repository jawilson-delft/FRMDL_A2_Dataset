"""Geometry, rasterization, and FMM eikonal solver for the corner-sharpness dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.path import Path

THETA_VALUES = (180, 150, 120, 90, 60, 30, 10)
TRAIN_RESOLUTION = 64
TEST_RESOLUTIONS = (64, 128, 256, 512)
DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0


@dataclass(frozen=True)
class SampleGeometry:
    """Continuous geometry parameters for one sample."""

    theta_deg: float
    center: tuple[float, float]
    rotation_deg: float
    scale: float
    goal: tuple[float, float]
    vertices: tuple[tuple[float, float], ...]
    sample_id: int

    def param_key(self) -> tuple[Any, ...]:
        """Key for deduplication (excludes goal)."""
        return (
            self.theta_deg,
            round(self.center[0], 10),
            round(self.center[1], 10),
            round(self.rotation_deg, 10),
            round(self.scale, 10),
        )

    def full_key(self) -> tuple[Any, ...]:
        return self.param_key() + (
            round(self.goal[0], 10),
            round(self.goal[1], 10),
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "theta_deg": self.theta_deg,
            "center": list(self.center),
            "rotation_deg": self.rotation_deg,
            "scale": self.scale,
            "goal": list(self.goal),
            "vertices": [list(v) for v in self.vertices],
            "sample_id": self.sample_id,
        }


def build_wedge_vertices(theta_deg: float, scale: float) -> np.ndarray:
    """Build local obstacle polygon vertices before rigid transform.

    The polygon is a rectangle with an optional symmetric V-notch on the top edge.
    The interior reflex angle at the notch tip equals ``theta_deg`` (180° = flat wall).
    """
    width = scale
    height = scale * 0.85
    half_w = width / 2.0
    half_h = height / 2.0

    if theta_deg >= 179.9:
        return np.array(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ],
            dtype=np.float64,
        )

    theta_rad = np.deg2rad(theta_deg)
    notch_depth = height * 0.35
    half_opening = notch_depth * np.tan((np.pi - theta_rad) / 2.0)
    half_opening = min(half_opening, half_w * 0.95)

    tip_y = half_h - notch_depth
    return np.array(
        [
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [half_opening, half_h],
            [0.0, tip_y],
            [-half_opening, half_h],
            [-half_w, half_h],
        ],
        dtype=np.float64,
    )


def transform_vertices(
    local_vertices: np.ndarray,
    center: tuple[float, float],
    rotation_deg: float,
) -> np.ndarray:
    """Apply rotation then translation to place the obstacle in the unit square."""
    theta = np.deg2rad(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    rotated = local_vertices @ rot.T
    rotated[:, 0] += center[0]
    rotated[:, 1] += center[1]
    return rotated


def rasterize_polygon(vertices: np.ndarray, resolution: int) -> np.ndarray:
    """Rasterize polygon onto a unit-square grid.

    Returns occupancy with shape (resolution, resolution) where 1 = free, 0 = obstacle.
    Grid coordinates: x along columns, y along rows, both in [0, 1].
    """
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = Path(vertices, closed=True).contains_points(points)
    obstacle = inside.reshape(resolution, resolution)
    return (~obstacle).astype(np.float32)


def polygon_inside_domain(vertices: np.ndarray, margin: float = 0.0) -> bool:
    lo = DOMAIN_MIN + margin
    hi = DOMAIN_MAX - margin
    return bool(
        np.all(vertices[:, 0] >= lo)
        and np.all(vertices[:, 0] <= hi)
        and np.all(vertices[:, 1] >= lo)
        and np.all(vertices[:, 1] <= hi)
    )


def notch_tip_local(theta_deg: float, scale: float) -> np.ndarray | None:
    """Local coordinates of the reflex-corner tip, or None for flat wall."""
    if theta_deg >= 179.9:
        return None
    local = build_wedge_vertices(theta_deg, scale)
    # CCW polygon: tip is the reflex vertex on the notched edge.
    return local[4].copy()


def notch_tip_world(geom: SampleGeometry) -> tuple[float, float]:
    """Return the reflex-corner (notch tip) in world coordinates."""
    tip_local = notch_tip_local(geom.theta_deg, geom.scale)
    if tip_local is None:
        return geom.center
    tip_world = transform_vertices(
        tip_local.reshape(1, 2), geom.center, geom.rotation_deg
    )[0]
    return float(tip_world[0]), float(tip_world[1])


def min_distance_to_polygon(point: tuple[float, float], vertices: np.ndarray) -> float:
    """Minimum Euclidean distance from point to polygon edges."""
    px, py = point
    dists = []
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-14:
            dists.append(np.hypot(px - x1, py - y1))
            continue
        t = np.clip(((px - x1) * dx + (py - y1) * dy) / seg_len_sq, 0.0, 1.0)
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        dists.append(np.hypot(px - proj_x, py - proj_y))
    return float(min(dists))


def solve_eikonal(occupancy: np.ndarray, goal: tuple[float, float]) -> np.ndarray:
    """Solve |∇V| = 1 in free space (c=1) with V(goal)=0 via scikit-fmm."""
    import skfmm

    resolution = occupancy.shape[0]
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    xx, yy = np.meshgrid(xs, ys)

    gx, gy = goal
    col = int(np.clip(np.round(gx * (resolution - 1)), 0, resolution - 1))
    row = int(np.clip(np.round(gy * (resolution - 1)), 0, resolution - 1))

    phi = np.ones((resolution, resolution), dtype=np.float64)
    phi[row, col] = -1.0

    mask = occupancy < 0.5
    if mask[row, col]:
        raise ValueError("Goal lies inside the obstacle.")

    speed = np.ma.MaskedArray(np.ones_like(occupancy, dtype=np.float64), mask=mask)
    dx = (DOMAIN_MAX - DOMAIN_MIN) / max(resolution - 1, 1)
    travel_time = skfmm.travel_time(phi, speed, dx=dx)
    solution = np.asarray(travel_time, dtype=np.float64)
    solution[mask] = 0.0
    solution[row, col] = 0.0
    return solution.astype(np.float32)


def geometry_hash(key: tuple[Any, ...]) -> str:
  return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def make_sample_geometry(
    rng: np.random.Generator,
    theta_deg: float,
    sample_id: int,
    used_param_keys: set[tuple[Any, ...]],
    used_full_keys: set[tuple[Any, ...]],
    *,
    min_goal_obstacle_dist: float = 0.04,
    min_goal_tip_dist: float = 0.06,
    boundary_margin: float = 0.05,
    scale_range: tuple[float, float] = (0.08, 0.18),
    max_attempts: int = 5000,
) -> SampleGeometry:
    """Sample obstacle pose/scale and goal with rejection constraints."""
    for _ in range(max_attempts):
        scale = float(rng.uniform(*scale_range))
        local_vertices = build_wedge_vertices(theta_deg, scale)
        extent = np.max(np.abs(local_vertices)) + 0.01
        lo = DOMAIN_MIN + boundary_margin + extent
        hi = DOMAIN_MAX - boundary_margin - extent
        if lo >= hi:
            continue

        center = (float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi)))
        rotation_deg = float(rng.uniform(0.0, 360.0))
        vertices = transform_vertices(local_vertices, center, rotation_deg)

        if not polygon_inside_domain(vertices, margin=0.01):
            continue

        param_key = (
            theta_deg,
            round(center[0], 10),
            round(center[1], 10),
            round(rotation_deg, 10),
            round(scale, 10),
        )
        if param_key in used_param_keys:
            continue

        for _goal_try in range(200):
            goal = (float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95)))
            full_key = param_key + (round(goal[0], 10), round(goal[1], 10))
            if full_key in used_full_keys:
                continue
            if min_distance_to_polygon(goal, vertices) < min_goal_obstacle_dist:
                continue

            tip = notch_tip_world(
                SampleGeometry(
                    theta_deg=theta_deg,
                    center=center,
                    rotation_deg=rotation_deg,
                    scale=scale,
                    goal=goal,
                    vertices=tuple(map(tuple, vertices.tolist())),
                    sample_id=sample_id,
                )
            )
            if np.hypot(goal[0] - tip[0], goal[1] - tip[1]) < min_goal_tip_dist:
                continue

            vertex_tuples = tuple((float(v[0]), float(v[1])) for v in vertices)
            return SampleGeometry(
                theta_deg=theta_deg,
                center=center,
                rotation_deg=rotation_deg,
                scale=scale,
                goal=goal,
                vertices=vertex_tuples,
                sample_id=sample_id,
            )

    raise RuntimeError(
        f"Failed to sample valid geometry for theta={theta_deg} after {max_attempts} attempts."
    )


def render_sample(geom: SampleGeometry, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize and solve FMM for a geometry at the given resolution."""
    vertices = np.array(geom.vertices, dtype=np.float64)
    occupancy = rasterize_polygon(vertices, resolution)
    travel_time = solve_eikonal(occupancy, geom.goal)
    return occupancy, travel_time
