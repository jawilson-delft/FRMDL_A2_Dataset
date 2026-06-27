#!/usr/bin/env python3
"""Standalone diagnostic: analytical convex-corner V vs FMM on a bare wedge."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon
from matplotlib.path import Path as MplPath

from geometry import (
    DOMAIN_MAX,
    DOMAIN_MIN,
    SampleGeometry,
    rasterize_polygon,
    render_sample,
    solve_eikonal,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "diagnosis"

DOMAIN_DIAGONAL = float(np.hypot(DOMAIN_MAX - DOMAIN_MIN, DOMAIN_MAX - DOMAIN_MIN))
NEAR_CORNER_FRAC = 0.05
RESOLUTIONS = (64, 128, 256, 512, 1024, 2048)
PANEL_RES = 2048
SANITY_RES = 512

# Fixed bare wedge: convex apex O, 90° opening left, goal to the right.
O = (0.5, 0.5)
G = (0.85, 0.5)
THETA_DEG = 90.0
FACE_LENGTH = 0.25
FACE_ANGLES_DEG = (135.0, 225.0)  # CCW interior wedge opening left through 180°


def build_wedge() -> tuple[tuple[tuple[float, float], ...], np.ndarray]:
    """Triangle obstacle: apex O + two face endpoints."""
    f1 = (
        O[0] + FACE_LENGTH * np.cos(np.deg2rad(FACE_ANGLES_DEG[0])),
        O[1] + FACE_LENGTH * np.sin(np.deg2rad(FACE_ANGLES_DEG[0])),
    )
    f2 = (
        O[0] + FACE_LENGTH * np.cos(np.deg2rad(FACE_ANGLES_DEG[1])),
        O[1] + FACE_LENGTH * np.sin(np.deg2rad(FACE_ANGLES_DEG[1])),
    )
    vertices = (O, f1, f2)
    return vertices, np.array(vertices, dtype=np.float64)


def _normalize_angle(angle: float) -> float:
    return float(np.mod(angle, 2.0 * np.pi))


def _angle_in_ccw_arc(theta: float, start: float, end: float) -> bool:
    """True if theta lies on the CCW arc from start to end (inclusive)."""
    theta = _normalize_angle(theta)
    start = _normalize_angle(start)
    end = _normalize_angle(end)
    if start <= end:
        return start <= theta <= end
    return theta >= start or theta <= end


def _ccw_arc_span(start: float, end: float) -> float:
    start = _normalize_angle(start)
    end = _normalize_angle(end)
    span = end - start
    if span < 0.0:
        span += 2.0 * np.pi
    return float(span)


def _angular_distance(a: float, b: float) -> float:
    d = abs(_normalize_angle(a) - _normalize_angle(b))
    return float(min(d, 2.0 * np.pi - d))


def obstacle_interior_arc(face_angles_rad: tuple[float, float]) -> tuple[float, float]:
    """CCW angular interval (size theta) occupied by the wedge interior at O."""
    a0, a1 = face_angles_rad
    if _ccw_arc_span(a0, a1) <= np.deg2rad(THETA_DEG) + 1e-9:
        return a0, a1
    return a1, a0


def shadow_boundary_angle(O_pt: tuple[float, float], G_pt: tuple[float, float]) -> float:
    """Ray from O extending G-O past O (away from G)."""
    ox, oy = O_pt
    gx, gy = G_pt
    return float(np.arctan2(oy - gy, ox - gx))


def far_face_angle(
    face_angles_rad: tuple[float, float], shadow_angle: float
) -> float:
    """Wedge face farthest from the shadow-boundary ray."""
    a0, a1 = face_angles_rad
    if _angular_distance(a0, shadow_angle) >= _angular_distance(a1, shadow_angle):
        return a0
    return a1


def valid_sector_arc(
    face_angles_rad: tuple[float, float], shadow_angle: float
) -> tuple[float, float]:
    """Free-space angular sector from shadow boundary to the far wedge face."""
    far = far_face_angle(face_angles_rad, shadow_angle)
    obs_start, obs_end = obstacle_interior_arc(face_angles_rad)

    # Two candidate CCW arcs from shadow to far face; pick the one outside obstacle interior.
    span_ccw = _ccw_arc_span(shadow_angle, far)
    span_cw = 2.0 * np.pi - span_ccw
    mid_ccw = _normalize_angle(shadow_angle + 0.5 * span_ccw)
    mid_cw = _normalize_angle(shadow_angle - 0.5 * span_cw)

    if not _angle_in_ccw_arc(mid_ccw, obs_start, obs_end):
        return shadow_angle, far
    if not _angle_in_ccw_arc(mid_cw, obs_start, obs_end):
        return far, shadow_angle
    # Fallback: shorter free-space arc.
    if span_ccw <= span_cw:
        return shadow_angle, far
    return far, shadow_angle


def _segments_properly_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
    eps: float = 1e-12,
) -> bool:
    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    if (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps) and (
        o3 > eps and o4 < -eps or o3 < -eps and o4 > eps
    ):
        return True
    return False


def segment_hits_obstacle_interior(
    p: tuple[float, float],
    q: tuple[float, float],
    vertices: np.ndarray,
) -> bool:
    """True if segment p-q intersects the closed polygon interior (blocking LOS)."""
    path = MplPath(vertices, closed=True)
    if path.contains_point(p, radius=-1e-9) or path.contains_point(q, radius=-1e-9):
        return True
    n = len(vertices)
    for i in range(n):
        a = tuple(vertices[i])
        b = tuple(vertices[(i + 1) % n])
        if _segments_properly_intersect(p, q, a, b):
            return True
    return False


def compute_analytical_V(
    xx: np.ndarray,
    yy: np.ndarray,
    O_pt: tuple[float, float],
    G_pt: tuple[float, float],
    wedge_vertices: np.ndarray,
    face_angles_deg: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical corner V with validity mask (True = valid free-space sector)."""
    face_angles_rad = tuple(np.deg2rad(a) for a in face_angles_deg)
    shadow_ang = shadow_boundary_angle(O_pt, G_pt)
    valid_start, valid_end = valid_sector_arc(face_angles_rad, shadow_ang)

    ox, oy = O_pt
    gx, gy = G_pt
    d = float(np.hypot(ox - gx, oy - gy))

    path = MplPath(wedge_vertices, closed=True)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    in_obstacle = path.contains_points(points).reshape(xx.shape)

    theta_pts = np.arctan2(yy - oy, xx - ox)
    in_valid_sector = np.zeros_like(xx, dtype=bool)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            in_valid_sector[i, j] = _angle_in_ccw_arc(
                float(theta_pts[i, j]), valid_start, valid_end
            )

    free = ~in_obstacle
    valid_mask = free & in_valid_sector

    dist_O = np.hypot(xx - ox, yy - oy)
    dist_G = np.hypot(xx - gx, yy - gy)
    V = np.full(xx.shape, np.nan, dtype=np.float64)

    blocked = np.zeros(xx.shape, dtype=bool)
    rows, cols = np.where(valid_mask)
    for r, c in zip(rows, cols):
        x_pt = (float(xx[r, c]), float(yy[r, c]))
        blocked[r, c] = segment_hits_obstacle_interior(x_pt, G_pt, wedge_vertices)

    V[valid_mask & blocked] = dist_O[valid_mask & blocked] + d
    V[valid_mask & ~blocked] = dist_G[valid_mask & ~blocked]
    return V.astype(np.float32), valid_mask


def relative_l2_masked(
    pred: np.ndarray, ref: np.ndarray, mask: np.ndarray, eps: float = 1e-8
) -> float:
    if not np.any(mask):
        return float("nan")
    diff = (pred - ref)[mask]
    tgt = ref[mask]
    return float(np.linalg.norm(diff) / max(np.linalg.norm(tgt), eps))


def near_corner_mask(
    xx: np.ndarray, yy: np.ndarray, center: tuple[float, float], radius_frac: float
) -> np.ndarray:
    radius = radius_frac * DOMAIN_DIAGONAL
    return np.hypot(xx - center[0], yy - center[1]) <= radius


def check_shadow_boundary(
    O_pt: tuple[float, float], G_pt: tuple[float, float], d: float
) -> tuple[bool, float]:
    """Sample along shadow ray; |x-O|+d must equal |x-G| to fp precision."""
    ox, oy = O_pt
    gx, gy = G_pt
    direction = np.array([ox - gx, oy - gy], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    ts = np.logspace(-4, 0, 200)  # 1e-4 .. 1.0 past O
    max_disc = 0.0
    for t in ts:
        x = ox + t * direction[0]
        y = oy + t * direction[1]
        disc = abs(np.hypot(x - ox, y - oy) + d - np.hypot(x - gx, y - gy))
        max_disc = max(max_disc, disc)
    passed = max_disc < 64.0 * np.finfo(np.float64).eps * max(d, 1.0)
    return passed, float(max_disc)


def make_geometry(vertices: tuple[tuple[float, float], ...]) -> SampleGeometry:
    return SampleGeometry(
        theta_deg=THETA_DEG,
        center=O,
        rotation_deg=0.0,
        scale=FACE_LENGTH,
        goal=G,
        vertices=vertices,
        sample_id=0,
    )


def coord_grids(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    return np.meshgrid(xs, ys)


def plot_analytical_sanity(
    V: np.ndarray,
    valid_mask: np.ndarray,
    wedge_vertices: np.ndarray,
    face_angles_deg: tuple[float, float],
    out_path: Path,
) -> None:
    res = V.shape[0]
    xx, yy = coord_grids(res)
    face_angles_rad = tuple(np.deg2rad(a) for a in face_angles_deg)
    shadow_ang = shadow_boundary_angle(O, G)
    far = far_face_angle(face_angles_rad, shadow_ang)
    valid_start, valid_end = valid_sector_arc(face_angles_rad, shadow_ang)

    fig, ax = plt.subplots(figsize=(7, 6))
    V_plot = np.where(valid_mask, V, np.nan)
    im = ax.imshow(
        V_plot,
        origin="lower",
        extent=[DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX],
        cmap="viridis",
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, label="V (analytical)")

    poly = Polygon(wedge_vertices, closed=True, fill=False, edgecolor="white", linewidth=2)
    ax.add_patch(poly)
    ax.plot(*O, "r^", markersize=10, label="O (apex)")
    ax.plot(*G, "g*", markersize=14, label="G (goal)")

    ray_len = 0.45
    for ang, color, label in (
        (shadow_ang, "cyan", "shadow boundary"),
        (far, "orange", "far face ray"),
    ):
        ax.plot(
            [O[0], O[0] + ray_len * np.cos(ang)],
            [O[1], O[1] + ray_len * np.sin(ang)],
            color=color,
            linewidth=2,
            linestyle="--",
            label=label,
        )

    # Valid-sector boundary arcs (short rays at endpoints).
    for ang, color in ((valid_start, "magenta"), (valid_end, "magenta")):
        ax.plot(
            [O[0], O[0] + 0.35 * np.cos(ang)],
            [O[1], O[1] + 0.35 * np.sin(ang)],
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )
    ax.set_title("Analytical V: valid sector, shadow & face rays")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(
    whole_errors: dict[int, float],
    corner_errors: dict[int, float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rs = list(RESOLUTIONS)
    ax.plot(rs, [whole_errors[r] for r in rs], "o-", label="whole valid region")
    ax.plot(rs, [corner_errors[r] for r in rs], "s-", label="near-corner band")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Rel. L2 vs analytical")
    ax.set_title("FMM vs analytical convergence (bare 90° wedge)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_panels(
    fmm: np.ndarray,
    analytical: np.ndarray,
    valid_mask: np.ndarray,
    occupancy: np.ndarray,
    out_path: Path,
) -> None:
    res = fmm.shape[0]
    xx, yy = coord_grids(res)
    corner = near_corner_mask(xx, yy, O, NEAR_CORNER_FRAC) & valid_mask
    diff = np.abs(fmm - analytical)
    diff[~valid_mask] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    extent = [DOMAIN_MIN, DOMAIN_MAX, DOMAIN_MIN, DOMAIN_MAX]
    titles = ("FMM", "Analytical", "|FMM − analytical|")
    fields = (
        np.where(occupancy > 0.5, fmm, np.nan),
        np.where(valid_mask, analytical, np.nan),
        diff,
    )
    for ax, field, title in zip(axes, fields, titles):
        im = ax.imshow(field, origin="lower", extent=extent, cmap="viridis", aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(title)
        if np.any(corner):
            cy, cx = np.where(corner)
            ax.scatter(
                xx[0, cx],
                yy[cy, 0],
                s=0.05,
                c="none",
                edgecolors="red",
                linewidths=0.2,
                alpha=0.35,
            )
        circ = Circle(O, NEAR_CORNER_FRAC * DOMAIN_DIAGONAL, fill=False, edgecolor="red", lw=1.2)
        ax.add_patch(circ)
    fig.suptitle(f"FMM vs analytical at res={res}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def assess_corner_trend(corner_errors: dict[int, float]) -> tuple[str, float, bool]:
    ordered = [corner_errors[r] for r in RESOLUTIONS]
    floor_val = ordered[-1]
    shrinks = ordered[-1] < ordered[0] * 0.5 and ordered[-1] < ordered[0] - 1e-4
    if ordered[-1] < 0.25 * ordered[0] and ordered[-1] < 1e-2:
        trend = "CONVERGES TOWARD ZERO"
    elif shrinks and ordered[-1] < 0.05:
        trend = "CONVERGES TOWARD ZERO"
    elif ordered[-1] >= 0.9 * ordered[0]:
        trend = "PERSISTS AT A FLOOR"
    elif ordered[-1] < ordered[0]:
        trend = "CONVERGES TOWARD ZERO"
    else:
        trend = "PERSISTS AT A FLOOR"
    return trend, floor_val, shrinks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vertices_tuple, wedge_vertices = build_wedge()
    geom = make_geometry(vertices_tuple)
    d = float(np.hypot(O[0] - G[0], O[1] - G[1]))

    # Step 2: analytical sanity plot at moderate resolution.
    xx_s, yy_s = coord_grids(SANITY_RES)
    V_analytical_s, valid_s = compute_analytical_V(
        xx_s, yy_s, O, G, wedge_vertices, FACE_ANGLES_DEG
    )
    sanity_path = OUT_DIR / "analytical_solution_sanity.png"
    plot_analytical_sanity(
        V_analytical_s, valid_s, wedge_vertices, FACE_ANGLES_DEG, sanity_path
    )
    print(f"Saved {sanity_path}")

    # Shadow-boundary consistency (must pass before FMM comparison).
    shadow_ok, max_disc = check_shadow_boundary(O, G, d)
    if not shadow_ok:
        print(f"Shadow-boundary consistency check: FAIL [max discrepancy {max_disc:.3e}]")
        print("Stopping: shadow-boundary check failed.")
        sys.exit(1)

    whole_errors: dict[int, float] = {}
    corner_errors: dict[int, float] = {}
    fmm_panel = None
    occ_panel = None
    V_panel = None
    valid_panel = None

    for res in RESOLUTIONS:
        xx, yy = coord_grids(res)
        V_ref, valid_mask = compute_analytical_V(
            xx, yy, O, G, wedge_vertices, FACE_ANGLES_DEG
        )
        occ, V_fmm = render_sample(geom, res)
        free = occ > 0.5
        eval_mask = valid_mask & free

        whole_errors[res] = relative_l2_masked(V_fmm, V_ref, eval_mask)
        corner = near_corner_mask(xx, yy, O, NEAR_CORNER_FRAC) & eval_mask
        corner_errors[res] = relative_l2_masked(V_fmm, V_ref, corner)

        if res == PANEL_RES:
            fmm_panel, occ_panel, V_panel, valid_panel = V_fmm, occ, V_ref, valid_mask

    conv_path = OUT_DIR / "fmm_vs_analytical_convergence.png"
    plot_convergence(whole_errors, corner_errors, conv_path)
    print(f"Saved {conv_path}")

    panel_path = OUT_DIR / "fmm_vs_analytical_2048_panels.png"
    plot_panels(fmm_panel, V_panel, valid_panel, occ_panel, panel_path)
    print(f"Saved {panel_path}")

    trend, floor_val, shrinks = assess_corner_trend(corner_errors)
    shrinks_txt = "yes" if shrinks else "no"

    print()
    print("Shadow-boundary consistency check: PASS" if shadow_ok else "Shadow-boundary consistency check: FAIL", end="")
    print(f" [max discrepancy {max_disc:.3e}]")
    whole_parts = ", ".join(f"res={r}:{whole_errors[r]:.4e}" for r in RESOLUTIONS)
    corner_parts = ", ".join(f"res={r}:{corner_errors[r]:.4e}" for r in RESOLUTIONS)
    print(f"FMM vs analytical error, whole valid region: [{whole_parts}]")
    print(f"FMM vs analytical error, near-corner band:   [{corner_parts}]")
    print(f"Near-corner error trend: {trend}")
    print(f"  floor={floor_val:.4e} at res=2048, shrinks 64→2048: {shrinks_txt}")


if __name__ == "__main__":
    main()
