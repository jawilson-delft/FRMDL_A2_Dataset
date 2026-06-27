#!/usr/bin/env python3
"""Cropped ultra-high-resolution FMM convergence near the kink (standalone diagnostic)."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from geometry import DOMAIN_MAX, DOMAIN_MIN, notch_tip_world, render_sample
from kink_utils import (
    DEFAULT_KINK_RADIUS_FRAC,
    DOMAIN_DIAGONAL,
    FLAT_WALL_THETA_THRESHOLD,
    geom_from_npz_row,
    get_kink_mask,
    relative_l2_masked_numpy,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "diagnosis"

CONVERGENCE_THETAS = (180, 90, 10)
SAMPLE_IDX = 0
RESOLUTIONS = (256, 512, 1024, 2048, 4096, 8192)
CROP_RADIUS_MULTIPLIER = 2.5
REF_RESOLUTION = 8192


def assess_dirichlet_crop_feasible() -> tuple[str, str]:
    """skfmm only supports zero-contour (phi=0) sources and masked obstacles."""
    try:
        import skfmm  # noqa: F401
    except ImportError:
        return "(b)", "skfmm unavailable; full-domain solve required"
    return (
        "(b)",
        "skfmm.travel_time lacks Dirichlet BC on crop boundaries; "
        "full-domain solve + crop-and-analyze",
    )


def crop_bbox_world(geom) -> tuple[tuple[float, float, float, float], float, float]:
    """Return world bbox (x_min, x_max, y_min, y_max), kink radius, half-extent."""
    tip = notch_tip_world(geom)
    kink_radius = DEFAULT_KINK_RADIUS_FRAC * DOMAIN_DIAGONAL
    half_extent = CROP_RADIUS_MULTIPLIER * kink_radius
    x_min = max(DOMAIN_MIN, tip[0] - half_extent)
    x_max = min(DOMAIN_MAX, tip[0] + half_extent)
    y_min = max(DOMAIN_MIN, tip[1] - half_extent)
    y_max = min(DOMAIN_MAX, tip[1] + half_extent)
    return (x_min, x_max, y_min, y_max), kink_radius, half_extent


def bbox_area_fraction(bbox: tuple[float, float, float, float]) -> float:
    x_min, x_max, y_min, y_max = bbox
    domain_area = (DOMAIN_MAX - DOMAIN_MIN) ** 2
    return float((x_max - x_min) * (y_max - y_min) / domain_area)


def world_to_index_ranges(
    bbox: tuple[float, float, float, float], resolution: int
) -> tuple[slice, slice]:
    """Map world bbox to row/col slices on the [0,1]^2 raster grid."""
    x_min, x_max, y_min, y_max = bbox
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    ys = np.linspace(DOMAIN_MIN, DOMAIN_MAX, resolution)
    col_idx = np.where((xs >= x_min) & (xs <= x_max))[0]
    row_idx = np.where((ys >= y_min) & (ys <= y_max))[0]
    if col_idx.size == 0 or row_idx.size == 0:
        raise ValueError(f"Empty crop at resolution {resolution} for bbox {bbox}")
    return slice(int(row_idx[0]), int(row_idx[-1]) + 1), slice(int(col_idx[0]), int(col_idx[-1]) + 1)


def sample_field_at_resolution(
    field: np.ndarray, res_src: int, res_tgt: int
) -> np.ndarray:
    """Bilinear sample field defined on res_src grid onto res_tgt grid."""
    coords = np.linspace(DOMAIN_MIN, DOMAIN_MAX, res_src)
    interp = RegularGridInterpolator(
        (coords, coords), field, bounds_error=False, fill_value=0.0
    )
    xs = np.linspace(DOMAIN_MIN, DOMAIN_MAX, res_tgt)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([yy, xx], axis=-1)
    return interp(pts).astype(np.float32)


def extract_crop(field: np.ndarray, row_sl: slice, col_sl: slice) -> np.ndarray:
    return field[row_sl, col_sl].copy()


def analysis_mask(
    geom,
    resolution: int,
    row_sl: slice,
    col_sl: slice,
    occupancy: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Kink mask within crop; fallback to crop free pixels for flat wall."""
    full_kink = get_kink_mask(
        geom.theta_deg,
        geom.center,
        geom.rotation_deg,
        geom.scale,
        resolution,
        radius_frac=DEFAULT_KINK_RADIUS_FRAC,
    )
    crop_kink = full_kink[row_sl, col_sl]
    free = occupancy[row_sl, col_sl] > 0.5
    mask = crop_kink & free
    label = "near-kink mask"
    if geom.theta_deg >= FLAT_WALL_THETA_THRESHOLD or not np.any(mask):
        mask = free
        label = "crop interior free (θ=180 control; no kink mask)"
    return mask, label


def describe_trend(errors: dict[int, float]) -> str:
    ordered = [errors[r] for r in RESOLUTIONS if r in errors and r < REF_RESOLUTION]
    if len(ordered) < 2:
        return "insufficient data"
    if ordered[-1] < ordered[0] * 0.25 and ordered[-1] < 1e-2:
        return "decreasing toward highest resolution"
    if ordered[-1] < ordered[0] * 0.5:
        return "decreasing but not fully converged at highest resolution"
    if ordered[-1] >= ordered[0] * 0.9:
        return "flat / persistent error across resolutions"
    return "mixed / slow decrease"


def run_convergence() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    method_label, method_note = assess_dirichlet_crop_feasible()

    # Use theta=10 sample 0 for canonical crop definition (same bbox logic for all thetas).
    ref_npz = DATA_DIR / "test" / "theta_10" / "res_512" / "samples.npz"
    ref_geom = geom_from_npz_row(np.load(ref_npz, allow_pickle=True), SAMPLE_IDX)
    bbox, kink_radius, half_extent = crop_bbox_world(ref_geom)
    area_frac = bbox_area_fraction(bbox)

    print("=== Cropped kink FMM convergence diagnostic ===\n")
    print(f"Crop center: notch tip {notch_tip_world(ref_geom)} (θ=10 reference pose)")
    print(
        f"Kink radius: {kink_radius:.6f} ({DEFAULT_KINK_RADIUS_FRAC * 100:.1f}% domain diagonal); "
        f"half-extent per side: {half_extent:.6f} ({CROP_RADIUS_MULTIPLIER}× kink radius)"
    )
    print(
        f"World bbox: x=[{bbox[0]:.6f}, {bbox[1]:.6f}], y=[{bbox[2]:.6f}, {bbox[3]:.6f}]"
    )
    print(f"BBox area fraction of full domain: {area_frac * 100:.4f}%")

    per_theta_errors: dict[int, dict[int, float]] = {}
    per_theta_labels: dict[int, str] = {}

    for theta in CONVERGENCE_THETAS:
        npz_path = DATA_DIR / "test" / f"theta_{int(theta)}" / "res_512" / "samples.npz"
        geom = geom_from_npz_row(np.load(npz_path, allow_pickle=True), SAMPLE_IDX)
        # Same relative crop size/position: center on tip (center for θ=180).
        theta_bbox, _, _ = crop_bbox_world(geom)

        occ_ref, v_ref = render_sample(geom, REF_RESOLUTION)
        row_ref, col_ref = world_to_index_ranges(theta_bbox, REF_RESOLUTION)
        _, mask_label = analysis_mask(geom, REF_RESOLUTION, row_ref, col_ref, occ_ref)

        errors: dict[int, float] = {}
        for res in RESOLUTIONS:
            if res == REF_RESOLUTION:
                errors[res] = 0.0
                continue
            occ, v = render_sample(geom, res)
            row_sl, col_sl = world_to_index_ranges(theta_bbox, res)
            v_crop = extract_crop(v, row_sl, col_sl)
            v_ref_on_res = sample_field_at_resolution(v_ref, REF_RESOLUTION, res)
            v_ref_crop_on_res = extract_crop(v_ref_on_res, row_sl, col_sl)
            mask, label = analysis_mask(geom, res, row_sl, col_sl, occ)
            errors[res] = relative_l2_masked_numpy(v_crop, v_ref_crop_on_res, mask)

        per_theta_labels[theta] = mask_label
        per_theta_errors[theta] = errors

    # Wall-clock for a single highest-resolution solve (θ=10 pose).
    geom_timing = geom_from_npz_row(np.load(ref_npz, allow_pickle=True), SAMPLE_IDX)
    t0 = time.perf_counter()
    render_sample(geom_timing, REF_RESOLUTION)
    highest_res_wallclock = time.perf_counter() - t0

    # Plot: one line per theta (exclude reference point at 0).
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for theta in CONVERGENCE_THETAS:
        errs = per_theta_errors[theta]
        rs = [r for r in RESOLUTIONS if r < REF_RESOLUTION]
        ys = [errs[r] for r in rs]
        ax.plot(rs, ys, "o-", linewidth=2, label=f"θ={theta}°")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Near-kink rel. L2 vs 8192 crop reference")
    ax.set_title(f"Cropped kink FMM convergence (test sample {SAMPLE_IDX})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    plot_path = OUT_DIR / "cropped_kink_convergence.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {plot_path}")

    # Conclusion from sharp-wedge (θ=90, 10) errors at highest compared resolution.
    sharp_trends = [describe_trend(per_theta_errors[t]) for t in (90, 10)]
    hi_res = max(r for r in RESOLUTIONS if r < REF_RESOLUTION)
    final_sharp = [per_theta_errors[t][hi_res] for t in (90, 10)]
    monotonic_down = all("decreasing" in t for t in sharp_trends)
    if monotonic_down and max(final_sharp) < 0.005:
        conclusion = "CONVERGES WITH ENOUGH LOCAL RESOLUTION"
    elif max(final_sharp) < 0.005:
        conclusion = "CONVERGES WITH ENOUGH LOCAL RESOLUTION"
    elif any("persistent" in t or "flat" in t for t in sharp_trends):
        conclusion = "PERSISTENT ERROR EVEN AT HIGHEST TRACTABLE RESOLUTION"
    elif max(final_sharp) < 0.02 and monotonic_down:
        conclusion = "CONVERGES WITH ENOUGH LOCAL RESOLUTION"
    else:
        conclusion = "PERSISTENT ERROR EVEN AT HIGHEST TRACTABLE RESOLUTION"

    def fmt_errors(theta: int) -> str:
        errs = per_theta_errors[theta]
        parts = [f"res={r}:{errs[r]:.4e}" for r in RESOLUTIONS if r < REF_RESOLUTION]
        trend = describe_trend(errs)
        note = f" ({per_theta_labels[theta]})" if theta == 180 else ""
        return f"[{', '.join(parts)}], trend={trend}{note}"

    print("\n" + "=" * 72)
    print("FINAL REPORT")
    print("=" * 72)
    print(
        f"Crop definition: [x=[{bbox[0]:.6f},{bbox[1]:.6f}], "
        f"y=[{bbox[2]:.6f},{bbox[3]:.6f}], {area_frac * 100:.4f}% of domain area]"
    )
    print(f"Method used: {method_label} full-domain solve, crop-and-analyze")
    print(f"  {method_note}")
    print(
        f"  Wall-clock at highest resolution ({REF_RESOLUTION}): "
        f"{highest_res_wallclock:.2f}s"
    )
    print(f"  Crop area fraction of full domain: {area_frac * 100:.4f}%")
    print("Convergence result per theta:")
    for theta in (180, 90, 10):
        print(f"  theta={theta}: {fmt_errors(theta)}")
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    run_convergence()
