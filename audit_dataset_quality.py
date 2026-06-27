#!/usr/bin/env python3
"""Read-only audit: FMM convergence, kink-mask consistency, train/test generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from geometry import THETA_VALUES, render_sample
from kink_utils import (
    DEFAULT_KINK_RADIUS_FRAC,
    DOMAIN_DIAGONAL,
    geom_from_npz_row,
    get_kink_mask,
    get_kink_mask_from_npz,
    relative_l2_masked_numpy,
)

ROOT = Path(__file__).resolve().parent
CONVERGENCE_RESOLUTIONS = (64, 128, 256, 512, 1024, 2048)
CONVERGENCE_THETAS = (180, 90, 10)
REGEN_SPOT_THETAS = (180, 90, 10)  # low / mid / high severity
REGEN_SAMPLE_IDX = 0


def _rel_l2_whole(pred: np.ndarray, ref: np.ndarray, free: np.ndarray) -> float:
    p = pred[free]
    r = ref[free]
    return float(np.linalg.norm(p - r) / max(np.linalg.norm(r), 1e-12))


def _interp_ref_to_grid(ref: np.ndarray, res_ref: int, res_tgt: int) -> np.ndarray:
    """Bilinear sample high-res field onto target resolution grid in [0,1]^2."""
    coords = np.linspace(0.0, 1.0, res_ref)
    interp = RegularGridInterpolator((coords, coords), ref, bounds_error=False, fill_value=0.0)
    xs = np.linspace(0.0, 1.0, res_tgt)
    yy, xx = np.meshgrid(xs, xs, indexing="ij")
    pts = np.stack([yy, xx], axis=-1)
    return interp(pts).astype(np.float32)


def fmm_convergence_audit(
    data_dir: Path,
    out_dir: Path,
    sample_idx: int = 0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    for theta in CONVERGENCE_THETAS:
        npz_path = data_dir / "test" / f"theta_{int(theta)}" / "res_512" / "samples.npz"
        data = np.load(npz_path, allow_pickle=True)
        geom = geom_from_npz_row(data, sample_idx)

        ref_res = 2048
        _, v_ref = render_sample(geom, ref_res)

        rows = []
        for res in CONVERGENCE_RESOLUTIONS:
            occ, v = render_sample(geom, res)
            free = occ > 0.5
            v_ref_on_grid = _interp_ref_to_grid(v_ref, ref_res, res)
            kink_mask = get_kink_mask(
                geom.theta_deg,
                geom.center,
                geom.rotation_deg,
                geom.scale,
                res,
                radius_frac=DEFAULT_KINK_RADIUS_FRAC,
            )
            whole_err = _rel_l2_whole(v, v_ref_on_grid, free)
            kink_err = relative_l2_masked_numpy(v, v_ref_on_grid, kink_mask & free)
            n_kink = int((kink_mask & free).sum())
            rows.append(
                {
                    "resolution": res,
                    "whole_rel_l2_vs_ref": whole_err,
                    "kink_rel_l2_vs_ref": kink_err,
                    "kink_pixels": n_kink,
                }
            )

        results[int(theta)] = rows

        fig, ax = plt.subplots(figsize=(7, 4.5))
        rs = [r["resolution"] for r in rows if r["resolution"] < ref_res]
        whole = [r["whole_rel_l2_vs_ref"] for r in rows if r["resolution"] < ref_res]
        kink = [r["kink_rel_l2_vs_ref"] for r in rows if r["resolution"] < ref_res]
        ax.plot(rs, whole, "o-", linewidth=2, label="whole domain (free)")
        ax.plot(rs, kink, "s-", linewidth=2, label="near-kink mask")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Resolution")
        ax.set_ylabel("Rel. L2 vs 2048 reference")
        ax.set_title(f"FMM convergence θ={theta}° (test sample {sample_idx})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"fmm_convergence_theta_{int(theta)}.png", dpi=150)
        plt.close(fig)

        print(f"\n--- FMM convergence θ={theta}° (sample {sample_idx}) ---")
        for r in rows:
            print(
                f"  res={r['resolution']:4d}  whole={r['whole_rel_l2_vs_ref']:.6f}  "
                f"kink={r['kink_rel_l2_vs_ref']:.6f}  kink_px={r['kink_pixels']}"
            )

    return results


def mask_pixel_scaling_audit(data_dir: Path, sample_idx: int = 0) -> list[dict]:
    rows: list[dict] = []
    for theta in THETA_VALUES:
        npz_path = data_dir / "test" / f"theta_{int(theta)}" / "res_512" / "samples.npz"
        data = np.load(npz_path, allow_pickle=True)
        for res in (64, 128, 256, 512):
            data_r = np.load(
                data_dir / "test" / f"theta_{int(theta)}" / f"res_{res}" / "samples.npz",
                allow_pickle=True,
            )
            mask = get_kink_mask_from_npz(data_r, sample_idx, res)
            occ = data_r["occupancy"][sample_idx]
            free = occ > 0.5
            n_mask = int(mask.sum())
            n_mk_free = int((mask & free).sum())
            rows.append(
                {
                    "theta": int(theta),
                    "resolution": res,
                    "mask_pixels": n_mask,
                    "mask_free_pixels": n_mk_free,
                    "expected_area_ratio_vs_64": (res / 64.0) ** 2,
                }
            )
    return rows


def tip_spot_check(data_dir: Path, n_per_theta: int = 5) -> list[dict]:
    """Compare notch_tip_world to reflex vertex (index 4) in stored polygon."""
    from geometry import SampleGeometry, notch_tip_world, transform_vertices, notch_tip_local

    rows = []
    for theta in THETA_VALUES:
        data = np.load(
            data_dir / "test" / f"theta_{int(theta)}" / "res_512" / "samples.npz",
            allow_pickle=True,
        )
        for idx in range(min(n_per_theta, len(data["occupancy"]))):
            geom = geom_from_npz_row(data, idx)
            tip_fn = notch_tip_world(geom)
            if theta >= 179.9:
                rows.append({"theta": theta, "idx": idx, "tip_dist": 0.0, "note": "flat wall"})
                continue
            tip_local = notch_tip_local(theta, geom.scale)
            tip_from_vertex = transform_vertices(
                tip_local.reshape(1, 2), geom.center, geom.rotation_deg
            )[0]
            dist = float(np.hypot(tip_fn[0] - tip_from_vertex[0], tip_fn[1] - tip_from_vertex[1]))
            rows.append({"theta": theta, "idx": idx, "tip_dist": dist, "note": "ok" if dist < 1e-10 else "MISMATCH"})
    return rows


def mask_overlay_figure(data_dir: Path, out_path: Path, sample_idx: int = 0) -> None:
    n_theta = len(THETA_VALUES)
    fig, axes = plt.subplots(2, n_theta, figsize=(2.2 * n_theta, 4.5))
    if n_theta == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j, theta in enumerate(THETA_VALUES):
        data = np.load(
            data_dir / "test" / f"theta_{int(theta)}" / "res_512" / "samples.npz",
            allow_pickle=True,
        )
        occ = data["occupancy"][sample_idx]
        tt = data["travel_time"][sample_idx]
        mask = get_kink_mask_from_npz(data, sample_idx, 512)
        geom = geom_from_npz_row(data, sample_idx)
        tip = geom.center if theta >= 179.9 else __import__("geometry").notch_tip_world(geom)

        ax0 = axes[0, j]
        ax0.imshow(occ, origin="lower", extent=[0, 1, 0, 1], cmap="gray")
        ax0.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.0, origin="lower", extent=[0, 1, 0, 1])
        ax0.plot(tip[0], tip[1], "r+", markersize=6)
        ax0.set_title(f"θ={int(theta)} occ+mask")
        ax0.set_xticks([])
        ax0.set_yticks([])

        ax1 = axes[1, j]
        im = ax1.imshow(tt, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
        ax1.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=1.0, origin="lower", extent=[0, 1, 0, 1])
        ax1.set_title("GT V")
        ax1.set_xticks([])
        ax1.set_yticks([])

    fig.suptitle(f"Kink mask overlay (test sample {sample_idx}, 512²)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def regen_spot_check(data_dir: Path) -> list[dict]:
    rows = []
    # Test buckets
    for theta in REGEN_SPOT_THETAS:
        for res in (64, 512):
            npz_path = data_dir / "test" / f"theta_{int(theta)}" / f"res_{res}" / "samples.npz"
            data = np.load(npz_path, allow_pickle=True)
            idx = REGEN_SAMPLE_IDX
            geom = geom_from_npz_row(data, idx)
            occ_new, tt_new = render_sample(geom, res)
            occ_st = data["occupancy"][idx]
            tt_st = data["travel_time"][idx]
            occ_match = np.array_equal(occ_new, occ_st)
            max_tt_diff = float(np.max(np.abs(tt_new - tt_st)))
            mean_tt_diff = float(np.mean(np.abs(tt_new - tt_st)))
            rows.append(
                {
                    "split": "test",
                    "theta": int(theta),
                    "resolution": res,
                    "idx": idx,
                    "occupancy_exact_match": occ_match,
                    "max_abs_tt_diff": max_tt_diff,
                    "mean_abs_tt_diff": mean_tt_diff,
                }
            )
    # Train sample
    train_path = data_dir / "train" / "train_64.npz"
    if train_path.exists():
        data = np.load(train_path, allow_pickle=True)
        idx = 0
        geom = geom_from_npz_row(data, idx)
        occ_new, tt_new = render_sample(geom, 64)
        rows.append(
            {
                "split": "train",
                "theta": int(data["theta_deg"][idx]),
                "resolution": 64,
                "idx": idx,
                "occupancy_exact_match": np.array_equal(occ_new, data["occupancy"][idx]),
                "max_abs_tt_diff": float(np.max(np.abs(tt_new - data["travel_time"][idx]))),
                "mean_abs_tt_diff": float(np.mean(np.abs(tt_new - data["travel_time"][idx]))),
            }
        )
    return rows


def print_call_site_audit() -> None:
    print("\n=== Mask function call sites (code inspection) ===")
    print("  get_kink_mask / get_kink_mask_from_npz / precompute_kink_masks: kink_utils.py")
    print("  train.py: precompute_kink_masks() at dataset init (NOT in generate_dataset.py)")
    print("  evaluate.py: precompute_kink_masks() + get_kink_mask_from_npz() for qualitative")
    print("  diagnose_loss_sensitivity.py: get_kink_mask_from_npz()")
    print("  generate_dataset.py: does NOT precompute or store masks")
    print("  All paths call kink_utils.get_kink_mask() (directly or via _from_npz / precompute)")

    print("\n=== FMM generation path (code inspection) ===")
    print("  generate_dataset.py train: render_sample(geom, TRAIN_RESOLUTION)")
    print("  generate_dataset.py test:  render_sample(geom, resolution)")
    print("  expand_train_set.py:       render_sample(geom, TRAIN_RESOLUTION)")
    print("  render_sample -> rasterize_polygon + solve_eikonal (geometry.py)")
    print("  solve_eikonal: skfmm.travel_time(phi, speed, dx=(1/(res-1)))")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "diagnosis")
    args = parser.parse_args()

    print_call_site_audit()

    print("\n" + "=" * 72)
    print("CHECK 1 — FMM convergence vs 2048 reference")
    print("=" * 72)
    fmm_convergence_audit(args.data_dir, args.out_dir)

    print("\n" + "=" * 72)
    print("CHECK 2 — Mask consistency")
    print("=" * 72)
    tip_rows = tip_spot_check(args.data_dir, n_per_theta=5)
    bad_tips = [r for r in tip_rows if r.get("tip_dist", 0) > 1e-8]
    print(f"Tip spot-check (5 samples × 7 θ): {len(tip_rows)} checks, {len(bad_tips)} mismatches")
    if bad_tips:
        for r in bad_tips[:5]:
            print(f"  MISMATCH θ={r['theta']} idx={r['idx']} dist={r['tip_dist']:.3e}")

    mask_overlay_figure(args.data_dir, args.out_dir / "mask_overlay_check.png")

    print("\nMask pixel counts (test sample 0, mask & free∩mask):")
    print(f"{'theta':>5} {'res':>5} {'mask_px':>8} {'free∩mask':>10} {'ratio_vs_64':>12}")
    base_counts: dict[int, int] = {}
    for row in mask_pixel_scaling_audit(args.data_dir):
        th = row["theta"]
        if row["resolution"] == 64:
            base_counts[th] = row["mask_pixels"]
        ratio = row["mask_pixels"] / max(base_counts.get(th, row["mask_pixels"]), 1)
        expected = (row["resolution"] / 64.0) ** 2
        print(
            f"{th:5d} {row['resolution']:5d} {row['mask_pixels']:8d} "
            f"{row['mask_free_pixels']:10d} {ratio:8.3f} (expect ~{expected:.3f})"
        )

    print("\n" + "=" * 72)
    print("CHECK 3 — Regenerate FMM from stored geometry vs .npz")
    print("=" * 72)
    for row in regen_spot_check(args.data_dir):
        print(
            f"  [{row['split']}] θ={row['theta']} res={row['resolution']} idx={row['idx']}: "
            f"occ_match={row['occupancy_exact_match']}  "
            f"max|ΔV|={row['max_abs_tt_diff']:.3e}  mean|ΔV|={row['mean_abs_tt_diff']:.3e}"
        )

    print(f"\nSaved FMM plots to {args.out_dir}/fmm_convergence_theta_*.png")
    print(f"Saved mask overlay to {args.out_dir}/mask_overlay_check.png")


if __name__ == "__main__":
    main()
