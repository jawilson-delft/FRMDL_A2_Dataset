#!/usr/bin/env python3
"""Pre-training read-only verification of the eikonal control dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from geometry import (
    THETA_VALUES,
    TEST_RESOLUTIONS,
    SampleGeometry,
    render_sample,
)

RESOLUTIONS = tuple(TEST_RESOLUTIONS)
CHECK1_POSE_THETAS = (180, 90, 10)
MAX_V_TOLERANCE = 0.10  # 10% vs finest resolution
MEAN_V_FLAG_THRESHOLD = 0.30  # 30% cross-theta spread


def _free_space_stats(travel_time: np.ndarray, occupancy: np.ndarray) -> dict[str, float]:
    free = travel_time[occupancy > 0.5]
    if free.size == 0:
        return {"max_v": np.nan, "p95_v": np.nan}
    return {
        "max_v": float(np.max(free)),
        "p95_v": float(np.percentile(free, 95)),
    }


def _geom_from_npz_row(data: np.lib.npyio.NpzFile, idx: int) -> SampleGeometry:
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


def _first_train_index_for_theta(train_data: np.lib.npyio.NpzFile, theta: float) -> int:
    thetas = train_data["theta_deg"]
    matches = np.where(np.isclose(thetas, theta))[0]
    if len(matches) == 0:
        raise ValueError(f"No training samples found for theta={theta}")
    return int(matches[0])


def _same_pose_exists_across_test_resolutions(
    data_dir: Path, theta: float, sample_index: int
) -> bool:
    """Return True if test sample_index shares param_hash across all resolutions."""
    hashes = []
    for res in RESOLUTIONS:
        path = data_dir / "test" / f"theta_{int(theta)}" / f"res_{res}" / "samples.npz"
        if not path.exists():
            return False
        data = np.load(path, allow_pickle=True)
        if sample_index >= len(data["param_hash"]):
            return False
        hashes.append(str(data["param_hash"][sample_index]))
    return len(set(hashes)) == 1


def check1_resolution_scaling(
    data_dir: Path, output_dir: Path
) -> dict:
    """Verify FMM travel times use physical dx, not grid-cell units."""
    train_path = data_dir / "train" / "train_64.npz"
    train_data = np.load(train_path, allow_pickle=True)

    poses: list[tuple[str, SampleGeometry]] = []
    for theta in CHECK1_POSE_THETAS:
        if _same_pose_exists_across_test_resolutions(data_dir, theta, 0):
            test_path = data_dir / "test" / f"theta_{int(theta)}" / "res_64" / "samples.npz"
            geom = _geom_from_npz_row(np.load(test_path, allow_pickle=True), 0)
            poses.append((f"test θ={int(theta)} idx=0", geom))
        else:
            idx = _first_train_index_for_theta(train_data, theta)
            geom = _geom_from_npz_row(train_data, idx)
            poses.append(
                (
                    f"train θ={int(theta)} idx={idx} (re-rendered at 4 resolutions; "
                    "test buckets are independent per resolution)",
                    geom,
                )
            )

    results_per_pose: list[dict] = []
    all_pass = True

    for pose_label, geom in poses:
        by_res: dict[int, dict[str, float]] = {}
        for res in RESOLUTIONS:
            occ, tt = render_sample(geom, res)
            by_res[res] = _free_space_stats(tt, occ)

        ref_max = by_res[512]["max_v"]
        ref_p95 = by_res[512]["p95_v"]
        ratios_max = {res: by_res[res]["max_v"] / by_res[64]["max_v"] for res in RESOLUTIONS}
        ratios_p95 = {res: by_res[res]["p95_v"] / by_res[64]["p95_v"] for res in RESOLUTIONS}

        pose_pass = True
        for res in (64, 128, 256):
            max_rel_err = abs(by_res[res]["max_v"] - ref_max) / max(ref_max, 1e-8)
            p95_rel_err = abs(by_res[res]["p95_v"] - ref_p95) / max(ref_p95, 1e-8)
            if max_rel_err > MAX_V_TOLERANCE or p95_rel_err > MAX_V_TOLERANCE:
                pose_pass = False

        # Detect cell-count scaling: V should not grow ~linearly with resolution index.
        res_scale = np.array([64, 128, 256, 512], dtype=float)
        max_vals = np.array([by_res[r]["max_v"] for r in RESOLUTIONS])
        if max_vals[0] > 1e-8:
            norm_growth = max_vals / max_vals[0]
            expected_cell_bug = res_scale / res_scale[0]
            bug_corr = np.corrcoef(norm_growth, expected_cell_bug)[0, 1]
            if bug_corr > 0.98 and ratios_max[512] > 1.5:
                pose_pass = False

        all_pass = all_pass and pose_pass
        results_per_pose.append(
            {
                "label": pose_label,
                "theta": geom.theta_deg,
                "by_resolution": by_res,
                "ratios_max_vs_64": ratios_max,
                "ratios_p95_vs_64": ratios_p95,
                "pass": pose_pass,
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for entry in results_per_pose:
        xs = list(RESOLUTIONS)
        maxs = [entry["by_resolution"][r]["max_v"] for r in RESOLUTIONS]
        p95s = [entry["by_resolution"][r]["p95_v"] for r in RESOLUTIONS]
        axes[0].plot(xs, maxs, marker="o", label=entry["label"])
        axes[1].plot(xs, p95s, marker="o", label=entry["label"])
    for ax, ylab in zip(axes, ("max(V) free space", "95th pct V free space")):
        ax.set_xlabel("Resolution")
        ax.set_ylabel(ylab)
        ax.set_xticks(list(RESOLUTIONS))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_title("Check 1: should be flat across resolution (physical dx)")
    fig.tight_layout()
    plot_path = output_dir / "check1_resolution_scaling.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "pass": all_pass,
        "poses": results_per_pose,
        "plot": str(plot_path),
        "note": (
            "Test npz buckets use independent poses per resolution; "
            "poses were taken from training metadata and re-rendered via geometry.render_sample."
        ),
    }


def check2_theta_scale_balance(data_dir: Path) -> dict:
    """Per-theta value-function scale on 64x64 training data."""
    train_path = data_dir / "train" / "train_64.npz"
    data = np.load(train_path, allow_pickle=True)

    rows = []
    for theta in THETA_VALUES:
        mask = np.isclose(data["theta_deg"], theta)
        indices = np.where(mask)[0]
        sample_maxes, sample_means, sample_stds = [], [], []
        for idx in indices:
            occ = data["occupancy"][idx]
            tt = data["travel_time"][idx]
            free = tt[occ > 0.5]
            if free.size == 0:
                continue
            sample_maxes.append(float(np.max(free)))
            sample_means.append(float(np.mean(free)))
            sample_stds.append(float(np.std(free)))

        rows.append(
            {
                "theta": int(theta),
                "n": len(sample_means),
                "mean_v": float(np.mean(sample_means)),
                "std_v": float(np.mean(sample_stds)),
                "max_v": float(np.mean(sample_maxes)),
            }
        )

    mean_vs = [r["mean_v"] for r in rows]
    global_mean = float(np.mean(mean_vs))
    flags = []
    for row in rows:
        rel_diff = abs(row["mean_v"] - global_mean) / max(global_mean, 1e-8)
        row["mean_v_rel_diff_from_global"] = rel_diff
        if rel_diff > MEAN_V_FLAG_THRESHOLD:
            flags.append(
                f"θ={row['theta']}: mean V differs by {rel_diff * 100:.1f}% from global mean"
            )

    normalization_status = (
        "No input or output normalization in train.py or model_fno.py: "
        "occupancy and travel_time are used raw; prepare_input() only appends "
        "(x,y) grid channels in [0,1]; loss is relative L2 (scale-normalized per sample)."
    )

    table_path = data_dir.parent / "results" / "verification" / "check2_theta_scale_table.txt"

    return {
        "rows": rows,
        "global_mean_v": global_mean,
        "flags": flags,
        "normalization_status": normalization_status,
        "table_path": str(table_path),
    }


def _param_distance(
    center_a: np.ndarray,
    rot_a: float,
    scale_a: float,
    center_b: np.ndarray,
    rot_b: float,
    scale_b: float,
) -> tuple[float, float, float, float]:
    """Return (pos_dist, rot_dist_deg, scale_rel_diff, combined) in interpretable units."""
    pos_dist = float(np.linalg.norm(center_a - center_b))  # unit-square coords
    rot_diff = abs(rot_a - rot_b) % 360.0
    rot_dist_deg = float(min(rot_diff, 360.0 - rot_diff))
    scale_rel_diff = float(abs(scale_a - scale_b) / max(scale_a, scale_b, 1e-8))
    combined = float(
        np.sqrt(
            (pos_dist) ** 2
            + (rot_dist_deg / 180.0) ** 2
            + scale_rel_diff**2
        )
    )
    return pos_dist, rot_dist_deg, scale_rel_diff, combined


def check3_train_test_separation(data_dir: Path) -> dict:
    """Nearest training neighbor in pose space for each test sample (same theta)."""
    train = np.load(data_dir / "train" / "train_64.npz", allow_pickle=True)

    train_by_theta: dict[int, list[dict]] = {int(t): [] for t in THETA_VALUES}
    for idx in range(len(train["sample_id"])):
        theta = int(round(float(train["theta_deg"][idx])))
        train_by_theta[theta].append(
            {
                "center": train["center"][idx],
                "rotation": float(train["rotation_deg"][idx]),
                "scale": float(train["scale"][idx]),
            }
        )

    nn_combined: list[float] = []
    nn_pos: list[float] = []
    nn_rot: list[float] = []
    nn_scale: list[float] = []
    near_duplicates: list[dict] = []

    for theta in THETA_VALUES:
        for res in RESOLUTIONS:
            path = data_dir / "test" / f"theta_{theta}" / f"res_{res}" / "samples.npz"
            test = np.load(path, allow_pickle=True)
            trainers = train_by_theta[theta]
            for idx in range(len(test["sample_id"])):
                tc = test["center"][idx]
                tr = float(test["rotation_deg"][idx])
                ts = float(test["scale"][idx])
                best = None
                for ref in trainers:
                    pd, rd, sd, combined = _param_distance(
                        tc, tr, ts, ref["center"], ref["rotation"], ref["scale"]
                    )
                    if best is None or combined < best["combined"]:
                        best = {
                            "pos_dist": pd,
                            "rot_dist_deg": rd,
                            "scale_rel_diff": sd,
                            "combined": combined,
                        }
                if best is None:
                    continue
                nn_combined.append(best["combined"])
                nn_pos.append(best["pos_dist"])
                nn_rot.append(best["rot_dist_deg"])
                nn_scale.append(best["scale_rel_diff"])

                if (
                    best["pos_dist"] < 0.01
                    and best["rot_dist_deg"] < 1.0
                    and best["scale_rel_diff"] < 0.01
                ):
                    near_duplicates.append(
                        {
                            "theta": theta,
                            "resolution": res,
                            "test_index": idx,
                            "nearest_pos_dist": best["pos_dist"],
                            "nearest_rot_dist_deg": best["rot_dist_deg"],
                            "nearest_scale_rel_diff": best["scale_rel_diff"],
                        }
                    )

    def _stats(vals: list[float]) -> dict[str, float]:
        arr = np.array(vals, dtype=float)
        return {"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max())}

    return {
        "n_test_samples": len(nn_combined),
        "combined_distance": _stats(nn_combined),
        "position_distance": _stats(nn_pos),
        "rotation_distance_deg": _stats(nn_rot),
        "scale_relative_diff": _stats(nn_scale),
        "near_duplicate_flags": near_duplicates,
        "n_near_duplicates": len(near_duplicates),
    }


def _print_check1(result: dict) -> None:
    print("\n=== CHECK 1 — Physical distance consistency across resolutions ===")
    print(result["note"])
    for entry in result["poses"]:
        print(f"\n  Pose: {entry['label']}")
        print(f"  {'res':>6}  {'max(V)':>10}  {'p95(V)':>10}")
        for res in RESOLUTIONS:
            stats = entry["by_resolution"][res]
            print(f"  {res:6d}  {stats['max_v']:10.5f}  {stats['p95_v']:10.5f}")
        rm = entry["ratios_max_vs_64"]
        rp = entry["ratios_p95_vs_64"]
        print(
            f"  max ratios vs 64: "
            f"128/64={rm[128]:.4f}, 256/64={rm[256]:.4f}, 512/64={rm[512]:.4f}"
        )
        print(
            f"  p95 ratios vs 64: "
            f"128/64={rp[128]:.4f}, 256/64={rp[256]:.4f}, 512/64={rp[512]:.4f}"
        )
        print(f"  pose result: {'PASS' if entry['pass'] else 'FAIL'}")


def _print_check2(result: dict, output_dir: Path) -> None:
    print("\n=== CHECK 2 — Per-theta value function scale (train 64×64) ===")
    print(f"  {'theta':>5}  {'mean V':>10}  {'std V':>10}  {'max V':>10}  {'Δ vs global':>12}")
    lines = [
        f"{'theta':>5}  {'mean V':>10}  {'std V':>10}  {'max V':>10}  {'rel_diff':>10}"
    ]
    for row in result["rows"]:
        rel = row["mean_v_rel_diff_from_global"]
        line = (
            f"{row['theta']:5d}  {row['mean_v']:10.5f}  {row['std_v']:10.5f}  "
            f"{row['max_v']:10.5f}  {rel * 100:10.1f}%"
        )
        print(f"  {line}")
        lines.append(line)
    print(f"\n  Global mean V: {result['global_mean_v']:.5f}")
    if result["flags"]:
        print("  FLAGS:")
        for flag in result["flags"]:
            print(f"    - {flag}")
    else:
        print("  FLAGS: none (no theta mean V differs >30% from global)")
    print(f"\n  Normalization: {result['normalization_status']}")
    table_path = output_dir / "check2_theta_scale_table.txt"
    table_path.write_text("\n".join(lines) + "\n")


def _print_check3(result: dict) -> None:
    print("\n=== CHECK 3 — Geometry uniqueness / train-test separation ===")
    print(f"  Test samples evaluated: {result['n_test_samples']}")
    for name, key in [
        ("Combined NN distance", "combined_distance"),
        ("Position NN dist (unit square)", "position_distance"),
        ("Rotation NN dist (degrees)", "rotation_distance_deg"),
        ("Scale NN rel diff", "scale_relative_diff"),
    ]:
        stats = result[key]
        print(
            f"  {name}: min={stats['min']:.5f}, "
            f"median={stats['median']:.5f}, max={stats['max']:.5f}"
        )
    print(f"  Near-duplicate flags (pos<1%, rot<1°, scale<1%): {result['n_near_duplicates']}")
    if result["near_duplicate_flags"]:
        for dup in result["near_duplicate_flags"][:10]:
            print(f"    - {dup}")
        if result["n_near_duplicates"] > 10:
            print(f"    ... and {result['n_near_duplicates'] - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "verification",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Eikonal control dataset — pre-training verification")
    print(f"Data directory: {args.data_dir}")

    c1 = check1_resolution_scaling(args.data_dir, args.output_dir)
    c2 = check2_theta_scale_balance(args.data_dir)
    c3 = check3_train_test_separation(args.data_dir)

    _print_check1(c1)
    _print_check2(c2, args.output_dir)
    _print_check3(c3)

    ratio_summary = []
    for entry in c1["poses"]:
        rm = entry["ratios_max_vs_64"]
        ratio_summary.append(
            f"θ={int(entry['theta'])} maxV 128/64={rm[128]:.3f} "
            f"256/64={rm[256]:.3f} 512/64={rm[512]:.3f}"
        )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(
        f"CHECK 1 (resolution scaling):  {'PASS' if c1['pass'] else 'FAIL'}  "
        f"[ratios: {'; '.join(ratio_summary)}]"
    )
    flag_str = (
        f"{len(c2['flags'])} flag(s): " + "; ".join(c2["flags"])
        if c2["flags"]
        else "no theta mean-V flags (>30%)"
    )
    print(f"CHECK 2 (theta scale balance): INFO        [{flag_str}; see table above]")
    print(
        "CHECK 3 (train/test separation): INFO       "
        f"[combined NN dist min/median/max = "
        f"{c3['combined_distance']['min']:.4f} / "
        f"{c3['combined_distance']['median']:.4f} / "
        f"{c3['combined_distance']['max']:.4f}; "
        f"near-duplicates = {c3['n_near_duplicates']}]"
    )
    print(f"\nPlot saved: {c1['plot']}")


if __name__ == "__main__":
    main()
