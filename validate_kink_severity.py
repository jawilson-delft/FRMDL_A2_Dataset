#!/usr/bin/env python3
"""Validate that θ controls gradient-kink severity in FMM ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from geometry import (
    THETA_VALUES,
    SampleGeometry,
    make_sample_geometry,
    notch_tip_world,
    render_sample,
)


def compute_gradient(field: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient; shape (2, H, W) for (gx, gy)."""
    gy, gx = np.gradient(field, dx, dx)
    return gx, gy


def _bilinear_sample(
    field: np.ndarray, x: float, y: float, resolution: int
) -> float:
    """Sample scalar field at continuous coords (x, y) in [0, 1]."""
    fx = x * (resolution - 1)
    fy = y * (resolution - 1)
    x0 = int(np.floor(fx))
    y0 = int(np.floor(fy))
    x1 = min(x0 + 1, resolution - 1)
    y1 = min(y0 + 1, resolution - 1)
    wx = fx - x0
    wy = fy - y0
    return float(
        (1 - wx) * (1 - wy) * field[y0, x0]
        + wx * (1 - wy) * field[y0, x1]
        + (1 - wx) * wy * field[y1, x0]
        + wx * wy * field[y1, x1]
    )


def _tip_face_directions(geom: SampleGeometry) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return tip position and unit directions into free space along each notch face."""
    vertices = np.array(geom.vertices, dtype=np.float64)
    if len(vertices) < 7:
        return None
    tip_idx = 4
    tip_pt = vertices[tip_idx]
    v_right = vertices[tip_idx - 1]
    v_left = vertices[tip_idx + 1]

    def unit(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-12 else vec

    # From tip toward opening along each V face.
    dir_right = unit(v_right - tip_pt)
    dir_left = unit(v_left - tip_pt)
    return tip_pt, dir_right, dir_left


def measure_kink_severity(
    travel_time: np.ndarray,
    occupancy: np.ndarray,
    geom: SampleGeometry,
    resolution: int,
) -> dict[str, float]:
    """Estimate kink severity by sampling gradients along both notch faces at the reflex tip."""
    dx = 1.0 / max(resolution - 1, 1)
    grad_x, grad_y = compute_gradient(travel_time, dx)

    face_dirs = _tip_face_directions(geom)
    if face_dirs is None:
        return {
            "gradient_jump": 0.0,
            "grad_peak": 1.0,
            "grad_direction_spread": 0.0,
            "kink_severity_score": 0.0,
            "kink_severity": 0.0,
        }

    tip_pt, dir_right, dir_left = face_dirs
    grads = []
    for direction in (dir_right, dir_left):
        for dist in (0.006, 0.010, 0.015, 0.022, 0.030, 0.040, 0.055):
            wx = float(tip_pt[0] + direction[0] * dist)
            wy = float(tip_pt[1] + direction[1] * dist)
            if not (0.0 < wx < 1.0 and 0.0 < wy < 1.0):
                continue
            col = int(np.clip(np.round(wx * (resolution - 1)), 0, resolution - 1))
            row = int(np.clip(np.round(wy * (resolution - 1)), 0, resolution - 1))
            if occupancy[row, col] < 0.5:
                continue
            gx = _bilinear_sample(grad_x, wx, wy, resolution)
            gy = _bilinear_sample(grad_y, wx, wy, resolution)
            grads.append(np.array([gx, gy]))
            break

    if len(grads) < 2:
        return {
            "gradient_jump": 0.0,
            "grad_peak": 0.0,
            "grad_direction_spread": 0.0,
            "kink_severity_score": 0.0,
            "kink_severity": 180.0 - geom.theta_deg,
        }

    grad_mags = [float(np.linalg.norm(g)) for g in grads]
    grad_peak = float(max(grad_mags))
    angles = [float(np.arctan2(g[1], g[0])) for g in grads]
    cx = float(np.mean(np.cos(angles)))
    sx = float(np.mean(np.sin(angles)))
    direction_spread = float(1.0 - np.hypot(cx, sx))
    gradient_jump = float(np.linalg.norm(grads[0] - grads[1]))
    severity_score = max(0.0, grad_peak - 1.0) + 3.0 * direction_spread + gradient_jump

    return {
        "gradient_jump": gradient_jump,
        "grad_peak": grad_peak,
        "grad_direction_spread": direction_spread,
        "kink_severity_score": severity_score,
        "kink_severity": 180.0 - geom.theta_deg,
    }


def plot_example_panel(
    ax_row: list[plt.Axes],
    travel_time: np.ndarray,
    geom: SampleGeometry,
    resolution: int,
    title: str,
) -> None:
    dx = 1.0 / max(resolution - 1, 1)
    grad_x, grad_y = compute_gradient(travel_time, dx)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    im0 = ax_row[0].imshow(travel_time, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    tip = notch_tip_world(geom)
    ax_row[0].plot(tip[0], tip[1], "r*", markersize=10)
    ax_row[0].plot(geom.goal[0], geom.goal[1], "c^", markersize=8)
    ax_row[0].set_title(f"{title}\nV(x) ground truth")
    plt.colorbar(im0, ax=ax_row[0], fraction=0.046)

    im1 = ax_row[1].imshow(grad_mag, origin="lower", extent=[0, 1, 0, 1], cmap="magma")
    ax_row[1].plot(tip[0], tip[1], "r*", markersize=10)
    ax_row[1].set_title("|∇V|")
    plt.colorbar(im1, ax=ax_row[1], fraction=0.046)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples-per-theta", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "validation",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    used_param_keys: set = set()
    used_full_keys: set = set()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    fig, axes = plt.subplots(
        len(THETA_VALUES),
        2,
        figsize=(10, 2.5 * len(THETA_VALUES)),
        squeeze=False,
    )

    sample_id = 0
    for i, theta in enumerate(THETA_VALUES):
        jumps = []
        for _ in tqdm(range(args.samples_per_theta), desc=f"validate θ={theta}"):
            geom = make_sample_geometry(
                rng, theta, sample_id, used_param_keys, used_full_keys
            )
            used_param_keys.add(geom.param_key())
            used_full_keys.add(geom.full_key())
            occ, travel_time = render_sample(geom, args.resolution)
            metrics = measure_kink_severity(travel_time, occ, geom, args.resolution)
            metrics["theta_deg"] = theta
            records.append(metrics)
            jumps.append(metrics["gradient_jump"])
            sample_id += 1

        # Plot the last sample for this theta.
        _, travel_time = render_sample(geom, args.resolution)
        plot_example_panel(
            [axes[i, 0], axes[i, 1]],
            travel_time,
            geom,
            args.resolution,
            f"θ={theta}° (avg jump={np.mean(jumps):.3f})",
        )

    fig.tight_layout()
    examples_path = args.output_dir / "kink_examples.png"
    fig.savefig(examples_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Aggregate gradient jump vs θ.
    by_theta = {theta: [] for theta in THETA_VALUES}
    for rec in records:
        by_theta[rec["theta_deg"]].append(rec["gradient_jump"])

    mean_jump = [float(np.mean(by_theta[t])) for t in THETA_VALUES]
    std_jump = [float(np.std(by_theta[t])) for t in THETA_VALUES]
    severity = [180.0 - t for t in THETA_VALUES]

    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].errorbar(severity, mean_jump, yerr=std_jump, marker="o", capsize=4, linewidth=2)
    axes[0].set_xlabel("Kink severity (180° − θ)")
    axes[0].set_ylabel("Gradient jump across notch faces")
    axes[0].set_title("Gradient discontinuity at reflex corner")
    axes[0].grid(True, alpha=0.3)

    notched = [t for t in THETA_VALUES if t < 180]
    notched_severity = [180.0 - t for t in notched]
    notched_jump = [float(np.mean(by_theta[t])) for t in notched]
    axes[1].bar([str(s) for s in notched_severity], notched_jump, color="steelblue")
    axes[1].set_xlabel("Kink severity (180° − θ)")
    axes[1].set_ylabel("Mean gradient jump")
    axes[1].set_title("Notched obstacles (θ < 180°): kink present")
    axes[1].grid(True, axis="y", alpha=0.3)

    baseline_jump = mean_jump[0]
    notched_mean = float(np.mean(notched_jump))
    monotonic = all(
        mean_jump[i] <= mean_jump[i + 1] + 1e-6 for i in range(len(mean_jump) - 1)
    )
    spearman = float(np.corrcoef(severity, mean_jump)[0, 1]) if len(severity) > 2 else 0.0
    fig2.suptitle(
        f"θ=180° baseline jump≈{baseline_jump:.3f}; notched mean≈{notched_mean:.1f}",
        fontsize=11,
    )
    fig2.tight_layout()
    severity_path = args.output_dir / "kink_severity_vs_theta.png"
    fig2.savefig(severity_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "theta_values": list(THETA_VALUES),
        "mean_gradient_jump": dict(zip(THETA_VALUES, mean_jump)),
        "std_gradient_jump": dict(zip(THETA_VALUES, std_jump)),
        "theta_180_baseline_jump": baseline_jump,
        "notched_mean_gradient_jump": notched_mean,
        "pearson_corr_severity_jump": spearman,
        "monotonic_in_severity": monotonic,
        "records": records,
    }
    summary_path = args.output_dir / "kink_validation.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved examples to {examples_path}")
    print(f"Saved severity plot to {severity_path}")
    print(f"Saved metrics to {summary_path}")
    print(f"θ=180° baseline jump: {baseline_jump:.4f}; notched mean: {notched_mean:.2f}")
    print(f"Monotonic kink severity vs θ: {monotonic}")


if __name__ == "__main__":
    main()
