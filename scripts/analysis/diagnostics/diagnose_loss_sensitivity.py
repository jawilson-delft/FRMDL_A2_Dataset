#!/usr/bin/env python3
"""Diagnose whether relative L2 loss is sensitive to the kink (notch-tip) region."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from geometry import THETA_VALUES, TRAIN_RESOLUTION
from kink_utils import (
    DEFAULT_KINK_RADIUS_FRAC,
    DOMAIN_DIAGONAL,
    get_kink_mask_from_npz,
    relative_l2_masked_numpy,
)
from model_fno import FNO2d, prepare_input

RADIUS_FRACTIONS = (0.02, 0.05, 0.10)
PRIMARY_RADIUS_FRAC = DEFAULT_KINK_RADIUS_FRAC
TRAIN_SAMPLES_PER_THETA = 20
VAL_SAMPLES_PER_THETA = 10


def _sample_indices_for_theta(
    all_indices: np.ndarray,
    theta_deg: np.ndarray,
    theta: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pool = all_indices[np.isclose(theta_deg[all_indices], theta)]
    if len(pool) == 0:
        return np.array([], dtype=np.int64)
    if len(pool) <= n:
        return pool
    choice = rng.choice(pool, size=n, replace=False)
    return np.sort(choice)


def step1_pixel_fractions(
    data_dir: Path,
    rng: np.random.Generator,
) -> tuple[list[dict], None]:
    """Quantify near-kink pixel fraction on training samples."""
    train_path = data_dir / "train" / "train_64.npz"
    data = np.load(train_path, allow_pickle=True)
    all_idx = np.arange(len(data["sample_id"]))

    print("\n=== STEP 1 — Near-kink pixel fraction (training sample) ===\n")
    print(f"Domain diagonal = {DOMAIN_DIAGONAL:.4f}")
    print(f"Samples per θ: {TRAIN_SAMPLES_PER_THETA}\n")
    print(f"{'theta':>6}  {'radius%':>8}  {'frac_free_px':>14}  {'frac_all_px':>12}")
    print("-" * 46)

    rows: list[dict] = []

    for theta in THETA_VALUES:
        indices = _sample_indices_for_theta(
            all_idx, data["theta_deg"], theta, TRAIN_SAMPLES_PER_THETA, rng
        )
        for radius_frac in RADIUS_FRACTIONS:
            fracs_free: list[float] = []
            fracs_all: list[float] = []
            for idx in indices:
                near = get_kink_mask_from_npz(
                    data, int(idx), TRAIN_RESOLUTION, radius_frac=radius_frac
                )
                free = data["occupancy"][idx] > 0.5
                n_free = int(free.sum())
                if n_free == 0:
                    continue
                fracs_free.append(float((near & free).sum()) / n_free)
                fracs_all.append(float(near.sum()) / near.size)

            mean_free = float(np.mean(fracs_free)) if fracs_free else float("nan")
            mean_all = float(np.mean(fracs_all)) if fracs_all else float("nan")
            print(f"{int(theta):6d}  {radius_frac * 100:7.1f}%  {mean_free:14.5f}  {mean_all:12.5f}")
            rows.append(
                {
                    "theta": int(theta),
                    "radius_frac": radius_frac,
                    "radius_abs": radius_frac * DOMAIN_DIAGONAL,
                    "mean_fraction_of_free_pixels": mean_free,
                    "mean_fraction_of_all_pixels": mean_all,
                    "n_samples": len(fracs_free),
                }
            )

    return rows, None


@torch.no_grad()
def step2_loss_decomposition(
    data_dir: Path,
    checkpoint_dir: Path,
    device: torch.device,
    rng: np.random.Generator,
) -> list[dict]:
    """Decompose validation relative L2 into near-kink vs far-from-kink."""
    split = np.load(data_dir / "train" / "val_split.npz")
    val_idx = split["val_indices"]
    data = np.load(data_dir / "train" / "train_64.npz", allow_pickle=True)

    ckpt_path = checkpoint_dir / "fno_best_val.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = FNO2d(
        in_channels=3,
        out_channels=1,
        width=cfg.get("width", 32),
        modes=cfg.get("modes", 12),
        n_layers=cfg.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"\n=== STEP 2 — Loss decomposition (val, radius={PRIMARY_RADIUS_FRAC * 100:.0f}% "
        f"of diagonal, checkpoint={ckpt_path.name}, epoch={ckpt.get('best_epoch', '?')}) ===\n"
    )
    print(
        f"{'theta':>6}  {'whole':>10}  {'near_kink':>10}  {'far_kink':>10}  "
        f"{'frac_free':>10}"
    )
    print("-" * 52)

    per_theta_rows: list[dict] = []
    radius_frac = PRIMARY_RADIUS_FRAC

    for theta in THETA_VALUES:
        indices = _sample_indices_for_theta(
            val_idx, data["theta_deg"], theta, VAL_SAMPLES_PER_THETA, rng
        )
        whole_list, near_list, far_list, frac_list = [], [], [], []

        for idx in indices:
            near_mask = get_kink_mask_from_npz(
                data, int(idx), TRAIN_RESOLUTION, radius_frac=radius_frac
            )
            far_mask = ~near_mask

            occ = (
                torch.from_numpy(data["occupancy"][idx])
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )
            target = data["travel_time"][idx]
            pred = model(prepare_input(occ)).squeeze().cpu().numpy()

            whole_list.append(
                relative_l2_masked_numpy(pred, target, np.ones_like(target, dtype=bool))
            )
            near_list.append(relative_l2_masked_numpy(pred, target, near_mask))
            far_list.append(relative_l2_masked_numpy(pred, target, far_mask))

            free = data["occupancy"][idx] > 0.5
            n_free = int(free.sum())
            if n_free > 0:
                frac_list.append(float((near_mask & free).sum()) / n_free)

        row = {
            "theta": int(theta),
            "pixel_fraction_near_kink_5pct": float(np.mean(frac_list)) if frac_list else float("nan"),
            "whole_domain_relL2": float(np.mean(whole_list)),
            "near_kink_relL2": float(np.mean(near_list)),
            "far_from_kink_relL2": float(np.mean(far_list)),
            "n_val_samples": len(indices),
        }
        per_theta_rows.append(row)
        print(
            f"{row['theta']:6d}  {row['whole_domain_relL2']:10.4f}  "
            f"{row['near_kink_relL2']:10.4f}  {row['far_from_kink_relL2']:10.4f}  "
            f"{row['pixel_fraction_near_kink_5pct']:10.5f}"
        )

    return per_theta_rows


def step3_interpretation_data(
    step1_rows: list[dict],
    step2_rows: list[dict],
) -> None:
    """Report side-by-side numbers for θ=180 (control) vs θ=10 (sharpest)."""
    print("\n=== STEP 3 — θ=180 (no kink) vs θ=10 (sharpest kink) ===\n")

    def _step1_at(theta: int, radius: float) -> float:
        for r in step1_rows:
            if r["theta"] == theta and np.isclose(r["radius_frac"], radius):
                return r["mean_fraction_of_free_pixels"]
        return float("nan")

    def _step2_row(theta: int) -> dict:
        for r in step2_rows:
            if r["theta"] == theta:
                return r
        return {}

    for label, theta in [("θ=180 (flat wall, control)", 180), ("θ=10 (sharpest notch)", 10)]:
        s2 = _step2_row(theta)
        print(f"{label}:")
        print(f"  Near-kink free-pixel fraction (5% radius, train avg): {_step1_at(theta, PRIMARY_RADIUS_FRAC):.5f}")
        print(f"  whole_domain_relL2:    {s2.get('whole_domain_relL2', float('nan')):.4f}")
        print(f"  near_kink_relL2:       {s2.get('near_kink_relL2', float('nan')):.4f}")
        print(f"  far_from_kink_relL2:   {s2.get('far_from_kink_relL2', float('nan')):.4f}")
        near = s2.get("near_kink_relL2", float("nan"))
        far = s2.get("far_from_kink_relL2", float("nan"))
        if not np.isnan(near) and not np.isnan(far) and far > 0:
            print(f"  near/far error ratio:  {near / far:.3f}")
        print()

    s180 = _step2_row(180)
    s10 = _step2_row(10)
    frac180 = _step1_at(180, PRIMARY_RADIUS_FRAC)
    frac10 = _step1_at(10, PRIMARY_RADIUS_FRAC)
    whole_gap = s10.get("whole_domain_relL2", 0) - s180.get("whole_domain_relL2", 0)
    near_gap = s10.get("near_kink_relL2", 0) - s180.get("near_kink_relL2", 0)
    far_gap = s10.get("far_from_kink_relL2", 0) - s180.get("far_from_kink_relL2", 0)

    print("Side-by-side deltas (θ=10 minus θ=180):")
    print(f"  Δ whole_domain_relL2:  {whole_gap:+.4f}")
    print(f"  Δ near_kink_relL2:     {near_gap:+.4f}")
    print(f"  Δ far_from_kink_relL2: {far_gap:+.4f}")
    print(f"  Near-kink pixel fraction: θ=180 {frac180:.5f}, θ=10 {frac10:.5f}")
    if frac10 > 0 and not np.isnan(near_gap):
        diluted = near_gap * frac10
        print(
            f"  If near-kink error excess ({near_gap:+.4f}) were averaged into whole-domain "
            f"loss weighted by pixel fraction ({frac10:.5f}), incremental contribution ≈ {diluted:+.4f}"
        )


def _save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "theta",
        "pixel_fraction_near_kink_5pct",
        "whole_domain_relL2",
        "near_kink_relL2",
        "far_from_kink_relL2",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def _save_bar_plot(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    thetas = [r["theta"] for r in rows]
    x = np.arange(len(thetas))
    width = 0.25
    whole = [r["whole_domain_relL2"] for r in rows]
    near = [r["near_kink_relL2"] for r in rows]
    far = [r["far_from_kink_relL2"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, whole, width, label="whole domain")
    ax.bar(x, near, width, label="near kink (5% diag)")
    ax.bar(x + width, far, width, label="far from kink")
    ax.set_xticks(x, [str(t) for t in thetas])
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Validation error decomposition by region (fno_best_val.pt)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.results_dir / "diagnosis"
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    step1_rows, _ = step1_pixel_fractions(args.data_dir, rng)
    step2_rows = step2_loss_decomposition(
        args.data_dir, args.checkpoint_dir, device, rng
    )
    step3_interpretation_data(step1_rows, step2_rows)

    csv_path = out_dir / "loss_sensitivity_breakdown.csv"
    plot_path = out_dir / "loss_sensitivity_breakdown.png"
    _save_csv(step2_rows, csv_path)
    _save_bar_plot(step2_rows, plot_path)

    print(f"\nSaved {csv_path}")
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
