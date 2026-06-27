#!/usr/bin/env python3
"""Read-only diagnostic: spatial structure away from kink/goal, correlation, contours."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr

from geometry import THETA_VALUES
from kink_utils import (
    DEFAULT_KINK_RADIUS_FRAC,
    DOMAIN_DIAGONAL,
    get_coord_grids,
    get_kink_mask_from_npz,
)
from model_fno import FNO2d, prepare_input


def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    if config.get("model_type") == "cnn":
        from model_cnn import CNN2d

        model = CNN2d(
            in_channels=3,
            out_channels=1,
            width=config.get("width", 32),
        ).to(device)
    else:
        model = FNO2d(
            in_channels=3,
            out_channels=1,
            width=config.get("width", 32),
            modes=config.get("modes", 12),
            n_layers=config.get("n_layers", 4),
        ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

ROOT = Path(__file__).resolve().parent
RESOLUTION = 512
GOAL_EXCLUDE_FRAC = 0.05  # task 1: mirror kink band scale
GOAL_NEAR_FRAC = 0.20  # task 2: near-goal correlation zone
SAMPLE_IDX = 0
DIAG_THETAS = tuple(int(t) for t in THETA_VALUES)
CONTOUR_THETA = 90
N_CONTOUR_LEVELS = 9


def _goal_mask(goal: tuple[float, float], resolution: int, radius_frac: float) -> np.ndarray:
    xx, yy = get_coord_grids(resolution)
    radius = radius_frac * DOMAIN_DIAGONAL
    return np.hypot(xx - goal[0], yy - goal[1]) <= radius


def _field_stats(values: np.ndarray) -> tuple[float, float]:
    return float(np.std(values)), float(values.max() - values.min())


def _pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    result = pearsonr(a, b)
    return float(result[0] if isinstance(result, tuple) else result.statistic)


@torch.no_grad()
def predict_field(model, occ_map: np.ndarray, device: torch.device) -> np.ndarray:
    occ = torch.from_numpy(occ_map).float().unsqueeze(0).unsqueeze(0).to(device)
    return model(prepare_input(occ)).squeeze().detach().cpu().numpy()


def analyze_sample(
    model,
    data: np.lib.npyio.NpzFile,
    idx: int,
    device: torch.device,
) -> dict:
    occ_map = data["occupancy"][idx]
    gt = data["travel_time"][idx]
    goal = (float(data["goal"][idx, 0]), float(data["goal"][idx, 1]))
    theta = float(data["theta_deg"][idx])

    pred = predict_field(model, occ_map, device)
    free = occ_map > 0.5
    kink = get_kink_mask_from_npz(data, idx, RESOLUTION, radius_frac=DEFAULT_KINK_RADIUS_FRAC)
    goal_excl = _goal_mask(goal, RESOLUTION, GOAL_EXCLUDE_FRAC)
    goal_near = _goal_mask(goal, RESOLUTION, GOAL_NEAR_FRAC)

    away = free & ~kink & ~goal_excl
    near_kink = free & kink
    whole_free = free

    gt_away = gt[away]
    pred_away = pred[away]
    gt_std_away, gt_range_away = _field_stats(gt_away)
    pred_std_away, pred_range_away = _field_stats(pred_away)
    std_ratio_away = pred_std_away / max(gt_std_away, 1e-12)
    range_ratio_away = pred_range_away / max(gt_range_away, 1e-12)

    gt_whole = gt[whole_free]
    pred_whole = pred[whole_free]
    gt_kink = gt[near_kink] if near_kink.any() else np.array([])
    pred_kink = pred[near_kink] if near_kink.any() else np.array([])

    whole_std_ratio = float(np.std(pred_whole) / max(np.std(gt_whole), 1e-12))
    kink_std_ratio = (
        float(np.std(pred_kink) / max(np.std(gt_kink), 1e-12)) if gt_kink.size else float("nan")
    )

    near_goal = free & goal_near
    far_from_goal = free & ~goal_near

    return {
        "theta_deg": theta,
        "sample_idx": idx,
        "sample_id": int(data["sample_id"][idx]),
        "goal": goal,
        "n_free": int(free.sum()),
        "n_away": int(away.sum()),
        "n_near_goal": int(near_goal.sum()),
        "n_far_from_goal": int(far_from_goal.sum()),
        "gt_std_away": gt_std_away,
        "gt_range_away": gt_range_away,
        "pred_std_away": pred_std_away,
        "pred_range_away": pred_range_away,
        "std_ratio_away_pct": std_ratio_away * 100.0,
        "range_ratio_away_pct": range_ratio_away * 100.0,
        "whole_std_ratio_pct": whole_std_ratio * 100.0,
        "kink_std_ratio_pct": kink_std_ratio * 100.0,
        "r_free_all": _pearson_safe(pred[free], gt[free]),
        "r_near_goal": _pearson_safe(pred[near_goal], gt[near_goal]),
        "r_far_from_goal": _pearson_safe(pred[far_from_goal], gt[far_from_goal]),
        "occ_map": occ_map,
        "gt": gt,
        "pred": pred,
        "free": free,
    }


def save_contour_plot(row: dict, out_path: Path) -> None:
    gt = row["gt"]
    pred = row["pred"]
    free = row["free"]
    gt_free = gt[free]
    pred_free = pred[free]

    gt_lo, gt_hi = float(gt_free.min()), float(gt_free.max())
    pred_lo, pred_hi = float(pred_free.min()), float(pred_free.max())
    gt_levels = np.linspace(gt_lo, gt_hi, N_CONTOUR_LEVELS)
    pred_levels = np.linspace(pred_lo, pred_hi, N_CONTOUR_LEVELS)

    fig, ax = plt.subplots(figsize=(7, 6))
    extent = [0, 1, 0, 1]
    ax.imshow(row["occ_map"], origin="lower", extent=extent, cmap="gray", alpha=0.35)
    cs_gt = ax.contour(
        gt,
        levels=gt_levels,
        colors="C0",
        linewidths=1.2,
        origin="lower",
        extent=extent,
    )
    cs_pred = ax.contour(
        pred,
        levels=pred_levels,
        colors="C3",
        linewidths=1.0,
        linestyles="--",
        origin="lower",
        extent=extent,
    )
    ax.plot(row["goal"][0], row["goal"][1], "g*", markersize=12, label="goal")
    ax.set_title(
        f"GT (blue, {N_CONTOUR_LEVELS} levels) vs FNO pred (red dashed, {N_CONTOUR_LEVELS} levels)\n"
        f"θ={int(row['theta_deg'])}°, sample {row['sample_idx']} (id={row['sample_id']})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    handles = [
        plt.Line2D([0], [0], color="C0", linewidth=1.5, label="GT V contours"),
        plt.Line2D([0], [0], color="C3", linewidth=1.5, linestyle="--", label="Pred V contours"),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _aggregate(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


def _pattern_label(r_near: float, r_far: float) -> str:
    if np.isnan(r_near) or np.isnan(r_far):
        return "INSUFFICIENT DATA"
    if r_near >= 0.5 and r_far < 0.3:
        return "MODEL TRACKS GOAL-DISTANCE MOSTLY"
    if r_near >= 0.5 and r_far >= 0.5:
        return "MODEL TRACKS FULL FIELD STRUCTURE"
    if r_near > r_far + 0.15:
        return "MODEL TRACKS GOAL-DISTANCE MOSTLY"
    if abs(r_near - r_far) <= 0.15:
        return "MODEL TRACKS FULL FIELD STRUCTURE"
    return "MIXED / WEAK CORRELATION"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "fno_best_val_kink.pt",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "diagnosis",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_checkpoint_model(args.checkpoint, device)

    rows: list[dict] = []
    for theta in DIAG_THETAS:
        npz_path = args.data_dir / "test" / f"theta_{theta}" / f"res_{RESOLUTION}" / "samples.npz"
        if not npz_path.exists():
            print(f"WARNING: missing {npz_path}")
            continue
        data = np.load(npz_path, allow_pickle=True)
        row = analyze_sample(model, data, SAMPLE_IDX, device)
        rows.append(row)
        print(
            f"θ={theta:3d} idx={SAMPLE_IDX}: away n={row['n_away']}  "
            f"std_ratio={row['std_ratio_away_pct']:.1f}%  "
            f"range_ratio={row['range_ratio_away_pct']:.1f}%  "
            f"r_near={row['r_near_goal']:.3f}  r_far={row['r_far_from_goal']:.3f}"
        )

    contour_row = next((r for r in rows if int(r["theta_deg"]) == CONTOUR_THETA), rows[0])
    contour_path = args.output_dir / "contour_comparison_theta90.png"
    save_contour_plot(contour_row, contour_path)

    mean_std_away, min_std_away, max_std_away = _aggregate(rows, "std_ratio_away_pct")
    mean_rng_away, min_rng_away, max_rng_away = _aggregate(rows, "range_ratio_away_pct")
    mean_gt_std_away = float(np.mean([r["gt_std_away"] for r in rows]))
    mean_gt_rng_away = float(np.mean([r["gt_range_away"] for r in rows]))
    mean_pred_std_away = float(np.mean([r["pred_std_away"] for r in rows]))
    mean_pred_rng_away = float(np.mean([r["pred_range_away"] for r in rows]))

    mean_whole = float(np.mean([r["whole_std_ratio_pct"] for r in rows]))
    kink_vals = [r["kink_std_ratio_pct"] for r in rows if not np.isnan(r["kink_std_ratio_pct"])]
    mean_kink = float(np.mean(kink_vals)) if kink_vals else float("nan")
    min_kink = float(np.min(kink_vals)) if kink_vals else float("nan")
    max_kink = float(np.max(kink_vals)) if kink_vals else float("nan")

    mean_r_near = float(np.mean([r["r_near_goal"] for r in rows]))
    mean_r_far = float(np.mean([r["r_far_from_goal"] for r in rows]))
    mean_r_all = float(np.mean([r["r_free_all"] for r in rows]))
    pattern = _pattern_label(mean_r_near, mean_r_far)

    gt_levels_span = (
        f"GT free V [{contour_row['gt'][contour_row['free']].min():.3f}, "
        f"{contour_row['gt'][contour_row['free']].max():.3f}]"
    )
    pred_levels_span = (
        f"pred free V [{contour_row['pred'][contour_row['free']].min():.3f}, "
        f"{contour_row['pred'][contour_row['free']].max():.3f}]"
    )

    report = f"""AWAY-FROM-EVERYTHING REGION (excl. kink mask AND goal-proximity 5% diag):
  Samples: thetas={list(int(r['theta_deg']) for r in rows)}, idx={SAMPLE_IDX}, res={RESOLUTION}
  GT std / range (mean over samples): {mean_gt_std_away:.4f} / {mean_gt_rng_away:.4f}
  Pred std / range (mean over samples): {mean_pred_std_away:.4f} / {mean_pred_rng_away:.4f}
  Ratio std: {mean_std_away:.1f}% (per-θ {min_std_away:.1f}–{max_std_away:.1f}%)
  Ratio range: {mean_rng_away:.1f}% (per-θ {min_rng_away:.1f}–{max_rng_away:.1f}%)
  Compare — near-kink std ratio (5% band): {min_kink:.1f}–{max_kink:.1f}% (mean {mean_kink:.1f}%)
  Compare — whole-domain free-space std ratio: {mean_whole:.1f}% (prior baseline ~19–22%)

POINTWISE CORRELATION (Pearson r, free pixels):
  All free pixels (mean over samples): {mean_r_all:.3f}
  Near goal (within 20% domain diagonal): {mean_r_near:.3f}
  Everywhere else (free, outside 20% goal zone): {mean_r_far:.3f}
  Pattern: {pattern}

CONTOUR COMPARISON: θ={int(contour_row['theta_deg'])} sample {SAMPLE_IDX} (id={contour_row['sample_id']}); saved {contour_path.name}. {gt_levels_span}; {pred_levels_span}. GT contours (blue, solid) and FNO pred contours (red, dashed), {N_CONTOUR_LEVELS} levels each on [0,1]^2 with occupancy underlay."""

    print("\n" + report)

    out_json = args.output_dir / "whole_domain_structure_audit.json"
    payload = {
        "checkpoint": str(args.checkpoint),
        "resolution": RESOLUTION,
        "sample_idx": SAMPLE_IDX,
        "goal_exclude_frac": GOAL_EXCLUDE_FRAC,
        "goal_near_frac": GOAL_NEAR_FRAC,
        "samples": [
            {k: v for k, v in r.items() if k not in ("occ_map", "gt", "pred", "free")}
            for r in rows
        ],
        "aggregate": {
            "mean_std_ratio_away_pct": mean_std_away,
            "mean_range_ratio_away_pct": mean_rng_away,
            "mean_whole_std_ratio_pct": mean_whole,
            "mean_kink_std_ratio_pct": mean_kink,
            "mean_r_near_goal": mean_r_near,
            "mean_r_far_from_goal": mean_r_far,
            "mean_r_free_all": mean_r_all,
            "pattern": pattern,
        },
        "contour_path": str(contour_path),
        "report": report,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved JSON: {out_json}")
    print(f"Saved contour: {contour_path}")


if __name__ == "__main__":
    main()
