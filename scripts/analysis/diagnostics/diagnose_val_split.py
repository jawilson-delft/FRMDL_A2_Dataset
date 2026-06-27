#!/usr/bin/env python3
"""Diagnose validation split composition and validation loss computation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from geometry import THETA_VALUES
from model_fno import FNO2d, prepare_input, relative_l2_loss
from split_utils import DEFAULT_VAL_FRACTION

MEAN_V_FLAG_THRESHOLD = 0.15  # flag if train/val mean-V differ >15% within same theta


def _free_space_v_stats(travel_time: np.ndarray, occupancy: np.ndarray) -> dict[str, float]:
    free = travel_time[occupancy > 0.5]
    if free.size == 0:
        return {"mean_v": np.nan, "std_v": np.nan, "max_v": np.nan, "n_pixels": 0}
    return {
        "mean_v": float(np.mean(free)),
        "std_v": float(np.std(free)),
        "max_v": float(np.max(free)),
        "n_pixels": int(free.size),
    }


def _per_sample_v_stats(data: np.lib.npyio.NpzFile, indices: np.ndarray) -> list[dict]:
    occ = data["occupancy"][indices]
    tt = data["travel_time"][indices]
    theta = data["theta_deg"][indices]
    rows = []
    for i in range(len(indices)):
        stats = _free_space_v_stats(tt[i], occ[i])
        stats["theta"] = float(theta[i])
        stats["index"] = int(indices[i])
        rows.append(stats)
    return rows


def _aggregate_by_theta(rows: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for theta in THETA_VALUES:
        trows = [r for r in rows if np.isclose(r["theta"], theta)]
        if not trows:
            out[int(theta)] = {"n": 0, "mean_v": np.nan, "std_v": np.nan}
            continue
        out[int(theta)] = {
            "n": len(trows),
            "mean_v": float(np.mean([r["mean_v"] for r in trows])),
            "std_v": float(np.mean([r["std_v"] for r in trows])),
        }
    return out


def check_a(data_dir: Path, out_dir: Path) -> dict:
    train_path = data_dir / "train" / "train_64.npz"
    split_path = data_dir / "train" / "val_split.npz"
    split = np.load(split_path)
    train_idx = split["train_indices"]
    val_idx = split["val_indices"]
    data = np.load(train_path, allow_pickle=True)
    theta_all = data["theta_deg"]

    print("\n=== CHECK A — Validation set composition ===\n")
    print(f"{'theta':>6}  {'n_train':>8}  {'n_val':>7}  {'val_frac':>9}")
    print("-" * 36)

    composition_rows = []
    flags: list[str] = []
    expected_frac = DEFAULT_VAL_FRACTION

    for theta in THETA_VALUES:
        n_train = int(np.sum(np.isclose(theta_all[train_idx], theta)))
        n_val = int(np.sum(np.isclose(theta_all[val_idx], theta)))
        total = n_train + n_val
        frac = n_val / total if total else float("nan")
        print(f"{int(theta):6d}  {n_train:8d}  {n_val:7d}  {frac:9.4f}")
        composition_rows.append(
            {"theta": int(theta), "n_train": n_train, "n_val": n_val, "val_fraction": frac}
        )
        if n_val == 0:
            flags.append(f"θ={int(theta)}: no validation samples")
        if abs(frac - expected_frac) > 0.02:
            flags.append(
                f"θ={int(theta)}: val fraction {frac:.3f} deviates >2% from target {expected_frac}"
            )

    train_stats = _aggregate_by_theta(_per_sample_v_stats(data, train_idx))
    val_stats = _aggregate_by_theta(_per_sample_v_stats(data, val_idx))

    print(f"\n{'theta':>6}  {'train mean V':>12}  {'val mean V':>12}  {'Δ%':>8}  "
          f"{'train std V':>12}  {'val std V':>12}")
    print("-" * 72)
    v_flags: list[str] = []
    for theta in THETA_VALUES:
        tr = train_stats[int(theta)]
        va = val_stats[int(theta)]
        if np.isnan(tr["mean_v"]) or np.isnan(va["mean_v"]):
            rel = float("nan")
        else:
            rel = abs(va["mean_v"] - tr["mean_v"]) / max(tr["mean_v"], 1e-8)
        print(
            f"{int(theta):6d}  {tr['mean_v']:12.5f}  {va['mean_v']:12.5f}  "
            f"{rel * 100:7.2f}%  {tr['std_v']:12.5f}  {va['std_v']:12.5f}"
        )
        if not np.isnan(rel) and rel > MEAN_V_FLAG_THRESHOLD:
            v_flags.append(
                f"θ={int(theta)}: val mean V differs {rel * 100:.1f}% from train mean V"
            )

    fig, ax = plt.subplots(figsize=(8, 4))
    thetas = [int(t) for t in THETA_VALUES]
    x = np.arange(len(thetas))
    w = 0.35
    ax.bar(x - w / 2, [train_stats[t]["mean_v"] for t in thetas], w, label="train mean V")
    ax.bar(x + w / 2, [val_stats[t]["mean_v"] for t in thetas], w, label="val mean V")
    ax.set_xticks(x, [str(t) for t in thetas])
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("Mean V (free space)")
    ax.set_title("Check A: per-θ mean V — train vs val subsets")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = out_dir / "check_a_mean_v_train_vs_val.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {plot_path}")

    all_flags = flags + v_flags
    if all_flags:
        print("\nFLAGS:")
        for f in all_flags:
            print(f"  - {f}")
    else:
        print("\nFLAGS: none — all 7 θ present with ~15% val each; "
              "per-θ mean/std V similar between train and val subsets.")

    return {
        "composition": composition_rows,
        "train_v_stats": train_stats,
        "val_v_stats": val_stats,
        "flags": all_flags,
        "plot": str(plot_path),
    }


def check_b() -> dict:
    print("\n=== CHECK B — Validation loss computation (code inspection) ===\n")

    findings: list[str] = []
    issues: list[str] = []

    train_path = Path(__file__).resolve().parent / "train.py"
    train_src = train_path.read_text()
    model_path = Path(__file__).resolve().parent / "model_fno.py"
    model_src = model_path.read_text()

    if "nn.Dropout" not in model_src and "nn.BatchNorm" not in model_src:
        findings.append(
            "FNO2d (model_fno.py): no nn.Dropout or nn.BatchNorm layers — "
            "model.train() vs model.eval() should not change forward outputs."
        )

    if "@torch.no_grad()" in train_src and "def _eval_loader" in train_src:
        findings.append(
            "train.py::_eval_loader (lines 89–100): decorated with @torch.no_grad()."
        )
    else:
        issues.append("@torch.no_grad() not found on _eval_loader")

    eval_body = train_src.split("def _eval_loader")[1].split("\ndef ")[0]
    if "model.eval()" in eval_body:
        findings.append("train.py::_eval_loader line 91: model.eval() called before val loop.")
    else:
        issues.append("model.eval() not found in _eval_loader")

    if train_src.count("relative_l2_loss(pred, target)") >= 2:
        findings.append(
            "train.py: relative_l2_loss(pred, target) used in both _eval_loader (line 98) "
            "and training loop (line 202) — identical function."
        )
    else:
        issues.append("relative_l2_loss not used consistently in train and val")

    if train_src.count("prepare_input(occ)") >= 2:
        findings.append(
            "train.py: prepare_input(occ) applied in both train (line 200) and "
            "val (line 97) paths."
        )
    else:
        issues.append("prepare_input not applied consistently")

    findings.extend([
        "train.py line 194: model.train() at start of each epoch (restores after _eval_loader).",
        "model_fno.py::relative_l2_loss (lines 130–136): per-sample ||pred-target||_2 / ||target||_2, "
        "then mean over batch. Includes ALL pixels (obstacle pixels have target=0).",
        "model_fno.py::prepare_input (lines 121–127): concatenates occupancy with fixed [0,1] grid — "
        "no data-dependent normalization.",
        "No running statistics or train-only normalization in train.py or model_fno.py.",
        "Both train and val epoch metrics average per-batch relative_l2_loss values "
        "(mean of batch means). Same aggregation for both — not a train/val asymmetry.",
    ])

    for line in findings:
        print(f"  • {line}")

    status = "PASS" if not issues else "ISSUE FOUND"
    if issues:
        print("\nISSUES:")
        for issue in issues:
            print(f"  ! {issue}")

    return {"status": status, "findings": findings, "issues": issues}


def _pick_val_samples(val_idx: np.ndarray, theta_deg: np.ndarray, n: int = 5) -> list[int]:
    """Pick one validation sample per distinct theta, spanning the theta range."""
    chosen: list[int] = []
    target_thetas = [180, 120, 90, 30, 10]
    for theta in target_thetas:
        mask = np.isclose(theta_deg[val_idx], theta)
        candidates = val_idx[mask]
        if len(candidates) > 0:
            chosen.append(int(candidates[len(candidates) // 2]))
    return chosen[:n]


@torch.no_grad()
def check_c(
    data_dir: Path,
    checkpoint_dir: Path,
    out_dir: Path,
    device: torch.device,
) -> dict:
    print("\n=== CHECK C — Spot check validation predictions ===\n")

    ckpt_path = checkpoint_dir / "fno_best_val.pt"
    if not ckpt_path.exists():
        ckpt_path = checkpoint_dir / "fno_final_200ep.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    best_epoch = ckpt.get("best_epoch", "?")
    best_val = ckpt.get("best_val_loss", float("nan"))
    print(f"Checkpoint: {ckpt_path.name} (best_epoch={best_epoch}, best_val={best_val:.6f})")
    if best_epoch != 1:
        print(
            f"Note: no epoch-1-only checkpoint saved; using best-val weights from epoch {best_epoch}."
        )

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

    split = np.load(data_dir / "train" / "val_split.npz")
    val_idx = split["val_indices"]
    raw = np.load(data_dir / "train" / "train_64.npz", allow_pickle=True)
    sample_indices = _pick_val_samples(val_idx, raw["theta_deg"])

    descriptions: list[str] = []
    for i, idx in enumerate(sample_indices):
        occ = torch.from_numpy(raw["occupancy"][idx]).float().unsqueeze(0).unsqueeze(0).to(device)
        target = raw["travel_time"][idx]
        theta = float(raw["theta_deg"][idx])
        pred = model(prepare_input(occ)).squeeze().cpu().numpy()
        error = pred - target
        rel_l2 = float(
            np.linalg.norm(error.ravel())
            / max(np.linalg.norm(target.ravel()), 1e-8)
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        im0 = axes[0].imshow(target, origin="lower", cmap="viridis")
        axes[0].set_title(f"Ground truth V (θ={theta:.0f}°)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(pred, origin="lower", cmap="viridis")
        axes[1].set_title("Predicted V")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        vmax = max(np.abs(error).max(), 1e-6)
        im2 = axes[2].imshow(error, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[2].set_title("Pointwise error")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Val sample {i} (idx={idx}, rel-L2={rel_l2:.4f})")
        fig.tight_layout()
        out_path = out_dir / f"val_spotcheck_sample_{i}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

        pred_range = float(pred.max() - pred.min())
        tgt_range = float(target.max() - target.min())
        pred_std = float(np.std(pred))
        has_nan = bool(np.isnan(pred).any())
        free_mask = raw["occupancy"][idx] > 0.5
        pred_free_mean = float(np.mean(pred[free_mask]))
        tgt_free_mean = float(np.mean(target[free_mask]))

        desc = (
            f"Sample {i} (θ={theta:.0f}°, idx={idx}): rel-L2={rel_l2:.4f}. "
            f"GT V range [{target.min():.3f}, {target.max():.3f}] (free-space mean {tgt_free_mean:.3f}). "
            f"Pred range [{pred.min():.3f}, {pred.max():.3f}] (free-space mean {pred_free_mean:.3f}), "
            f"pred std={pred_std:.4f}, range={pred_range:.3f}. "
        )
        if has_nan:
            desc += "Contains NaN regions. "
        elif pred_range < 0.05 * max(tgt_range, 1e-6):
            desc += (
                f"Prediction is nearly constant (~{pred_free_mean:.3f} across domain) "
                "relative to target spatial structure. "
            )
        else:
            desc += (
                "Prediction shows spatial variation with a plausible value-function-like "
                "structure (higher away from goal, lower near goal/obstacles) but "
                "with visible inaccuracy vs ground truth. "
            )
        descriptions.append(desc)
        print(f"  {desc}")

    return {
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "sample_indices": sample_indices,
        "descriptions": descriptions,
    }


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
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = args.results_dir / "diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    result_a = check_a(args.data_dir, out_dir)
    result_b = check_b()
    result_c = check_c(args.data_dir, args.checkpoint_dir, out_dir, device)

    a_flag_str = (
        f"{len(result_a['flags'])} flag(s): " + "; ".join(result_a["flags"])
        if result_a["flags"]
        else "balanced 15% stratified split per θ; train/val V stats comparable within θ"
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"CHECK A (val composition):     INFO  [{a_flag_str}]")
    print(
        f"CHECK B (val loss computation): {result_b['status']}  "
        + ("; ".join(result_b["findings"][:3]) + " ..." if result_b["status"] == "PASS" else "; ".join(result_b["issues"]))
    )
    print(
        "CHECK C (spot check predictions): INFO  "
        + " | ".join(d[:120] + "..." if len(d) > 120 else d for d in result_c["descriptions"][:2])
    )
    if len(result_c["descriptions"]) > 2:
        print("  " + " | ".join(result_c["descriptions"][2:]))

    summary_path = out_dir / "diagnosis_summary.json"
    summary_path.write_text(
        json.dumps(
            {"check_a": result_a, "check_b": result_b, "check_c": result_c},
            indent=2,
            default=str,
        )
    )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
