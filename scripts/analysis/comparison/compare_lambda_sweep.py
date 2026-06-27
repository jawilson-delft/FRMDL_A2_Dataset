#!/usr/bin/env python3
"""Compare validation curves across lambda_kink sweep runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MEAN_KINK_PIXEL_FRAC = 0.00569  # from lambda=100 training diagnostic

RUNS = [
    ("100", "lambda_100", None),
    ("175", "lambda_175", 175.0),
    ("350", "lambda_350", 350.0),
    ("700", "lambda_700", 700.0),
]


def _load_history(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _resolve_log_path(training_root: Path, label: str, baseline: bool) -> Path:
    if baseline:
        # lambda=100 baseline: prefer dedicated subdir, fall back to root training log
        dedicated = training_root / label / "loss_log.csv"
        if dedicated.exists():
            return dedicated
        return training_root / "loss_log.csv"
    return training_root / label / "loss_log.csv"


def _analyze_run(history: list[dict], lambda_kink: float) -> dict:
    val_kink = [float(h["val_kink_loss"]) for h in history]
    val_whole = [float(h["val_whole_loss"]) for h in history]
    epochs = [int(h["epoch"]) for h in history]

    best_idx = int(np.argmin(val_kink))
    best_epoch = epochs[best_idx]
    best_val_kink = val_kink[best_idx]

    ep20_kink = val_kink[19] if len(val_kink) >= 20 else float("nan")
    late_min = min(val_kink[20:]) if len(val_kink) > 20 else float("nan")
    late_improvement_vs_ep20 = bool(late_min < ep20_kink) if len(val_kink) > 20 else False

    effective_weight = lambda_kink * MEAN_KINK_PIXEL_FRAC

    return {
        "lambda_kink": lambda_kink,
        "best_val_kink": best_val_kink,
        "best_epoch": best_epoch,
        "val_kink_at_200": val_kink[-1],
        "val_whole_at_200": val_whole[-1],
        "effective_weight_ratio": effective_weight,
        "late_improvement_vs_ep20": late_improvement_vs_ep20,
        "epochs": epochs,
        "val_kink": val_kink,
        "val_whole": val_whole,
    }


def _plot_comparison(
    summaries: list[dict],
    y_key: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for s in summaries:
        lam = int(s["lambda_kink"])
        ax.plot(s["epochs"], s[y_key], linewidth=1.5, label=f"λ={lam}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "training",
    )
    args = parser.parse_args()

    summaries: list[dict] = []
    print("\n=== Lambda sweep comparison ===\n")
    print(
        f"{'lambda':>6}  {'best_val_kink':>13}  {'best_ep':>7}  "
        f"{'val_kink@200':>12}  {'val_whole@200':>13}  {'eff_weight':>10}  {'late↓vs_ep20':>12}"
    )
    print("-" * 82)

    for lam_str, label, lam_val in RUNS:
        lam_float = float(lam_str) if lam_val is None else lam_val
        log_path = _resolve_log_path(args.training_root, label, baseline=(lam_str == "100"))
        if not log_path.exists():
            print(f"WARNING: missing {log_path}, skipping λ={lam_str}")
            continue
        history = _load_history(log_path)
        summary = _analyze_run(history, lam_float)
        summaries.append(summary)
        late_flag = "yes" if summary["late_improvement_vs_ep20"] else "no"
        print(
            f"{lam_float:6.0f}  {summary['best_val_kink']:13.5f}  {summary['best_epoch']:7d}  "
            f"{summary['val_kink_at_200']:12.5f}  {summary['val_whole_at_200']:13.5f}  "
            f"{summary['effective_weight_ratio']:10.3f}  {late_flag:>12}"
        )

    if not summaries:
        raise SystemExit("No sweep logs found.")

    _plot_comparison(
        summaries,
        "val_kink",
        "Validation near-kink relative L2",
        "Val kink loss across lambda_kink sweep",
        args.training_root / "lambda_sweep_val_kink_comparison.png",
    )
    _plot_comparison(
        summaries,
        "val_whole",
        "Validation whole-domain relative L2",
        "Val whole-domain loss across lambda_kink sweep",
        args.training_root / "lambda_sweep_val_whole_comparison.png",
    )

    print(f"\nSaved {args.training_root / 'lambda_sweep_val_kink_comparison.png'}")
    print(f"Saved {args.training_root / 'lambda_sweep_val_whole_comparison.png'}")

    # Factual trend notes
    print("\n--- Observed trends (factual) ---")
    best_epochs = [s["best_epoch"] for s in summaries]
    print(
        f"Best val_kink epochs: "
        + ", ".join(f"λ={int(s['lambda_kink'])}@{s['best_epoch']}" for s in summaries)
    )
    late_flags = [s["late_improvement_vs_ep20"] for s in summaries]
    if all(not f for f in late_flags):
        print(
            "No run shows val_kink loss falling below its epoch-20 value in epochs 21–200."
        )
    else:
        improved = [int(s["lambda_kink"]) for s, f in zip(summaries, late_flags) if f]
        print(
            f"Runs with val_kink below epoch-20 value after epoch 20: λ={improved}"
        )

    final_kink = [s["val_kink_at_200"] for s in summaries]
    print(
        "Final val_kink at epoch 200: "
        + ", ".join(
            f"λ={int(s['lambda_kink'])}={s['val_kink_at_200']:.4f}" for s in summaries
        )
    )
    if all(v > 0.5 for v in final_kink):
        print("All four runs have val_kink_at_200 above 0.5.")


if __name__ == "__main__":
    main()
