#!/usr/bin/env python3
"""Compare training curves across baseline and reduced-capacity runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from model_fno import (
    DEFAULT_MODES,
    DEFAULT_N_LAYERS,
    DEFAULT_WIDTH,
    FNO2d,
    REDUCED_MODES,
    REDUCED_N_LAYERS,
    REDUCED_WIDTH,
    count_parameters,
)

RUNS = [
    (
        "baseline (original capacity, no WD)",
        "baseline",
        DEFAULT_WIDTH,
        DEFAULT_MODES,
        DEFAULT_N_LAYERS,
        0.0,
        Path("results/training/loss_log.csv"),
    ),
    (
        "reduced capacity, no WD",
        "reduced_capacity",
        REDUCED_WIDTH,
        REDUCED_MODES,
        REDUCED_N_LAYERS,
        0.0,
        Path("results/training/reduced_capacity/loss_log.csv"),
    ),
    (
        "reduced capacity + WD=1e-4",
        "reduced_capacity_wd",
        REDUCED_WIDTH,
        REDUCED_MODES,
        REDUCED_N_LAYERS,
        1e-4,
        Path("results/training/reduced_capacity_wd/loss_log.csv"),
    ),
]


def _load_history(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _summarize(name: str, width: int, modes: int, n_layers: int, wd: float, history: list[dict]) -> dict:
    val_kink = [float(h["val_kink_loss"]) for h in history]
    train_kink = [float(h["train_kink_loss"]) for h in history]
    best_idx = int(np.argmin(val_kink))
    model = FNO2d(width=width, modes=modes, n_layers=n_layers)
    return {
        "name": name,
        "param_count": count_parameters(model),
        "best_val_kink": val_kink[best_idx],
        "best_epoch": int(history[best_idx]["epoch"]),
        "val_kink_at_200": val_kink[-1],
        "train_kink_at_200": train_kink[-1],
        "weight_decay": wd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "training",
    )
    args = parser.parse_args()

    print("\n=== Capacity experiment comparison ===\n")
    print(
        f"{'configuration':<38}  {'params':>8}  {'best_vk':>8}  {'best_ep':>7}  "
        f"{'vk@200':>8}  {'tk@200':>8}"
    )
    print("-" * 88)

    summaries: list[dict] = []
    for label, _, width, modes, n_layers, wd, rel_path in RUNS:
        path = Path(__file__).resolve().parent / rel_path
        if not path.exists():
            print(f"WARNING: missing {path}, skipping {label}")
            continue
        history = _load_history(path)
        summary = _summarize(label, width, modes, n_layers, wd, history)
        summaries.append(summary)
        print(
            f"{summary['name']:<38}  {summary['param_count']:8,d}  "
            f"{summary['best_val_kink']:8.5f}  {summary['best_epoch']:7d}  "
            f"{summary['val_kink_at_200']:8.5f}  {summary['train_kink_at_200']:8.5f}"
        )

    if len(summaries) < 2:
        raise SystemExit("Need at least baseline + one new run.")

    baseline = summaries[0]
    print("\n--- Observed patterns (factual) ---")
    for s in summaries[1:]:
        later_or_lower = (
            s["best_val_kink"] < baseline["best_val_kink"]
            or s["best_epoch"] > baseline["best_epoch"]
        )
        smaller_gap = s["train_kink_at_200"] > baseline["train_kink_at_200"] * 1.5
        print(f"{s['name']}:")
        print(
            f"  best val_kink {s['best_val_kink']:.5f} @ ep{s['best_epoch']} "
            f"(baseline {baseline['best_val_kink']:.5f} @ ep{baseline['best_epoch']})"
        )
        print(
            f"  train_kink@200 {s['train_kink_at_200']:.5f} vs baseline {baseline['train_kink_at_200']:.5f}"
        )
        if later_or_lower:
            print("  -> lower best val_kink and/or later best epoch than baseline")
        if smaller_gap:
            print("  -> train kink at epoch 200 substantially higher than baseline (smaller train collapse)")
        if not later_or_lower and not smaller_gap:
            print("  -> no clear improvement vs baseline on these metrics")


if __name__ == "__main__":
    main()
