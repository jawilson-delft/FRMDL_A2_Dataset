#!/usr/bin/env python3
"""Compare training curves across all completed experiment runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from evaluate import load_model
from model_fno import (
    DEFAULT_MODES,
    DEFAULT_N_LAYERS,
    DEFAULT_WIDTH,
    FNO2d,
    REDUCED_MODES,
    REDUCED_N_LAYERS,
    REDUCED_WIDTH,
    REFERENCE_MODES_HALF_RES,
    count_parameters,
    prepare_input,
)

ROOT = Path(__file__).resolve().parent
THETAS_AMP = (180, 150, 90, 30, 10)
RES_AMP = 512

# (label, log_path, width, modes, n_layers, checkpoint_path or None, amp_note)
RUN_SPECS: list[tuple] = [
    (
        "baseline λ=100",
        "results/training/loss_log.csv",
        DEFAULT_WIDTH,
        DEFAULT_MODES,
        DEFAULT_N_LAYERS,
        "checkpoints/fno_best_val_kink.pt",
        "prior",
    ),
    (
        "λ=175",
        "results/training/lambda_175/loss_log.csv",
        DEFAULT_WIDTH,
        DEFAULT_MODES,
        DEFAULT_N_LAYERS,
        None,
        None,
    ),
    (
        "λ=350",
        "results/training/lambda_350/loss_log.csv",
        DEFAULT_WIDTH,
        DEFAULT_MODES,
        DEFAULT_N_LAYERS,
        None,
        None,
    ),
    (
        "λ=700",
        "results/training/lambda_700/loss_log.csv",
        DEFAULT_WIDTH,
        DEFAULT_MODES,
        DEFAULT_N_LAYERS,
        None,
        None,
    ),
    (
        "extended schedule (early stop)",
        "results/training/extended_run/loss_log.csv",
        DEFAULT_WIDTH,
        DEFAULT_MODES,
        DEFAULT_N_LAYERS,
        "checkpoints/extended_run/fno_best_val_kink.pt",
        "prior",
    ),
    (
        "reduced capacity",
        "results/training/reduced_capacity/loss_log.csv",
        REDUCED_WIDTH,
        REDUCED_MODES,
        REDUCED_N_LAYERS,
        "checkpoints/reduced_capacity/fno_best_val_kink.pt",
        None,
    ),
    (
        "reduced capacity + WD",
        "results/training/reduced_capacity_wd/loss_log.csv",
        REDUCED_WIDTH,
        REDUCED_MODES,
        REDUCED_N_LAYERS,
        "checkpoints/reduced_capacity_wd/fno_best_val_kink.pt",
        None,
    ),
    (
        "modes=32 (half-res ref)",
        "results/training/modes32/loss_log.csv",
        DEFAULT_WIDTH,
        REFERENCE_MODES_HALF_RES,
        DEFAULT_N_LAYERS,
        "checkpoints/modes32/fno_best_val_kink.pt",
        None,
    ),
]

# Documented amplitude ratios from earlier diagnostics (mean over 5 θ, test sample 0).
PRIOR_AMP = {
    "baseline λ=100": "~19-22%",
    "extended schedule (early stop)": "~24-31%",
}


def _load_history(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _summarize_log(history: list[dict]) -> dict:
    val_kink = [float(h["val_kink_loss"]) for h in history]
    train_kink = [float(h["train_kink_loss"]) for h in history]
    best_idx = int(np.argmin(val_kink))
    final_idx = len(history) - 1
    return {
        "best_val_kink": val_kink[best_idx],
        "best_epoch": int(history[best_idx]["epoch"]),
        "final_epoch": int(history[final_idx]["epoch"]),
        "val_kink_final": val_kink[final_idx],
        "train_kink_final": train_kink[final_idx],
    }


@torch.no_grad()
def _amplitude_ratio(checkpoint: Path, device: torch.device) -> str:
    if not checkpoint.exists():
        return "n/a"
    model = load_model(checkpoint, device)
    ratios: list[float] = []
    for theta in THETAS_AMP:
        npz = ROOT / "data" / "test" / f"theta_{theta}" / f"res_{RES_AMP}" / "samples.npz"
        data = np.load(npz)
        occ_map = data["occupancy"][0]
        tgt = data["travel_time"][0]
        free = occ_map > 0.5
        occ = torch.from_numpy(occ_map).float().unsqueeze(0).unsqueeze(0).to(device)
        pred = model(prepare_input(occ)).squeeze().detach().cpu().numpy()
        ratios.append(float(np.std(pred[free]) / max(np.std(tgt[free]), 1e-8)))
    lo, hi = min(ratios), max(ratios)
    return f"{lo*100:.1f}-{hi*100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print("\n=== All training runs comparison ===\n")
    header = (
        f"{'configuration':<32}  {'params':>9}  {'best_vk':>8}  {'best_ep':>7}  "
        f"{'vk@final':>8}  {'tk@final':>8}  {'amp_ratio':>12}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    for label, rel_log, width, modes, n_layers, rel_ckpt, amp_note in RUN_SPECS:
        log_path = ROOT / rel_log
        if not log_path.exists():
            print(f"WARNING: missing {log_path}, skipping {label}")
            continue
        history = _load_history(log_path)
        stats = _summarize_log(history)
        n_params = count_parameters(FNO2d(width=width, modes=modes, n_layers=n_layers))
        if amp_note == "prior" and label in PRIOR_AMP:
            amp = PRIOR_AMP[label]
        elif rel_ckpt:
            amp = _amplitude_ratio(ROOT / rel_ckpt, device)
        else:
            amp = "n/a"
        row = {
            "label": label,
            "params": n_params,
            **stats,
            "amp_ratio": amp,
        }
        rows.append(row)
        print(
            f"{label:<32}  {n_params:9,d}  {stats['best_val_kink']:8.5f}  "
            f"{stats['best_epoch']:7d}  {stats['val_kink_final']:8.5f}  "
            f"{stats['train_kink_final']:8.5f}  {amp:>12}"
        )

    if rows:
        m32 = next((r for r in rows if "modes=32" in r["label"]), None)
        base = next((r for r in rows if r["label"] == "baseline λ=100"), None)
        if m32 and base:
            print("\n--- modes=32 vs baseline (factual) ---")
            print(
                f"best val_kink: {m32['best_val_kink']:.5f} @ ep{m32['best_epoch']} "
                f"vs baseline {base['best_val_kink']:.5f} @ ep{base['best_epoch']}"
            )
            print(
                f"val_kink @ final: {m32['val_kink_final']:.5f} (ep{m32['final_epoch']}) "
                f"vs baseline {base['val_kink_final']:.5f}"
            )


if __name__ == "__main__":
    main()
