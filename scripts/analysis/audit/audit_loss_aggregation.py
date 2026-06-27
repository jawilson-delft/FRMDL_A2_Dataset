#!/usr/bin/env python3
"""Read-only: compare batch-mean vs per-sample epoch aggregation (one epoch)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from geometry import TRAIN_RESOLUTION
from kink_utils import DEFAULT_LAMBDA_KINK
from model_fno import FNO2d, combined_kink_loss, prepare_input
from split_utils import load_or_create_split
from train import EikonalTrainDataset


def _epoch_metrics(loader, model, device, lambda_kink: float) -> dict:
    """Replicate train.py mean-of-batch-means vs true per-sample mean."""
    whole_sum = kink_sum = total_sum = 0.0
    n_batches = 0
    sample_whole: list[float] = []
    sample_kink: list[float] = []
    sample_total: list[float] = []

    model.eval()
    with torch.no_grad():
        for occ, target, kink_mask in loader:
            occ = occ.to(device)
            target = target.to(device)
            kink_mask = kink_mask.to(device)
            pred = model(prepare_input(occ))
            total, whole, kink = combined_kink_loss(pred, target, kink_mask, lambda_kink)
            whole_sum += whole.item()
            kink_sum += kink.item()
            total_sum += total.item()
            n_batches += 1
            for i in range(pred.shape[0]):
                pi = pred[i : i + 1]
                ti = target[i : i + 1]
                mi = kink_mask[i : i + 1]
                t, w, k = combined_kink_loss(pi, ti, mi, lambda_kink)
                sample_whole.append(w.item())
                sample_kink.append(k.item())
                sample_total.append(t.item())

    batch_mean_whole = whole_sum / n_batches
    batch_mean_kink = kink_sum / n_batches
    batch_mean_total = total_sum / n_batches
    return {
        "n_batches": n_batches,
        "n_samples": len(sample_whole),
        "batch_mean_whole": batch_mean_whole,
        "batch_mean_kink": batch_mean_kink,
        "batch_mean_total": batch_mean_total,
        "sample_mean_whole": float(np.mean(sample_whole)),
        "sample_mean_kink": float(np.mean(sample_kink)),
        "sample_mean_total": float(np.mean(sample_total)),
        "recomposed_total": batch_mean_whole + lambda_kink * batch_mean_kink,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_path = args.data_dir / "train" / "train_64.npz"
    split_path = args.data_dir / "train" / "val_split.npz"
    train_idx, val_idx = load_or_create_split(data_path, split_path)

    train_ds = EikonalTrainDataset(data_path, train_idx)
    val_ds = EikonalTrainDataset(data_path, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    n_train = len(train_ds)
    n_val = len(val_ds)
    bs = args.batch_size
    train_last = n_train - (n_train // bs) * bs or bs
    val_last = n_val - (n_val // bs) * bs or bs

    print("=== Batch geometry ===")
    print(f"Train: {n_train} samples, batch_size={bs}, "
          f"full_batches={n_train // bs}, last_batch_size={train_last}")
    print(f"Val:   {n_val} samples, last_batch_size={val_last}")

    model = FNO2d().to(device)
    lam = DEFAULT_LAMBDA_KINK

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        m = _epoch_metrics(loader, model, device, lam)
        print(f"\n=== {name} (untrained model, one pass) ===")
        print(f"  logged (batch-mean):  whole={m['batch_mean_whole']:.6f}  "
              f"kink={m['batch_mean_kink']:.6f}  total={m['batch_mean_total']:.6f}")
        print(f"  true per-sample:    whole={m['sample_mean_whole']:.6f}  "
              f"kink={m['sample_mean_kink']:.6f}  total={m['sample_mean_total']:.6f}")
        print(f"  recomposed total:   {m['recomposed_total']:.6f}")
        print(f"  |batch-sample| whole: {abs(m['batch_mean_whole']-m['sample_mean_whole']):.6e}")
        print(f"  |batch-sample| kink:  {abs(m['batch_mean_kink']-m['sample_mean_kink']):.6e}")
        print(f"  |total - recomposed|: {abs(m['batch_mean_total']-m['recomposed_total']):.6e}")


if __name__ == "__main__":
    main()
