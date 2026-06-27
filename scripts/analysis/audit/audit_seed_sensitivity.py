#!/usr/bin/env python3
"""Run train.py baseline config with explicit training seed (does not modify train.py)."""

from __future__ import annotations

import argparse
import random
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader

# Must set seeds before model init inside train.train()
import train as train_mod


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_dataloader_with_seed(seed: int):
    _orig = TorchDataLoader

    def _patched(*args, **kwargs):
        if kwargs.get("shuffle") and "generator" not in kwargs:
            gen = torch.Generator()
            gen.manual_seed(seed)
            kwargs["generator"] = gen
        return _orig(*args, **kwargs)

    return _patched, _orig


def run_with_seed(seed: int, run_label: str, epochs: int = 200) -> None:
    _seed_everything(seed)
    patched, orig = _make_dataloader_with_seed(seed)
    train_mod.DataLoader = patched  # type: ignore[misc]
    try:
        base = Path(__file__).resolve().parent
        argv = [
            "train.py",
            "--run-label",
            run_label,
            "--epochs",
            str(epochs),
            "--lambda-kink",
            "100",
            "--lr-decay-gamma",
            "0.95",
            "--lr-decay-every-epochs",
            "5",
        ]
        old_argv = sys.argv
        sys.argv = argv
        print(f"=== Training seed={seed} run_label={run_label} ===", flush=True)
        train_mod.main()
        sys.argv = old_argv
    finally:
        train_mod.DataLoader = orig  # type: ignore[misc]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-label", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    run_with_seed(args.seed, args.run_label, args.epochs)


if __name__ == "__main__":
    main()
