#!/usr/bin/env python3
"""Read-only: verify realized LR at selected epochs matches manual decay rule."""

from __future__ import annotations

import math

LR_INITIAL = 1e-3
GAMMA = 0.95
DECAY_EVERY_EPOCHS = 5
BATCH_SIZE = 16
N_TRAIN = 5355
STEPS_PER_EPOCH = math.ceil(N_TRAIN / BATCH_SIZE)  # 335
DECAY_EVERY_STEPS = STEPS_PER_EPOCH * DECAY_EVERY_EPOCHS  # 1675


def lr_after_epoch(epoch: int) -> float:
    n_decays = (epoch * STEPS_PER_EPOCH) // DECAY_EVERY_STEPS
    return LR_INITIAL * (GAMMA ** n_decays)


def main() -> None:
    print("=== LR schedule simulation (baseline: gamma=0.95, every 5 epochs) ===")
    print(f"steps/epoch={STEPS_PER_EPOCH}, decay every {DECAY_EVERY_STEPS} steps")
    for ep in (1, 10, 50, 100):
        n = (ep * STEPS_PER_EPOCH) // DECAY_EVERY_STEPS
        print(f"  after epoch {ep:3d}: lr={lr_after_epoch(ep):.6e}  ({n} decay events)")


if __name__ == "__main__":
    main()
