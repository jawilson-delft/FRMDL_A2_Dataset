"""Train/validation index split for 64×64 training data (stratified by θ)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geometry import THETA_VALUES

DEFAULT_SPLIT_SEED = 42
DEFAULT_VAL_FRACTION = 0.15


def create_stratified_split(
    theta_deg: np.ndarray,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, val_indices) stratified by theta."""
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for theta in THETA_VALUES:
        indices = np.where(np.isclose(theta_deg, theta))[0]
        if len(indices) == 0:
            raise ValueError(f"No training samples for theta={theta}")
        perm = indices.copy()
        rng.shuffle(perm)
        n_val = int(round(len(perm) * val_fraction))
        n_val = max(1, min(n_val, len(perm) - 1))
        val_parts.append(np.sort(perm[:n_val]))
        train_parts.append(np.sort(perm[n_val:]))

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def save_split(
    split_path: Path,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    seed: int,
    val_fraction: float,
) -> None:
    split_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "seed": seed,
        "val_fraction": val_fraction,
        "n_train": int(len(train_indices)),
        "n_val": int(len(val_indices)),
        "theta_values": list(THETA_VALUES),
        "per_theta": {},
    }
    np.savez_compressed(
        split_path,
        train_indices=train_indices.astype(np.int64),
        val_indices=val_indices.astype(np.int64),
    )
    meta_path = split_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))


def load_or_create_split(
    train_npz: Path,
    split_path: Path,
    *,
    seed: int = DEFAULT_SPLIT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    if split_path.exists():
        data = np.load(split_path)
        return data["train_indices"], data["val_indices"]

    raw = np.load(train_npz)
    train_idx, val_idx = create_stratified_split(
        raw["theta_deg"], seed=seed, val_fraction=val_fraction
    )
    save_split(split_path, train_idx, val_idx, seed=seed, val_fraction=val_fraction)

    meta_path = split_path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    for theta in THETA_VALUES:
        mask_tr = np.isclose(raw["theta_deg"][train_idx], theta)
        mask_va = np.isclose(raw["theta_deg"][val_idx], theta)
        meta["per_theta"][str(int(theta))] = {
            "n_train": int(mask_tr.sum()),
            "n_val": int(mask_va.sum()),
        }
    meta_path.write_text(json.dumps(meta, indent=2))
    return train_idx, val_idx
