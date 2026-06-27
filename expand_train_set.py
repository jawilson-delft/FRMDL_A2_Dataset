#!/usr/bin/env python3
"""Expand 64×64 training data without regenerating the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from generate_dataset import _save_npz_bundle
from geometry import THETA_VALUES, TEST_RESOLUTIONS, geometry_hash, make_sample_geometry, render_sample
from split_utils import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VAL_FRACTION,
    create_stratified_split,
    save_split,
)

# Continuation seed: original manifest seed (42) + total samples drawn in initial run.
ORIGINAL_SEED = 42
INITIAL_TOTAL_SAMPLES = 3500  # 2100 train + 1400 test from manifest.json
EXPANSION_SEED = ORIGINAL_SEED + INITIAL_TOTAL_SAMPLES  # 3542
TARGET_PER_THETA = 900


def _param_key_from_row(data: np.lib.npyio.NpzFile, idx: int) -> tuple:
    return (
        float(data["theta_deg"][idx]),
        round(float(data["center"][idx, 0]), 10),
        round(float(data["center"][idx, 1]), 10),
        round(float(data["rotation_deg"][idx]), 10),
        round(float(data["scale"][idx]), 10),
    )


def _full_key_from_row(data: np.lib.npyio.NpzFile, idx: int) -> tuple:
    pk = _param_key_from_row(data, idx)
    return pk + (
        round(float(data["goal"][idx, 0]), 10),
        round(float(data["goal"][idx, 1]), 10),
    )


def _load_used_keys(data_dir: Path) -> tuple[set, set]:
    used_param: set = set()
    used_full: set = set()

    train_path = data_dir / "train" / "train_64.npz"
    train = np.load(train_path, allow_pickle=True)
    for idx in range(len(train["sample_id"])):
        used_param.add(_param_key_from_row(train, idx))
        used_full.add(_full_key_from_row(train, idx))

    for theta in THETA_VALUES:
        for resolution in TEST_RESOLUTIONS:
            path = (
                data_dir
                / "test"
                / f"theta_{int(theta)}"
                / f"res_{resolution}"
                / "samples.npz"
            )
            test = np.load(path, allow_pickle=True)
            for idx in range(len(test["sample_id"])):
                used_param.add(_param_key_from_row(test, idx))
                used_full.add(_full_key_from_row(test, idx))

    return used_param, used_full


def _param_distance(
    c1: np.ndarray,
    r1: float,
    s1: float,
    c2: np.ndarray,
    r2: float,
    s2: float,
) -> tuple[float, float, float, float]:
    pos_dist = float(np.linalg.norm(c1 - c2))
    rot_dist_deg = abs(r1 - r2) % 360.0
    rot_dist_deg = min(rot_dist_deg, 360.0 - rot_dist_deg)
    scale_rel_diff = abs(s1 - s2) / max(abs(s1), abs(s2), 1e-12)
    combined = pos_dist + rot_dist_deg / 180.0 + scale_rel_diff
    return pos_dist, rot_dist_deg, scale_rel_diff, combined


def check_new_samples_vs_test(
    data_dir: Path,
    new_start_idx: int,
    train_path: Path,
) -> dict:
    """Quick Check-3-style collision screen for newly appended training rows."""
    train = np.load(train_path, allow_pickle=True)
    train_by_theta: dict[int, list[dict]] = {int(t): [] for t in THETA_VALUES}
    for idx in range(new_start_idx, len(train["sample_id"])):
        theta = int(round(float(train["theta_deg"][idx])))
        train_by_theta[theta].append(
            {
                "train_index": idx,
                "center": train["center"][idx],
                "rotation": float(train["rotation_deg"][idx]),
                "scale": float(train["scale"][idx]),
            }
        )

    near_duplicates: list[dict] = []
    for theta in THETA_VALUES:
        new_rows = train_by_theta[theta]
        if not new_rows:
            continue
        for res in TEST_RESOLUTIONS:
            path = data_dir / "test" / f"theta_{theta}" / f"res_{res}" / "samples.npz"
            test = np.load(path, allow_pickle=True)
            for tidx in range(len(test["sample_id"])):
                tc = test["center"][tidx]
                tr = float(test["rotation_deg"][tidx])
                ts = float(test["scale"][tidx])
                for nrow in new_rows:
                    pd, rd, sd, _ = _param_distance(
                        tc, tr, ts, nrow["center"], nrow["rotation"], nrow["scale"]
                    )
                    if pd < 0.01 and rd < 1.0 and sd < 0.01:
                        near_duplicates.append(
                            {
                                "theta": theta,
                                "resolution": res,
                                "test_index": tidx,
                                "train_index": nrow["train_index"],
                                "nearest_pos_dist": pd,
                                "nearest_rot_dist_deg": rd,
                                "nearest_scale_rel_diff": sd,
                            }
                        )
    return {"n_near_duplicates": len(near_duplicates), "near_duplicate_flags": near_duplicates}


def append_training_samples(
    data_dir: Path,
    *,
    target_per_theta: int,
    expansion_seed: int,
) -> dict:
    train_path = data_dir / "train" / "train_64.npz"
    existing = np.load(train_path, allow_pickle=True)
    n_existing = len(existing["sample_id"])

    per_theta_counts = {
        int(theta): int(np.sum(np.isclose(existing["theta_deg"], theta)))
        for theta in THETA_VALUES
    }
    for theta, count in per_theta_counts.items():
        if count > target_per_theta:
            raise ValueError(
                f"theta={theta} already has {count} samples (target {target_per_theta})"
            )

    used_param_keys, used_full_keys = _load_used_keys(data_dir)
    rng = np.random.default_rng(expansion_seed)

    new_occ, new_tt, new_meta = [], [], []
    sample_id = n_existing
    from geometry import TRAIN_RESOLUTION

    for theta in THETA_VALUES:
        n_add = target_per_theta - per_theta_counts[int(theta)]
        for _ in tqdm(range(n_add), desc=f"append train θ={theta}"):
            for attempt in range(5000):
                geom = make_sample_geometry(
                    rng,
                    theta,
                    sample_id,
                    used_param_keys,
                    used_full_keys,
                )
                try:
                    occ, tt = render_sample(geom, TRAIN_RESOLUTION)
                except ValueError:
                    used_param_keys.discard(geom.param_key())
                    used_full_keys.discard(geom.full_key())
                    continue
                used_param_keys.add(geom.param_key())
                used_full_keys.add(geom.full_key())
                new_occ.append(occ)
                new_tt.append(tt)
                meta = geom.to_metadata_dict()
                meta["param_hash"] = geometry_hash(geom.param_key())
                new_meta.append(meta)
                sample_id += 1
                break
            else:
                raise RuntimeError(
                    f"Failed to sample valid geometry for theta={theta} after 5000 attempts"
                )

    if not new_occ:
        print("No new samples needed.")
        return {"n_added": 0, "n_total": n_existing, "train_path": str(train_path)}

    occ = np.concatenate([existing["occupancy"], np.stack(new_occ)], axis=0)
    tt = np.concatenate([existing["travel_time"], np.stack(new_tt)], axis=0)

    old_meta = []
    for idx in range(n_existing):
        old_meta.append(
            {
                "theta_deg": float(existing["theta_deg"][idx]),
                "goal": existing["goal"][idx].tolist(),
                "center": existing["center"][idx].tolist(),
                "rotation_deg": float(existing["rotation_deg"][idx]),
                "scale": float(existing["scale"][idx]),
                "sample_id": int(existing["sample_id"][idx]),
                "vertices": json.loads(str(existing["vertices_json"][idx])),
                "param_hash": str(existing["param_hash"][idx]),
            }
        )
    for i, meta in enumerate(new_meta):
        meta["sample_id"] = n_existing + i
    all_meta = old_meta + new_meta

    _save_npz_bundle(train_path, occ, tt, all_meta)
    return {
        "n_added": len(new_meta),
        "n_total": len(all_meta),
        "train_path": str(train_path),
        "expansion_seed": expansion_seed,
        "per_theta_before": per_theta_counts,
        "per_theta_after": {int(t): target_per_theta for t in THETA_VALUES},
    }


def recreate_val_split(data_dir: Path, *, seed: int, val_fraction: float) -> dict:
    train_path = data_dir / "train" / "train_64.npz"
    split_path = data_dir / "train" / "val_split.npz"
    json_path = split_path.with_suffix(".json")
    for path in (split_path, json_path):
        if path.exists():
            path.unlink()

    raw = np.load(train_path)
    train_idx, val_idx = create_stratified_split(
        raw["theta_deg"], seed=seed, val_fraction=val_fraction
    )
    save_split(split_path, train_idx, val_idx, seed=seed, val_fraction=val_fraction)

    meta = json.loads(json_path.read_text())
    for theta in THETA_VALUES:
        mask_tr = np.isclose(raw["theta_deg"][train_idx], theta)
        mask_va = np.isclose(raw["theta_deg"][val_idx], theta)
        meta["per_theta"][str(int(theta))] = {
            "n_train": int(mask_tr.sum()),
            "n_val": int(mask_va.sum()),
        }
    json_path.write_text(json.dumps(meta, indent=2))
    return {
        "split_path": str(split_path),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "per_theta": meta["per_theta"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--target-per-theta", type=int, default=TARGET_PER_THETA)
    parser.add_argument("--expansion-seed", type=int, default=EXPANSION_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    args = parser.parse_args()

    train_path = args.data_dir / "train" / "train_64.npz"
    n_before = len(np.load(train_path)["sample_id"])

    print(
        f"Expanding training set to {args.target_per_theta}/θ "
        f"(expansion_seed={args.expansion_seed}, original_seed={ORIGINAL_SEED})"
    )
    expand_info = append_training_samples(
        args.data_dir,
        target_per_theta=args.target_per_theta,
        expansion_seed=args.expansion_seed,
    )
    print(f"Added {expand_info['n_added']} samples → {expand_info['n_total']} total")

    collision = check_new_samples_vs_test(args.data_dir, n_before, train_path)
    print(
        f"New-vs-test collision check: {collision['n_near_duplicates']} near-duplicates "
        f"(threshold: pos<1%, rot<1°, scale<1%)"
    )
    if collision["n_near_duplicates"] > 0:
        for dup in collision["near_duplicate_flags"][:5]:
            print(f"  FLAG: {dup}")
        raise SystemExit("Collision check failed.")

    split_info = recreate_val_split(
        args.data_dir, seed=args.split_seed, val_fraction=args.val_fraction
    )
    print(
        f"Recreated val split: {split_info['n_train']} train / {split_info['n_val']} val "
        f"(seed={args.split_seed}, fraction={args.val_fraction})"
    )
    print(json.dumps(split_info["per_theta"], indent=2))

    manifest_path = args.data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["train_per_theta"] = args.target_per_theta
        manifest["train"]["n_samples"] = expand_info["n_total"]
        manifest["train_expansion"] = {
            "expansion_seed": args.expansion_seed,
            "original_seed": ORIGINAL_SEED,
            "n_added": expand_info["n_added"],
            "val_split_seed": args.split_seed,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
