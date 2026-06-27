#!/usr/bin/env python3
"""Generate the eikonal corner-sharpness control dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from geometry import (
    THETA_VALUES,
    TEST_RESOLUTIONS,
    TRAIN_RESOLUTION,
    SampleGeometry,
    geometry_hash,
    make_sample_geometry,
    render_sample,
)


def _save_npz_bundle(
    path: Path,
    occupancy: np.ndarray,
    travel_time: np.ndarray,
    metadata: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    theta = np.array([m["theta_deg"] for m in metadata], dtype=np.float32)
    goal = np.array([m["goal"] for m in metadata], dtype=np.float32)
    center = np.array([m["center"] for m in metadata], dtype=np.float32)
    rotation = np.array([m["rotation_deg"] for m in metadata], dtype=np.float32)
    scale = np.array([m["scale"] for m in metadata], dtype=np.float32)
    sample_id = np.array([m["sample_id"] for m in metadata], dtype=np.int64)
    param_hash = np.array([m["param_hash"] for m in metadata], dtype="<U16")
    vertices_json = np.array(
        [json.dumps(m["vertices"]) for m in metadata], dtype=object
    )
    np.savez_compressed(
        path,
        occupancy=occupancy,
        travel_time=travel_time,
        theta_deg=theta,
        goal=goal,
        center=center,
        rotation_deg=rotation,
        scale=scale,
        sample_id=sample_id,
        param_hash=param_hash,
        vertices_json=vertices_json,
    )


def generate_training_set(
    output_dir: Path,
    rng: np.random.Generator,
    samples_per_theta: int,
    used_param_keys: set,
    used_full_keys: set,
) -> dict:
    all_occ, all_tt, all_meta = [], [], []
    sample_id = 0
    for theta in THETA_VALUES:
        for _ in tqdm(range(samples_per_theta), desc=f"train θ={theta}"):
            geom = make_sample_geometry(
                rng,
                theta,
                sample_id,
                used_param_keys,
                used_full_keys,
            )
            used_param_keys.add(geom.param_key())
            used_full_keys.add(geom.full_key())
            occ, tt = render_sample(geom, TRAIN_RESOLUTION)
            all_occ.append(occ)
            all_tt.append(tt)
            meta = geom.to_metadata_dict()
            meta["param_hash"] = geometry_hash(geom.param_key())
            all_meta.append(meta)
            sample_id += 1

    occupancy = np.stack(all_occ, axis=0)
    travel_time = np.stack(all_tt, axis=0)
    train_path = output_dir / "train" / "train_64.npz"
    _save_npz_bundle(train_path, occupancy, travel_time, all_meta)
    return {
        "path": str(train_path),
        "n_samples": len(all_meta),
        "resolution": TRAIN_RESOLUTION,
    }


def generate_test_sets(
    output_dir: Path,
    rng: np.random.Generator,
    samples_per_cell: int,
    used_param_keys: set,
    used_full_keys: set,
    start_sample_id: int,
) -> dict:
    summary = {}
    sample_id = start_sample_id
    for theta in THETA_VALUES:
        for resolution in TEST_RESOLUTIONS:
            occ_list, tt_list, meta_list = [], [], []
            desc = f"test θ={theta} res={resolution}"
            for _ in tqdm(range(samples_per_cell), desc=desc):
                geom = make_sample_geometry(
                    rng,
                    theta,
                    sample_id,
                    used_param_keys,
                    used_full_keys,
                )
                used_param_keys.add(geom.param_key())
                used_full_keys.add(geom.full_key())
                occ, tt = render_sample(geom, resolution)
                occ_list.append(occ)
                tt_list.append(tt)
                meta = geom.to_metadata_dict()
                meta["param_hash"] = geometry_hash(geom.param_key())
                meta["resolution"] = resolution
                meta_list.append(meta)
                sample_id += 1

            occ_arr = np.stack(occ_list, axis=0)
            tt_arr = np.stack(tt_list, axis=0)
            out_path = (
                output_dir
                / "test"
                / f"theta_{int(theta)}"
                / f"res_{resolution}"
                / "samples.npz"
            )
            _save_npz_bundle(out_path, occ_arr, tt_arr, meta_list)
            summary[f"theta_{int(theta)}_res_{resolution}"] = {
                "path": str(out_path),
                "n_samples": len(meta_list),
                "theta_deg": theta,
                "resolution": resolution,
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-per-theta", type=int, default=300)
    parser.add_argument("--test-per-cell", type=int, default=50)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small dataset for smoke tests (30 train/θ, 5 test/cell).",
    )
    args = parser.parse_args()

    if args.quick:
        args.train_per_theta = 30
        args.test_per_cell = 5

    rng = np.random.default_rng(args.seed)
    used_param_keys: set = set()
    used_full_keys: set = set()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating training set: {args.train_per_theta} samples × {len(THETA_VALUES)} θ")
    train_info = generate_training_set(
        output_dir,
        rng,
        args.train_per_theta,
        used_param_keys,
        used_full_keys,
    )

    print("Generating zero-shot test sets (7 θ × 4 resolutions)")
    test_info = generate_test_sets(
        output_dir,
        rng,
        args.test_per_cell,
        used_param_keys,
        used_full_keys,
        start_sample_id=train_info["n_samples"],
    )

    manifest = {
        "seed": args.seed,
        "theta_values": list(THETA_VALUES),
        "train_resolution": TRAIN_RESOLUTION,
        "test_resolutions": list(TEST_RESOLUTIONS),
        "train_per_theta": args.train_per_theta,
        "test_per_cell": args.test_per_cell,
        "train": train_info,
        "test_buckets": test_info,
        "n_unique_param_keys": len(used_param_keys),
        "n_unique_full_keys": len(used_full_keys),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Done. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
