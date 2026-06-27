#!/usr/bin/env python3
"""Evaluate FNO on zero-shot (θ, resolution) test buckets and produce plots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from geometry import TEST_RESOLUTIONS, THETA_VALUES
from model_fno import FNO2d, prepare_input
from kink_utils import DEFAULT_KINK_RADIUS_FRAC, get_kink_mask_from_npz, precompute_kink_masks

QUAL_THETAS = (180, 90, 10)
KINK_VALIDATION_PATH = Path(__file__).resolve().parent / "results" / "validation" / "kink_validation.json"


def load_model(checkpoint_path: Path, device: torch.device) -> FNO2d:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = FNO2d(
        in_channels=3,
        out_channels=1,
        width=config.get("width", 32),
        modes=config.get("modes", 12),
        n_layers=config.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _per_sample_relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> list[float]:
    diff = (pred - target).reshape(pred.shape[0], -1)
    tgt = target.reshape(target.shape[0], -1)
    num = torch.linalg.vector_norm(diff, dim=1)
    den = torch.linalg.vector_norm(tgt, dim=1).clamp_min(eps)
    return (num / den).cpu().tolist()


def _per_sample_masked_relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> list[float]:
    if mask.dtype != torch.bool:
        mask = mask.bool()
    m = mask.unsqueeze(1)
    diff = (pred - target) * m
    tgt = target * m
    diff_flat = diff.reshape(pred.shape[0], -1)
    tgt_flat = tgt.reshape(target.shape[0], -1)
    num = torch.linalg.vector_norm(diff_flat, dim=1)
    den = torch.linalg.vector_norm(tgt_flat, dim=1)
    valid = mask.reshape(mask.shape[0], -1).any(dim=1)
    out: list[float] = []
    per = num / den.clamp_min(eps)
    for i in range(pred.shape[0]):
        out.append(float(per[i].item()) if bool(valid[i]) else 0.0)
    return out


@torch.no_grad()
def evaluate_bucket(
    model: FNO2d,
    npz_path: Path,
    device: torch.device,
    batch_size: int = 8,
    *,
    kink_radius_frac: float = DEFAULT_KINK_RADIUS_FRAC,
) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    resolution = int(data["occupancy"].shape[-1])
    occupancy = torch.from_numpy(data["occupancy"]).float().unsqueeze(1)
    travel_time = torch.from_numpy(data["travel_time"]).float().unsqueeze(1)
    indices = np.arange(len(data["sample_id"]))
    kink_masks = torch.from_numpy(
        precompute_kink_masks(data, indices, resolution, radius_frac=kink_radius_frac)
    )

    whole_errors: list[float] = []
    kink_errors: list[float] = []
    for start in range(0, len(occupancy), batch_size):
        occ = occupancy[start : start + batch_size].to(device)
        tgt = travel_time[start : start + batch_size].to(device)
        masks = kink_masks[start : start + batch_size].to(device)
        pred = model(prepare_input(occ))
        whole_errors.extend(_per_sample_relative_l2(pred, tgt))
        kink_errors.extend(_per_sample_masked_relative_l2(pred, tgt, masks))

    return {
        "mean_whole_domain_rel_l2": float(np.mean(whole_errors)),
        "std_whole_domain_rel_l2": float(np.std(whole_errors)),
        "mean_near_kink_rel_l2": float(np.mean(kink_errors)),
        "std_near_kink_rel_l2": float(np.std(kink_errors)),
        "mean_rel_l2": float(np.mean(whole_errors)),
        "std_rel_l2": float(np.std(whole_errors)),
        "n_samples": int(len(occupancy)),
        "per_sample_whole": whole_errors,
        "per_sample_kink": kink_errors,
    }


def save_error_csv(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "theta_deg",
                "kink_severity",
                "resolution",
                "whole_domain_relL2",
                "whole_domain_std",
                "near_kink_relL2",
                "near_kink_std",
                "n_samples",
            ]
        )
        for theta in THETA_VALUES:
            for resolution in TEST_RESOLUTIONS:
                key = f"theta_{int(theta)}_res_{resolution}"
                if key not in results:
                    continue
                row = results[key]
                writer.writerow(
                    [
                        theta,
                        180 - theta,
                        resolution,
                        row["mean_whole_domain_rel_l2"],
                        row["std_whole_domain_rel_l2"],
                        row["mean_near_kink_rel_l2"],
                        row["std_near_kink_rel_l2"],
                        row["n_samples"],
                    ]
                )


def plot_error_grid(
    results: dict,
    output_path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    severity = [180.0 - t for t in THETA_VALUES]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    mean_key = f"mean_{metric}"
    std_key = f"std_{metric}"

    for resolution in TEST_RESOLUTIONS:
        means = np.array([results[f"theta_{int(t)}_res_{resolution}"][mean_key] for t in THETA_VALUES])
        stds = np.array([results[f"theta_{int(t)}_res_{resolution}"][std_key] for t in THETA_VALUES])
        sev = np.array(severity)
        line, = ax.plot(sev, means, marker="o", linewidth=2, label=f"{resolution}×{resolution}")
        ax.fill_between(
            sev,
            means - stds,
            means + stds,
            alpha=0.2,
            color=line.get_color(),
        )

    ax.set_xlabel("Kink severity (180° − θ)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_qualitative_examples(
    model: FNO2d,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    *,
    kink_radius_frac: float = DEFAULT_KINK_RADIUS_FRAC,
) -> None:
    qual_dir = output_dir / "qualitative_512"
    qual_dir.mkdir(parents=True, exist_ok=True)
    resolution = 512

    for theta in QUAL_THETAS:
        npz_path = data_dir / "test" / f"theta_{int(theta)}" / f"res_{resolution}" / "samples.npz"
        if not npz_path.exists():
            print(f"Skipping qualitative example for θ={theta}: {npz_path} missing")
            continue

        data = np.load(npz_path, allow_pickle=True)
        occ_map = data["occupancy"][0]
        tgt = data["travel_time"][0]
        kink_mask = get_kink_mask_from_npz(data, 0, resolution, radius_frac=kink_radius_frac)
        occ = torch.from_numpy(occ_map).float().unsqueeze(0).unsqueeze(0).to(device)
        pred = model(prepare_input(occ)).squeeze().detach().cpu().numpy()
        err = np.abs(tgt - pred)

        vmin = float(min(tgt.min(), pred.min()))
        vmax = float(max(tgt.max(), pred.max()))

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        panels = [
            (occ_map, "Occupancy (input)", "gray", None, None, False),
            (tgt, "Ground-truth V (FMM)", "viridis", vmin, vmax, False),
            (pred, "FNO prediction V̂", "viridis", vmin, vmax, False),
            (err, "|V − V̂| (+ kink region)", "hot", None, None, True),
        ]
        for ax, (field, title, cmap, lo, hi, show_kink) in zip(axes, panels):
            kw = {"origin": "lower", "extent": [0, 1, 0, 1], "cmap": cmap}
            if lo is not None and hi is not None:
                kw["vmin"] = lo
                kw["vmax"] = hi
            im = ax.imshow(field, **kw)
            if show_kink and np.any(kink_mask):
                ax.contour(
                    kink_mask.astype(float),
                    levels=[0.5],
                    colors="cyan",
                    linewidths=1.5,
                    origin="lower",
                    extent=[0, 1, 0, 1],
                )
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"θ = {theta}°  (512×512, sample 0)")
        fig.tight_layout()
        out_path = qual_dir / f"theta_{int(theta)}_example.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def save_severity_vs_error_table(results: dict, output_path: Path) -> None:
    kink_path = KINK_VALIDATION_PATH
    if kink_path.exists():
        kink = json.loads(kink_path.read_text())
        mean_jump = kink.get("mean_gradient_jump", {})
    else:
        mean_jump = {str(int(t)): float("nan") for t in THETA_VALUES}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["theta_deg", "kink_severity", "mean_gradient_jump"]
        for res in TEST_RESOLUTIONS:
            header.extend(
                [
                    f"whole_domain_relL2_res{res}",
                    f"whole_domain_std_res{res}",
                    f"near_kink_relL2_res{res}",
                    f"near_kink_std_res{res}",
                ]
            )
        writer.writerow(header)
        for theta in THETA_VALUES:
            row = [
                theta,
                180 - theta,
                mean_jump.get(str(int(theta)), mean_jump.get(int(theta), "")),
            ]
            for res in TEST_RESOLUTIONS:
                key = f"theta_{int(theta)}_res_{res}"
                row.append(results[key]["mean_whole_domain_rel_l2"])
                row.append(results[key]["std_whole_domain_rel_l2"])
                row.append(results[key]["mean_near_kink_rel_l2"])
                row.append(results[key]["std_near_kink_rel_l2"])
            writer.writerow(row)


def _factual_theta_sweep_description(
    results: dict, resolution: int, metric: str, metric_label: str
) -> str:
    mean_key = f"mean_{metric}"
    means = {
        int(t): results[f"theta_{int(t)}_res_{resolution}"][mean_key] for t in THETA_VALUES
    }
    parts = [f"{metric_label} at {resolution}×{resolution}:"]
    for t in THETA_VALUES:
        parts.append(f"  θ={int(t)} (severity {int(180-t)}): {means[int(t)]:.5f}")
    t180, t10 = means[180], means[10]
    delta = t10 - t180
    parts.append(
        f"  θ=180 → θ=10: {t180:.5f} → {t10:.5f} (Δ = {delta:+.5f})"
    )
    return "\n".join(parts)


def save_train_vs_test_kink_csv(
    results: dict,
    output_path: Path,
    val_kink_epoch8: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "theta_deg",
                "test_near_kink_mean_res64",
                "test_near_kink_std_res64",
                "val_kink_epoch8_aggregate",
                "lambda_kink",
            ]
        )
        for theta in THETA_VALUES:
            key = f"theta_{int(theta)}_res_64"
            row = results[key]
            writer.writerow(
                [
                    theta,
                    row["mean_near_kink_rel_l2"],
                    row["std_near_kink_rel_l2"],
                    val_kink_epoch8,
                    100,
                ]
            )


def write_summary(
    results: dict,
    output_path: Path,
    checkpoint: Path,
    *,
    ckpt_epoch: int,
    lambda_kink: float,
) -> str:
    lines = [
        "Eikonal FNO — final evaluation summary",
        f"Checkpoint: {checkpoint.name} (epoch {ckpt_epoch}, lambda_kink={lambda_kink})",
        "Note: lambda_kink=100 is the reported configuration. A lambda sweep (175, 350, 700)",
        "was started but not completed in time; other lambda values were not fully evaluated",
        "and are not ruled out — this is a reporting limitation, not an experimental finding.",
        "",
        "=== Whole-domain relative L2 (mean ± std over 50 test samples) ===",
    ]
    header = f"{'theta':>6} {'sev':>4} " + " ".join(f"{'res'+str(r):>16}" for r in TEST_RESOLUTIONS)
    lines.append(header)
    for theta in THETA_VALUES:
        sev = int(180 - theta)
        cells = []
        for res in TEST_RESOLUTIONS:
            key = f"theta_{int(theta)}_res_{res}"
            m = results[key]["mean_whole_domain_rel_l2"]
            s = results[key]["std_whole_domain_rel_l2"]
            cells.append(f"{m:.4f}±{s:.4f}")
        lines.append(f"{int(theta):6d} {sev:4d} " + " ".join(f"{c:>16}" for c in cells))

    lines.append("")
    lines.append("=== Near-kink relative L2 (mean ± std over 50 test samples) ===")
    lines.append(header)
    for theta in THETA_VALUES:
        sev = int(180 - theta)
        cells = []
        for res in TEST_RESOLUTIONS:
            key = f"theta_{int(theta)}_res_{res}"
            m = results[key]["mean_near_kink_rel_l2"]
            s = results[key]["std_near_kink_rel_l2"]
            cells.append(f"{m:.4f}±{s:.4f}")
        lines.append(f"{int(theta):6d} {sev:4d} " + " ".join(f"{c:>16}" for c in cells))

    lines.append("")
    lines.append("=== Per-resolution sweep θ=180 → θ=10 (factual) ===")
    for res in TEST_RESOLUTIONS:
        lines.append("")
        lines.append(_factual_theta_sweep_description(
            results, res, "whole_domain_rel_l2", "Whole-domain rel-L2"
        ))
        lines.append(_factual_theta_sweep_description(
            results, res, "near_kink_rel_l2", "Near-kink rel-L2"
        ))

    text = "\n".join(lines) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints" / "fno_best_val_kink.pt",
    )
    parser.add_argument(
        "--training-log",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "training" / "loss_log.csv",
    )
    parser.add_argument("--lambda-kink", type=float, default=100.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_epoch = int(ckpt.get("best_epoch", ckpt.get("history", [{}])[-1].get("epoch", "?")))

    val_kink_epoch8 = float("nan")
    if args.training_log.exists():
        with args.training_log.open() as f:
            for row in csv.DictReader(f):
                if int(row["epoch"]) == ckpt_epoch:
                    val_kink_epoch8 = float(row["val_kink_loss"])
                    break

    results = {}
    for theta in THETA_VALUES:
        for resolution in TEST_RESOLUTIONS:
            npz_path = (
                args.data_dir
                / "test"
                / f"theta_{int(theta)}"
                / f"res_{resolution}"
                / "samples.npz"
            )
            bucket_key = f"theta_{int(theta)}_res_{resolution}"
            if not npz_path.exists():
                print(f"Warning: missing {npz_path}")
                continue
            metrics = evaluate_bucket(model, npz_path, device, args.batch_size)
            results[bucket_key] = {
                **metrics,
                "theta_deg": theta,
                "resolution": resolution,
            }
            print(
                f"{bucket_key}: whole={metrics['mean_whole_domain_rel_l2']:.5f} "
                f"kink={metrics['mean_near_kink_rel_l2']:.5f} "
                f"(n={metrics['n_samples']})"
            )

    json_results = {
        k: {kk: vv for kk, vv in v.items() if not str(kk).startswith("per_sample")}
        for k, v in results.items()
    }
    json_path = args.output_dir / "error_grid.json"
    json_path.write_text(json.dumps(json_results, indent=2))

    csv_path = args.output_dir / "error_grid.csv"
    save_error_csv(results, csv_path)

    plot_error_grid(
        results,
        args.output_dir / "error_vs_kink_severity_whole_domain.png",
        metric="whole_domain_rel_l2",
        ylabel="Mean whole-domain relative L2 error",
        title="FNO zero-shot whole-domain error vs. corner sharpness",
    )
    plot_error_grid(
        results,
        args.output_dir / "error_vs_kink_severity_near_kink.png",
        metric="near_kink_rel_l2",
        ylabel="Mean near-kink relative L2 error",
        title="FNO zero-shot near-kink error vs. corner sharpness",
    )
    # legacy alias
    plot_error_grid(
        results,
        args.output_dir / "error_vs_kink_severity.png",
        metric="whole_domain_rel_l2",
        ylabel="Mean relative L2 error",
        title="FNO zero-shot error vs. corner sharpness and resolution",
    )

    severity_table = args.output_dir / "severity_vs_error_table.csv"
    save_severity_vs_error_table(results, severity_table)

    save_qualitative_examples(model, args.data_dir, args.output_dir, device)

    cross_path = args.output_dir / "train_vs_test_kink_error.csv"
    if not np.isnan(val_kink_epoch8):
        save_train_vs_test_kink_csv(results, cross_path, val_kink_epoch8)
        print(f"Saved train vs test kink cross-check to {cross_path}")
        print(f"Val kink loss at epoch {ckpt_epoch} (aggregate, from training log): {val_kink_epoch8:.6f}")

    summary_path = args.output_dir / "summary.txt"
    summary_text = write_summary(
        results,
        summary_path,
        args.checkpoint,
        ckpt_epoch=ckpt_epoch,
        lambda_kink=args.lambda_kink,
    )
    print(summary_text)
    print(f"Saved error grid CSV to {csv_path}")
    print(f"Saved whole-domain plot to {args.output_dir / 'error_vs_kink_severity_whole_domain.png'}")
    print(f"Saved near-kink plot to {args.output_dir / 'error_vs_kink_severity_near_kink.png'}")
    print(f"Saved severity vs error table to {severity_table}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
