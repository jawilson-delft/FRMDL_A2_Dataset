#!/usr/bin/env python3
"""Train FNO2d once on the combined 64×64 training set (all θ mixed)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from geometry import TRAIN_RESOLUTION
from kink_utils import (
    DEFAULT_KINK_RADIUS_FRAC,
    DEFAULT_LAMBDA_KINK,
    mean_kink_pixel_fraction,
    precompute_kink_masks,
)
from model_fno import (
    DEFAULT_MODES,
    DEFAULT_N_LAYERS,
    DEFAULT_WIDTH,
    FNO2d,
    REDUCED_MODES,
    REDUCED_N_LAYERS,
    REDUCED_WIDTH,
    combined_kink_loss,
    count_parameters,
    prepare_input,
)
from split_utils import DEFAULT_SPLIT_SEED, DEFAULT_VAL_FRACTION, load_or_create_split

# Default exponential LR decay (baseline lambda=100 run): gamma every N epochs.
LR_INITIAL = 1e-3
DEFAULT_LR_DECAY_GAMMA = 0.95
DEFAULT_LR_DECAY_EVERY_EPOCHS = 5
LR_MILESTONE_EPOCHS = (100, 200, 300, 400, 500)

DEFAULT_WEIGHT_DECAY = 0.0
CAPACITY_WEIGHT_DECAY = 1e-4

LOSS_CSV_FIELDS = [
    "epoch",
    "train_whole_loss",
    "train_kink_loss",
    "val_whole_loss",
    "val_kink_loss",
    "train_total_loss",
    "val_total_loss",
    # legacy aliases for backward compatibility
    "train_rel_l2",
    "val_rel_l2",
]


class EikonalTrainDataset(Dataset):
    def __init__(
        self,
        npz_path: Path,
        indices: np.ndarray | None = None,
        *,
        kink_radius_frac: float = DEFAULT_KINK_RADIUS_FRAC,
    ):
        data = np.load(npz_path)
        occ = data["occupancy"]
        tt = data["travel_time"]
        if indices is None:
            indices = np.arange(len(occ))
        else:
            occ = occ[indices]
            tt = tt[indices]
        self.occupancy = torch.from_numpy(occ).float().unsqueeze(1)
        self.travel_time = torch.from_numpy(tt).float().unsqueeze(1)
        masks = precompute_kink_masks(
            data, indices, TRAIN_RESOLUTION, radius_frac=kink_radius_frac
        )
        self.kink_masks = torch.from_numpy(masks)

    def __len__(self) -> int:
        return self.occupancy.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.occupancy[idx], self.travel_time[idx], self.kink_masks[idx]


def _save_loss_csv(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOSS_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def _save_loss_plot(history: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, whole_key, kink_key, title in (
        (axes[0], "train_whole_loss", "train_kink_loss", "Train"),
        (axes[1], "val_whole_loss", "val_kink_loss", "Validation"),
    ):
        ax.plot(epochs, [h[whole_key] for h in history], linewidth=1.5, label="whole domain")
        ax.plot(epochs, [h[kink_key] for h in history], linewidth=1.5, label="near kink")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Relative L2 loss")
        ax.set_title(f"{title} loss components (kink-weighted training)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _note_plateau(history: list[dict], window: int = 20, rtol: float = 0.02) -> str | None:
    if len(history) < window + 1:
        return None
    recent = [h["train_kink_loss"] for h in history[-window:]]
    spread = (max(recent) - min(recent)) / max(min(recent), 1e-12)
    if spread < rtol:
        epoch = history[-window]["epoch"]
        return (
            f"Train kink loss changed by <{rtol * 100:.0f}% over epochs {epoch}–"
            f"{history[-1]['epoch']} (possible plateau)."
        )
    return None


@torch.no_grad()
def _eval_loader(
    model: FNO2d,
    loader: DataLoader,
    device: torch.device,
    lambda_kink: float,
) -> tuple[float, float, float]:
    model.eval()
    whole_total = kink_total = 0.0
    n_batches = 0
    for occ, target, kink_mask in loader:
        occ = occ.to(device)
        target = target.to(device)
        kink_mask = kink_mask.to(device)
        pred = model(prepare_input(occ))
        _, whole, kink = combined_kink_loss(pred, target, kink_mask, lambda_kink)
        whole_total += whole.item()
        kink_total += kink.item()
        n_batches += 1
    avg_whole = whole_total / max(n_batches, 1)
    avg_kink = kink_total / max(n_batches, 1)
    avg_total = avg_whole + lambda_kink * avg_kink
    return avg_total, avg_whole, avg_kink


def _lr_after_epoch(
    epoch: int,
    initial_lr: float,
    gamma: float,
    decay_every_epochs: int,
    steps_per_epoch: int,
) -> float:
    """LR after completing `epoch` (1-indexed), matching step-based decay in the train loop."""
    decay_every_steps = max(steps_per_epoch * decay_every_epochs, 1)
    n_decays = (epoch * steps_per_epoch) // decay_every_steps
    return initial_lr * (gamma ** n_decays)


def _print_lr_schedule_preview(
    initial_lr: float,
    gamma: float,
    decay_every_epochs: int,
    steps_per_epoch: int,
    max_epochs: int,
) -> None:
    decay_every_steps = max(steps_per_epoch * decay_every_epochs, 1)
    n_decay_total = (max_epochs * steps_per_epoch) // decay_every_steps
    final_lr = initial_lr * (gamma ** n_decay_total)
    print("--- Learning rate schedule ---")
    print(f"  Initial LR:           {initial_lr:.6e}")
    print(f"  Decay gamma:          {gamma}")
    print(f"  Decay every:          {decay_every_steps} steps "
          f"(~{decay_every_epochs} epochs, {steps_per_epoch} steps/epoch)")
    print(f"  Expected decay events over {max_epochs} epochs: {n_decay_total}")
    print(f"  Estimated final LR:   {final_lr:.6e}")
    milestones = [e for e in LR_MILESTONE_EPOCHS if e <= max_epochs]
    if milestones:
        print("  LR after completing milestone epochs (computed):")
        for ep in milestones:
            lr_ep = _lr_after_epoch(ep, initial_lr, gamma, decay_every_epochs, steps_per_epoch)
            print(f"    epoch {ep:3d}: {lr_ep:.6e}  ({lr_ep / initial_lr:.4f}× initial)")


def _late_val_kink_improvement(history: list[dict], after_epoch: int = 20) -> dict:
    """Check for val_kink below epoch-`after_epoch` value in later epochs."""
    if len(history) <= after_epoch:
        return {"late_improvement_vs_ep20": False, "late_min_epoch": None, "late_min": None}
    ep_ref = float(history[after_epoch - 1]["val_kink_loss"])
    late = history[after_epoch:]
    late_vals = [float(h["val_kink_loss"]) for h in late]
    late_min = min(late_vals)
    late_min_epoch = late[late_vals.index(late_min)]["epoch"]
    return {
        "late_improvement_vs_ep20": late_min < ep_ref,
        "late_min_epoch": late_min_epoch,
        "late_min": late_min,
        "ep20_val_kink": ep_ref,
    }


def _overfit_summary(best_epoch: int, final_epoch: int, metric: str) -> str:
    gap = final_epoch - best_epoch
    if gap <= 5:
        return (
            f"Best validation {metric} epoch ({best_epoch}) is within {gap} epoch(s) of the "
            f"final epoch ({final_epoch})."
        )
    if gap <= 20:
        return (
            f"Best validation {metric} epoch ({best_epoch}) is {gap} epochs before the final "
            f"epoch ({final_epoch}) — mild regression after best checkpoint."
        )
    return (
        f"Best validation {metric} epoch ({best_epoch}) is {gap} epochs before the final "
        f"epoch ({final_epoch}) — validation likely worsened well before training ended."
    )


def _save_checkpoint(path: Path, model, args, history, **extra) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "history": history,
        **extra,
    }
    torch.save(payload, path)


def train(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_path = args.data_dir / "train" / "train_64.npz"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. Run generate_dataset.py first."
        )

    split_path = args.data_dir / "train" / "val_split.npz"
    train_idx, val_idx = load_or_create_split(
        data_path,
        split_path,
        seed=args.split_seed,
        val_fraction=args.val_fraction,
    )

    train_ds = EikonalTrainDataset(
        data_path, train_idx, kink_radius_frac=args.kink_radius_frac
    )
    val_ds = EikonalTrainDataset(
        data_path, val_idx, kink_radius_frac=args.kink_radius_frac
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    raw = np.load(data_path)
    mean_kink_frac = mean_kink_pixel_fraction(
        train_ds.kink_masks.numpy(), raw["occupancy"][train_idx]
    )
    lambda_kink = args.lambda_kink
    effective_kink_weight = lambda_kink * mean_kink_frac

    print(f"Device: {device}")
    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)} (64×64)")
    print(f"Split file: {split_path} (seed={args.split_seed}, val_fraction={args.val_fraction})")
    print("--- Kink-weighted loss ---")
    print(f"  lambda_kink:              {lambda_kink}")
    print(f"  kink_radius_frac:         {args.kink_radius_frac} "
          f"({args.kink_radius_frac * np.sqrt(2):.4f} abs, {args.kink_radius_frac * 100:.1f}% diagonal)")
    print(f"  mean kink free-pixel frac: {mean_kink_frac:.5f} (train set)")
    print(f"  loss = whole_relL2 + lambda_kink * near_kink_relL2")
    print(f"  effective kink term weight ≈ lambda * frac = {effective_kink_weight:.3f} "
          f"(vs 1.0 for whole-domain term at unit loss)")

    model = FNO2d(
        in_channels=3,
        out_channels=1,
        width=args.width,
        modes=args.modes,
        n_layers=args.n_layers,
    ).to(device)

    baseline_ref = FNO2d(
        in_channels=3,
        out_channels=1,
        width=DEFAULT_WIDTH,
        modes=DEFAULT_MODES,
        n_layers=DEFAULT_N_LAYERS,
    )
    n_params = count_parameters(model)
    n_params_baseline = count_parameters(baseline_ref)
    print("--- Model capacity ---")
    print(f"  width={args.width}  modes={args.modes}  n_layers={args.n_layers}")
    print(f"  trainable parameters (this run): {n_params:,}")
    print(f"  trainable parameters (baseline): {n_params_baseline:,}  "
          f"(width={DEFAULT_WIDTH}, modes={DEFAULT_MODES}, n_layers={DEFAULT_N_LAYERS})")
    if n_params_baseline > 0:
        print(f"  capacity ratio vs baseline: {n_params / n_params_baseline:.3f}")

    initial_lr = args.lr if args.lr != 1e-3 else LR_INITIAL
    lr_gamma = args.lr_decay_gamma
    lr_decay_every_epochs = args.lr_decay_every_epochs
    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=initial_lr, weight_decay=args.weight_decay
        )
        optimizer_name = "AdamW"
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
        optimizer_name = "Adam"
    print("--- Optimizer ---")
    print(f"  {optimizer_name}  lr={initial_lr:.6e}  weight_decay={args.weight_decay:.6e}")

    steps_per_epoch = len(train_loader)
    lr_decay_every_steps = max(steps_per_epoch * lr_decay_every_epochs, 1)

    _print_lr_schedule_preview(
        initial_lr, lr_gamma, lr_decay_every_epochs, steps_per_epoch, args.epochs
    )
    if args.early_stop_patience > 0:
        print("--- Early stopping ---")
        print(f"  Monitor:              val_kink_loss")
        print(f"  Patience:             {args.early_stop_patience} epochs without improvement")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_total = float("inf")
    best_total_epoch = 0
    best_total_state: dict | None = None
    best_kink = float("inf")
    best_kink_epoch = 0
    best_kink_state: dict | None = None

    history: list[dict] = []
    plateau_note: str | None = None
    global_step = 0
    n_lr_decays = 0
    epochs_without_kink_improvement = 0
    early_stopped = False
    stop_epoch: int | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_whole_sum = train_kink_sum = train_total_sum = 0.0
        n_batches = 0
        for occ, target, kink_mask in train_loader:
            occ = occ.to(device)
            target = target.to(device)
            kink_mask = kink_mask.to(device)
            x = prepare_input(occ)
            pred = model(x)
            total_loss, whole_loss, kink_loss = combined_kink_loss(
                pred, target, kink_mask, lambda_kink
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % lr_decay_every_steps == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] *= lr_gamma
                n_lr_decays += 1

            train_whole_sum += whole_loss.item()
            train_kink_sum += kink_loss.item()
            train_total_sum += total_loss.item()
            n_batches += 1

        avg_train_whole = train_whole_sum / max(n_batches, 1)
        avg_train_kink = train_kink_sum / max(n_batches, 1)
        avg_train_total = train_total_sum / max(n_batches, 1)
        avg_val_total, avg_val_whole, avg_val_kink = _eval_loader(
            model, val_loader, device, lambda_kink
        )

        row = {
            "epoch": epoch,
            "train_whole_loss": avg_train_whole,
            "train_kink_loss": avg_train_kink,
            "val_whole_loss": avg_val_whole,
            "val_kink_loss": avg_val_kink,
            "train_total_loss": avg_train_total,
            "val_total_loss": avg_val_total,
            "train_rel_l2": avg_train_whole,
            "val_rel_l2": avg_val_whole,
        }
        history.append(row)

        if avg_val_total < best_total:
            best_total = avg_val_total
            best_total_epoch = epoch
            best_total_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            _save_checkpoint(
                args.checkpoint_dir / "fno_best_val.pt",
                model,
                args,
                history,
                best_val_loss=best_total,
                best_epoch=best_total_epoch,
                best_metric="combined_total",
            )

        if avg_val_kink < best_kink:
            best_kink = avg_val_kink
            best_kink_epoch = epoch
            best_kink_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_kink_improvement = 0
            _save_checkpoint(
                args.checkpoint_dir / "fno_best_val_kink.pt",
                model,
                args,
                history,
                best_val_kink_loss=best_kink,
                best_epoch=best_kink_epoch,
                best_metric="val_kink_only",
            )
        elif args.early_stop_patience > 0:
            epochs_without_kink_improvement += 1

        if (
            args.early_stop_patience > 0
            and epochs_without_kink_improvement >= args.early_stop_patience
        ):
            early_stopped = True
            stop_epoch = epoch
            print(
                f"Early stopping at epoch {epoch}: val_kink has not improved for "
                f"{args.early_stop_patience} epochs (best={best_kink:.6f} at epoch {best_kink_epoch})",
                flush=True,
            )
            break

        note = _note_plateau(history)
        if note and plateau_note is None:
            plateau_note = note
            print(f"NOTE: {note}")

        if epoch % max(args.epochs // 10, 1) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs}  "
                f"train whole={avg_train_whole:.4f} kink={avg_train_kink:.4f} total={avg_train_total:.4f}  "
                f"val whole={avg_val_whole:.4f} kink={avg_val_kink:.4f} total={avg_val_total:.4f}",
                flush=True,
            )

    ckpt_path = args.checkpoint_dir / args.checkpoint_name
    _save_checkpoint(
        ckpt_path,
        model,
        args,
        history,
        final_train_loss=history[-1]["train_total_loss"],
        final_val_loss=history[-1]["val_total_loss"],
        best_val_loss=best_total,
        best_epoch=best_total_epoch,
        best_val_kink_loss=best_kink,
        best_kink_epoch=best_kink_epoch,
    )

    if best_total_state is not None:
        torch.save(
            {
                "model_state_dict": best_total_state,
                "config": vars(args),
                "history": history,
                "best_val_loss": best_total,
                "best_epoch": best_total_epoch,
                "best_metric": "combined_total",
            },
            args.checkpoint_dir / "fno_best_val.pt",
        )
    if best_kink_state is not None:
        torch.save(
            {
                "model_state_dict": best_kink_state,
                "config": vars(args),
                "history": history,
                "best_val_kink_loss": best_kink,
                "best_epoch": best_kink_epoch,
                "best_metric": "val_kink_only",
            },
            args.checkpoint_dir / "fno_best_val_kink.pt",
        )

    if args.run_label:
        csv_path = args.results_dir / "loss_log.csv"
        plot_path = args.results_dir / "loss_curve.png"
    else:
        csv_path = args.results_dir / "training" / "loss_log.csv"
        plot_path = args.results_dir / "training" / "loss_curve.png"
    _save_loss_csv(history, csv_path)
    _save_loss_plot(history, plot_path)

    metrics_path = args.checkpoint_dir / f"{Path(args.checkpoint_name).stem}_history.json"
    metrics_path.write_text(json.dumps(history, indent=2))

    final_epoch = history[-1]["epoch"]
    late_kink = _late_val_kink_improvement(history)
    final = history[-1]
    summary_total = _overfit_summary(best_total_epoch, final_epoch, "combined total")
    summary_kink = _overfit_summary(best_kink_epoch, final_epoch, "kink-only")
    print(f"Saved final checkpoint to {ckpt_path}")
    print(f"Saved best combined-val checkpoint to {args.checkpoint_dir / 'fno_best_val.pt'}")
    if best_kink_epoch != best_total_epoch:
        print(
            f"Saved best kink-val checkpoint to {args.checkpoint_dir / 'fno_best_val_kink.pt'} "
            f"(epoch {best_kink_epoch}, differs from combined-best epoch {best_total_epoch})"
        )
    else:
        print(
            f"Best combined and best kink checkpoints coincide at epoch {best_total_epoch}"
        )
    print(f"Saved loss log to {csv_path}")
    print(f"Saved loss curve to {plot_path}")
    print("--- Training summary ---")
    print(f"lambda_kink used: {lambda_kink}  (effective weight ≈ {effective_kink_weight:.3f})")
    print(f"Final train whole / kink / total:  "
          f"{final['train_whole_loss']:.6f} / {final['train_kink_loss']:.6f} / {final['train_total_loss']:.6f}")
    print(f"Final val whole / kink / total:    "
          f"{final['val_whole_loss']:.6f} / {final['val_kink_loss']:.6f} / {final['val_total_loss']:.6f}")
    print(f"Best val combined: {best_total:.6f} (epoch {best_total_epoch})")
    print(f"Best val kink:     {best_kink:.6f} (epoch {best_kink_epoch})")
    if args.early_stop_patience > 0:
        if early_stopped:
            print(f"Early stopping:    triggered at epoch {stop_epoch}")
        else:
            print(f"Early stopping:    not triggered (completed {final_epoch}/{args.epochs} epochs)")
    if late_kink["late_improvement_vs_ep20"]:
        print(
            f"Late val_kink dip: yes — min {late_kink['late_min']:.6f} at epoch "
            f"{late_kink['late_min_epoch']} (below epoch-20 value {late_kink['ep20_val_kink']:.6f})"
        )
    else:
        print(
            f"Late val_kink dip: no — no epoch after 20 with val_kink below "
            f"epoch-20 value ({late_kink.get('ep20_val_kink', float('nan')):.6f})"
        )
    ep1_kink = history[0]["train_kink_loss"]
    epN_kink = history[-1]["train_kink_loss"]
    print(f"Train kink loss epoch 1 → {final_epoch}: {ep1_kink:.6f} → {epN_kink:.6f} "
          f"({'decreasing' if epN_kink < ep1_kink else 'not decreasing'})")
    final_lr = optimizer.param_groups[0]["lr"]
    print(f"Final learning rate: {final_lr:.6e}  ({n_lr_decays} decay events)")
    print(summary_total)
    print(summary_kink)
    if plateau_note:
        print(f"Plateau note: {plateau_note}")

    return {
        "checkpoint": str(ckpt_path),
        "best_val_checkpoint": str(args.checkpoint_dir / "fno_best_val.pt"),
        "best_kink_checkpoint": str(args.checkpoint_dir / "fno_best_val_kink.pt"),
        "lambda_kink": lambda_kink,
        "effective_kink_weight": effective_kink_weight,
        "history": history,
        "best_val_kink": best_kink,
        "best_kink_epoch": best_kink_epoch,
        "early_stopped": early_stopped,
        "stop_epoch": stop_epoch,
        "final_epoch": final_epoch,
        "late_kink": late_kink,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Final-epoch checkpoint filename (default: fno_final_200ep.pt).",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lambda-kink", type=float, default=DEFAULT_LAMBDA_KINK)
    parser.add_argument("--kink-radius-frac", type=float, default=DEFAULT_KINK_RADIUS_FRAC)
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Label for sweep runs (e.g. lambda_175) -> results/training/<label>/ and checkpoints/<label>/",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=DEFAULT_LR_DECAY_GAMMA,
        help="Multiply LR by this factor each decay event.",
    )
    parser.add_argument(
        "--lr-decay-every-epochs",
        type=int,
        default=DEFAULT_LR_DECAY_EVERY_EPOCHS,
        help="Apply LR decay every this many epochs (via optimizer steps).",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if val_kink_loss does not improve for this many epochs (0=disabled).",
    )
    parser.add_argument(
        "--reduced-capacity",
        action="store_true",
        help=(
            f"Use reduced FNO capacity (width {REDUCED_WIDTH}, modes {REDUCED_MODES}, "
            f"n_layers {REDUCED_N_LAYERS})."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Optimizer weight decay (uses AdamW when > 0, else Adam).",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Short run for smoke test.")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 5
        args.batch_size = 8
    if args.reduced_capacity:
        args.width = REDUCED_WIDTH
        args.modes = REDUCED_MODES
        args.n_layers = REDUCED_N_LAYERS
    if args.run_label:
        base = Path(__file__).resolve().parent
        args.results_dir = base / "results" / "training" / args.run_label
        args.checkpoint_dir = base / "checkpoints" / args.run_label
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_name is None:
        args.checkpoint_name = "fno_eikonal.pt" if args.quick else "fno_final_200ep.pt"
    args.checkpoint_name = Path(args.checkpoint_name).name
    train(args)


if __name__ == "__main__":
    main()
