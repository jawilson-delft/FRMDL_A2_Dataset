#!/usr/bin/env python3
"""Train a small CNN baseline on the combined 64×64 training set (whole-domain rel-L2 only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model_cnn import CNN2d, count_parameters
from model_fno import prepare_input, relative_l2_loss
from split_utils import DEFAULT_SPLIT_SEED, DEFAULT_VAL_FRACTION, load_or_create_split
from train import (
    DEFAULT_LR_DECAY_EVERY_EPOCHS,
    DEFAULT_LR_DECAY_GAMMA,
    EikonalTrainDataset,
    LR_INITIAL,
    _print_lr_schedule_preview,
)


@torch.no_grad()
def _eval_whole_loader(
    model: CNN2d,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    for occ, target, _kink_mask in loader:
        occ = occ.to(device)
        target = target.to(device)
        pred = model(prepare_input(occ))
        total += relative_l2_loss(pred, target).item()
        n_batches += 1
    return total / max(n_batches, 1)


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

    train_ds = EikonalTrainDataset(data_path, train_idx)
    val_ds = EikonalTrainDataset(data_path, val_idx)
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

    print(f"Device: {device}")
    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)} (64×64)")
    print("Loss: whole-domain relative L2 only (no kink term)")

    model = CNN2d(in_channels=3, out_channels=1, width=args.width).to(device)
    n_params = count_parameters(model)
    print(f"CNN trainable parameters: {n_params:,}")

    initial_lr = args.lr if args.lr != 1e-3 else LR_INITIAL
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    steps_per_epoch = len(train_loader)
    lr_decay_every_steps = max(steps_per_epoch * args.lr_decay_every_epochs, 1)
    _print_lr_schedule_preview(
        initial_lr,
        args.lr_decay_gamma,
        args.lr_decay_every_epochs,
        steps_per_epoch,
        args.epochs,
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = 0
    best_state: dict | None = None
    history: list[dict] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = 0.0
        n_batches = 0
        for occ, target, _kink_mask in train_loader:
            occ = occ.to(device)
            target = target.to(device)
            pred = model(prepare_input(occ))
            loss = relative_l2_loss(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            if global_step % lr_decay_every_steps == 0:
                for pg in optimizer.param_groups:
                    pg["lr"] *= args.lr_decay_gamma

            train_sum += loss.item()
            n_batches += 1

        avg_train = train_sum / max(n_batches, 1)
        avg_val = _eval_whole_loader(model, val_loader, device)
        history.append(
            {"epoch": epoch, "train_whole_loss": avg_train, "val_whole_loss": avg_val}
        )

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(args.epochs // 10, 1) == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs}  "
                f"train whole={avg_train:.4f}  val whole={avg_val:.4f}",
                flush=True,
            )

    ckpt_path = args.checkpoint_dir / args.checkpoint_name
    payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            **vars(args),
            "model_type": "cnn",
            "width": args.width,
        },
        "history": history,
        "best_val_whole_loss": best_val,
        "best_epoch": best_epoch,
    }
    torch.save(payload, ckpt_path)

    if best_state is not None:
        best_path = args.checkpoint_dir / "cnn_best_val.pt"
        torch.save(
            {
                "model_state_dict": best_state,
                "config": payload["config"],
                "history": history,
                "best_val_whole_loss": best_val,
                "best_epoch": best_epoch,
                "best_metric": "val_whole_only",
            },
            best_path,
        )
        print(f"Saved best checkpoint to {best_path} (epoch {best_epoch}, val={best_val:.6f})")

    hist_path = args.checkpoint_dir / "cnn_history.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"Saved final checkpoint to {ckpt_path}")
    return {"checkpoint": str(ckpt_path), "best_epoch": best_epoch, "best_val": best_val}


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
        default=Path(__file__).resolve().parent / "checkpoints" / "cnn_baseline",
    )
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=DEFAULT_LR_DECAY_GAMMA,
    )
    parser.add_argument(
        "--lr-decay-every-epochs",
        type=int,
        default=DEFAULT_LR_DECAY_EVERY_EPOCHS,
    )
    parser.add_argument("--checkpoint-name", type=str, default="cnn_final_200ep.pt")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.epochs = 5
        args.batch_size = 8
    train(args)


if __name__ == "__main__":
    main()
