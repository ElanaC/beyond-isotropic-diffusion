#!/usr/bin/env python3
"""Train class-conditioned DDPM ε_θ(x_t, t, c) on 50k CIFAR-10 train images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from ddpm import ClassConditionalUNet, GaussianDiffusion
from ddpm.data import default_cifar10_batches_root, make_cifar10_labeled_loader  # noqa: E402
from ddpm.paths import CONDITIONAL_DIR  # noqa: E402
from ddpm.utils import resolve_device, save_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train class-conditioned DDPM")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=CONDITIONAL_DIR)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    data_root = args.data_root or default_cifar10_batches_root()
    out = args.output_dir
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = make_cifar10_labeled_loader(
        root=data_root,
        train=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = ClassConditionalUNet(base_channels=args.base_channels).to(device)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, schedule=args.schedule).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        diffusion.load_state_dict(ckpt["diffusion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["data_root"] = str(data_root)
    config["model_type"] = "conditional"
    with open(out / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = diffusion.training_loss(model, x, y=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            n_batches += 1
            epoch_loss += loss.item()
            if global_step % args.log_every == 0:
                print(f"epoch {epoch} step {global_step} loss {loss.item():.4f}")
            if args.max_batches is not None and n_batches >= args.max_batches:
                break
        print(f"=== epoch {epoch} | avg loss {epoch_loss / max(n_batches, 1):.4f} ===")
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(path, model, diffusion, optimizer, epoch, global_step, args)
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
