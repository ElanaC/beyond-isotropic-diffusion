#!/usr/bin/env python3
"""Sample from class-conditioned DDPM (default 10k = 1k per class)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from ddpm import ClassConditionalUNet, GaussianDiffusion
from ddpm.paths import CONDITIONAL_SAMPLES  # noqa: E402
from ddpm.utils import resolve_device, save_image_tensor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample class-conditioned DDPM")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=10_000)
    p.add_argument("--samples-per-class", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=Path, default=CONDITIONAL_SAMPLES)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_labels(num_samples: int, samples_per_class: int | None) -> np.ndarray:
    if samples_per_class is not None:
        return np.repeat(np.arange(10), samples_per_class)
    per_class = num_samples // 10
    labels = np.repeat(np.arange(10), per_class)
    if len(labels) < num_samples:
        labels = np.concatenate([labels, np.arange(num_samples - len(labels))])
    return labels


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    num_samples = 10 * args.samples_per_class if args.samples_per_class else args.num_samples
    labels = build_labels(num_samples, args.samples_per_class)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    model = ClassConditionalUNet(base_channels=train_args.get("base_channels", 64)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    diffusion = GaussianDiffusion(
        timesteps=train_args.get("timesteps", 1000),
        schedule=train_args.get("schedule", "linear"),
    ).to(device)
    diffusion.load_state_dict(ckpt["diffusion"])

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    pbar = tqdm(total=num_samples, desc="Sampling (conditional)")
    while idx < num_samples:
        n = min(args.batch_size, num_samples - idx)
        y = torch.from_numpy(labels[idx : idx + n]).long().to(device)
        with torch.no_grad():
            samples = diffusion.sample(model, (n, 3, 32, 32), device, y=y)
        for i in range(n):
            save_image_tensor(out_dir / f"{idx + i:05d}.png", samples[i])
        idx += n
        pbar.update(n)
    pbar.close()
    np.save(out_dir / "labels.npy", labels)
    print(f"Wrote {num_samples} images + labels.npy to {out_dir}")


if __name__ == "__main__":
    main()
