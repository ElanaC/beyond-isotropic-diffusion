#!/usr/bin/env python3
"""Sample images from a trained baseline DDPM checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from ddpm.cuda_env import prepend_nvidia_libs  # noqa: E402

prepend_nvidia_libs()

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from ddpm import GaussianDiffusion, SmallUNet  # noqa: E402
from ddpm.paths import BASELINE_SAMPLES  # noqa: E402
from ddpm.utils import resolve_device, save_image_tensor  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from baseline DDPM")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=Path, default=BASELINE_SAMPLES)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    model = SmallUNet(base_channels=train_args.get("base_channels", 64)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diffusion = GaussianDiffusion(
        timesteps=train_args.get("timesteps", 1000),
        schedule=train_args.get("schedule", "linear"),
    ).to(device)
    diffusion.load_state_dict(ckpt["diffusion"])

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    pbar = tqdm(total=args.num_samples, desc="Sampling (baseline)")
    while generated < args.num_samples:
        n = min(args.batch_size, args.num_samples - generated)
        with torch.no_grad():
            samples = diffusion.sample(model, (n, 3, 32, 32), device)
        for i in range(n):
            save_image_tensor(out_dir / f"{generated + i:05d}.png", samples[i])
        generated += n
        pbar.update(n)
    pbar.close()
    print(f"Wrote {generated} images to {out_dir}")


if __name__ == "__main__":
    main()
