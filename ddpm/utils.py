"""Shared helpers for training, sampling, and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    diffusion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "diffusion": diffusion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": vars(args),
    }
    torch.save(payload, path)


def save_image_tensor(path: Path, x: torch.Tensor) -> None:
    """Save CHW tensor in [-1, 1] as PNG."""
    arr = (
        ((x.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0)
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(arr).save(path)


def tensor_to_uint8_nchw(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float BCHW -> uint8 BCHW for metrics."""
    return ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
