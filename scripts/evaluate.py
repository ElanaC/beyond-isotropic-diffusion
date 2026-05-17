#!/usr/bin/env python3
"""FID + ResNet-32 accuracy/confidence on generated PNG folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance

import os
import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from ddpm.data import default_cifar10_batches_root
from ddpm.utils import resolve_device, tensor_to_uint8_nchw  # noqa: E402

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


class GeneratedImageDataset(Dataset):
    def __init__(self, image_dir: Path, max_images: int | None = None) -> None:
        paths = sorted(image_dir.glob("*.png"))
        if max_images is not None:
            paths = paths[:max_images]
        if not paths:
            raise FileNotFoundError(f"No PNGs in {image_dir}")
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def load_cifar10_test_tensors(root: Path, max_images: int) -> torch.Tensor:
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cifar10_pytorch import CIFAR10PickleDataset

    ds = CIFAR10PickleDataset(root, train=False)
    n = min(max_images, len(ds))
    return torch.stack([ds[i][0] for i in range(n)])


def load_resnet_cifar10(device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True
    )
    return model.eval().to(device)


def preprocess_for_resnet(x: torch.Tensor) -> torch.Tensor:
    x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    mean = torch.tensor(CIFAR_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def classifier_metrics(model, images, device, true_labels, batch_size):
    correct, total, conf_sum = 0, 0, 0.0
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].to(device)
        probs = F.softmax(model(preprocess_for_resnet(batch)), dim=1)
        conf, pred = probs.max(dim=1)
        conf_sum += conf.sum().item()
        if true_labels is not None:
            y = torch.from_numpy(true_labels[start : start + batch_size]).to(device)
            correct += (pred == y).sum().item()
        total += batch.shape[0]
    out = {"mean_confidence": conf_sum / max(total, 1)}
    if true_labels is not None:
        out["accuracy_vs_label"] = correct / max(total, 1)
    return out


def _fid_device(device: torch.device) -> torch.device:
    """Inception FID uses float64 internally; MPS does not support that."""
    if device.type == "mps":
        return torch.device("cpu")
    return device


@torch.no_grad()
def compute_fid(real, fake_loader, device, batch_size):
    fid_dev = _fid_device(device)
    if fid_dev != device:
        print(f"FID on {fid_dev} (MPS incompatible with FID float64)")
    fid = FrechetInceptionDistance(feature=2048).to(fid_dev)
    n_real = len(real)
    print(f"FID: processing {n_real} real test images (CPU, no progress bar — slow)...")
    for start in range(0, n_real, batch_size):
        batch = tensor_to_uint8_nchw(real[start : start + batch_size]).to(fid_dev)
        fid.update(batch, real=True)
        if (start // batch_size) % 20 == 0 and start > 0:
            print(f"  real: {start}/{n_real}")
    print("FID: processing generated images...")
    n_fake = 0
    for batch in fake_loader:
        fid.update(tensor_to_uint8_nchw(batch).to(fid_dev), real=False)
        n_fake += batch.shape[0]
        if n_fake % (batch_size * 20) == 0:
            print(f"  fake: {n_fake}")
    print("FID: computing score...")
    return float(fid.compute().cpu())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generated-dir", type=Path, required=True)
    p.add_argument("--labels-file", type=Path, default=None)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    device = resolve_device(args.device)
    data_root = args.data_root or default_cifar10_batches_root()
    gen_ds = GeneratedImageDataset(args.generated_dir, max_images=args.max_samples)
    gen_loader = DataLoader(gen_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    true_labels = np.load(args.labels_file)[: len(gen_ds)] if args.labels_file else None

    print(f"Evaluating {len(gen_ds)} images from {args.generated_dir}")
    real = load_cifar10_test_tensors(data_root, len(gen_ds))
    fid_score = compute_fid(real, gen_loader, device, args.batch_size)
    gen_images = torch.stack([gen_ds[i] for i in range(len(gen_ds))])
    classifier = load_resnet_cifar10(device)
    cls_metrics = classifier_metrics(classifier, gen_images, device, true_labels, args.batch_size)

    results = {"generated_dir": str(args.generated_dir), "num_images": len(gen_ds), "fid": fid_score, **cls_metrics}
    print(json.dumps(results, indent=2))
    out_path = args.output_json or (args.generated_dir / "metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
