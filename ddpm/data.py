"""Dataset wiring for training scripts."""

from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

from ddpm.paths import CIFAR_BATCHES_DIR

if str(CIFAR_BATCHES_DIR) not in sys.path:
    sys.path.insert(0, str(CIFAR_BATCHES_DIR))

from cifar10_pytorch import CIFAR10PickleDataset  # noqa: E402


def default_cifar10_batches_root() -> Path:
    return CIFAR_BATCHES_DIR


class CIFAR10ImagesOnly(Dataset):
    """Returns images only (baseline ignores labels)."""

    def __init__(self, root: Path, train: bool = True) -> None:
        self._ds = CIFAR10PickleDataset(root, train=train)

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        x, _y = self._ds[idx]
        return x


def make_cifar10_loader(
    root: Path | None = None,
    train: bool = True,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool | None = None,
) -> DataLoader:
    if root is None:
        root = default_cifar10_batches_root()
    if shuffle is None:
        shuffle = train
    ds = CIFAR10ImagesOnly(root, train=train)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )


def make_cifar10_labeled_loader(
    root: Path | None = None,
    train: bool = True,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool | None = None,
) -> DataLoader:
    if root is None:
        root = default_cifar10_batches_root()
    if shuffle is None:
        shuffle = train
    ds = CIFAR10PickleDataset(root, train=train)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
