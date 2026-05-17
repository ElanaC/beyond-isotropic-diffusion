"""CIFAR-10 PyTorch Dataset loaded from official pickle batches."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset


def unpickle(path: Path) -> dict:
    with open(path, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


def _load_cifar10_tensors(
    root: Path, train: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns images float32 in [-1, 1], shape (N, 3, 32, 32), and labels int64 (N,).
    """
    root = root.resolve()
    if train:
        chunks = []
        labels_parts = []
        for i in range(1, 6):
            batch = unpickle(root / f"data_batch_{i}")
            chunks.append(torch.from_numpy(batch[b"data"]))
            labels_parts.extend(batch[b"labels"])
        data = torch.cat(chunks, dim=0)
        labels = torch.tensor(labels_parts, dtype=torch.long)
    else:
        batch = unpickle(root / "test_batch")
        data = torch.from_numpy(batch[b"data"])
        labels = torch.tensor(batch[b"labels"], dtype=torch.long)

    # uint8 (N, 3072): R block, G block, B block — reshape to (N, 3, 32, 32)
    images = data.reshape(-1, 3, 32, 32).to(torch.float32)
    images = images / 255.0 * 2.0 - 1.0
    return images, labels


class CIFAR10PickleDataset(Dataset):
    """
    Loads CIFAR-10 from the official ``cifar-10-batches-py`` directory.

    ``root`` should be the folder containing ``data_batch_*``, ``test_batch``,
    and ``batches.meta`` (not the parent ``CIFAR10_dataset`` unless files live there).
    """

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.train = train
        self.transform = transform
        self.images, self.labels = _load_cifar10_tensors(self.root, train)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx]
        y = self.labels[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def default_cifar10_batches_root() -> Path:
    """Folder that contains ``data_batch_*`` — same directory as this file when placed in the official extract."""
    return Path(__file__).resolve().parent


if __name__ == "__main__":
    root = default_cifar10_batches_root()
    ds = CIFAR10PickleDataset(root, train=True)
    x, y = ds[0]
    print("root:", root)
    print("len(train):", len(ds), "expected 50000")
    print("x shape, dtype, min, max:", x.shape, x.dtype, x.min().item(), x.max().item())
    print("y:", y.item())
