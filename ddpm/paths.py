"""Repository paths — all scripts run from repo root (CS229/)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Official CIFAR-10 pickle batches (data_batch_1 … test_batch)
CIFAR_BATCHES_DIR = REPO_ROOT / "CIFAR10_dataset" / "cifar-10-batches-py"

# Training / sampling outputs
OUTPUTS_DIR = REPO_ROOT / "outputs"
BASELINE_DIR = OUTPUTS_DIR / "baseline"
CONDITIONAL_DIR = OUTPUTS_DIR / "conditional"

BASELINE_CHECKPOINTS = BASELINE_DIR / "checkpoints"
BASELINE_SAMPLES = BASELINE_DIR / "samples"
CONDITIONAL_CHECKPOINTS = CONDITIONAL_DIR / "checkpoints"
CONDITIONAL_SAMPLES = CONDITIONAL_DIR / "samples"


def setup_repo() -> Path:
    """Add repo root to sys.path and chdir there (call before other ddpm imports in CLI scripts)."""
    import os
    import sys

    root = REPO_ROOT
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    os.chdir(root)
    return root
