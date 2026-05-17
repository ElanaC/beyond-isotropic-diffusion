"""Set LD_LIBRARY_PATH for pip-bundled NVIDIA libs before importing torch."""

from __future__ import annotations

import glob
import os
import site


def prepend_nvidia_libs() -> None:
    """Must run before ``import torch`` on some GCP DL VM images."""
    paths: list[str] = []
    for sp in site.getsitepackages():
        paths.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
    paths = sorted(set(p for p in paths if os.path.isdir(p)))
    if not paths:
        return
    prefix = ":".join(paths)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{current}" if current else prefix
