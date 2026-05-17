#!/usr/bin/env python3
"""
2D class embedding plot for generated CIFAR samples.

- ResNet-32 penultimate features (64-d) → PCA → scatter colored by predicted class
- Large markers = per-class means in PCA space (the "10 peaks")
- Optional metrics JSON includes class-histogram TVD (not shown on the figure).

Baseline: points cluster around ResNet-predicted class means (no intended label).
Conditional: pass --labels-file to also color by intended class in a second panel.

Compare baseline vs conditional:
  python scripts/plot_class_embedding.py \\
    --generated-dir outputs/baseline/samples \\
    --compare-dir outputs/conditional/samples \\
    --compare-labels-file outputs/conditional/samples/labels.npy \\
    --output outputs/class_embedding_compare.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from ddpm.data import default_cifar10_batches_root  # noqa: E402
from ddpm.utils import resolve_device  # noqa: E402

sys.path.insert(0, str(_ROOT / "scripts"))
import evaluate as ev  # noqa: E402

GeneratedImageDataset = ev.GeneratedImageDataset
load_resnet_cifar10 = ev.load_resnet_cifar10
preprocess_for_resnet = ev.preprocess_for_resnet

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# tab10-like distinct colors for 10 classes
CLASS_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def cifar10_test_class_histogram(data_root: Path) -> np.ndarray:
    import sys as _sys

    if str(data_root) not in _sys.path:
        _sys.path.insert(0, str(data_root))
    from cifar10_pytorch import CIFAR10PickleDataset

    ds = CIFAR10PickleDataset(data_root, train=False)
    labels = np.array([lbl for _, lbl in ds], dtype=np.int64)
    counts = np.bincount(labels, minlength=10).astype(np.float64)
    return counts / counts.sum()


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.abs(p - q).sum())


def class_histogram(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    if counts.sum() == 0:
        return np.ones(num_classes) / num_classes
    return counts / counts.sum()


@torch.no_grad()
def extract_resnet_features(
    model: torch.nn.Module,
    image_dir: Path,
    device: torch.device,
    batch_size: int,
    max_images: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    ds = GeneratedImageDataset(image_dir, max_images=max_images)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    feats_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []

    def _hook(_module, inputs) -> None:
        feats_list.append(inputs[0].detach().cpu().numpy())

    handle = model.fc.register_forward_pre_hook(_hook)
    try:
        for batch in loader:
            batch = batch.to(device)
            logits = model(preprocess_for_resnet(batch))
            pred_list.append(logits.argmax(dim=1).cpu().numpy())
    finally:
        handle.remove()

    features = np.concatenate(feats_list, axis=0)
    predictions = np.concatenate(pred_list, axis=0)
    return features, predictions


def pca_project(features: np.ndarray, pca: PCA | None = None) -> tuple[np.ndarray, PCA]:
    if pca is None:
        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(features)
    else:
        coords = pca.transform(features)
    return coords, pca


def class_centroids_2d(coords: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    centroids = np.full((num_classes, 2), np.nan, dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            centroids[c] = coords[mask].mean(axis=0)
    return centroids


def plot_panel(
    ax: plt.Axes,
    coords: np.ndarray,
    color_labels: np.ndarray,
    centroids: np.ndarray,
    title: str,
    point_alpha: float,
    show_legend: bool,
) -> None:
    for c in range(10):
        mask = color_labels == c
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[CLASS_COLORS[c]],
            s=6,
            alpha=point_alpha,
            linewidths=0,
            rasterized=True,
        )
        if not np.isnan(centroids[c, 0]):
            ax.scatter(
                centroids[c, 0],
                centroids[c, 1],
                c=[CLASS_COLORS[c]],
                s=220,
                marker="*",
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
                label=CIFAR10_CLASSES[c] if show_legend else None,
            )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_aspect("equal", adjustable="datalim")
    if show_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=7,
            frameon=True,
            title="class (★ = mean)",
        )


def process_one_set(
    name: str,
    image_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    max_images: int | None,
    intended_labels: np.ndarray | None,
    reference_hist: np.ndarray,
) -> dict:
    features, pred = extract_resnet_features(model, image_dir, device, batch_size, max_images)
    pred_hist = class_histogram(pred)
    metrics = {
        "name": name,
        "num_images": int(len(pred)),
        "tvd_pred_vs_cifar_test": total_variation_distance(pred_hist, reference_hist),
        "predicted_histogram": pred_hist.tolist(),
    }
    if intended_labels is not None:
        intended = intended_labels[: len(pred)]
        metrics["tvd_intended_vs_cifar_test"] = total_variation_distance(
            class_histogram(intended), reference_hist
        )
        metrics["tvd_pred_vs_intended"] = total_variation_distance(pred_hist, class_histogram(intended))
        metrics["accuracy_pred_vs_intended"] = float((pred == intended).mean())
    return {
        "features": features,
        "pred": pred,
        "intended": intended_labels[: len(pred)] if intended_labels is not None else None,
        "metrics": metrics,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="PCA class embedding plot")
    p.add_argument("--generated-dir", type=Path, required=True, help="Primary sample folder (e.g. baseline)")
    p.add_argument("--labels-file", type=Path, default=None, help="Intended labels .npy (conditional)")
    p.add_argument("--compare-dir", type=Path, default=None, help="Second folder for side-by-side compare")
    p.add_argument("--compare-labels-file", type=Path, default=None)
    p.add_argument("--name", type=str, default="generated", help="Panel title for primary set")
    p.add_argument("--compare-name", type=str, default="conditional")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=5000, help="Cap images for speed (use 10000 for full)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--point-alpha", type=float, default=0.35)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--metrics-json", type=Path, default=None)
    args = p.parse_args()

    device = resolve_device(args.device)
    data_root = args.data_root or default_cifar10_batches_root()
    reference_hist = cifar10_test_class_histogram(data_root)

    out_path = args.output or (args.generated_dir / "class_embedding.png")
    metrics_path = args.metrics_json or (out_path.with_suffix(".json"))

    print(f"Loading ResNet-32 on {device}...")
    model = load_resnet_cifar10(device)

    intended = np.load(args.labels_file) if args.labels_file else None
    primary = process_one_set(
        args.name,
        args.generated_dir,
        model,
        device,
        args.batch_size,
        args.max_samples,
        intended,
        reference_hist,
    )

    sets = [primary]
    if args.compare_dir is not None:
        compare_intended = (
            np.load(args.compare_labels_file) if args.compare_labels_file else None
        )
        sets.append(
            process_one_set(
                args.compare_name,
                args.compare_dir,
                model,
                device,
                args.batch_size,
                args.max_samples,
                compare_intended,
                reference_hist,
            )
        )

    # Shared PCA basis when comparing (fair geometry)
    all_features = np.concatenate([s["features"] for s in sets], axis=0)
    pca = PCA(n_components=2, random_state=0)
    pca.fit(all_features)

    n_panels = 0
    if args.compare_dir is not None:
        n_panels = 2
    elif primary["intended"] is not None:
        n_panels = 2  # intended + predicted coloring
    else:
        n_panels = 1

    fig_w = 7.5 * n_panels if n_panels > 1 else 8
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 6), squeeze=False)
    axes_flat = axes.ravel()

    panel_idx = 0
    if args.compare_dir is not None:
        for s, ax in zip(sets, axes_flat[:2]):
            coords, _ = pca_project(s["features"], pca)
            centroids = class_centroids_2d(coords, s["pred"])
            title = f"{s['metrics']['name']}\n(ResNet predicted class, ★ = class mean)"
            plot_panel(ax, coords, s["pred"], centroids, title, args.point_alpha, show_legend=(panel_idx == 1))
            panel_idx += 1
    else:
        coords, _ = pca_project(primary["features"], pca)
        if primary["intended"] is not None:
            cent_i = class_centroids_2d(coords, primary["intended"])
            plot_panel(
                axes_flat[0],
                coords,
                primary["intended"],
                cent_i,
                f"{args.name}: intended label\n(★ = class mean)",
                args.point_alpha,
                show_legend=False,
            )
            cent_p = class_centroids_2d(coords, primary["pred"])
            plot_panel(
                axes_flat[1],
                coords,
                primary["pred"],
                cent_p,
                f"{args.name}: ResNet predicted\n(★ = class mean)",
                args.point_alpha,
                show_legend=True,
            )
        else:
            centroids = class_centroids_2d(coords, primary["pred"])
            plot_panel(
                axes_flat[0],
                coords,
                primary["pred"],
                centroids,
                f"{args.name}: ResNet-predicted class\n(★ = class mean)",
                args.point_alpha,
                show_legend=True,
            )

    fig.suptitle(
        "Generated images in ResNet-32 feature space (PCA)\n"
        "Points = images; ★ = per-class mean in 2D",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {out_path}")

    all_metrics = {
        "reference": "cifar10_test_class_histogram",
        "cifar10_test_histogram": reference_hist.tolist(),
        "sets": [s["metrics"] for s in sets],
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics → {metrics_path}")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
