r"""
CIFAR-10 diffusion experiment: baseline isotropic DDPM vs estimated-discrete-latent
observation process.

The generator never uses CIFAR-10 ground-truth labels.  We only assume K=10 latent
classes.  We estimate a discrete latent \hat C by unsupervised clustering, estimate
p(\hat C=k) empirically from cluster sizes, then train a conditional denoiser
on \hat C.  This is the CIFAR analogue of Montanari's Section 5 idea: estimate a
low-dimensional latent partition from data, sample the latent from its empirical
law, and denoise within the selected branch.

Usage examples:

  # 1) Build pseudo-labels by low-resolution K-means.
  python cifar_observation_diffusion.py cluster --data_dir ./data --out ./runs/clusters_lowres8_k10.joblib \
      --feature_mode lowres --lowres 8 --k 10 --pca_dim 64

  # 2) Train unconditional baseline.
  python cifar_observation_diffusion.py train --data_dir ./data --run_dir ./runs/base \
      --method baseline --epochs 50 --batch_size 128 --base_channels 64

  # 3) Train estimated-latent observation model.
  python cifar_observation_diffusion.py train --data_dir ./data --run_dir ./runs/cluster \
      --method cluster --cluster_file ./runs/clusters_lowres8_k10.joblib \
      --epochs 50 --batch_size 128 --base_channels 64

  # 4) Sample from either checkpoint.
  python cifar_observation_diffusion.py sample --run_dir ./runs/base --checkpoint ./runs/base/ckpt_last.pt \
      --method baseline --n 64

  python cifar_observation_diffusion.py sample --run_dir ./runs/cluster --checkpoint ./runs/cluster/ckpt_last.pt \
      --method cluster --cluster_file ./runs/clusters_lowres8_k10.joblib --n 64

Notes:
  * This is intentionally compact research code, not production code.
  * It uses a standard discrete-time DDPM epsilon-prediction implementation.  This
    is easier to debug than a continuous-time SDE, but the observation-process
    comparison is the same: baseline observes noisy x_t; the modified model observes
    (\hat C, x_t).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image


# ----------------------------- utilities ----------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def to_uint_grid(x: torch.Tensor) -> torch.Tensor:
    """Convert model images in [-1,1] to [0,1]."""
    return (x.clamp(-1, 1) + 1) / 2


def num_groups(channels: int, max_groups: int = 8) -> int:
    for g in reversed(range(1, max_groups + 1)):
        if channels % g == 0:
            return g
    return 1


# ----------------------------- clustering ---------------------------------


def cifar_tensor_dataset(data_dir: str, train: bool = True) -> torchvision.datasets.CIFAR10:
    return torchvision.datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=T.ToTensor(),  # [0,1]
    )


@torch.no_grad()
def extract_features(
    data_dir: str,
    batch_size: int,
    feature_mode: str = "lowres",
    lowres: int = 8,
    num_workers: int = 2,
    return_labels: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract unsupervised features from CIFAR-10 training images.

    Ground-truth labels are ignored unless return_labels=True for optional diagnostics.
    They are not used by clustering or generator training.
    """
    ds = cifar_tensor_dataset(data_dir, train=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats, true_labels = [], []
    for x, y in loader:
        if feature_mode == "lowres":
            # Captures global layout/color more than local texture.
            z = F.interpolate(x, size=(lowres, lowres), mode="area")
            z = z.flatten(1)
        elif feature_mode == "pixels":
            z = x.flatten(1)
        elif feature_mode == "channel_means":
            z = x.mean(dim=(2, 3))
        else:
            raise ValueError(f"unknown feature_mode={feature_mode}")
        feats.append(z.cpu().numpy())
        if return_labels:
            true_labels.append(y.numpy())
    labels_out = np.concatenate(true_labels, axis=0) if return_labels else None
    return np.concatenate(feats, axis=0), labels_out


def cluster_purity(assignments: np.ndarray, labels: np.ndarray, k: int) -> float:
    # Optional diagnostic only; labels are not used in training.
    total = 0
    for c in range(k):
        idx = np.where(assignments == c)[0]
        if len(idx) == 0:
            continue
        counts = np.bincount(labels[idx], minlength=10)
        total += counts.max()
    return float(total) / float(len(assignments))


def cmd_cluster(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(Path(args.out).parent)
    feats, true_labels = extract_features(
        args.data_dir,
        batch_size=args.batch_size,
        feature_mode=args.feature_mode,
        lowres=args.lowres,
        num_workers=args.num_workers,
        return_labels=args.diagnose_labels,
    )

    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    pca = None
    feats_for_kmeans = feats_scaled
    if args.pca_dim > 0 and args.pca_dim < feats_scaled.shape[1]:
        pca = PCA(n_components=args.pca_dim, random_state=args.seed, whiten=False)
        feats_for_kmeans = pca.fit_transform(feats_scaled)

    kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        batch_size=args.kmeans_batch_size,
        n_init=10,
        random_state=args.seed,
        max_iter=args.kmeans_max_iter,
    )
    assignments = kmeans.fit_predict(feats_for_kmeans).astype(np.int64)
    counts = np.bincount(assignments, minlength=args.k).astype(np.float64)
    probs = counts / counts.sum()

    bundle = {
        "assignments": assignments,
        "probs": probs,
        "counts": counts,
        "k": args.k,
        "feature_mode": args.feature_mode,
        "lowres": args.lowres,
        "pca_dim": args.pca_dim,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
    }
    joblib.dump(bundle, args.out)

    print(f"Saved clusters to {args.out}")
    print("cluster counts:", counts.astype(int).tolist())
    print("cluster probs:", np.round(probs, 4).tolist())
    if args.diagnose_labels and true_labels is not None:
        print(f"optional diagnostic purity against true CIFAR labels: {cluster_purity(assignments, true_labels, args.k):.3f}")
        print("Ground-truth labels were used only for this diagnostic, not for clustering.")
    else:
        print("Skipped ground-truth label diagnostics; clustering used no CIFAR labels.")




def features_from_images_for_bundle(x01: torch.Tensor, bundle: dict) -> np.ndarray:
    """Apply the same unsupervised feature map used to create clusters.

    x01 should be a CPU tensor in [0,1] with shape [B,3,32,32].
    """
    feature_mode = bundle.get("feature_mode", "lowres")
    lowres = int(bundle.get("lowres", 8))
    with torch.no_grad():
        if feature_mode == "lowres":
            z = F.interpolate(x01, size=(lowres, lowres), mode="area").flatten(1)
        elif feature_mode == "pixels":
            z = x01.flatten(1)
        elif feature_mode == "channel_means":
            z = x01.mean(dim=(2, 3))
        else:
            raise ValueError(f"unknown feature_mode={feature_mode}")
    feats = z.numpy()
    feats = bundle["scaler"].transform(feats)
    if bundle.get("pca") is not None:
        feats = bundle["pca"].transform(feats)
    return feats


def predict_clusters_for_images(x01: torch.Tensor, bundle: dict) -> np.ndarray:
    feats = features_from_images_for_bundle(x01, bundle)
    return bundle["kmeans"].predict(feats).astype(np.int64)

# ----------------------------- datasets -----------------------------------


class CIFARObservationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        train: bool = True,
        assignments: Optional[np.ndarray] = None,
        subset: int = 0,
    ):
        self.ds = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=train,
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1,1]
            ]),
        )
        self.assignments = assignments
        n = len(self.ds)
        if subset and subset > 0:
            n = min(n, subset)
        self.indices = np.arange(n)
        if assignments is not None and len(assignments) < len(self.ds):
            raise ValueError("assignments length is shorter than the CIFAR training set")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x, true_y = self.ds[idx]
        pseudo_y = -1 if self.assignments is None else int(self.assignments[idx])
        return x, pseudo_y, true_y, idx


# ----------------------------- model --------------------------------------


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t is integer timestep, shape [B]. Scale roughly to [0,1].
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / max(half - 1, 1))
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups(out_ch), out_ch)
        self.emb = nn.Linear(emb_dim, 2 * out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = F.silu(self.norm1(h))
        scale_shift = self.emb(emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.conv2(h)
        h = F.silu(self.norm2(h))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """A deliberately small CIFAR U-Net.

    Conditional model: time embedding + pseudo-class embedding. Baseline model:
    time embedding only.  This keeps the denoising architecture the same up to a
    small embedding table.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 256,
        num_classes: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_emb = nn.Embedding(num_classes, time_dim) if num_classes > 0 else None

        c0 = base_channels
        c1 = base_channels * 2
        c2 = base_channels * 4

        self.init = nn.Conv2d(in_channels, c0, 3, padding=1)
        self.down1 = ResBlock(c0, c0, time_dim)          # 32x32
        self.downsample1 = nn.Conv2d(c0, c1, 4, stride=2, padding=1)  # 16x16
        self.down2 = ResBlock(c1, c1, time_dim)
        self.downsample2 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)  # 8x8
        self.down3 = ResBlock(c2, c2, time_dim)

        self.mid1 = ResBlock(c2, c2, time_dim)
        self.mid2 = ResBlock(c2, c2, time_dim)

        self.upsample2 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)  # 16x16
        self.up2 = ResBlock(c1 + c1, c1, time_dim)
        self.upsample1 = nn.ConvTranspose2d(c1, c0, 4, stride=2, padding=1)  # 32x32
        self.up1 = ResBlock(c0 + c0, c0, time_dim)

        self.out_norm = nn.GroupNorm(num_groups(c0), c0)
        self.out = nn.Conv2d(c0, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.time_mlp(t)
        if self.class_emb is not None:
            if y is None:
                raise ValueError("conditional model requires y")
            emb = emb + self.class_emb(y)

        x = self.init(x)
        s1 = self.down1(x, emb)
        x = self.downsample1(s1)
        s2 = self.down2(x, emb)
        x = self.downsample2(s2)
        x = self.down3(x, emb)
        x = self.mid1(x, emb)
        x = self.mid2(x, emb)
        x = self.upsample2(x)
        x = torch.cat([x, s2], dim=1)
        x = self.up2(x, emb)
        x = self.upsample1(x)
        x = torch.cat([x, s1], dim=1)
        x = self.up1(x, emb)
        x = F.silu(self.out_norm(x))
        return self.out(x)


# ----------------------------- diffusion ----------------------------------


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class DDPMSchedule:
    def __init__(self, cfg: DiffusionConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, device=device), self.alpha_bars[:-1]], dim=0)
        self.posterior_variance = self.betas * (1.0 - alpha_bars_prev) / (1.0 - self.alpha_bars)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return a_bar.sqrt() * x0 + (1.0 - a_bar).sqrt() * noise


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        old = copy.deepcopy(model.state_dict())
        state = model.state_dict()
        for k, v in self.shadow.items():
            state[k].copy_(v)
        return old

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]


@torch.no_grad()
def ddpm_sample(
    model: nn.Module,
    schedule: DDPMSchedule,
    n: int,
    device: torch.device,
    y: Optional[torch.Tensor] = None,
    batch_size: int = 64,
) -> torch.Tensor:
    model.eval()
    outs = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        x = torch.randn(b, 3, 32, 32, device=device)
        y_b = None if y is None else y[start:start + b].to(device)
        for ti in reversed(range(schedule.cfg.timesteps)):
            t = torch.full((b,), ti, device=device, dtype=torch.long)
            eps = model(x, t, y_b)
            beta = schedule.betas[ti]
            alpha = schedule.alphas[ti]
            alpha_bar = schedule.alpha_bars[ti]
            mean = (x - beta / torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha)
            if ti > 0:
                var = schedule.posterior_variance[ti]
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        outs.append(x.cpu())
    return torch.cat(outs, dim=0)


# ----------------------------- train/sample commands -----------------------


def load_cluster_bundle(path: str) -> dict:
    bundle = joblib.load(path)
    required = ["assignments", "probs", "k"]
    for r in required:
        if r not in bundle:
            raise ValueError(f"cluster file missing {r}")
    return bundle


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[EMA],
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "global_step": global_step,
    }
    torch.save(ckpt, path)


def build_model_from_args(args: argparse.Namespace, num_classes: int) -> TinyUNet:
    return TinyUNet(
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        num_classes=num_classes,
    )


def cmd_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.run_dir)
    device = pick_device(args.device)
    print(f"Using device: {device}")

    assignments = None
    cluster_probs = None
    num_classes = 0
    if args.method == "cluster":
        bundle = load_cluster_bundle(args.cluster_file)
        assignments = bundle["assignments"]
        cluster_probs = torch.tensor(bundle["probs"], dtype=torch.float32)
        num_classes = int(bundle["k"])
        print("Using empirical pseudo-label distribution:", np.round(bundle["probs"], 4).tolist())

    ds = CIFARObservationDataset(args.data_dir, train=True, assignments=assignments, subset=args.subset)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = build_model_from_args(args, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = DDPMSchedule(DiffusionConfig(args.timesteps, args.beta_start, args.beta_end), device)
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    config = vars(args).copy()
    if cluster_probs is not None:
        config["cluster_probs"] = cluster_probs.tolist()
    with open(Path(args.run_dir) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x0, pseudo_y, _true_y, _idx in loader:
            x0 = x0.to(device)
            y = None
            if args.method == "cluster":
                y = pseudo_y.to(device=device, dtype=torch.long)

            t = torch.randint(0, args.timesteps, (x0.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = schedule.q_sample(x0, t, noise)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xt, t, y)
                loss = F.mse_loss(pred, noise)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            running += loss.item()
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"epoch {epoch:03d} step {global_step:06d} loss {running / args.log_every:.4f}")
                running = 0.0

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            # Temporarily use EMA weights for samples, if available.
            old_state = None
            if ema is not None:
                old_state = ema.apply_to(model)
            with torch.no_grad():
                sample_y = None
                if args.method == "cluster":
                    probs = cluster_probs.to(device)
                    sample_y = torch.multinomial(probs, args.preview_n, replacement=True)
                imgs = ddpm_sample(model, schedule, args.preview_n, device, y=sample_y, batch_size=args.sample_batch_size)
                save_image(to_uint_grid(imgs), Path(args.run_dir) / f"samples_epoch_{epoch:03d}.png", nrow=int(math.sqrt(args.preview_n)))
            if old_state is not None:
                model.load_state_dict(old_state)

        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(Path(args.run_dir) / "ckpt_last.pt", model, optimizer, ema, args, epoch, global_step)
            save_checkpoint(Path(args.run_dir) / f"ckpt_epoch_{epoch:03d}.pt", model, optimizer, ema, args, epoch, global_step)


def cmd_sample(args: argparse.Namespace) -> None:
    ensure_dir(args.run_dir)
    device = pick_device(args.device)
    print(f"Using device: {device}")

    num_classes = 0
    cluster_probs = None
    bundle = None
    if args.cluster_file:
        bundle = load_cluster_bundle(args.cluster_file)
    if args.method == "cluster":
        if bundle is None:
            raise ValueError("--cluster_file is required for method=cluster")
        num_classes = int(bundle["k"])
        cluster_probs = torch.tensor(bundle["probs"], dtype=torch.float32, device=device)
        print("Sampling pseudo-labels from empirical distribution:", np.round(bundle["probs"], 4).tolist())

    # Prefer checkpoint's architecture args when available.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    if "base_channels" in train_args and not args.override_arch:
        args.base_channels = int(train_args["base_channels"])
        args.time_dim = int(train_args["time_dim"])
        args.timesteps = int(train_args["timesteps"])
        args.beta_start = float(train_args["beta_start"])
        args.beta_end = float(train_args["beta_end"])

    model = build_model_from_args(args, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model"])
    if args.use_ema and ckpt.get("ema") is not None:
        ema = EMA(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_to(model)
        print("Using EMA weights")

    schedule = DDPMSchedule(DiffusionConfig(args.timesteps, args.beta_start, args.beta_end), device)
    y = None
    if args.method == "cluster":
        y = torch.multinomial(cluster_probs, args.n, replacement=True)
        counts = torch.bincount(y.cpu(), minlength=num_classes).numpy()
        print("sampled pseudo-label counts:", counts.tolist())

    imgs = ddpm_sample(model, schedule, args.n, device, y=y, batch_size=args.sample_batch_size)
    x01 = to_uint_grid(imgs)
    out_path = Path(args.run_dir) / args.out
    save_image(x01, out_path, nrow=args.nrow)
    tensor_path = out_path.with_suffix(".pt")
    torch.save({"images_0_1": x01, "sampled_pseudo_y": None if y is None else y.cpu()}, tensor_path)
    print(f"Saved samples to {out_path}")
    print(f"Saved sample tensor to {tensor_path}")

    if bundle is not None:
        pred = predict_clusters_for_images(x01.cpu(), bundle)
        counts = np.bincount(pred, minlength=int(bundle["k"])).astype(np.float64)
        q = counts / max(counts.sum(), 1.0)
        p = bundle["probs"]
        tv = 0.5 * np.abs(q - p).sum()
        print("generated-image cluster counts under q(x):", counts.astype(int).tolist())
        print("generated-image cluster probs under q(x):", np.round(q, 4).tolist())
        print(f"TV distance to empirical cluster distribution: {tv:.4f}")


# ----------------------------- cli ----------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CIFAR observation-process diffusion experiment")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("cluster")
    pc.add_argument("--data_dir", type=str, default="./data")
    pc.add_argument("--out", type=str, required=True)
    pc.add_argument("--k", type=int, default=10)
    pc.add_argument("--feature_mode", type=str, default="lowres", choices=["lowres", "pixels", "channel_means"])
    pc.add_argument("--lowres", type=int, default=8)
    pc.add_argument("--pca_dim", type=int, default=64)
    pc.add_argument("--batch_size", type=int, default=512)
    pc.add_argument("--kmeans_batch_size", type=int, default=2048)
    pc.add_argument("--kmeans_max_iter", type=int, default=200)
    pc.add_argument("--num_workers", type=int, default=2)
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--diagnose_labels", action="store_true", help="optional: compute cluster purity using true CIFAR labels after clustering")
    pc.set_defaults(func=cmd_cluster)

    pt = sub.add_parser("train")
    pt.add_argument("--data_dir", type=str, default="./data")
    pt.add_argument("--run_dir", type=str, required=True)
    pt.add_argument("--method", type=str, required=True, choices=["baseline", "cluster"])
    pt.add_argument("--cluster_file", type=str, default="")
    pt.add_argument("--device", type=str, default="auto")
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--batch_size", type=int, default=128)
    pt.add_argument("--sample_batch_size", type=int, default=64)
    pt.add_argument("--num_workers", type=int, default=2)
    pt.add_argument("--subset", type=int, default=0, help="use first N train images; 0 = all")
    pt.add_argument("--base_channels", type=int, default=64)
    pt.add_argument("--time_dim", type=int, default=256)
    pt.add_argument("--timesteps", type=int, default=1000)
    pt.add_argument("--beta_start", type=float, default=1e-4)
    pt.add_argument("--beta_end", type=float, default=2e-2)
    pt.add_argument("--lr", type=float, default=2e-4)
    pt.add_argument("--weight_decay", type=float, default=1e-4)
    pt.add_argument("--grad_clip", type=float, default=1.0)
    pt.add_argument("--ema_decay", type=float, default=0.999)
    pt.add_argument("--amp", action="store_true", help="CUDA mixed precision only")
    pt.add_argument("--preview_n", type=int, default=64)
    pt.add_argument("--log_every", type=int, default=100)
    pt.add_argument("--sample_every", type=int, default=10)
    pt.add_argument("--save_every", type=int, default=10)
    pt.add_argument("--seed", type=int, default=0)
    pt.set_defaults(func=cmd_train)

    ps = sub.add_parser("sample")
    ps.add_argument("--run_dir", type=str, required=True)
    ps.add_argument("--checkpoint", type=str, required=True)
    ps.add_argument("--method", type=str, required=True, choices=["baseline", "cluster"])
    ps.add_argument("--cluster_file", type=str, default="")
    ps.add_argument("--device", type=str, default="auto")
    ps.add_argument("--n", type=int, default=64)
    ps.add_argument("--nrow", type=int, default=8)
    ps.add_argument("--sample_batch_size", type=int, default=64)
    ps.add_argument("--out", type=str, default="samples.png")
    ps.add_argument("--use_ema", action="store_true")
    # Architecture defaults; overwritten from checkpoint unless --override_arch.
    ps.add_argument("--override_arch", action="store_true")
    ps.add_argument("--base_channels", type=int, default=64)
    ps.add_argument("--time_dim", type=int, default=256)
    ps.add_argument("--timesteps", type=int, default=1000)
    ps.add_argument("--beta_start", type=float, default=1e-4)
    ps.add_argument("--beta_end", type=float, default=2e-2)
    ps.set_defaults(func=cmd_sample)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "method", None) == "cluster" and not getattr(args, "cluster_file", ""):
        raise ValueError("--cluster_file is required for method=cluster")
    args.func(args)


if __name__ == "__main__":
    main()
