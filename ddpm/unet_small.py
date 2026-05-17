"""Small U-Net for baseline (unconditional) DDPM on 32×32 CIFAR-10.

Deliberately modest capacity so the baseline can struggle to separate
the 10 class modes without explicit conditioning.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class SmallUNet(nn.Module):
    """
    Denoiser ε_θ(x_t, t) — no class input (baseline).

    32×32 → 16×16 → 8×8 bottleneck, then upsample with skips.
    Default 64 base channels (~few M parameters).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        c = base_channels
        self.in_conv = nn.Conv2d(in_channels, c, 3, padding=1)

        self.enc1 = nn.ModuleList([ResBlock(c, time_dim), ResBlock(c, time_dim)])
        self.down1 = Downsample(c)
        self.enc2 = nn.ModuleList([ResBlock(c, time_dim), ResBlock(c, time_dim)])
        self.down2 = Downsample(c)

        self.mid = nn.ModuleList([ResBlock(c, time_dim), ResBlock(c, time_dim)])

        self.up2 = Upsample(c)
        self.dec2 = nn.ModuleList([ResBlock(c, time_dim), ResBlock(c, time_dim)])
        self.up1 = Upsample(c)
        self.dec1 = nn.ModuleList([ResBlock(c, time_dim), ResBlock(c, time_dim)])

        self.out_norm = nn.GroupNorm(8, c)
        self.out_conv = nn.Conv2d(c, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        h = self.in_conv(x)
        for block in self.enc1:
            h = block(h, t_emb)
        skip32 = h
        h = self.down1(h)

        for block in self.enc2:
            h = block(h, t_emb)
        skip16 = h
        h = self.down2(h)

        for block in self.mid:
            h = block(h, t_emb)

        h = self.up2(h) + skip16
        for block in self.dec2:
            h = block(h, t_emb)

        h = self.up1(h) + skip32
        for block in self.dec1:
            h = block(h, t_emb)

        return self.out_conv(F.silu(self.out_norm(h)))
