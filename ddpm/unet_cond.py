"""Class-conditioned U-Net: ε_θ(x_t, t, c) for CIFAR-10 (10 classes)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.unet_small import (
    Downsample,
    ResBlock,
    Upsample,
    sinusoidal_time_embedding,
)


class ClassConditionalUNet(nn.Module):
    """
    Same spatial architecture as ``SmallUNet``, but class label c is embedded
    and added to the timestep embedding (revealed at generation time).
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 3,
        base_channels: int = 64,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.class_emb = nn.Embedding(num_classes, time_dim)

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

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb) + self.class_emb(y)
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
