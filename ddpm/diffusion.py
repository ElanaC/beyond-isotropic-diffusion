"""DDPM forward process, training loss, and ancestral sampling."""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ddpm.schedule import cosine_beta_schedule, linear_beta_schedule


ScheduleName = Literal["linear", "cosine"]


class GaussianDiffusion(nn.Module):
    """
    Isotropic Gaussian diffusion on images x in R^{C×H×W}.

    Forward (Eq. style from Montanari §4.1): q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I).
    We train a denoiser ε_θ(x_t, t) to predict the noise added at step t.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        schedule: ScheduleName = "linear",
    ) -> None:
        super().__init__()
        self.timesteps = timesteps

        if schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas", torch.sqrt(1.0 / alphas)
        )
        # posterior variance for q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

    def _gather(self, coeffs: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Index schedule coeffs at batch timesteps t and broadcast to x shape."""
        out = coeffs.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus = self._gather(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    @staticmethod
    def _predict_noise(
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y is None:
            return model(x_t, t)
        return model(x_t, t, y)

    def training_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        E_{t, ε} ||ε - ε_θ(x_t, t, ·)||^2.
        Pass ``y`` (class labels) for the class-conditioned denoiser.
        """
        b = x_start.shape[0]
        device = x_start.device
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        noise_pred = self._predict_noise(model, x_t, t, y)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: int,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One reverse DDPM step: sample x_{t-1} from p_θ(x_{t-1} | x_t)."""
        b = x.shape[0]
        t_batch = torch.full((b,), t, device=x.device, dtype=torch.long)

        betas_t = self._gather(self.betas, t_batch, x.shape)
        sqrt_one_minus = self._gather(
            self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape
        )
        sqrt_recip_alpha = self._gather(self.sqrt_recip_alphas, t_batch, x.shape)

        eps_theta = self._predict_noise(model, x, t_batch, y)
        # recover x_0 estimate from epsilon parameterization
        model_mean = sqrt_recip_alpha * (
            x - betas_t * eps_theta / sqrt_one_minus
        )

        if t == 0:
            return model_mean

        posterior_var = self._gather(self.posterior_variance, t_batch, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ancestral sampling: x_T ~ N(0,I), then T reverse steps to x_0."""
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t, y=y)
        return x.clamp(-1.0, 1.0)
