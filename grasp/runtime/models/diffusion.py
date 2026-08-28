from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DDPMSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_cumprod: torch.Tensor
    sqrt_alpha_cumprod: torch.Tensor
    sqrt_one_minus_alpha_cumprod: torch.Tensor

    @property
    def num_steps(self) -> int:
        return int(self.betas.shape[0])

    @classmethod
    def create(
        cls,
        *,
        num_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device,
    ) -> DDPMSchedule:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if beta_start <= 0.0 or beta_end <= 0.0:
            raise ValueError("beta values must be positive")
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        return cls(
            betas=betas,
            alphas=alphas,
            alpha_cumprod=alpha_cumprod,
            sqrt_alpha_cumprod=torch.sqrt(alpha_cumprod),
            sqrt_one_minus_alpha_cumprod=torch.sqrt(1.0 - alpha_cumprod),
        )

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return torch.randint(
            low=0,
            high=self.num_steps,
            size=(batch_size,),
            dtype=torch.long,
            device=self.betas.device,
        )

    def add_noise(
        self,
        clean: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clean.ndim != 2:
            raise ValueError(f"clean must have shape (B, D), got {tuple(clean.shape)}")
        if timesteps.ndim != 1 or timesteps.shape[0] != clean.shape[0]:
            raise ValueError("timesteps must have shape (B,) and match clean batch size")
        if noise is None:
            noise = torch.randn_like(clean)
        if noise.shape != clean.shape:
            raise ValueError("noise must match clean shape")
        sqrt_alpha = self.sqrt_alpha_cumprod[timesteps].unsqueeze(-1).to(clean.dtype)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cumprod[timesteps].unsqueeze(-1).to(
            clean.dtype
        )
        return sqrt_alpha * clean + sqrt_one_minus_alpha * noise, noise

    def denoise_step(
        self,
        *,
        noisy: torch.Tensor,
        noise_prediction: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy.shape != noise_prediction.shape:
            raise ValueError("noisy and noise_prediction must have matching shape")
        if timesteps.ndim != 1 or timesteps.shape[0] != noisy.shape[0]:
            raise ValueError("timesteps must have shape (B,) and match noisy batch size")
        beta_t = self.betas[timesteps].unsqueeze(-1).to(noisy.dtype)
        alpha_t = self.alphas[timesteps].unsqueeze(-1).to(noisy.dtype)
        alpha_bar_t = self.alpha_cumprod[timesteps].unsqueeze(-1).to(noisy.dtype)
        mean = (noisy - beta_t * noise_prediction / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(
            alpha_t
        )
        nonzero = (timesteps > 0).to(noisy.dtype).unsqueeze(-1)
        noise = torch.randn_like(noisy)
        return mean + nonzero * torch.sqrt(beta_t) * noise

