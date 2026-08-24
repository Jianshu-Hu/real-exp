from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PosteriorOutput:
    mu: torch.Tensor
    logvar: torch.Tensor
    latent: torch.Tensor


class PriorEncoder(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = 384,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (512, 256),
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.net = _MLP(feature_dim, hidden_dims)
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(
        self,
        *,
        partial_feature: torch.Tensor,
        sample_latent: bool = True,
    ) -> PosteriorOutput:
        _validate_2d("partial_feature", partial_feature, self.feature_dim)
        hidden = self.net(partial_feature)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        latent = reparameterize(mu, logvar) if sample_latent else mu
        return PosteriorOutput(mu=mu, logvar=logvar, latent=latent)


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int = 384,
        grasp_target_dim: int = 51,
        latent_dim: int = 128,
        hidden_dims: tuple[int, ...] = (512, 256),
        conditioning: str = "target_film",
    ) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if grasp_target_dim <= 0:
            raise ValueError("grasp_target_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        if conditioning not in ("target_film", "full_feature_only"):
            raise ValueError("conditioning must be 'target_film' or 'full_feature_only'")
        self.feature_dim = feature_dim
        self.grasp_target_dim = grasp_target_dim
        self.latent_dim = latent_dim
        self.conditioning = conditioning
        if self.conditioning == "target_film":
            self.film = nn.Linear(grasp_target_dim, feature_dim * 2)
        else:
            self.film = None
        self.net = _MLP(feature_dim, hidden_dims)
        self.mu_head = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_head = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(
        self,
        *,
        full_feature: torch.Tensor,
        grasp_target: torch.Tensor | None = None,
        sample_latent: bool = True,
    ) -> PosteriorOutput:
        _validate_2d("full_feature", full_feature, self.feature_dim)
        if self.conditioning == "target_film":
            if grasp_target is None:
                raise ValueError("grasp_target is required when conditioning='target_film'")
            _validate_2d("grasp_target", grasp_target, self.grasp_target_dim)
            if self.film is None:
                raise RuntimeError("target_film conditioning requires a FiLM layer")
            gamma, beta = torch.chunk(self.film(grasp_target), 2, dim=-1)
            conditioned_feature = full_feature * (1.0 + gamma) + beta
        else:
            conditioned_feature = full_feature
        hidden = self.net(conditioned_feature)
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)
        latent = reparameterize(mu, logvar) if sample_latent else mu
        return PosteriorOutput(mu=mu, logvar=logvar, latent=latent)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if mu.shape != logvar.shape:
        raise ValueError("mu and logvar must have matching shape")
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


def standard_normal_sample(
    *,
    batch_size: int,
    latent_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    return torch.randn(batch_size, latent_dim, dtype=dtype, device=device)


def _validate_2d(name: str, value: torch.Tensor, dim: int) -> None:
    if value.ndim != 2 or value.shape[-1] != dim:
        raise ValueError(f"{name} must have shape (B, {dim}), got {tuple(value.shape)}")


class _MLP(nn.Module):
    def __init__(self, in_dim: int, dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = in_dim
        for index, dim in enumerate(dims):
            if dim <= 0:
                raise ValueError("MLP dimensions must be positive")
            layers.append(nn.Linear(current_dim, dim))
            if index != len(dims) - 1:
                layers.append(nn.ReLU(inplace=True))
            current_dim = dim
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError(f"values must have shape (B, C), got {tuple(values.shape)}")
        return self.net(values)
