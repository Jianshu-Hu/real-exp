from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class GraspDiffusionDecoderOutput:
    noise_prediction: torch.Tensor
    grasp_pose_noise_prediction: torch.Tensor
    hand_pose_noise_prediction: torch.Tensor
    hidden: torch.Tensor


class GraspDiffusionDecoder(nn.Module):
    """Diffusion-style joint grasp decoder shell.

    The decoder models wrist and MANO parameters as one grasp target so the
    generated hand pose and wrist frame stay coupled.
    """

    def __init__(
        self,
        *,
        condition_dim: int = 384,
        latent_dim: int = 128,
        grasp_pose_dim: int = 6,
        hand_pose_dim: int = 45,
        hidden_dim: int = 128,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if grasp_pose_dim <= 0:
            raise ValueError("grasp_pose_dim must be positive")
        if hand_pose_dim <= 0:
            raise ValueError("hand_pose_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.grasp_pose_dim = grasp_pose_dim
        self.hand_pose_dim = hand_pose_dim
        self.grasp_target_dim = grasp_pose_dim + hand_pose_dim
        time_dim = hidden_dim
        global_cond_dim = condition_dim + latent_dim + time_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.input_proj = nn.Conv1d(1, hidden_dim, kernel_size=1)
        self.down_block_0 = ConditionalResidualBlock1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            cond_dim=global_cond_dim,
            kernel_size=kernel_size,
        )
        self.downsample_0 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1)
        self.down_block_1 = ConditionalResidualBlock1d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim * 2,
            cond_dim=global_cond_dim,
            kernel_size=kernel_size,
        )
        self.downsample_1 = nn.Conv1d(
            hidden_dim * 2,
            hidden_dim * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.down_block_2 = ConditionalResidualBlock1d(
            in_channels=hidden_dim * 4,
            out_channels=hidden_dim * 4,
            cond_dim=global_cond_dim,
            kernel_size=kernel_size,
        )
        self.downsample_2 = nn.Conv1d(
            hidden_dim * 4,
            hidden_dim * 8,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.down_block_3 = ConditionalResidualBlock1d(
            in_channels=hidden_dim * 8,
            out_channels=hidden_dim * 8,
            cond_dim=global_cond_dim,
            kernel_size=kernel_size,
        )
        self.downsample_3 = nn.Conv1d(
            hidden_dim * 8,
            hidden_dim * 16,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.mid_block = ConditionalResidualBlock1d(
            in_channels=hidden_dim * 16,
            out_channels=hidden_dim * 16,
            cond_dim=global_cond_dim,
            kernel_size=kernel_size,
        )
        self.upsample_3 = nn.ConvTranspose1d(
            hidden_dim * 16,
            hidden_dim * 8,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_block_3 = ConditionalResidualBlock1d(
            in_channels=hidden_dim * 16,
            out_channels=hidden_dim * 8,
            cond_dim=global_cond_dim,
            kernel_size=kernel_size,
        )
        self.upsample_2 = nn.ConvTranspose1d(
            hidden_dim * 8,
            hidden_dim * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_block_2 = ConditionalResidualBlock1d(
            in_channels=hidden_dim * 8,
            out_channels=hidden_dim * 4,
            cond_dim=global_cond_dim,
            kernel_size=kernel_size,
        )
        self.upsample_1 = nn.ConvTranspose1d(
            hidden_dim * 4,
            hidden_dim * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_block_1 = ConditionalResidualBlock1d(
            in_channels=hidden_dim * 4,
            out_channels=hidden_dim * 2,
            cond_dim=global_cond_dim,
            kernel_size=kernel_size,
        )
        self.upsample_0 = nn.ConvTranspose1d(
            hidden_dim * 2,
            hidden_dim,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_block_0 = ConditionalResidualBlock1d(
            in_channels=hidden_dim * 2,
            out_channels=hidden_dim,
            cond_dim=global_cond_dim,
            kernel_size=kernel_size,
        )
        self.grasp_pose_output_proj = nn.Linear(
            hidden_dim * self.grasp_target_dim,
            grasp_pose_dim,
        )
        self.hand_pose_output_proj = nn.Linear(
            hidden_dim * self.grasp_target_dim,
            hand_pose_dim,
        )

    def forward(
        self,
        condition_feature: torch.Tensor,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        noisy_grasp_target: torch.Tensor | None = None,
    ) -> GraspDiffusionDecoderOutput:
        if condition_feature.ndim != 2 or condition_feature.shape[-1] != self.condition_dim:
            raise ValueError(
                "condition_feature must have shape "
                f"(B, {self.condition_dim}), got {tuple(condition_feature.shape)}"
            )
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent must have shape (B, {self.latent_dim}), got {tuple(latent.shape)}"
            )
        if condition_feature.shape[0] != latent.shape[0]:
            raise ValueError("condition_feature and latent must share batch dimension")
        if timesteps.ndim != 1 or timesteps.shape[0] != latent.shape[0]:
            raise ValueError("timesteps must have shape (B,) and match batch size")
        if noisy_grasp_target is None:
            noisy_grasp_target = torch.zeros(
                latent.shape[0],
                self.grasp_target_dim,
                dtype=latent.dtype,
                device=latent.device,
            )
        if (
            noisy_grasp_target.ndim != 2
            or noisy_grasp_target.shape[-1] != self.grasp_target_dim
        ):
            raise ValueError(
                "noisy_grasp_target must have shape "
                f"(B, {self.grasp_target_dim}), got {tuple(noisy_grasp_target.shape)}"
            )
        time_feature = self.time_embedding(timesteps, dtype=condition_feature.dtype)
        global_cond = torch.cat([condition_feature, latent, time_feature], dim=-1)
        hidden = self.input_proj(noisy_grasp_target.unsqueeze(1))
        skip_0 = self.down_block_0(hidden, global_cond)
        hidden = self.downsample_0(skip_0)
        skip_1 = self.down_block_1(hidden, global_cond)
        hidden = self.downsample_1(skip_1)
        skip_2 = self.down_block_2(hidden, global_cond)
        hidden = self.downsample_2(skip_2)
        skip_3 = self.down_block_3(hidden, global_cond)
        hidden = self.downsample_3(skip_3)
        hidden = self.mid_block(hidden, global_cond)
        hidden = self.upsample_3(hidden)
        hidden = _match_sequence_length(hidden, skip_3.shape[-1])
        hidden = self.up_block_3(torch.cat([hidden, skip_3], dim=1), global_cond)
        hidden = self.upsample_2(hidden)
        hidden = _match_sequence_length(hidden, skip_2.shape[-1])
        hidden = self.up_block_2(torch.cat([hidden, skip_2], dim=1), global_cond)
        hidden = self.upsample_1(hidden)
        hidden = _match_sequence_length(hidden, skip_1.shape[-1])
        hidden = self.up_block_1(torch.cat([hidden, skip_1], dim=1), global_cond)
        hidden = self.upsample_0(hidden)
        hidden = _match_sequence_length(hidden, skip_0.shape[-1])
        hidden = self.up_block_0(torch.cat([hidden, skip_0], dim=1), global_cond)
        flat_hidden = hidden.flatten(start_dim=1)
        grasp_pose_noise_prediction = self.grasp_pose_output_proj(flat_hidden)
        hand_pose_noise_prediction = self.hand_pose_output_proj(flat_hidden)
        noise_prediction = torch.cat(
            [grasp_pose_noise_prediction, hand_pose_noise_prediction],
            dim=-1,
        )
        return GraspDiffusionDecoderOutput(
            noise_prediction=noise_prediction,
            grasp_pose_noise_prediction=grasp_pose_noise_prediction,
            hand_pose_noise_prediction=hand_pose_noise_prediction,
            hidden=hidden,
        )

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timesteps: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(f"timesteps must have shape (B,), got {tuple(timesteps.shape)}")
        half_dim = self.dim // 2
        if half_dim == 0:
            embedding = timesteps.to(dtype=dtype).unsqueeze(-1)
        else:
            frequencies = torch.exp(
                torch.arange(half_dim, device=timesteps.device, dtype=dtype)
                * (-math.log(10000.0) / max(half_dim - 1, 1))
            )
            angles = timesteps.to(dtype=dtype).unsqueeze(-1) * frequencies.unsqueeze(0)
            embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            if embedding.shape[-1] < self.dim:
                embedding = torch.nn.functional.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return self.proj(embedding)


class ConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("channel counts must be positive")
        if cond_dim <= 0:
            raise ValueError("cond_dim must be positive")
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = nn.GroupNorm(8 if out_channels % 8 == 0 else 1, out_channels)
        self.norm2 = nn.GroupNorm(8 if out_channels % 8 == 0 else 1, out_channels)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.residual_proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"values must have shape (B, C, T), got {tuple(values.shape)}")
        if cond.ndim != 2 or cond.shape[0] != values.shape[0]:
            raise ValueError("cond must have shape (B, D) and match values batch size")
        gamma, beta = torch.chunk(self.cond_proj(cond), 2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        hidden = self.activation(self.norm1(self.conv1(values)))
        hidden = hidden * (1.0 + gamma) + beta
        hidden = self.activation(self.norm2(self.conv2(hidden)))
        return self.residual_proj(values) + hidden


def _match_sequence_length(values: torch.Tensor, target_length: int) -> torch.Tensor:
    if values.shape[-1] == target_length:
        return values
    return torch.nn.functional.interpolate(
        values,
        size=target_length,
        mode="linear",
        align_corners=False,
    )
