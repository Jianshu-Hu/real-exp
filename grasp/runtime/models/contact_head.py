from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ContactHeadOutput:
    logits: torch.Tensor
    scores: torch.Tensor
    binary_logits: torch.Tensor
    binary: torch.Tensor
    finger_logits: torch.Tensor
    finger_labels: torch.Tensor


class ContactMapHead(nn.Module):
    """Per-point soft, binary, and finger-semantic contact prediction head."""

    def __init__(
        self,
        *,
        point_dim: int = 3,
        condition_dim: int = 384,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.0,
        num_fingers: int = 5,
        binary_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if point_dim <= 0:
            raise ValueError("point_dim must be positive")
        if condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if num_fingers <= 0:
            raise ValueError("num_fingers must be positive")
        if not 0.0 < binary_threshold < 1.0:
            raise ValueError("binary_threshold must be in (0, 1)")
        self.point_dim = point_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        self.num_fingers = num_fingers
        self.binary_threshold = binary_threshold
        input_dim = point_dim + condition_dim + latent_dim
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.score_head = nn.Linear(current_dim, 1)
        self.binary_head = nn.Linear(current_dim, 1)
        self.finger_head = nn.Linear(current_dim, num_fingers)

    def forward(
        self,
        *,
        points: torch.Tensor,
        condition_feature: torch.Tensor,
        prior_mu: torch.Tensor,
    ) -> ContactHeadOutput:
        if points.ndim != 3 or points.shape[-1] != self.point_dim:
            raise ValueError(
                f"points must have shape (B, N, {self.point_dim}), got {tuple(points.shape)}"
            )
        if condition_feature.ndim != 2 or condition_feature.shape[-1] != self.condition_dim:
            raise ValueError(
                "condition_feature must have shape "
                f"(B, {self.condition_dim}), got {tuple(condition_feature.shape)}"
            )
        if prior_mu.ndim != 2 or prior_mu.shape[-1] != self.latent_dim:
            raise ValueError(
                f"prior_mu must have shape (B, {self.latent_dim}), got {tuple(prior_mu.shape)}"
            )
        if points.shape[0] != condition_feature.shape[0] or points.shape[0] != prior_mu.shape[0]:
            raise ValueError("points, condition_feature, and prior_mu must share batch size")
        batch_size, point_count, _ = points.shape
        global_condition = torch.cat([condition_feature, prior_mu], dim=-1)
        global_condition = global_condition[:, None, :].expand(batch_size, point_count, -1)
        features = self.trunk(torch.cat([points, global_condition], dim=-1))
        logits = self.score_head(features).squeeze(-1)
        binary_logits = self.binary_head(features).squeeze(-1)
        binary = torch.sigmoid(binary_logits) >= self.binary_threshold
        finger_logits = self.finger_head(features)
        finger_labels = finger_logits.argmax(dim=-1) + 1
        finger_labels = torch.where(binary, finger_labels, torch.zeros_like(finger_labels))
        return ContactHeadOutput(
            logits=logits,
            scores=torch.sigmoid(logits),
            binary_logits=binary_logits,
            binary=binary,
            finger_logits=finger_logits,
            finger_labels=finger_labels,
        )
