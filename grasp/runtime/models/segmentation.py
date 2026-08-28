from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .encoder import (
    PointNetSetAbstractionMSG,
    gather_points,
)


@dataclass(frozen=True)
class PointSegmentationOutput:
    logits: torch.Tensor
    scores: torch.Tensor


class PointNetPlusPlusSegmentation(nn.Module):
    """PointNet++ object/table segmentation baseline for visible scene points."""

    def __init__(
        self,
        *,
        point_dim: int = 3,
        hidden_dim: int = 128,
        first_center_count: int = 256,
        second_center_count: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if point_dim < 3:
            raise ValueError("point_dim must include xyz coordinates")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if first_center_count <= 0 or second_center_count <= 0:
            raise ValueError("center counts must be positive")
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        scale_dim = max(16, hidden_dim // 3)
        self.point_dim = point_dim
        self.sa1_out_dim = scale_dim * 3
        self.sa2_out_dim = scale_dim * 3
        self.sa1 = PointNetSetAbstractionMSG(
            center_count=first_center_count,
            radii=(0.03, 0.06, 0.12),
            neighbor_counts=(16, 32, 64),
            in_dim=point_dim,
            mlp_dims_by_scale=tuple((scale_dim, scale_dim) for _ in range(3)),
        )
        self.sa2 = PointNetSetAbstractionMSG(
            center_count=second_center_count,
            radii=(0.06, 0.12, 0.24),
            neighbor_counts=(16, 32, 64),
            in_dim=self.sa1_out_dim,
            mlp_dims_by_scale=tuple((hidden_dim, scale_dim) for _ in range(3)),
        )
        self.fp2 = PointFeaturePropagation(
            in_dim=self.sa1_out_dim + self.sa2_out_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        self.fp1 = PointFeaturePropagation(
            in_dim=point_dim + hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
        )
        layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> PointSegmentationOutput:
        if points.ndim != 3 or points.shape[-1] != self.point_dim:
            raise ValueError(
                f"points must have shape (B, N, {self.point_dim}), got {tuple(points.shape)}"
            )
        xyz = points[..., :3]
        sa1_points, sa1_features = self.sa1(xyz, points)
        sa2_points, sa2_features = self.sa2(sa1_points, sa1_features)
        up_sa1 = self.fp2(
            target_points=sa1_points,
            source_points=sa2_points,
            target_features=sa1_features,
            source_features=sa2_features,
        )
        up_points = self.fp1(
            target_points=xyz,
            source_points=sa1_points,
            target_features=points,
            source_features=up_sa1,
        )
        logits = self.classifier(up_points).squeeze(-1)
        return PointSegmentationOutput(logits=logits, scores=torch.sigmoid(logits))


class PointFeaturePropagation(nn.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        if in_dim <= 0 or hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("feature propagation dimensions must be positive")
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        *,
        target_points: torch.Tensor,
        source_points: torch.Tensor,
        target_features: torch.Tensor,
        source_features: torch.Tensor,
    ) -> torch.Tensor:
        if target_points.ndim != 3 or target_points.shape[-1] != 3:
            raise ValueError(f"target_points must have shape (B, N, 3), got {tuple(target_points.shape)}")
        if source_points.ndim != 3 or source_points.shape[-1] != 3:
            raise ValueError(f"source_points must have shape (B, M, 3), got {tuple(source_points.shape)}")
        if target_features.ndim != 3 or target_features.shape[:2] != target_points.shape[:2]:
            raise ValueError("target_features must match target_points batch and point dimensions")
        if source_features.ndim != 3 or source_features.shape[:2] != source_points.shape[:2]:
            raise ValueError("source_features must match source_points batch and point dimensions")
        distances = torch.cdist(target_points, source_points)
        k = min(3, int(source_points.shape[1]))
        nearest = torch.topk(distances, k=k, dim=-1, largest=False, sorted=False)
        weights = torch.reciprocal(torch.clamp(nearest.values, min=1e-8))
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-8)
        gathered = gather_points(source_features, nearest.indices)
        interpolated = torch.sum(gathered * weights.unsqueeze(-1), dim=2)
        return self.mlp(torch.cat([target_features, interpolated], dim=-1))


def resample_object_points(
    points: torch.Tensor,
    scores: torch.Tensor,
    *,
    point_count: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Select predicted object points and randomly resample them to a fixed size."""
    sampled_points, _ = resample_object_points_with_indices(
        points,
        scores,
        point_count=point_count,
        threshold=threshold,
    )
    return sampled_points


def resample_object_points_with_indices(
    points: torch.Tensor,
    scores: torch.Tensor,
    *,
    point_count: int,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample predicted object points and return their source-point indices."""

    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (B, N, 3), got {tuple(points.shape)}")
    if scores.shape != points.shape[:2]:
        raise ValueError("scores must have shape (B, N)")
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    batch_indices = []
    for sample_scores in scores:
        selected = torch.nonzero(sample_scores >= float(threshold), as_tuple=False).flatten()
        if selected.shape[0] == 0:
            selected = torch.argmax(sample_scores).reshape(1)
        if selected.shape[0] >= point_count:
            order = torch.randperm(selected.shape[0], device=selected.device)[:point_count]
            selected = selected[order]
        else:
            repeat = (point_count + selected.shape[0] - 1) // selected.shape[0]
            selected = selected.repeat(repeat)[:point_count]
        batch_indices.append(selected)
    indices = torch.stack(batch_indices, dim=0)
    sampled_points = torch.gather(points, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
    return sampled_points, indices
