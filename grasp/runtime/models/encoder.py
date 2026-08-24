from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GraspEncoderOutput:
    point_features: torch.Tensor
    sa2_points: torch.Tensor
    global_feature: torch.Tensor
    latent_input: torch.Tensor


class PointNetPlusPlusPointEncoder(nn.Module):
    def __init__(
        self,
        *,
        point_dim: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        if point_dim < 3:
            raise ValueError("point_dim must include xyz coordinates")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if out_dim <= 0:
            raise ValueError("out_dim must be positive")
        self.point_dim = point_dim
        self.encoder = PointNetPlusPlusEncoder(
            in_dim=point_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )

    def forward(self, points: torch.Tensor) -> GraspEncoderOutput:
        if points.ndim != 3 or points.shape[-1] != self.point_dim:
            raise ValueError(
                f"points must have shape (B, N, {self.point_dim}), got {tuple(points.shape)}"
            )
        hierarchy = self.encoder.encode_hierarchy(points)
        return GraspEncoderOutput(
            point_features=hierarchy.sa1_features,
            sa2_points=hierarchy.sa2_points,
            global_feature=hierarchy.global_feature,
            latent_input=hierarchy.global_feature,
        )


class GraspEncoder(nn.Module):
    def __init__(
        self,
        *,
        point_dim: int = 3,
        point_feature_dim: int = 256,
        latent_input_dim: int = 384,
    ) -> None:
        super().__init__()
        self.point_encoder = PointNetPlusPlusPointEncoder(
            point_dim=point_dim,
            hidden_dim=max(64, point_feature_dim // 2),
            out_dim=point_feature_dim,
        )
        self.latent_projection = nn.Sequential(
            nn.Linear(point_feature_dim, latent_input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_input_dim, latent_input_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> GraspEncoderOutput:
        point_output = self.point_encoder(points)
        latent_input = self.latent_projection(point_output.global_feature)
        return GraspEncoderOutput(
            point_features=point_output.point_features,
            sa2_points=point_output.sa2_points,
            global_feature=point_output.global_feature,
            latent_input=latent_input,
        )


class _SharedMLP(nn.Module):
    def __init__(self, in_dim: int, dims: tuple[int, ...]) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if not dims:
            raise ValueError("dims must not be empty")
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
        if values.ndim != 3:
            raise ValueError(f"values must have shape (B, N, C), got {tuple(values.shape)}")
        return self.net(values)


@dataclass(frozen=True)
class PointNetPlusPlusHierarchy:
    sa1_points: torch.Tensor
    sa1_features: torch.Tensor
    sa2_points: torch.Tensor
    sa2_features: torch.Tensor
    global_feature: torch.Tensor


class PointNetPlusPlusEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int = 3,
        hidden_dim: int = 128,
        out_dim: int = 256,
        first_center_count: int = 256,
        second_center_count: int = 64,
        radii_by_layer: tuple[tuple[float, ...], tuple[float, ...]] = (
            (0.03, 0.06, 0.12),
            (0.06, 0.12, 0.24),
        ),
        neighbor_counts_by_layer: tuple[tuple[int, ...], tuple[int, ...]] = (
            (16, 32, 64),
            (16, 32, 64),
        ),
    ) -> None:
        super().__init__()
        if in_dim < 3:
            raise ValueError("in_dim must include xyz coordinates")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if out_dim <= 0:
            raise ValueError("out_dim must be positive")
        if first_center_count <= 0 or second_center_count <= 0:
            raise ValueError("center counts must be positive")
        if len(radii_by_layer) != 2 or len(neighbor_counts_by_layer) != 2:
            raise ValueError("PointNet++ encoder expects two abstraction layers")

        scale_dim = max(16, hidden_dim // 3)
        self.sa1_out_dim = scale_dim * len(radii_by_layer[0])
        self.sa2_out_dim = scale_dim * len(radii_by_layer[1])
        self.sa1 = PointNetSetAbstractionMSG(
            center_count=first_center_count,
            radii=radii_by_layer[0],
            neighbor_counts=neighbor_counts_by_layer[0],
            in_dim=in_dim,
            mlp_dims_by_scale=tuple(
                (scale_dim, scale_dim) for _ in radii_by_layer[0]
            ),
        )
        self.sa2 = PointNetSetAbstractionMSG(
            center_count=second_center_count,
            radii=radii_by_layer[1],
            neighbor_counts=neighbor_counts_by_layer[1],
            in_dim=self.sa1_out_dim,
            mlp_dims_by_scale=tuple(
                (hidden_dim, scale_dim) for _ in radii_by_layer[1]
            ),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(self.sa2_out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.encode_hierarchy(points).global_feature

    def encode_hierarchy(self, points: torch.Tensor) -> PointNetPlusPlusHierarchy:
        if points.ndim != 3:
            raise ValueError(f"points must have shape (B, N, C), got {tuple(points.shape)}")
        if points.shape[-1] < 3:
            raise ValueError("points must include xyz coordinates in the first 3 channels")
        xyz = points[..., :3]
        sa1_points, sa1_features = self.sa1(xyz, points)
        sa2_points, sa2_features = self.sa2(sa1_points, sa1_features)
        global_feature = self.global_mlp(sa2_features).max(dim=1).values
        return PointNetPlusPlusHierarchy(
            sa1_points=sa1_points,
            sa1_features=sa1_features,
            sa2_points=sa2_points,
            sa2_features=sa2_features,
            global_feature=global_feature,
        )


class PointNetSetAbstractionMSG(nn.Module):
    def __init__(
        self,
        *,
        center_count: int,
        radii: tuple[float, ...],
        neighbor_counts: tuple[int, ...],
        in_dim: int,
        mlp_dims_by_scale: tuple[tuple[int, ...], ...],
    ) -> None:
        super().__init__()
        if center_count <= 0:
            raise ValueError("center_count must be positive")
        if not radii:
            raise ValueError("radii must not be empty")
        if len(radii) != len(neighbor_counts) or len(radii) != len(mlp_dims_by_scale):
            raise ValueError("radii, neighbor_counts, and mlp_dims_by_scale must match")
        if any(radius <= 0.0 for radius in radii):
            raise ValueError("radii must be positive")
        if any(count <= 0 for count in neighbor_counts):
            raise ValueError("neighbor_counts must be positive")
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        self.center_count = center_count
        self.radii = radii
        self.neighbor_counts = neighbor_counts
        self.local_mlps = nn.ModuleList(
            [_make_point_mlp(3 + in_dim, dims) for dims in mlp_dims_by_scale]
        )

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"points must have shape (B, N, 3), got {tuple(points.shape)}")
        if features.ndim != 3:
            raise ValueError(f"features must have shape (B, N, C), got {tuple(features.shape)}")
        if points.shape[:2] != features.shape[:2]:
            raise ValueError("points and features must share batch and point dimensions")

        center_count = min(self.center_count, points.shape[1])
        center_indices = farthest_point_sample(points, center_count)
        centers = gather_points(points, center_indices)
        distances = torch.cdist(centers, points)
        pooled_features = []
        for radius, neighbor_count, local_mlp in zip(
            self.radii,
            self.neighbor_counts,
            self.local_mlps,
            strict=True,
        ):
            neighbor_indices = ball_query(
                distances,
                radius=radius,
                neighbor_count=min(neighbor_count, points.shape[1]),
            )
            neighbor_points = gather_points(points, neighbor_indices)
            neighbor_features = gather_points(features, neighbor_indices)
            local_input = torch.cat(
                [neighbor_points - centers.unsqueeze(2), neighbor_features],
                dim=-1,
            )
            pooled_features.append(local_mlp(local_input).max(dim=2).values)
        return centers, torch.cat(pooled_features, dim=-1)


def farthest_point_sample(points: torch.Tensor, sample_count: int) -> torch.Tensor:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape (B, N, 3), got {tuple(points.shape)}")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    batch_size, point_count, _ = points.shape
    if sample_count > point_count:
        sample_count = point_count
    indices = torch.zeros(batch_size, sample_count, dtype=torch.long, device=points.device)
    distances = torch.full(
        (batch_size, point_count),
        float("inf"),
        dtype=points.dtype,
        device=points.device,
    )
    farthest = torch.zeros(batch_size, dtype=torch.long, device=points.device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=points.device)
    for sample_index in range(sample_count):
        indices[:, sample_index] = farthest
        centroid = points[batch_indices, farthest].unsqueeze(1)
        new_distances = torch.sum((points - centroid) ** 2, dim=-1)
        distances = torch.minimum(distances, new_distances)
        farthest = torch.max(distances, dim=1).indices
    return indices


def ball_query(
    distances: torch.Tensor,
    *,
    radius: float,
    neighbor_count: int,
) -> torch.Tensor:
    if distances.ndim != 3:
        raise ValueError(f"distances must have shape (B, C, N), got {tuple(distances.shape)}")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if neighbor_count <= 0:
        raise ValueError("neighbor_count must be positive")
    masked_distances = distances.masked_fill(distances > radius, float("inf"))
    nearest = torch.topk(
        masked_distances,
        k=neighbor_count,
        dim=-1,
        largest=False,
        sorted=False,
    ).indices
    fallback = torch.topk(
        distances,
        k=1,
        dim=-1,
        largest=False,
        sorted=False,
    ).indices
    has_neighbor = torch.isfinite(masked_distances.gather(-1, nearest))
    return torch.where(has_neighbor, nearest, fallback.expand_as(nearest))


def gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if points.ndim != 3:
        raise ValueError(f"points must have shape (B, N, C), got {tuple(points.shape)}")
    if indices.ndim == 2:
        expanded = indices.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        return torch.gather(points, 1, expanded)
    if indices.ndim == 3:
        batch_size, center_count, neighbor_count = indices.shape
        flat_indices = indices.reshape(batch_size, center_count * neighbor_count)
        gathered = gather_points(points, flat_indices)
        return gathered.reshape(batch_size, center_count, neighbor_count, points.shape[-1])
    raise ValueError(f"indices must have shape (B, M) or (B, M, K), got {tuple(indices.shape)}")


def _make_point_mlp(in_dim: int, dims: tuple[int, ...]) -> nn.Module:
    if not dims:
        raise ValueError("dims must not be empty")
    layers: list[nn.Module] = []
    current_dim = in_dim
    for index, dim in enumerate(dims):
        if dim <= 0:
            raise ValueError("MLP dimensions must be positive")
        layers.append(nn.Linear(current_dim, dim))
        if index != len(dims) - 1:
            layers.append(nn.ReLU(inplace=True))
        current_dim = dim
    return nn.Sequential(*layers)
