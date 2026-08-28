from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from grasp.runtime.retargeting.robot_hand_model import RobotHandModel, RobotHandSpec


@dataclass(frozen=True)
class ContactRefinementConfig:
    steps: int = 10
    learning_rate: float = 1e-2
    top_contact_points: int = 128
    contact_weight: float = 5.0
    contact_normal_weight: float = 1e-3
    seed_weight: float = 1.0
    joint_limit_weight: float = 5.0
    wrist_weight: float = 2.0
    penetration_weight: float = 200.0
    penetration_margin_m: float = 4e-3
    sdf_normal_neighbors: int = 16
    collision_surface_samples_per_geometry: int = 64
    max_wrist_translation_m: float = 0.04
    max_wrist_rotvec_rad: float = 0.35
    device: str = "auto"
    record_history: bool = False


@dataclass(frozen=True)
class ContactRefinementResult:
    robot_trans: np.ndarray
    robot_global_orient: np.ndarray
    robot_joints: np.ndarray
    history: list[dict[str, float]]
    metadata: dict[str, Any]


class ContactRefiner:
    def __init__(
        self,
        *,
        robot_spec: RobotHandSpec,
        config: ContactRefinementConfig | None = None,
    ) -> None:
        self.config = ContactRefinementConfig() if config is None else config
        if self.config.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.config.penetration_weight < 0:
            raise ValueError("penetration_weight must be non-negative")
        if self.config.contact_normal_weight < 0:
            raise ValueError("contact_normal_weight must be non-negative")
        if self.config.penetration_margin_m < 0:
            raise ValueError("penetration_margin_m must be non-negative")
        if self.config.sdf_normal_neighbors < 3:
            raise ValueError("sdf_normal_neighbors must be at least 3")
        if self.config.collision_surface_samples_per_geometry <= 0:
            raise ValueError("collision_surface_samples_per_geometry must be positive")
        self.device = _resolve_device(self.config.device)
        self.hand_model = RobotHandModel(robot_spec).to(
            device=self.device,
            dtype=torch.float32,
        )

    def refine(
        self,
        *,
        object_points: np.ndarray,
        contact_scores: np.ndarray,
        sdf_object_points: np.ndarray | None = None,
        seed_trans: np.ndarray,
        seed_global_orient: np.ndarray,
        seed_joints: np.ndarray,
    ) -> ContactRefinementResult:
        points_np, scores_np, selected_indices = _select_contact_points(
            object_points=object_points,
            contact_scores=contact_scores,
            count=int(self.config.top_contact_points),
        )
        points = torch.as_tensor(points_np, dtype=torch.float32, device=self.device)
        scores = torch.as_tensor(scores_np, dtype=torch.float32, device=self.device)
        scores = scores / torch.clamp(scores.sum(), min=1e-6)
        contact_surface_points_np = _validate_sdf_points(object_points)
        contact_surface_points = torch.as_tensor(
            contact_surface_points_np,
            dtype=torch.float32,
            device=self.device,
        )
        contact_surface_normals = _estimate_outward_normals(
            contact_surface_points,
            neighbors=int(self.config.sdf_normal_neighbors),
        )
        point_normals = contact_surface_normals[
            torch.as_tensor(selected_indices, dtype=torch.long, device=self.device)
        ]
        sdf_points_np = _validate_sdf_points(
            object_points if sdf_object_points is None else sdf_object_points
        )
        sdf_points = torch.as_tensor(
            sdf_points_np,
            dtype=torch.float32,
            device=self.device,
        )
        if sdf_object_points is None:
            sdf_normals = contact_surface_normals
        else:
            sdf_normals = _estimate_outward_normals(
                sdf_points,
                neighbors=int(self.config.sdf_normal_neighbors),
            )
        trans0 = torch.as_tensor(
            seed_trans,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 3)
        orient0 = torch.as_tensor(
            seed_global_orient,
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 3)
        joints0 = torch.as_tensor(seed_joints, dtype=torch.float32, device=self.device).reshape(
            1,
            self.hand_model.num_joints,
        )
        joints0 = self.hand_model.clamp_joints(joints0)

        trans_delta = torch.nn.Parameter(torch.zeros_like(trans0))
        orient_delta = torch.nn.Parameter(torch.zeros_like(orient0))
        joints = torch.nn.Parameter(joints0.clone())
        optimizer = torch.optim.Adam(
            [trans_delta, orient_delta, joints],
            lr=float(self.config.learning_rate),
        )
        history: list[dict[str, float]] = []
        initial_penetration: dict[str, float] | None = None

        for step in range(int(self.config.steps)):
            optimizer.zero_grad(set_to_none=True)
            trans, orient = self._bounded_wrist(trans0, orient0, trans_delta, orient_delta)
            if self.hand_model.spec.contact_links:
                contact_hand_points, contact_hand_normals = (
                    self.hand_model.collision_surface_samples(
                    trans=trans,
                    global_orient=orient,
                    joints=joints,
                    samples_per_geometry=int(
                        self.config.collision_surface_samples_per_geometry
                    ),
                    link_names=self.hand_model.spec.contact_links,
                    )
                )
                contact_hand_points = contact_hand_points[0]
                contact_hand_normals = contact_hand_normals[0]
            else:
                contact_hand_points = self.hand_model.landmarks(
                    trans=trans,
                    global_orient=orient,
                    joints=joints,
                )[0]
                contact_hand_normals = F.normalize(
                    contact_hand_points - trans,
                    dim=1,
                )
            contact_loss, contact_normal_loss = _weighted_contact_losses(
                contact_hand_points,
                contact_hand_normals,
                points,
                point_normals,
                scores,
            )
            collision_points = self.hand_model.collision_surface_points(
                trans=trans,
                global_orient=orient,
                joints=joints,
                samples_per_geometry=int(
                    self.config.collision_surface_samples_per_geometry
                ),
            )[0]
            signed_distances = _point_cloud_sdf(
                collision_points,
                sdf_points,
                sdf_normals,
            )
            penetration_loss = _penetration_loss(
                signed_distances,
                margin_m=float(self.config.penetration_margin_m),
            )
            if initial_penetration is None:
                initial_penetration = _penetration_metrics(signed_distances)
            record_step = self.config.record_history and (
                step == 0 or step == self.config.steps - 1 or (step + 1) % 25 == 0
            )
            if record_step:
                penetration_metrics = _penetration_metrics(signed_distances)
            seed_loss = torch.mean((joints - joints0).square())
            wrist_loss = torch.mean((trans - trans0).square()) + torch.mean(
                (orient - orient0).square()
            )
            joint_limit_loss = _joint_limit_loss(
                joints,
                self.hand_model.joint_limits.lower,
                self.hand_model.joint_limits.upper,
            )
            total = (
                float(self.config.contact_weight) * contact_loss
                + float(self.config.contact_normal_weight) * contact_normal_loss
                + float(self.config.seed_weight) * seed_loss
                + float(self.config.wrist_weight) * wrist_loss
                + float(self.config.joint_limit_weight) * joint_limit_loss
                + float(self.config.penetration_weight) * penetration_loss
            )
            total.backward()
            optimizer.step()
            with torch.no_grad():
                joints.copy_(self.hand_model.clamp_joints(joints))
            if record_step:
                history.append(
                    {
                        "step": float(step + 1),
                        "total": float(total.detach().cpu()),
                        "contact": float(contact_loss.detach().cpu()),
                        "contact_normal": float(contact_normal_loss.detach().cpu()),
                        "seed": float(seed_loss.detach().cpu()),
                        "wrist": float(wrist_loss.detach().cpu()),
                        "joint_limit": float(joint_limit_loss.detach().cpu()),
                        "penetration": float(penetration_loss.detach().cpu()),
                        **penetration_metrics,
                    }
                )

        with torch.no_grad():
            trans, orient = self._bounded_wrist(trans0, orient0, trans_delta, orient_delta)
            final_joints = self.hand_model.clamp_joints(joints)
            final_collision_points = self.hand_model.collision_surface_points(
                trans=trans,
                global_orient=orient,
                joints=final_joints,
                samples_per_geometry=int(
                    self.config.collision_surface_samples_per_geometry
                ),
            )[0]
            if self.hand_model.spec.contact_links:
                final_contact_hand_points = self.hand_model.collision_surface_points(
                    trans=trans,
                    global_orient=orient,
                    joints=final_joints,
                    samples_per_geometry=int(
                        self.config.collision_surface_samples_per_geometry
                    ),
                    link_names=self.hand_model.spec.contact_links,
                )[0]
            else:
                final_contact_hand_points = self.hand_model.landmarks(
                    trans=trans,
                    global_orient=orient,
                    joints=final_joints,
                )[0]
            final_signed_distances = _point_cloud_sdf(
                final_collision_points,
                sdf_points,
                sdf_normals,
            )
            final_penetration = _penetration_metrics(final_signed_distances)
            if initial_penetration is None:
                initial_penetration = dict(final_penetration)
        return ContactRefinementResult(
            robot_trans=trans[0].detach().cpu().numpy().astype(np.float32),
            robot_global_orient=orient[0].detach().cpu().numpy().astype(np.float32),
            robot_joints=final_joints[0].detach().cpu().numpy().astype(np.float32),
            history=history,
            metadata={
                "selected_contact_points": int(points_np.shape[0]),
                "hand_contact_surface_samples": int(
                    final_contact_hand_points.shape[0]
                ),
                "hand_contact_links": list(self.hand_model.spec.contact_links),
                "contact_normal_orientation": (
                    "object normals point away from the point-cloud centroid"
                ),
                "contact_score_min": float(scores_np.min()) if scores_np.size else 0.0,
                "contact_score_max": float(scores_np.max()) if scores_np.size else 0.0,
                "sdf_object_points": int(sdf_points_np.shape[0]),
                "collision_surface_samples": int(final_collision_points.shape[0]),
                "sdf_type": "oriented_point_cloud_point_to_plane",
                "sdf_note": (
                    "Approximate SDF from locally estimated point-cloud normals; "
                    "signs may be unreliable for sparse, partial, or strongly concave geometry."
                ),
                "initial_penetration": initial_penetration,
                "final_penetration": final_penetration,
                "config": self.config.__dict__,
            },
        )

    def _bounded_wrist(
        self,
        trans0: torch.Tensor,
        orient0: torch.Tensor,
        trans_delta: torch.Tensor,
        orient_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_t = float(self.config.max_wrist_translation_m)
        max_r = float(self.config.max_wrist_rotvec_rad)
        trans = trans0 + max_t * torch.tanh(trans_delta)
        orient = orient0 + max_r * torch.tanh(orient_delta)
        return trans, orient


def _select_contact_points(
    *,
    object_points: np.ndarray,
    contact_scores: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(object_points, dtype=np.float32)
    scores = np.asarray(contact_scores, dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"object_points must have shape (N, 3), got {points.shape}")
    if scores.shape[0] != points.shape[0]:
        raise ValueError("contact_scores and object_points must have the same length")
    if count <= 0:
        raise ValueError("top_contact_points must be positive")
    selected_count = min(int(count), points.shape[0])
    order = np.argsort(scores)[-selected_count:]
    return (
        points[order].astype(np.float32),
        scores[order].astype(np.float32),
        order.astype(np.int64),
    )


def _weighted_contact_losses(
    hand_points: torch.Tensor,
    hand_normals: torch.Tensor,
    points: torch.Tensor,
    point_normals: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = torch.cdist(points.unsqueeze(0), hand_points.unsqueeze(0), p=2)[0]
    nearest, nearest_indices = distances.min(dim=1)
    matched_hand_normals = hand_normals[nearest_indices]
    normal_cosine = torch.sum(point_normals * matched_hand_normals, dim=1)
    normal_error = 0.5 * (1.0 + torch.clamp(normal_cosine, min=-1.0, max=1.0))
    return (
        torch.sum(scores * nearest.square()),
        torch.sum(scores * normal_error.square()),
    )


def _validate_sdf_points(object_points: np.ndarray) -> np.ndarray:
    points = np.asarray(object_points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"sdf_object_points must have shape (N, 3), got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError("sdf_object_points must contain at least 3 points")
    if not np.all(np.isfinite(points)):
        raise ValueError("sdf_object_points must contain only finite values")
    return points


def _estimate_outward_normals(
    points: torch.Tensor,
    *,
    neighbors: int,
) -> torch.Tensor:
    if neighbors < 3:
        raise ValueError("sdf_normal_neighbors must be at least 3")
    neighbor_count = min(int(neighbors), int(points.shape[0]) - 1)
    distances = torch.cdist(points.unsqueeze(0), points.unsqueeze(0))[0]
    indices = torch.topk(distances, k=neighbor_count + 1, largest=False).indices[:, 1:]
    neighborhoods = points[indices]
    centered = neighborhoods - neighborhoods.mean(dim=1, keepdim=True)
    covariance = centered.transpose(1, 2) @ centered / float(neighbor_count)
    normals = torch.linalg.eigh(covariance).eigenvectors[:, :, 0]
    orientation_reference = points - points.mean(dim=0, keepdim=True)
    reference_norm = torch.linalg.vector_norm(
        orientation_reference,
        dim=1,
        keepdim=True,
    )
    fallback = orientation_reference / torch.clamp(reference_norm, min=1e-8)
    normal_norm = torch.linalg.vector_norm(normals, dim=1, keepdim=True)
    normals = torch.where(normal_norm > 1e-8, normals / normal_norm, fallback)
    orientation = torch.sum(normals * orientation_reference, dim=1, keepdim=True)
    normals = torch.where(orientation < 0.0, -normals, normals)
    return normals.detach()


def _point_cloud_sdf(
    query_points: torch.Tensor,
    surface_points: torch.Tensor,
    surface_normals: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(
        query_points.unsqueeze(0),
        surface_points.unsqueeze(0),
    )[0]
    nearest_indices = distances.argmin(dim=1)
    nearest_points = surface_points[nearest_indices]
    nearest_normals = surface_normals[nearest_indices]
    return torch.sum((query_points - nearest_points) * nearest_normals, dim=1)


def _penetration_loss(
    signed_distances: torch.Tensor,
    *,
    margin_m: float,
) -> torch.Tensor:
    if margin_m < 0.0:
        raise ValueError("penetration_margin_m must be non-negative")
    margin_violation = F.relu(float(margin_m) - signed_distances)
    return torch.mean(margin_violation.square())


def _penetration_metrics(signed_distances: torch.Tensor) -> dict[str, float]:
    penetration_depth = F.relu(-signed_distances)
    metrics = {
        "penetrating_collision_sample_count": float(
            torch.count_nonzero(penetration_depth > 0).cpu()
        ),
        "max_penetration_depth_m": float(penetration_depth.max().detach().cpu()),
        "mean_penetration_depth_m": float(penetration_depth.mean().detach().cpu()),
        "min_signed_distance_m": float(signed_distances.min().detach().cpu()),
    }
    return metrics


def _joint_limit_loss(
    joints: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    lower_violation = F.relu(lower.unsqueeze(0) - joints)
    upper_violation = F.relu(joints - upper.unsqueeze(0))
    return torch.mean(lower_violation.square() + upper_violation.square())


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
