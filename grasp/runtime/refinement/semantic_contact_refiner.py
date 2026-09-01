from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from grasp.runtime.rl.contact_surface_regions import (
    build_finger_surface_regions,
    pose_finger_surface_regions,
)
from grasp.runtime.refinement.contact_refiner import (
    _estimate_outward_normals,
    _joint_limit_loss,
    _penetration_loss,
    _penetration_metrics,
    _point_cloud_sdf,
    _resolve_device,
    _validate_sdf_points,
    _weighted_contact_losses,
)
from grasp.runtime.retargeting.robot_hand_model import (
    FINGER_NAMES,
    RobotHandModel,
    RobotHandSpec,
)


@dataclass(frozen=True)
class SemanticContactRefinementConfig:
    steps: int = 40
    learning_rate: float = 1e-2
    top_contact_points_per_finger: int = 56
    contact_weight: float = 5.0
    contact_normal_weight: float = 1.5e-2
    coverage_weight: float = 0.25
    thumb_loss_weight: float = 5.0
    pinky_loss_weight: float = 1.0
    seed_weight: float = 0.2
    joint_limit_weight: float = 5.0
    wrist_weight: float = 2.0
    penetration_weight: float = 10.0
    penetration_margin_m: float = 4e-3
    sdf_normal_neighbors: int = 16
    collision_surface_samples_per_geometry: int = 64
    use_inner_finger_surfaces: bool = True
    surface_reference_flexion: float = 0.55
    surface_inner_max_angle_deg: float = 40.0
    coverage_topk_fraction: float = 0.25
    max_wrist_translation_m: float = 0.04
    max_wrist_rotvec_rad: float = 0.35
    contact_recall_threshold_m: float = 5e-3
    device: str = "auto"
    record_history: bool = False


@dataclass(frozen=True)
class SemanticContactRefinementResult:
    robot_trans: np.ndarray
    robot_global_orient: np.ndarray
    robot_joints: np.ndarray
    history: list[dict[str, float]]
    metadata: dict[str, Any]


class SemanticContactRefiner:
    def __init__(
        self,
        *,
        robot_spec: RobotHandSpec,
        config: SemanticContactRefinementConfig,
    ) -> None:
        self.config = config
        _validate_config(config)
        self.device = _resolve_device(config.device)
        self.hand_model = RobotHandModel(robot_spec).to(
            device=self.device,
            dtype=torch.float32,
        )
        self.finger_names = robot_spec.finger_names
        self.surface_regions = (
            build_finger_surface_regions(
                self.hand_model,
                samples_per_geometry=int(
                    config.collision_surface_samples_per_geometry
                ),
                reference_flexion=float(config.surface_reference_flexion),
                inner_max_angle_deg=float(config.surface_inner_max_angle_deg),
            )
            if config.use_inner_finger_surfaces
            and robot_spec.name
            in (
                "right_sharpa_wave",
                "right_wuji_hand",
                "right_wuji_hand2_beta1",
                "right_shadow_hand",
                "right_allegro_hand",
                "right_leap_hand",
            )
            else None
        )

    def refine(
        self,
        *,
        object_points: np.ndarray,
        contact_scores: np.ndarray,
        contact_binary: np.ndarray,
        contact_finger_labels: np.ndarray,
        sdf_object_points: np.ndarray,
        seed_trans: np.ndarray,
        seed_global_orient: np.ndarray,
        seed_joints: np.ndarray,
    ) -> SemanticContactRefinementResult:
        points_np = _validate_sdf_points(object_points)
        targets_np = _select_finger_targets(
            points_np,
            contact_scores,
            contact_binary,
            contact_finger_labels,
            count=int(self.config.top_contact_points_per_finger),
            active_fingers=self.finger_names,
        )
        surface_points = torch.as_tensor(
            points_np, dtype=torch.float32, device=self.device
        )
        surface_normals = _estimate_outward_normals(
            surface_points,
            neighbors=int(self.config.sdf_normal_neighbors),
        )
        sdf_np = _validate_sdf_points(sdf_object_points)
        sdf_points = torch.as_tensor(sdf_np, dtype=torch.float32, device=self.device)
        sdf_normals = _estimate_outward_normals(
            sdf_points,
            neighbors=int(self.config.sdf_normal_neighbors),
        )

        trans0 = torch.as_tensor(
            seed_trans, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        orient0 = torch.as_tensor(
            seed_global_orient, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        joints0 = self.hand_model.clamp_joints(
            torch.as_tensor(
                seed_joints, dtype=torch.float32, device=self.device
            ).reshape(1, self.hand_model.num_joints)
        )
        target_groups = self._target_groups(
            targets_np,
            surface_normals=surface_normals,
            trans=trans0,
            orient=orient0,
            joints=joints0,
        )
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
            trans, orient = self._bounded_wrist(
                trans0, orient0, trans_delta, orient_delta
            )
            contact_losses = []
            normal_losses = []
            coverage_losses = []
            loss_fingers = []
            posed_regions = self._posed_region_map(trans, orient, joints)
            for group in target_groups:
                hand_points, hand_normals = self._group_surface_samples(
                    group,
                    trans=trans,
                    orient=orient,
                    joints=joints,
                    posed_regions=posed_regions,
                )
                target_points = group["points"]
                scores = group["scores"]
                target_normals = group["normals"]
                normalized_scores = scores / torch.clamp(scores.sum(), min=1e-6)
                contact_loss, normal_loss = _weighted_contact_losses(
                    hand_points,
                    hand_normals,
                    target_points,
                    target_normals,
                    normalized_scores,
                )
                contact_losses.append(contact_loss)
                normal_losses.append(normal_loss)
                loss_fingers.append(str(group["finger"]))
                coverage_losses.append(
                    _trimmed_surface_distance_loss(
                        hand_points,
                        sdf_points,
                        fraction=float(self.config.coverage_topk_fraction),
                    )
                )
            finger_weights = {
                "thumb": float(self.config.thumb_loss_weight),
                "pinky": float(self.config.pinky_loss_weight),
            }
            contact_loss = _finger_weighted_mean(
                contact_losses, loss_fingers, finger_weights
            )
            normal_loss = _finger_weighted_mean(
                normal_losses, loss_fingers, finger_weights
            )
            coverage_loss = _finger_weighted_mean(
                coverage_losses, loss_fingers, finger_weights
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
                collision_points, sdf_points, sdf_normals
            )
            penetration_loss = _penetration_loss(
                signed_distances,
                margin_m=float(self.config.penetration_margin_m),
            )
            if initial_penetration is None:
                initial_penetration = _penetration_metrics(signed_distances)
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
                + float(self.config.contact_normal_weight) * normal_loss
                + float(self.config.coverage_weight) * coverage_loss
                + float(self.config.seed_weight) * seed_loss
                + float(self.config.wrist_weight) * wrist_loss
                + float(self.config.joint_limit_weight) * joint_limit_loss
                + float(self.config.penetration_weight) * penetration_loss
            )
            total.backward()
            optimizer.step()
            with torch.no_grad():
                joints.copy_(self.hand_model.clamp_joints(joints))
            if self.config.record_history and (
                step == 0 or step == self.config.steps - 1 or (step + 1) % 25 == 0
            ):
                history.append(
                    {
                        "step": float(step + 1),
                        "total": float(total.detach().cpu()),
                        "contact": float(contact_loss.detach().cpu()),
                        "contact_normal": float(normal_loss.detach().cpu()),
                        "coverage": float(coverage_loss.detach().cpu()),
                        "seed": float(seed_loss.detach().cpu()),
                        "wrist": float(wrist_loss.detach().cpu()),
                        "joint_limit": float(joint_limit_loss.detach().cpu()),
                        "penetration": float(penetration_loss.detach().cpu()),
                    }
                )

        with torch.no_grad():
            trans, orient = self._bounded_wrist(
                trans0, orient0, trans_delta, orient_delta
            )
            final_joints = self.hand_model.clamp_joints(joints)
            final_collision_points = self.hand_model.collision_surface_points(
                trans=trans,
                global_orient=orient,
                joints=final_joints,
                samples_per_geometry=int(
                    self.config.collision_surface_samples_per_geometry
                ),
            )[0]
            final_signed_distances = _point_cloud_sdf(
                final_collision_points, sdf_points, sdf_normals
            )
            final_penetration = _penetration_metrics(final_signed_distances)
            semantic_metrics = self._semantic_metrics(
                trans, orient, final_joints, self._finger_targets(target_groups)
            )
            segment_metrics = self._segment_metrics(
                trans,
                orient,
                final_joints,
                target_groups,
                sdf_points=sdf_points,
            )
            if initial_penetration is None:
                initial_penetration = dict(final_penetration)
        return SemanticContactRefinementResult(
            robot_trans=trans[0].detach().cpu().numpy().astype(np.float32),
            robot_global_orient=orient[0].detach().cpu().numpy().astype(np.float32),
            robot_joints=final_joints[0].detach().cpu().numpy().astype(np.float32),
            history=history,
            metadata={
                "refinement_stage": "semantic_contact",
                "active_fingers": list(targets_np),
                "active_segments": [
                    {
                        "finger": str(group["finger"]),
                        "segment": str(group["segment"]),
                        "link_name": group["link_name"],
                        "target_points": int(group["points"].shape[0]),
                    }
                    for group in target_groups
                ],
                "contact_surface_source": (
                    "calibrated_inner_finger_regions"
                    if self.surface_regions is not None
                    else "robot_spec_finger_contact_links"
                ),
                "selected_points_per_finger": {
                    finger: int(values[0].shape[0])
                    for finger, values in targets_np.items()
                },
                "semantic_metrics": semantic_metrics,
                "segment_metrics": segment_metrics,
                "initial_penetration": initial_penetration,
                "final_penetration": final_penetration,
                "config": self.config.__dict__,
            },
        )

    def _target_groups(
        self,
        targets_np: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        surface_normals: torch.Tensor,
        trans: torch.Tensor,
        orient: torch.Tensor,
        joints: torch.Tensor,
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        if self.surface_regions is None:
            for finger, (points, scores, indices) in targets_np.items():
                index_tensor = torch.as_tensor(
                    indices, dtype=torch.long, device=self.device
                )
                groups.append(
                    {
                        "finger": finger,
                        "segment": "finger",
                        "link_name": None,
                        "points": torch.as_tensor(
                            points, dtype=torch.float32, device=self.device
                        ),
                        "scores": torch.as_tensor(
                            scores, dtype=torch.float32, device=self.device
                        ),
                        "normals": surface_normals[index_tensor],
                    }
                )
            return groups

        with torch.no_grad():
            posed = pose_finger_surface_regions(
                self.hand_model,
                self.surface_regions,
                trans=trans,
                global_orient=orient,
                joints=joints,
            )
        for finger, (points_np, scores_np, indices_np) in targets_np.items():
            points = torch.as_tensor(
                points_np, dtype=torch.float32, device=self.device
            )
            scores = torch.as_tensor(
                scores_np, dtype=torch.float32, device=self.device
            )
            indices = torch.as_tensor(
                indices_np, dtype=torch.long, device=self.device
            )
            candidates = [item for item in posed if item[0].finger == finger]
            patch_distances = torch.stack(
                [
                    torch.cdist(points.unsqueeze(0), patch_points[inner].unsqueeze(0))[0]
                    .min(dim=1)
                    .values
                    for _, patch_points, _, inner in candidates
                ],
                dim=1,
            )
            assignments = patch_distances.argmin(dim=1)
            for patch_index, (patch, _, _, _) in enumerate(candidates):
                selected = assignments == patch_index
                if not bool(selected.any()):
                    continue
                groups.append(
                    {
                        "finger": finger,
                        "segment": patch.segment,
                        "link_name": patch.link_name,
                        "points": points[selected],
                        "scores": scores[selected],
                        "normals": surface_normals[indices[selected]],
                    }
                )
        return groups

    def _posed_region_map(
        self,
        trans: torch.Tensor,
        orient: torch.Tensor,
        joints: torch.Tensor,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]] | None:
        if self.surface_regions is None:
            return None
        return {
            patch.link_name: (points[inner], normals[inner])
            for patch, points, normals, inner in pose_finger_surface_regions(
                self.hand_model,
                self.surface_regions,
                trans=trans,
                global_orient=orient,
                joints=joints,
            )
        }

    def _segment_metrics(
        self,
        trans: torch.Tensor,
        orient: torch.Tensor,
        joints: torch.Tensor,
        groups: list[dict[str, Any]],
        *,
        sdf_points: torch.Tensor,
    ) -> dict[str, dict[str, Any]]:
        posed_regions = self._posed_region_map(trans, orient, joints)
        if posed_regions is None:
            return {}
        threshold = float(self.config.contact_recall_threshold_m)
        metrics = {}
        for group in groups:
            hand_points = posed_regions[str(group["link_name"])][0]
            target_distances = torch.cdist(
                group["points"].unsqueeze(0), hand_points.unsqueeze(0)
            )[0].min(dim=1).values
            patch_distances = torch.cdist(
                hand_points.unsqueeze(0), sdf_points.unsqueeze(0)
            )[0].min(dim=1).values
            key = f'{group["finger"]}.{group["segment"]}'
            metrics[key] = {
                "target_points": int(group["points"].shape[0]),
                "target_mean_distance_m": float(target_distances.mean().cpu()),
                "target_recall": float(
                    (target_distances <= threshold).float().mean().cpu()
                ),
                "inner_patch_contact_fraction": float(
                    (patch_distances <= threshold).float().mean().cpu()
                ),
            }
        return metrics

    def _group_surface_samples(
        self,
        group: dict[str, Any],
        *,
        trans: torch.Tensor,
        orient: torch.Tensor,
        joints: torch.Tensor,
        posed_regions: dict[str, tuple[torch.Tensor, torch.Tensor]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if posed_regions is not None:
            return posed_regions[str(group["link_name"])]
        points, normals = self.hand_model.collision_surface_samples(
            trans=trans,
            global_orient=orient,
            joints=joints,
            samples_per_geometry=int(
                self.config.collision_surface_samples_per_geometry
            ),
            link_names=self.hand_model.spec.finger_contact_links[
                str(group["finger"])
            ],
        )
        return points[0], normals[0]

    @staticmethod
    def _finger_targets(
        groups: list[dict[str, Any]],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        by_finger: dict[str, list[torch.Tensor]] = {}
        for group in groups:
            by_finger.setdefault(str(group["finger"]), []).append(group["points"])
        return {
            finger: (
                torch.cat(points, dim=0),
                torch.empty(0, device=points[0].device),
                torch.empty(0, device=points[0].device),
            )
            for finger, points in by_finger.items()
        }

    def evaluate(
        self,
        *,
        object_points: np.ndarray,
        contact_scores: np.ndarray,
        contact_binary: np.ndarray,
        contact_finger_labels: np.ndarray,
        sdf_object_points: np.ndarray,
        robot_trans: np.ndarray,
        robot_global_orient: np.ndarray,
        robot_joints: np.ndarray,
    ) -> dict[str, Any]:
        points_np = _validate_sdf_points(object_points)
        selected = _select_finger_targets(
            points_np,
            contact_scores,
            contact_binary,
            contact_finger_labels,
            count=points_np.shape[0],
            active_fingers=self.finger_names,
        )
        targets = {
            finger: (
                torch.as_tensor(points, dtype=torch.float32, device=self.device),
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
            )
            for finger, (points, _, _) in selected.items()
        }
        trans = torch.as_tensor(
            robot_trans, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        orient = torch.as_tensor(
            robot_global_orient, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        joints = torch.as_tensor(
            robot_joints, dtype=torch.float32, device=self.device
        ).reshape(1, self.hand_model.num_joints)
        sdf_points = torch.as_tensor(
            _validate_sdf_points(sdf_object_points),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            collision_points = self.hand_model.collision_surface_points(
                trans=trans,
                global_orient=orient,
                joints=joints,
                samples_per_geometry=int(
                    self.config.collision_surface_samples_per_geometry
                ),
            )[0]
            sdf_normals = _estimate_outward_normals(
                sdf_points,
                neighbors=int(self.config.sdf_normal_neighbors),
            )
            signed_distances = _point_cloud_sdf(
                collision_points, sdf_points, sdf_normals
            )
            return {
                "semantic": self._semantic_metrics(trans, orient, joints, targets),
                "penetration": _penetration_metrics(signed_distances),
            }

    def _bounded_wrist(
        self,
        trans0: torch.Tensor,
        orient0: torch.Tensor,
        trans_delta: torch.Tensor,
        orient_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trans = trans0 + float(self.config.max_wrist_translation_m) * torch.tanh(
            trans_delta
        )
        orient = orient0 + float(self.config.max_wrist_rotvec_rad) * torch.tanh(
            orient_delta
        )
        return trans, orient

    def _semantic_metrics(
        self,
        trans: torch.Tensor,
        orient: torch.Tensor,
        joints: torch.Tensor,
        targets: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> dict[str, Any]:
        posed_regions = self._posed_region_map(trans, orient, joints)
        hand_groups: list[torch.Tensor] = []
        hand_labels = []
        finger_ids = {finger: FINGER_NAMES.index(finger) + 1 for finger in self.finger_names}
        for finger in self.finger_names:
            finger_id = finger_ids[finger]
            if posed_regions is None:
                points = self.hand_model.collision_surface_points(
                    trans=trans,
                    global_orient=orient,
                    joints=joints,
                    samples_per_geometry=int(
                        self.config.collision_surface_samples_per_geometry
                    ),
                    link_names=self.hand_model.spec.finger_contact_links[finger],
                )[0]
            else:
                finger_points = [
                    posed_regions[patch.link_name][0]
                    for patch in self.surface_regions.patches
                    if patch.finger == finger
                ]
                points = torch.cat(finger_points, dim=0)
            hand_groups.append(points)
            hand_labels.append(
                torch.full(
                    (points.shape[0],), finger_id, dtype=torch.long, device=self.device
                )
            )
        all_hand_points = torch.cat(hand_groups)
        all_hand_labels = torch.cat(hand_labels)
        per_finger: dict[str, dict[str, float]] = {}
        correct_distances = []
        semantic_matches = []
        for finger in self.finger_names:
            finger_id = finger_ids[finger]
            if finger not in targets:
                continue
            target_points = targets[finger][0]
            group_index = self.finger_names.index(finger)
            correct = torch.cdist(
                target_points.unsqueeze(0), hand_groups[group_index].unsqueeze(0)
            )[0].min(dim=1).values
            nearest_all = torch.cdist(
                target_points.unsqueeze(0), all_hand_points.unsqueeze(0)
            )[0].argmin(dim=1)
            matches = all_hand_labels[nearest_all] == finger_id
            correct_distances.append(correct)
            semantic_matches.append(matches)
            per_finger[finger] = {
                "mean_distance_m": float(correct.mean().cpu()),
                "max_distance_m": float(correct.max().cpu()),
                "recall": float(
                    (correct <= float(self.config.contact_recall_threshold_m))
                    .float()
                    .mean()
                    .cpu()
                ),
                "semantic_match_rate": float(matches.float().mean().cpu()),
            }
        distances = torch.cat(correct_distances)
        matches = torch.cat(semantic_matches)
        return {
            "mean_distance_m": float(distances.mean().cpu()),
            "max_distance_m": float(distances.max().cpu()),
            "recall": float(
                (distances <= float(self.config.contact_recall_threshold_m))
                .float()
                .mean()
                .cpu()
            ),
            "semantic_match_rate": float(matches.float().mean().cpu()),
            "per_finger": per_finger,
        }


def _select_finger_targets(
    object_points: np.ndarray,
    contact_scores: np.ndarray,
    contact_binary: np.ndarray,
    contact_finger_labels: np.ndarray,
    *,
    count: int,
    active_fingers: tuple[str, ...] = FINGER_NAMES,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    scores = np.asarray(contact_scores, dtype=np.float32).reshape(-1)
    binary = np.asarray(contact_binary, dtype=np.float32).reshape(-1)
    labels = np.asarray(contact_finger_labels, dtype=np.int64).reshape(-1)
    expected = object_points.shape[0]
    if (
        scores.shape[0] != expected
        or binary.shape[0] != expected
        or labels.shape[0] != expected
    ):
        raise ValueError("contact arrays must align with object_points")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(binary)):
        raise ValueError("contact scores and binary labels must be finite")
    if np.any((labels < 0) | (labels > len(FINGER_NAMES))):
        raise ValueError("contact_finger_labels must be in [0, 5]")
    binary_mask = binary > 0.5
    if not np.array_equal(labels > 0, binary_mask):
        raise ValueError("contact binary mask and finger labels disagree")
    if count <= 0:
        raise ValueError("top_contact_points_per_finger must be positive")
    selected = {}
    for finger in active_fingers:
        finger_id = FINGER_NAMES.index(finger) + 1
        indices = np.flatnonzero(labels == finger_id)
        if indices.size == 0:
            continue
        order = _spatially_diverse_contact_indices(
            object_points,
            scores,
            indices,
            count=min(count, indices.size),
        )
        selected[finger] = (
            object_points[order].astype(np.float32),
            scores[order].astype(np.float32),
            order.astype(np.int64),
        )
    if not selected:
        raise ValueError("sample has no finger contact targets")
    return selected


def _spatially_diverse_contact_indices(
    points: np.ndarray,
    scores: np.ndarray,
    candidates: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    """Select confident contact points without collapsing onto one local peak."""

    if count >= candidates.size:
        return candidates
    candidate_points = np.asarray(points[candidates], dtype=np.float32)
    candidate_scores = np.asarray(scores[candidates], dtype=np.float32)
    score_range = float(candidate_scores.max() - candidate_scores.min())
    normalized_scores = (
        (candidate_scores - candidate_scores.min()) / score_range
        if score_range > 1e-8
        else np.ones_like(candidate_scores)
    )
    selected = [int(candidate_scores.argmax())]
    min_squared_distance = np.sum(
        np.square(candidate_points - candidate_points[selected[0]]), axis=1
    )
    for _ in range(1, count):
        priority = min_squared_distance * (0.25 + 0.75 * normalized_scores)
        priority[np.asarray(selected, dtype=np.int64)] = -1.0
        next_index = int(priority.argmax())
        selected.append(next_index)
        squared_distance = np.sum(
            np.square(candidate_points - candidate_points[next_index]), axis=1
        )
        min_squared_distance = np.minimum(min_squared_distance, squared_distance)
    return candidates[np.asarray(selected, dtype=np.int64)]


def _trimmed_surface_distance_loss(
    hand_points: torch.Tensor,
    object_points: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    distances = torch.cdist(
        hand_points.unsqueeze(0), object_points.unsqueeze(0)
    )[0].min(dim=1).values
    count = max(1, int(np.ceil(float(fraction) * distances.shape[0])))
    return torch.topk(distances, k=count, largest=False).values.square().mean()


def _finger_weighted_mean(
    losses: list[torch.Tensor],
    fingers: list[str],
    weights: dict[str, float],
) -> torch.Tensor:
    if not losses or len(losses) != len(fingers):
        raise ValueError("finger losses and labels must be non-empty and aligned")
    losses_by_finger: dict[str, list[torch.Tensor]] = {}
    for loss, finger in zip(losses, fingers, strict=True):
        losses_by_finger.setdefault(finger, []).append(loss)
    finger_losses = []
    active_weights = []
    for finger, values in losses_by_finger.items():
        finger_losses.append(torch.stack(values).mean())
        active_weights.append(float(weights.get(finger, 1.0)))
    weight_tensor = torch.as_tensor(
        active_weights,
        dtype=finger_losses[0].dtype,
        device=finger_losses[0].device,
    )
    weighted_losses = torch.stack(finger_losses) * weight_tensor
    return torch.sum(weighted_losses) / weight_tensor.sum()


def _validate_config(config: SemanticContactRefinementConfig) -> None:
    if config.steps < 0:
        raise ValueError("steps must be non-negative")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.top_contact_points_per_finger <= 0:
        raise ValueError("top_contact_points_per_finger must be positive")
    if config.sdf_normal_neighbors < 3:
        raise ValueError("sdf_normal_neighbors must be at least 3")
    if config.collision_surface_samples_per_geometry <= 0:
        raise ValueError("collision_surface_samples_per_geometry must be positive")
    if not 0.0 <= config.surface_reference_flexion <= 1.0:
        raise ValueError("surface_reference_flexion must be in [0, 1]")
    if not 0.0 < config.surface_inner_max_angle_deg < 90.0:
        raise ValueError("surface_inner_max_angle_deg must be in (0, 90)")
    if not 0.0 < config.coverage_topk_fraction <= 1.0:
        raise ValueError("coverage_topk_fraction must be in (0, 1]")
    nonnegative = (
        config.contact_weight,
        config.contact_normal_weight,
        config.coverage_weight,
        config.thumb_loss_weight,
        config.pinky_loss_weight,
        config.seed_weight,
        config.joint_limit_weight,
        config.wrist_weight,
        config.penetration_weight,
        config.penetration_margin_m,
        config.max_wrist_translation_m,
        config.max_wrist_rotvec_rad,
        config.contact_recall_threshold_m,
    )
    if any(value < 0 for value in nonnegative):
        raise ValueError(
            "semantic refinement weights and distance bounds must be non-negative"
        )
    if config.thumb_loss_weight <= 0 or config.pinky_loss_weight <= 0:
        raise ValueError("thumb and pinky loss weights must be positive")
