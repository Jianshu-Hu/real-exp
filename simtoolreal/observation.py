"""Build the exact SimToolReal actor observation outside Isaac Lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from kinematics import PolicyKinematics, matrix_to_quat_xyzw
from policy_contract import (
    KEYPOINT_CORNERS,
    KEYPOINT_SCALE,
    OBJECT_BASE_SIZE,
    OBS_FIELDS,
    observation_dim,
)


def checked_transform(value: object, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3) or np.linalg.det(rotation) < 0.0:
        raise ValueError(f"{name} rotation is not a proper orthonormal matrix")
    return matrix


def object_keypoints(pose: np.ndarray, scales: np.ndarray) -> np.ndarray:
    offsets = KEYPOINT_CORNERS * (OBJECT_BASE_SIZE * KEYPOINT_SCALE * 0.5 * scales)
    return pose[:3, 3] + offsets @ pose[:3, :3].T


@dataclass
class ObservationResult:
    vector: np.ndarray
    fields: dict[str, np.ndarray]


class ObservationBuilder:
    def __init__(
        self,
        kinematics: PolicyKinematics,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        *,
        fields: Iterable[str] = OBS_FIELDS,
        clip: float = 10.0,
    ) -> None:
        self.kinematics = kinematics
        self.lower = np.asarray(lower_limits, dtype=np.float64)
        self.upper = np.asarray(upper_limits, dtype=np.float64)
        self.fields = tuple(fields)
        self.dimension = observation_dim(self.fields)
        self.clip = float(clip)
        if self.lower.shape != (27,) or self.upper.shape != (27,):
            raise ValueError("observation joint limits must each contain 27 values")
        if not np.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("observation clip must be positive and finite")

    def build(
        self,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        previous_targets: np.ndarray,
        object_pose_world: np.ndarray,
        goal_pose_world: np.ndarray,
        object_scales: np.ndarray,
        world_from_robot: np.ndarray,
    ) -> ObservationResult:
        q = np.asarray(joint_position, dtype=np.float64)
        qd = np.asarray(joint_velocity, dtype=np.float64)
        prev = np.asarray(previous_targets, dtype=np.float64)
        scales = np.asarray(object_scales, dtype=np.float64)
        if q.shape != (27,) or qd.shape != (27,) or prev.shape != (27,):
            raise ValueError("joint position, velocity, and previous targets must be 27-vectors")
        if scales.shape != (3,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("object scales must be three positive finite values")
        object_pose = checked_transform(object_pose_world, name="object pose")
        goal_pose = checked_transform(goal_pose_world, name="goal pose")
        robot_pose = checked_transform(world_from_robot, name="policy world-from-robot")
        palm_pos, palm_rot_xyzw, fingertips = self.kinematics.evaluate(q, robot_pose)
        object_points = object_keypoints(object_pose, scales)
        goal_points = object_keypoints(goal_pose, scales)
        values = {
            "joint_pos": 2.0 * (q - self.lower) / (self.upper - self.lower) - 1.0,
            "joint_vel": qd,
            "prev_action_targets": prev,
            "palm_pos": palm_pos,
            "palm_rot": palm_rot_xyzw,
            "object_rot": matrix_to_quat_xyzw(object_pose[:3, :3]),
            "fingertip_pos_rel_palm": (fingertips - palm_pos).reshape(-1),
            "keypoints_rel_palm": (object_points - palm_pos).reshape(-1),
            "keypoints_rel_goal": (object_points - goal_points).reshape(-1),
            "object_scales": scales,
        }
        vector = np.concatenate([values[name].reshape(-1) for name in self.fields]).astype(np.float32)
        vector = np.clip(vector, -self.clip, self.clip)
        if vector.shape != (self.dimension,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"invalid actor observation shape/content: {vector.shape}")
        return ObservationResult(vector=vector, fields=values)

