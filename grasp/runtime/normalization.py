"""Checkpoint normalization contracts used during deployed inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class GraspDataNormalizer:
    def __init__(
        self,
        *,
        point_scale_m: float = 0.2,
        wrist_translation_scale_m: float = 0.2,
        mano_pose_scale_rad: float = float(np.pi),
        hand_joint_scale_rad: float = float(np.pi),
    ) -> None:
        self.point_scale_m = float(point_scale_m)
        self.wrist_translation_scale_m = float(wrist_translation_scale_m)
        self.mano_pose_scale_rad = float(mano_pose_scale_rad)
        self.hand_joint_scale_rad = float(hand_joint_scale_rad)

    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        return _scale_array(points, self.point_scale_m)

    def denormalize_points(self, points: np.ndarray) -> np.ndarray:
        return _unscale_array(points, self.point_scale_m)

    def config(self) -> dict[str, float]:
        return {
            "point_scale_m": self.point_scale_m,
            "wrist_translation_scale_m": self.wrist_translation_scale_m,
            "mano_pose_scale_rad": self.mano_pose_scale_rad,
            "hand_joint_scale_rad": self.hand_joint_scale_rad,
        }


class GraspTargetNormalizer:
    def __init__(
        self,
        *,
        wrist_translation_scale_m: float = 1.0,
        mano_pose_scale_rad: float = 1.0,
        hand_pose_scale_rad: float = 1.0,
    ) -> None:
        self.wrist_translation_scale_m = float(wrist_translation_scale_m)
        self.mano_pose_scale_rad = float(mano_pose_scale_rad)
        self.hand_pose_scale_rad = float(hand_pose_scale_rad)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "GraspTargetNormalizer":
        if not config:
            return cls()
        return cls(
            wrist_translation_scale_m=float(config.get("wrist_translation_scale_m", 1.0)),
            mano_pose_scale_rad=float(config.get("mano_pose_scale_rad", 1.0)),
            hand_pose_scale_rad=float(config.get("hand_pose_scale_rad", config.get("mano_pose_scale_rad", 1.0))),
        )

    @classmethod
    def from_data_normalizer(cls, normalizer: GraspDataNormalizer) -> "GraspTargetNormalizer":
        return cls(
            wrist_translation_scale_m=normalizer.wrist_translation_scale_m,
            mano_pose_scale_rad=normalizer.mano_pose_scale_rad,
            hand_pose_scale_rad=normalizer.mano_pose_scale_rad,
        )

    def denormalize_tensor(self, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-1] != 51:
            raise ValueError(f"grasp target must have 51 dims, got {tuple(target.shape)}")
        wrist, orient, hand_pose = torch.split(target, [3, 3, 45], dim=-1)
        return torch.cat((
            wrist * self._tensor_scale(self.wrist_translation_scale_m, target),
            orient * self._tensor_scale(self.mano_pose_scale_rad, target),
            hand_pose * self._tensor_scale(self.hand_pose_scale_rad, target),
        ), dim=-1)

    def denormalize_wrist_translation(self, wrist_translation: np.ndarray) -> np.ndarray:
        return _unscale_array(wrist_translation, self.wrist_translation_scale_m)

    def denormalize_mano_orient(self, mano_orient: np.ndarray) -> np.ndarray:
        return _unscale_array(mano_orient, self.mano_pose_scale_rad)

    def denormalize_hand_pose(self, hand_pose: np.ndarray) -> np.ndarray:
        return _unscale_array(hand_pose, self.hand_pose_scale_rad)

    def config(self) -> dict[str, float | str]:
        return {
            "schema": "wrist_meters_mano_rad_scales",
            "wrist_translation_scale_m": self.wrist_translation_scale_m,
            "mano_pose_scale_rad": self.mano_pose_scale_rad,
            "hand_pose_scale_rad": self.hand_pose_scale_rad,
        }

    @staticmethod
    def _tensor_scale(scale: float, reference: torch.Tensor) -> torch.Tensor:
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"target normalization scale must be positive, got {scale}")
        return torch.as_tensor(scale, dtype=reference.dtype, device=reference.device)


def _scale_array(array: np.ndarray, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"normalization scale must be positive, got {scale}")
    return (np.asarray(array, dtype=np.float32) / np.float32(scale)).astype(np.float32)


def _unscale_array(array: np.ndarray, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"normalization scale must be positive, got {scale}")
    return (np.asarray(array, dtype=np.float32) * np.float32(scale)).astype(np.float32)
