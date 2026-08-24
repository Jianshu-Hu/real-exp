from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grasp.runtime.retargeting.mano_landmarks import select_mano_landmarks
from grasp.runtime.retargeting.mano_landmarks import MANO_LANDMARK_NAMES
from grasp.runtime.retargeting.mano_model import ManoModel, ManoModelConfig
from grasp.runtime.retargeting.robot_hand_model import (
    RobotHandLandmarkFitConfig,
    RobotHandLandmarkFitResult,
    RobotHandLandmarkFitter,
    RobotHandModel,
    RobotHandSpec,
)


@dataclass(frozen=True)
class ManoToRobotRetargetConfig:
    mano_root: Path = Path("data/HandsAndObejects/mano")
    mano_side: str = "right"
    device: str = "auto"
    landmark_fit_steps: int = 50
    flat_hand_mean: bool = False


@dataclass(frozen=True)
class ManoToRobotRetargetResult:
    robot_trans: np.ndarray
    robot_global_orient: np.ndarray
    robot_joints: np.ndarray
    robot_joint_names: list[str]
    robot_mapping: dict[str, Any]
    fit_error: dict[str, float]


class ManoToRobotRetargeter:
    """Retarget MANO pose to a robot hand by fitting semantic landmarks."""

    def __init__(
        self,
        *,
        robot_spec: RobotHandSpec,
        config: ManoToRobotRetargetConfig | None = None,
    ) -> None:
        self.config = ManoToRobotRetargetConfig() if config is None else config
        self._validate_config()
        self._mano_model = ManoModel(
            ManoModelConfig(
                mano_root=self.config.mano_root,
                mano_side=self.config.mano_side,
                flat_hand_mean=bool(self.config.flat_hand_mean),
            )
        )
        self._robot_model = RobotHandModel(robot_spec)
        self._fitter = RobotHandLandmarkFitter(
            hand_model=self._robot_model,
            config=RobotHandLandmarkFitConfig(
                device=self.config.device,
                steps=int(self.config.landmark_fit_steps),
            ),
        )

    @property
    def robot_joint_names(self) -> list[str]:
        return list(self._robot_model.joint_names)

    @property
    def robot_spec(self) -> RobotHandSpec:
        return self._robot_model.spec

    def retarget_sample(
        self,
        *,
        mano_global_orient: np.ndarray,
        mano_hand_pose: np.ndarray,
        mano_betas: np.ndarray | None = None,
        mano_transl: np.ndarray | None = None,
        init_robot_trans: np.ndarray | None = None,
        init_robot_global_orient: np.ndarray | None = None,
        init_robot_joints: np.ndarray | None = None,
    ) -> ManoToRobotRetargetResult:
        return self.retarget_batch(
            mano_global_orient=np.asarray(mano_global_orient, dtype=np.float32).reshape(1, 3),
            mano_hand_pose=np.asarray(mano_hand_pose, dtype=np.float32).reshape(1, 45),
            mano_betas=(
                None
                if mano_betas is None
                else np.asarray(mano_betas, dtype=np.float32).reshape(1, 10)
            ),
            mano_transl=(
                None
                if mano_transl is None
                else np.asarray(mano_transl, dtype=np.float32).reshape(1, 3)
            ),
            init_robot_trans=(
                None
                if init_robot_trans is None
                else np.asarray(init_robot_trans, dtype=np.float32).reshape(1, 3)
            ),
            init_robot_global_orient=(
                None
                if init_robot_global_orient is None
                else np.asarray(init_robot_global_orient, dtype=np.float32).reshape(1, 3)
            ),
            init_robot_joints=(
                None
                if init_robot_joints is None
                else np.asarray(init_robot_joints, dtype=np.float32).reshape(
                    1,
                    self._robot_model.num_joints,
                )
            ),
        )[0]

    def retarget_batch(
        self,
        *,
        mano_global_orient: np.ndarray,
        mano_hand_pose: np.ndarray,
        mano_betas: np.ndarray | None = None,
        mano_transl: np.ndarray | None = None,
        init_robot_trans: np.ndarray | None = None,
        init_robot_global_orient: np.ndarray | None = None,
        init_robot_joints: np.ndarray | None = None,
    ) -> list[ManoToRobotRetargetResult]:
        mano_global_orient = np.asarray(mano_global_orient, dtype=np.float32)
        mano_hand_pose = np.asarray(mano_hand_pose, dtype=np.float32)
        if mano_global_orient.ndim != 2 or mano_global_orient.shape[1] != 3:
            raise ValueError(
                "mano_global_orient must have shape (B, 3), got "
                f"{mano_global_orient.shape}"
            )
        batch_size = int(mano_global_orient.shape[0])
        mano_hand_pose = mano_hand_pose.reshape(batch_size, 45)
        mano_betas = (
            np.zeros((batch_size, 10), dtype=np.float32)
            if mano_betas is None
            else np.asarray(mano_betas, dtype=np.float32).reshape(batch_size, 10)
        )
        mano_transl = (
            np.zeros((batch_size, 3), dtype=np.float32)
            if mano_transl is None
            else np.asarray(mano_transl, dtype=np.float32).reshape(batch_size, 3)
        )
        target_landmarks = self._mano_landmarks_batch(
            global_orient=mano_global_orient,
            hand_pose=mano_hand_pose,
            betas=mano_betas,
            transl=mano_transl,
        )
        target_wrist_translation = target_landmarks[:, 0, :]
        init_robot_trans_provided = init_robot_trans is not None
        init_robot_trans = (
            target_wrist_translation
            if init_robot_trans is None
            else np.asarray(init_robot_trans, dtype=np.float32).reshape(batch_size, 3)
        )
        if init_robot_global_orient is None:
            if self._robot_model.spec.mano_to_global_orient_seed is None:
                init_robot_global_orient = mano_global_orient
            else:
                init_robot_global_orient = (
                    self._robot_model.spec.mano_to_global_orient_seed(mano_global_orient)
                )
        else:
            init_robot_global_orient = np.asarray(
                init_robot_global_orient,
                dtype=np.float32,
            ).reshape(batch_size, 3)
        if init_robot_joints is None:
            if self._robot_model.spec.mano_to_joint_seed is None:
                raise ValueError(
                    "init_robot_joints is required when robot_spec does not provide "
                    "mano_to_joint_seed"
                )
            init_robot_joints = self._robot_model.spec.mano_to_joint_seed(
                mano_hand_pose,
                list(self._robot_model.joint_names),
            )
        target_landmarks = self._scale_landmarks_to_robot_finger_lengths(
            target_landmarks=target_landmarks,
            init_robot_global_orient=init_robot_global_orient,
            init_robot_joints=init_robot_joints,
        )
        if not init_robot_trans_provided:
            init_robot_trans = self._align_root_to_finger_targets(
                target_landmarks=target_landmarks,
                init_robot_global_orient=init_robot_global_orient,
                init_robot_joints=init_robot_joints,
            )
        fit = self._fitter.fit_batch(
            target_landmarks=target_landmarks,
            init_trans=init_robot_trans,
            init_global_orient=init_robot_global_orient,
            init_joints=init_robot_joints,
        )
        return [
            self._result_from_fit(fit, index)
            for index in range(batch_size)
        ]

    def _scale_landmarks_to_robot_finger_lengths(
        self,
        *,
        target_landmarks: np.ndarray,
        init_robot_global_orient: np.ndarray,
        init_robot_joints: np.ndarray,
    ) -> np.ndarray:
        """Stretch MANO finger targets to the corresponding robot lengths."""
        with torch.no_grad():
            robot_landmarks = self._robot_model.landmarks(
                trans=torch.zeros(
                    (target_landmarks.shape[0], 3), device=self._fitter.device
                ),
                global_orient=torch.as_tensor(
                    init_robot_global_orient, device=self._fitter.device
                ),
                joints=torch.as_tensor(init_robot_joints, device=self._fitter.device),
            ).cpu().numpy()
        scaled = target_landmarks.copy()
        for middle, distal, tip in self._finger_landmark_triplets():
            mano_length = np.linalg.norm(
                target_landmarks[:, tip] - target_landmarks[:, middle], axis=1
            )
            robot_length = np.linalg.norm(
                robot_landmarks[:, tip] - robot_landmarks[:, middle], axis=1
            )
            ratio = robot_length / mano_length
            for index in (distal, tip):
                scaled[:, index] = target_landmarks[:, middle] + ratio[:, None] * (
                    target_landmarks[:, index] - target_landmarks[:, middle]
                )
        return scaled.astype(np.float32)

    def _align_root_to_finger_targets(
        self,
        *,
        target_landmarks: np.ndarray,
        init_robot_global_orient: np.ndarray,
        init_robot_joints: np.ndarray,
    ) -> np.ndarray:
        """Initialize root translation from distal and fingertip targets."""
        with torch.no_grad():
            robot_landmarks = self._robot_model.landmarks(
                trans=torch.zeros(
                    (target_landmarks.shape[0], 3), device=self._fitter.device
                ),
                global_orient=torch.as_tensor(
                    init_robot_global_orient, device=self._fitter.device
                ),
                joints=torch.as_tensor(init_robot_joints, device=self._fitter.device),
            ).cpu().numpy()
        indices = tuple(
            index
            for middle, distal, tip in self._finger_landmark_triplets()
            for index in (distal, tip)
        )
        return np.mean(target_landmarks[:, indices] - robot_landmarks[:, indices], axis=1)

    def _finger_landmark_triplets(self) -> tuple[tuple[int, int, int], ...]:
        index_by_name = {
            name: index for index, name in enumerate(self._robot_model.spec.landmark_names)
        }
        return tuple(
            (
                index_by_name[f"{finger}_middle"],
                index_by_name[f"{finger}_distal"],
                index_by_name[f"{finger}_tip"],
            )
            for finger in self._robot_model.spec.finger_names
        )

    def _mano_landmarks_batch(
        self,
        *,
        global_orient: np.ndarray,
        hand_pose: np.ndarray,
        betas: np.ndarray,
        transl: np.ndarray,
    ) -> np.ndarray:
        landmarks = []
        for index in range(global_orient.shape[0]):
            output = self._mano_model.forward(
                global_orient=global_orient[index],
                hand_pose=hand_pose[index],
                betas=betas[index],
                transl=transl[index],
            )
            all_landmarks = select_mano_landmarks(
                    output["joints"][0],
                    output["vertices"][0],
                ).astype(np.float32)
            indices = [
                MANO_LANDMARK_NAMES.index(
                    self._robot_model.spec.mano_landmark_name_map.get(name, name)
                    if self._robot_model.spec.mano_landmark_name_map is not None
                    else name
                )
                for name in self._robot_model.spec.landmark_names
            ]
            landmarks.append(all_landmarks[indices])
        return np.stack(landmarks, axis=0).astype(np.float32)

    def _result_from_fit(
        self,
        fit: RobotHandLandmarkFitResult,
        index: int,
    ) -> ManoToRobotRetargetResult:
        metadata = fit.metadata[index]
        return ManoToRobotRetargetResult(
            robot_trans=fit.trans[index].astype(np.float32),
            robot_global_orient=fit.global_orient[index].astype(np.float32),
            robot_joints=fit.joints[index].astype(np.float32),
            robot_joint_names=list(self._robot_model.joint_names),
            robot_mapping={
                "robot_hand": self._robot_model.spec.name,
                "robot_urdf_path": self._robot_model.urdf_path.as_posix(),
                "landmark_links": dict(self._robot_model.spec.landmark_links),
            },
            fit_error={
                "fit_loss": float(fit.loss[index]),
                "landmark_rmse_m": float(metadata["landmark_rmse_m"]),
                "weighted_landmark_rmse_m": float(
                    metadata["weighted_landmark_rmse_m"]
                ),
            },
        )

    def _validate_config(self) -> None:
        if self.config.mano_side not in ("right", "left"):
            raise ValueError("mano_side must be 'right' or 'left'")
        if self.config.landmark_fit_steps < 0:
            raise ValueError("landmark_fit_steps must be non-negative")
