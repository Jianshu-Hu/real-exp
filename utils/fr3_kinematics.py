"""Shared FR3 forward-kinematics and pose-representation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

TARGET_EE_SOURCE_PAIRED_JOINT_FK = "paired_target_joint_fk_v1"


def _rpy_rotation_matrix(rpy: Any) -> np.ndarray:
    """Return the URDF fixed-axis roll/pitch/yaw rotation matrix."""
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _transform(xyz: Any, rpy: Any = (0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = _rpy_rotation_matrix(rpy)
    result[:3, 3] = np.asarray(xyz, dtype=float)
    return result


def _rotation_about_z(angle: float) -> np.ndarray:
    sine, cosine = np.sin(angle), np.cos(angle)
    result = np.eye(4, dtype=float)
    result[:2, :2] = ((cosine, -sine), (sine, cosine))
    return result


# Joint origins from franka_description's robots/fr3/fr3.urdf.xacro.  Keeping
# this small serial chain locally lets the ROS bridge compute target EE poses on
# data-server hosts whose system Python does not provide Pinocchio.
_FR3_JOINT_ORIGINS = (
    _transform((0.0, 0.0, 0.333)),
    _transform((0.0, 0.0, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
    _transform((0.0, -0.316, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    _transform((0.0825, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    _transform((-0.0825, 0.384, 0.0), (-np.pi / 2.0, 0.0, 0.0)),
    _transform((0.0, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
    _transform((0.088, 0.0, 0.0), (np.pi / 2.0, 0.0, 0.0)),
)
_FR3_LINK7_TO_FLANGE = _transform((0.0, 0.0, 0.107))


def pose_vector_to_matrix(values: Any) -> np.ndarray:
    """Convert x/y/z/roll/pitch/yaw to a homogeneous transform."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Pose vector must contain six finite values, got {vector.shape}.")
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = vector[:3]
    transform[:3, :3] = _rpy_rotation_matrix(vector[3:])
    return transform


def matrix_to_pose_vector(matrix: Any) -> np.ndarray:
    """Convert a homogeneous transform to x/y/z/roll/pitch/yaw."""
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Pose matrix must be a finite 4x4 transform.")
    rotation = transform[:3, :3]
    pitch = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(np.cos(pitch)) > 1e-8:
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    else:
        # At gimbal lock, choose roll=0 and preserve the represented rotation
        # through yaw, matching the bridge's existing transform conversion.
        roll = 0.0
        yaw = float(np.arctan2(-rotation[0, 1], rotation[1, 1]))
    return np.concatenate((transform[:3, 3], (roll, pitch, yaw)))


def wrapped_pose_delta(current: Any, target: Any) -> np.ndarray:
    """Return the legacy additive XYZ/RPY delta with rotation wrapping."""
    current_vector = np.asarray(current, dtype=float)
    target_vector = np.asarray(target, dtype=float)
    if current_vector.shape != (6,) or target_vector.shape != (6,):
        raise ValueError("Current and target poses must each contain six values.")
    delta = target_vector - current_vector
    delta[3:] = (delta[3:] + np.pi) % (2.0 * np.pi) - np.pi
    return delta


def pose_error(actual: Any, expected: Any) -> tuple[float, float]:
    """Return translation distance and rotation distance between transforms."""
    actual_matrix = np.asarray(actual, dtype=float)
    expected_matrix = np.asarray(expected, dtype=float)
    position_error = float(np.linalg.norm(actual_matrix[:3, 3] - expected_matrix[:3, 3]))
    relative_rotation = expected_matrix[:3, :3].T @ actual_matrix[:3, :3]
    orientation_error = float(
        np.arccos(np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0))
    )
    return position_error, orientation_error


def build_fr3_model() -> tuple[Any, int]:
    """Build the no-gripper FR3 model used by the ROS controller stack."""
    import pinocchio as pin
    import xacro
    from ament_index_python.packages import get_package_share_directory

    xacro_path = (
        Path(get_package_share_directory("franka_description"))
        / "robots"
        / "fr3"
        / "fr3.urdf.xacro"
    )
    xml = xacro.process_file(
        str(xacro_path),
        mappings={
            "ros2_control": "false",
            "arm_id": "fr3",
            "arm_prefix": "",
            "robot_ip": "",
            "hand": "false",
            "use_fake_hardware": "false",
            "fake_sensor_commands": "false",
        },
    ).toxml()
    model = pin.buildModelFromXML(xml)
    frame_id = model.getFrameId("fr3_link8")
    if frame_id >= len(model.frames):
        raise RuntimeError("FR3 model does not contain the fr3_link8 flange frame.")
    return model, frame_id


class Fr3ForwardKinematics:
    """Reusable FR3 FK evaluator with Pinocchio and dependency-free NumPy backends."""

    def __init__(self, *, backend: str = "auto") -> None:
        if backend not in {"auto", "pinocchio", "numpy"}:
            raise ValueError(f"Unsupported FR3 FK backend: {backend!r}.")
        self.backend = backend
        self.model = None
        self.frame_id = None
        self.data = None
        if backend != "numpy":
            try:
                self.model, self.frame_id = build_fr3_model()
            except ModuleNotFoundError:
                if backend == "pinocchio":
                    raise
                self.backend = "numpy"
            else:
                self.data = self.model.createData()
                self.backend = "pinocchio"

    def flange_pose(self, q: Any) -> np.ndarray:
        joints = np.asarray(q, dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"FR3 joints must contain seven finite values, got {joints.shape}.")
        if self.backend == "numpy":
            result = np.eye(4, dtype=float)
            for origin, joint_position in zip(_FR3_JOINT_ORIGINS, joints, strict=True):
                result = result @ origin @ _rotation_about_z(float(joint_position))
            return result @ _FR3_LINK7_TO_FLANGE

        import pinocchio as pin

        pin.forwardKinematics(self.model, self.data, joints)
        pin.updateFramePlacements(self.model, self.data)
        return np.asarray(self.data.oMf[self.frame_id].homogeneous, dtype=float)

    def end_effector_pose(self, q: Any, flange_to_ee: Any) -> np.ndarray:
        tool = np.asarray(flange_to_ee, dtype=float)
        if tool.shape != (4, 4) or not np.all(np.isfinite(tool)):
            raise ValueError("F_T_EE must be a finite 4x4 transform.")
        return self.flange_pose(q) @ tool


def infer_flange_to_ee(
    kinematics: Fr3ForwardKinematics,
    joint_samples: Any,
    ee_pose_samples: Any,
    *,
    max_samples: int = 2000,
) -> np.ndarray:
    """Robustly infer a constant F_T_EE from synchronized q and EE observations."""
    from scipy.spatial.transform import Rotation

    joints = np.asarray(joint_samples, dtype=float)
    poses = np.asarray(ee_pose_samples, dtype=float)
    if joints.ndim != 2 or joints.shape[1] != 7 or poses.shape != (len(joints), 6):
        raise ValueError(
            f"Expected joint/pose samples shaped (N, 7)/(N, 6), got {joints.shape}/{poses.shape}."
        )
    if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(poses)):
        raise ValueError("Tool-transform inference requires finite joint and pose samples.")
    if len(joints) == 0:
        raise ValueError("Tool-transform inference requires at least one sample.")
    sample_indices = np.linspace(
        0, len(joints) - 1, min(len(joints), max_samples), dtype=int
    )
    inferred = [
        np.linalg.inv(kinematics.flange_pose(joints[index]))
        @ pose_vector_to_matrix(poses[index])
        for index in sample_indices
    ]
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.median(
        np.asarray([item[:3, 3] for item in inferred]), axis=0
    )
    rotation_vectors = Rotation.from_matrix(
        np.asarray([item[:3, :3] for item in inferred])
    ).as_rotvec()
    transform[:3, :3] = Rotation.from_rotvec(
        np.median(rotation_vectors, axis=0)
    ).as_matrix()
    return transform
