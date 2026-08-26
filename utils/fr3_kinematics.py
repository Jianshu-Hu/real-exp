"""Shared FR3 forward-kinematics and pose-representation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

TARGET_EE_SOURCE_PAIRED_JOINT_FK = "paired_target_joint_fk_v1"
EE_STATE_DIM = 9
EE_ACTION_DIM = 6


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


def rotation_matrix_to_6d(rotation: Any) -> np.ndarray:
    """Return the continuous 6D rotation representation (first two columns)."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Rotation must be a finite 3x3 matrix, got {matrix.shape}.")
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def rotation_6d_to_matrix(values: Any) -> np.ndarray:
    """Recover a proper rotation matrix from two predicted columns via Gram-Schmidt."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"6D rotation must contain six finite values, got {vector.shape}.")
    first_norm = float(np.linalg.norm(vector[:3]))
    if first_norm < 1e-8:
        raise ValueError("The first 6D rotation column is degenerate.")
    first = vector[:3] / first_norm
    second_raw = vector[3:] - first * float(np.dot(first, vector[3:]))
    second_norm = float(np.linalg.norm(second_raw))
    if second_norm < 1e-8:
        raise ValueError("The two 6D rotation columns are collinear.")
    second = second_raw / second_norm
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def matrix_to_ee_state(matrix: Any) -> np.ndarray:
    """Convert a transform to position plus continuous 6D rotation (9 values)."""
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("EE transform must be a finite 4x4 matrix.")
    return np.concatenate((transform[:3, 3], rotation_matrix_to_6d(transform[:3, :3])))


def ee_state_to_matrix(values: Any) -> np.ndarray:
    """Convert position plus continuous 6D rotation to a homogeneous transform."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (EE_STATE_DIM,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"EE state must contain {EE_STATE_DIM} finite values, got {vector.shape}.")
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = vector[:3]
    transform[:3, :3] = rotation_6d_to_matrix(vector[3:])
    return transform


def rotation_matrix_to_rotvec(rotation: Any) -> np.ndarray:
    """Return the principal SO(3) logarithm of a proper rotation matrix."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Rotation must be a finite 3x3 matrix, got {matrix.shape}.")
    # Project small numerical errors back onto SO(3) before taking the logarithm.
    u, _, vt = np.linalg.svd(matrix)
    matrix = u @ vt
    if np.linalg.det(matrix) < 0.0:
        u[:, -1] *= -1.0
        matrix = u @ vt
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    vee = np.asarray(
        (matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]),
        dtype=float,
    )
    if angle < 1e-7:
        return 0.5 * vee
    if np.pi - angle < 1e-5:
        # Near pi, the skew part vanishes. Recover a stable principal axis from R+I.
        symmetric = (matrix + np.eye(3)) / 2.0
        axis_index = int(np.argmax(np.diag(symmetric)))
        axis = symmetric[:, axis_index]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            axis = eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))]
            norm = float(np.linalg.norm(axis))
        axis = axis / norm
        if float(np.dot(axis, vee)) < 0.0:
            axis = -axis
        return angle * axis
    return (angle / (2.0 * np.sin(angle))) * vee


def rotvec_to_rotation_matrix(values: Any) -> np.ndarray:
    """Return the SO(3) exponential of a three-dimensional rotation vector."""
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Rotation vector must contain three finite values, got {vector.shape}.")
    angle = float(np.linalg.norm(vector))
    skew = np.asarray(
        ((0.0, -vector[2], vector[1]), (vector[2], 0.0, -vector[0]), (-vector[1], vector[0], 0.0)),
        dtype=float,
    )
    if angle < 1e-7:
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    return np.eye(3) + (np.sin(angle) / angle) * skew + ((1.0 - np.cos(angle)) / angle**2) * (skew @ skew)


def ee_delta(current: Any, target: Any) -> np.ndarray:
    """Return base-frame translation and spatial rotation-vector delta."""
    current_matrix = np.asarray(current, dtype=float)
    target_matrix = np.asarray(target, dtype=float)
    if current_matrix.shape != (4, 4) or target_matrix.shape != (4, 4):
        raise ValueError("Current and target EE transforms must both be 4x4 matrices.")
    translation = target_matrix[:3, 3] - current_matrix[:3, 3]
    relative_rotation = target_matrix[:3, :3] @ current_matrix[:3, :3].T
    return np.concatenate((translation, rotation_matrix_to_rotvec(relative_rotation)))


def apply_ee_delta(current: Any, delta: Any) -> np.ndarray:
    """Apply a base-frame translation/spatial-rotation delta to an EE transform."""
    current_matrix = np.asarray(current, dtype=float)
    delta_vector = np.asarray(delta, dtype=float)
    if current_matrix.shape != (4, 4) or not np.all(np.isfinite(current_matrix)):
        raise ValueError("Current EE transform must be a finite 4x4 matrix.")
    if delta_vector.shape != (EE_ACTION_DIM,) or not np.all(np.isfinite(delta_vector)):
        raise ValueError(f"EE delta must contain {EE_ACTION_DIM} finite values, got {delta_vector.shape}.")
    target = np.eye(4, dtype=float)
    target[:3, 3] = current_matrix[:3, 3] + delta_vector[:3]
    target[:3, :3] = rotvec_to_rotation_matrix(delta_vector[3:]) @ current_matrix[:3, :3]
    return target


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
    if joints.ndim != 2 or joints.shape[1] != 7 or poses.shape not in {
        (len(joints), 6),
        (len(joints), EE_STATE_DIM),
    }:
        raise ValueError(
            "Expected joint/pose samples shaped (N, 7)/(N, 6 or 9), "
            f"got {joints.shape}/{poses.shape}."
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
        @ (
            ee_state_to_matrix(poses[index])
            if poses.shape[1] == EE_STATE_DIM
            else pose_vector_to_matrix(poses[index])
        )
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
