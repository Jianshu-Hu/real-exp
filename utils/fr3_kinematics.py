"""Shared FR3 forward-kinematics and pose-representation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def pose_vector_to_matrix(values: Any) -> np.ndarray:
    """Convert x/y/z/roll/pitch/yaw to a homogeneous transform."""
    from scipy.spatial.transform import Rotation

    vector = np.asarray(values, dtype=float)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"Pose vector must contain six finite values, got {vector.shape}.")
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = vector[:3]
    transform[:3, :3] = Rotation.from_euler("xyz", vector[3:]).as_matrix()
    return transform


def matrix_to_pose_vector(matrix: Any) -> np.ndarray:
    """Convert a homogeneous transform to x/y/z/roll/pitch/yaw."""
    from scipy.spatial.transform import Rotation

    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Pose matrix must be a finite 4x4 transform.")
    return np.concatenate(
        (transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_euler("xyz"))
    )


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
    from scipy.spatial.transform import Rotation

    actual_matrix = np.asarray(actual, dtype=float)
    expected_matrix = np.asarray(expected, dtype=float)
    position_error = float(np.linalg.norm(actual_matrix[:3, 3] - expected_matrix[:3, 3]))
    orientation_error = float(
        np.linalg.norm(
            Rotation.from_matrix(
                expected_matrix[:3, :3].T @ actual_matrix[:3, :3]
            ).as_rotvec()
        )
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
    """Reusable FK evaluator that avoids allocating Pinocchio data per frame."""

    def __init__(self) -> None:
        self.model, self.frame_id = build_fr3_model()
        self.data = self.model.createData()

    def flange_pose(self, q: Any) -> np.ndarray:
        import pinocchio as pin

        joints = np.asarray(q, dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError(f"FR3 joints must contain seven finite values, got {joints.shape}.")
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
