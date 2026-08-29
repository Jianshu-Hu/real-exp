from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


COMMAND_FORMAT = "real_exp_wuji_grasp_command_v2"
INFERENCE_REQUEST_FORMAT = "real_exp_camera_grasp_request_v1"
INFERENCE_RESPONSE_FORMAT = "real_exp_camera_grasp_response_v1"
WUJI_RIGHT_JOINT_NAMES = tuple(
    f"right_finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)
WUJI_COMMAND_HAND_MODEL = "wuji_hand_2"
WUJI_COMMAND_JOINT_CONVENTION = "wuji_sdk_firmware_order"
WUJI_COMMAND_SOURCE_MODEL = "wuji_hand_v1_robodex"
WUJI_COMMAND_CONVERSION = "wuji_hand_v1_robodex_to_wuji_hand_2_lateral_sign_v1"
# The first-generation RoboDex model and the Wuji Hand 2 firmware use opposite
# positive axes for the MCP abduction/adduction joint of the four non-thumb
# fingers. Their remaining joint axes do not require a sign change. This is a
# temporary model-boundary conversion until grasp inference uses Hand 2 geometry.
WUJI_V1_TO_HAND2_NEGATED_JOINT_NAMES = (
    "right_finger2_joint2",
    "right_finger3_joint2",
    "right_finger4_joint2",
    "right_finger5_joint2",
)
EE_POSITION_LOWER_M = np.asarray([-0.40, -1.00, -0.60], dtype=np.float64)
EE_POSITION_UPPER_M = np.asarray([1.00, 1.00, 1.20], dtype=np.float64)
EE_POSITION_MAX_RADIUS_M = 1.25


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_transform(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = read_transform(transform, "transform")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = read_transform(transform, "point transform")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points must have finite shape (N, 3)")
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def load_calibration_transforms(
    camera_to_world_path: Path,
    camera_to_robot_base_path: Path,
    mount_path: Path,
) -> dict[str, np.ndarray]:
    world_data = read_json(camera_to_world_path)
    robot_data = read_json(camera_to_robot_base_path)
    mount_data = read_json(mount_path)
    world_t_camera = read_transform(world_data.get("world_T_camera"), "world_T_camera")
    base_t_camera = read_transform(robot_data.get("base_T_camera"), "base_T_camera")
    mount_value = mount_data.get("ee_T_hand", mount_data.get("flange_T_hand"))
    if mount_value is None:
        raise ValueError(f"{mount_path} must define ee_T_hand (or flange_T_hand)")
    ee_t_hand = read_transform(mount_value, "ee_T_hand")
    base_t_world = base_t_camera @ invert_transform(world_t_camera)
    return {
        "world_T_camera": world_t_camera,
        "base_T_camera": base_t_camera,
        "base_T_world": read_transform(base_t_world, "base_T_world"),
        "ee_T_hand": ee_t_hand,
    }


def hand_pose_to_ee_pose(
    world_t_hand: np.ndarray,
    base_t_world: np.ndarray,
    ee_t_hand: np.ndarray,
) -> np.ndarray:
    # B_T_E = B_T_W @ W_T_H @ H_T_E.
    return read_transform(
        read_transform(base_t_world, "base_T_world")
        @ read_transform(world_t_hand, "world_T_hand")
        @ invert_transform(ee_t_hand),
        "base_T_ee",
    )


def matrix_to_xyz_rpy(transform: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    transform = read_transform(transform, "pose")
    rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=False)
    return np.concatenate((transform[:3, 3], rpy)).astype(np.float64)


def xyz_rpy_to_matrix(pose: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("pose must contain six finite XYZ/RPY values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def reorder_wuji_joints(joints: np.ndarray, names: Sequence[str]) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float64).reshape(-1)
    names = tuple(str(name) for name in names)
    if joints.shape != (20,) or len(names) != 20 or len(set(names)) != 20:
        raise ValueError("retargeted Wuji result must contain 20 uniquely named joints")
    missing = set(WUJI_RIGHT_JOINT_NAMES) - set(names)
    extra = set(names) - set(WUJI_RIGHT_JOINT_NAMES)
    if missing or extra:
        raise ValueError(f"unexpected Wuji joint contract: missing={sorted(missing)}, extra={sorted(extra)}")
    by_name = dict(zip(names, joints, strict=True))
    ordered = np.asarray([by_name[name] for name in WUJI_RIGHT_JOINT_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(ordered)):
        raise ValueError("Wuji joint target contains non-finite values")
    return ordered


def wuji_v1_model_to_hand2_firmware(joints: np.ndarray) -> np.ndarray:
    """Convert canonical RoboDex/Wuji-v1 angles to Hand 2 SDK firmware angles.

    Input and output both use finger-major ``finger1..5 x joint1..4`` ordering.
    Only the four non-thumb lateral joints change sign.
    """
    converted = np.asarray(joints, dtype=np.float64).reshape(-1).copy()
    if converted.shape != (20,) or not np.all(np.isfinite(converted)):
        raise ValueError("Wuji v1-to-Hand-2 conversion requires 20 finite joint angles")
    name_to_index = {name: index for index, name in enumerate(WUJI_RIGHT_JOINT_NAMES)}
    for name in WUJI_V1_TO_HAND2_NEGATED_JOINT_NAMES:
        converted[name_to_index[name]] *= -1.0
    return converted


def validate_command(command: Any, *, max_age_s: float, expected_side: str) -> dict[str, Any]:
    if not isinstance(command, dict) or command.get("format") != COMMAND_FORMAT:
        raise ValueError(f"command format must be {COMMAND_FORMAT!r}")
    command_id = command.get("command_id")
    if not isinstance(command_id, str) or not command_id or len(command_id) > 128:
        raise ValueError("command_id must be a non-empty string of at most 128 characters")
    if command.get("side") != expected_side:
        raise ValueError(f"server only accepts side={expected_side!r}")
    expected_contract = {
        "hand_model": WUJI_COMMAND_HAND_MODEL,
        "hand_joint_convention": WUJI_COMMAND_JOINT_CONVENTION,
        "hand_joint_source_model": WUJI_COMMAND_SOURCE_MODEL,
        "hand_joint_conversion": WUJI_COMMAND_CONVERSION,
    }
    for field, expected in expected_contract.items():
        if command.get(field) != expected:
            raise ValueError(
                f"command {field} must be {expected!r}, got {command.get(field)!r}"
            )
    created = float(command.get("created_unix_s", float("nan")))
    age = time.time() - created
    if not math.isfinite(created) or age < -5.0 or age > max_age_s:
        raise ValueError(f"command timestamp is stale or invalid (age={age:.3f}s)")
    base_t_ee = read_transform(command.get("base_T_ee"), "base_T_ee")
    pose = np.asarray(command.get("ee_pose_xyz_rpy"), dtype=np.float64)
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise ValueError("ee_pose_xyz_rpy must contain six finite values")
    if not np.allclose(xyz_rpy_to_matrix(pose), base_t_ee, atol=1e-6):
        raise ValueError("ee_pose_xyz_rpy does not match base_T_ee")
    position = base_t_ee[:3, 3]
    if np.any(position < EE_POSITION_LOWER_M) or np.any(position > EE_POSITION_UPPER_M):
        raise ValueError(f"EE position {position.tolist()} is outside the conservative workspace")
    if float(np.linalg.norm(position)) > EE_POSITION_MAX_RADIUS_M:
        raise ValueError("EE position exceeds the conservative radial workspace")
    joints = np.asarray(command.get("hand_joints"), dtype=np.float64)
    names = command.get("hand_joint_names")
    canonical_joints = reorder_wuji_joints(joints, names if isinstance(names, list) else [])
    normalized = dict(command)
    normalized["base_T_ee"] = base_t_ee
    normalized["ee_pose_xyz_rpy"] = pose
    normalized["hand_joints"] = canonical_joints
    return normalized


def validate_inference_request(
    request: Any, *, max_age_s: float, expected_side: str
) -> dict[str, Any]:
    """Validate a control-host request to start one camera inference."""
    if not isinstance(request, dict) or request.get("format") != INFERENCE_REQUEST_FORMAT:
        raise ValueError(f"request format must be {INFERENCE_REQUEST_FORMAT!r}")
    request_id = request.get("request_id")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise ValueError("request_id must be a non-empty string of at most 128 characters")
    if request.get("action") != "infer_grasp":
        raise ValueError("request action must be 'infer_grasp'")
    if request.get("side") != expected_side:
        raise ValueError(f"inference server only accepts side={expected_side!r}")
    created = float(request.get("created_unix_s", float("nan")))
    age = time.time() - created
    if not math.isfinite(created) or age < -5.0 or age > max_age_s:
        raise ValueError(f"request timestamp is stale or invalid (age={age:.3f}s)")
    return dict(request)
