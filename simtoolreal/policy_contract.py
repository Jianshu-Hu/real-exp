"""FR3 + Wuji Hand 2 constants shared by real-policy deployment code.

The ordering and observation layout mirror the ``franka_wuji_right`` profile in
``libs/SimToolReal-Franka-Wuji2``.  Keeping the contract here makes deployment
independent of Isaac Lab while still making every checkpoint-facing assumption
visible and testable.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np


ARM_JOINT_NAMES = tuple(f"right_fr3_joint{i}" for i in range(1, 8))
HAND_JOINT_NAMES = (
    "r_thumb_cmc_flex",
    "r_thumb_cmc_abd",
    "r_thumb_mcp",
    "r_thumb_ip",
    "r_index_finger_mcp_flex",
    "r_index_finger_mcp_abd",
    "r_index_finger_pip",
    "r_index_finger_dip",
    "r_middle_finger_mcp_flex",
    "r_middle_finger_mcp_abd",
    "r_middle_finger_pip",
    "r_middle_finger_dip",
    "r_ring_finger_mcp_flex",
    "r_ring_finger_mcp_abd",
    "r_ring_finger_pip",
    "r_ring_finger_dip",
    "r_pinky_mcp_flex",
    "r_pinky_mcp_abd",
    "r_pinky_pip",
    "r_pinky_dip",
)
JOINT_NAMES = ARM_JOINT_NAMES + HAND_JOINT_NAMES
NUM_ARM_JOINTS = len(ARM_JOINT_NAMES)
NUM_HAND_JOINTS = len(HAND_JOINT_NAMES)
NUM_JOINTS = len(JOINT_NAMES)

OBS_FIELDS = (
    "joint_pos",
    "joint_vel",
    "prev_action_targets",
    "palm_pos",
    "palm_rot",
    "object_rot",
    "fingertip_pos_rel_palm",
    "keypoints_rel_palm",
    "keypoints_rel_goal",
    "object_scales",
)
OBS_FIELD_SIZES = {
    "joint_pos": NUM_JOINTS,
    "joint_vel": NUM_JOINTS,
    "prev_action_targets": NUM_JOINTS,
    "palm_pos": 3,
    "palm_rot": 4,
    "object_rot": 4,
    "fingertip_pos_rel_palm": 15,
    "keypoints_rel_palm": 12,
    "keypoints_rel_goal": 12,
    "object_scales": 3,
}
OBS_DIM = sum(OBS_FIELD_SIZES[name] for name in OBS_FIELDS)
ACTION_DIM = NUM_JOINTS
POLICY_RATE_HZ = 60.0

# Real FR3 position limits. The training asset intentionally expands some
# ranges; those expanded values remain authoritative for observation scaling,
# while these bounds are an additional final hardware-command clamp.
FR3_HARDWARE_LOWER = np.asarray(
    (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
    dtype=np.float64,
)
FR3_HARDWARE_UPPER = np.asarray(
    (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
    dtype=np.float64,
)

PALM_CENTER_OFFSET = np.asarray((0.00334318, -0.01757558, 0.15968175))
PALM_FRAME_QUAT_WXYZ = np.asarray(
    (0.004759216, -0.002238904, 0.018592786, 0.999813299)
)
FINGERTIP_OFFSETS = np.asarray(
    (
        (0.0, 0.0, -0.02978),
        (0.0, 0.0, -0.02475),
        (0.0, 0.0, -0.02475),
        (0.0, 0.0, -0.02475),
        (0.0, 0.0, -0.02475),
    )
)
FINGERTIP_LINK_NAMES = (
    "r_thumb_distal",
    "r_index_finger_distal",
    "r_middle_finger_distal",
    "r_ring_finger_distal",
    "r_pinky_distal",
)
KEYPOINT_CORNERS = np.asarray(
    ((1, 1, 1), (1, 1, -1), (-1, -1, 1), (-1, -1, -1)),
    dtype=np.float64,
)
OBJECT_BASE_SIZE = 0.04
KEYPOINT_SCALE = 1.5


def observation_dim(fields: Iterable[str]) -> int:
    """Return the actor width and reject unknown field names."""
    names = tuple(fields)
    unknown = [name for name in names if name not in OBS_FIELD_SIZES]
    if unknown:
        raise ValueError(f"unsupported policy observation fields: {unknown}")
    return sum(OBS_FIELD_SIZES[name] for name in names)


def canonical_name(name: str) -> str:
    """Normalize common ROS FR3 aliases to the policy's right-arm names."""
    value = str(name).strip()
    for prefix in ("right_", "right/", "right::"):
        if value.startswith(prefix) and value[len(prefix) :] in {
            f"fr3_joint{i}" for i in range(1, 8)
        }:
            return f"right_{value[len(prefix):]}"
    if value in {f"fr3_joint{i}" for i in range(1, 8)}:
        return f"right_{value}"
    return value


def reorder_joint_vector(values: Iterable[float], names: Iterable[str]) -> np.ndarray:
    """Reorder a named vector into the exact 27-DoF checkpoint order."""
    vector = np.asarray(list(values), dtype=np.float64)
    normalized_names = [canonical_name(name) for name in names]
    if vector.shape != (len(normalized_names),):
        raise ValueError("joint names and values must have the same length")
    if len(set(normalized_names)) != len(normalized_names):
        raise ValueError("joint state contains duplicate canonical names")
    by_name = dict(zip(normalized_names, vector, strict=True))
    missing = [name for name in JOINT_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"joint state is missing canonical joints: {missing}")
    result = np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("joint state contains non-finite values")
    return result


def load_joint_limits(robot_urdf: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all 27 trained limits by canonical name from the combined URDF.

    The selected URDF must be the same limit variant used to train the policy.
    This is deliberately name-based: XML declaration order and device order are
    not treated as interchangeable evidence.
    """
    try:
        root = ET.parse(robot_urdf).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"could not read policy robot URDF {robot_urdf}: {exc}") from exc
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            continue
        limits[str(joint.attrib.get("name", ""))] = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        )
    missing = [name for name in JOINT_NAMES if name not in limits]
    if missing:
        raise ValueError(f"policy robot URDF is missing canonical joints: {missing}")
    lower = np.asarray([limits[name][0] for name in JOINT_NAMES])
    upper = np.asarray([limits[name][1] for name in JOINT_NAMES])
    if np.any(lower >= upper):
        raise ValueError("robot joint limits must have positive ranges")
    return lower, upper


def hardware_command_limits(
    training_lower: np.ndarray, training_upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect trained target bounds with the real FR3 safety envelope."""
    lower = np.asarray(training_lower, dtype=np.float64).copy()
    upper = np.asarray(training_upper, dtype=np.float64).copy()
    if lower.shape != (NUM_JOINTS,) or upper.shape != (NUM_JOINTS,):
        raise ValueError("training limits must each contain 27 values")
    lower[:NUM_ARM_JOINTS] = np.maximum(lower[:NUM_ARM_JOINTS], FR3_HARDWARE_LOWER)
    upper[:NUM_ARM_JOINTS] = np.minimum(upper[:NUM_ARM_JOINTS], FR3_HARDWARE_UPPER)
    if np.any(lower >= upper):
        raise ValueError("training/hardware joint-limit intersection is empty")
    return lower, upper


assert NUM_JOINTS == 27
assert OBS_DIM == 134
