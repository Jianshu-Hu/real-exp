from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = REPO_ROOT / "simtoolreal"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from action import ActionPipeline
from kinematics import PolicyKinematics
from observation import ObservationBuilder, object_keypoints
from policy_contract import (
    FR3_HARDWARE_UPPER,
    JOINT_NAMES,
    OBS_DIM,
    hardware_command_limits,
    load_joint_limits,
    reorder_joint_vector,
)
from state_server import bridge_state, pose_in_world


ROBOT_URDF = (
    REPO_ROOT
    / "libs/SimToolReal-Franka-Wuji2/assets/urdf/franka_wuji_right/fr3v2_wuji_hand2_right.urdf"
)


def test_canonical_joint_reorder_accepts_ros_arm_aliases() -> None:
    names = list(reversed(JOINT_NAMES))
    names = [name.removeprefix("right_") if name.startswith("right_fr3") else name for name in names]
    values = list(range(27))
    reordered = reorder_joint_vector(values, names)
    by_name = dict(zip(["right_" + n if n.startswith("fr3_") else n for n in names], values))
    assert reordered.tolist() == [by_name[name] for name in JOINT_NAMES]


def test_training_limits_and_real_arm_intersection() -> None:
    lower, upper = load_joint_limits(ROBOT_URDF)
    command_lower, command_upper = hardware_command_limits(lower, upper)
    assert lower.shape == upper.shape == (27,)
    assert upper[5] > FR3_HARDWARE_UPPER[5]
    np.testing.assert_array_less(command_upper[:7], FR3_HARDWARE_UPPER + 1e-12)
    np.testing.assert_allclose(command_lower[7:], lower[7:])


def test_observation_has_upstream_134_layout_and_identity_goal_error() -> None:
    lower, upper = load_joint_limits(ROBOT_URDF)
    q = (lower + upper) / 2.0
    builder = ObservationBuilder(PolicyKinematics(ROBOT_URDF), lower, upper)
    pose = np.eye(4)
    pose[:3, 3] = (0.1, -0.2, 0.65)
    world_from_robot = np.eye(4)
    world_from_robot[1, 3] = 0.8
    result = builder.build(q, np.zeros(27), q, pose, pose, np.ones(3), world_from_robot)
    assert result.vector.shape == (OBS_DIM,) == (134,)
    np.testing.assert_allclose(result.fields["joint_pos"], 0.0, atol=1e-6)
    np.testing.assert_allclose(result.fields["keypoints_rel_goal"], 0.0, atol=1e-7)


def test_keypoint_rotation_and_scale() -> None:
    pose = np.eye(4)
    pose[:3, :3] = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    points = object_keypoints(pose, np.ones(3))
    np.testing.assert_allclose(points[0], (-0.03, 0.03, 0.03), atol=1e-12)


def test_action_semantics_and_final_hardware_clamp() -> None:
    lower, upper = load_joint_limits(ROBOT_URDF)
    command_lower, command_upper = hardware_command_limits(lower, upper)
    initial = (command_lower + command_upper) / 2.0
    pipeline = ActionPipeline(
        lower,
        upper,
        arm_moving_average=1.0,
        hand_moving_average=1.0,
        command_lower_limits=command_lower,
        command_upper_limits=command_upper,
    )
    pipeline.reset(initial)
    target = pipeline.targets(np.ones(27))
    np.testing.assert_allclose(target.arm, np.minimum(initial[:7] + 1.5 / 60.0, command_upper[:7]))
    np.testing.assert_allclose(target.hand, command_upper[7:])
    assert target.full.shape == (27,)


def test_bridge_contract_rejects_wrong_profile() -> None:
    with pytest.raises(ValueError, match="arm_mode=right"):
        bridge_state({"arm_mode": "left", "include_hand": True, "joint_state": [0.0] * 27})


def test_camera_pose_requires_and_applies_calibration() -> None:
    camera_from_object = np.eye(4)
    camera_from_object[2, 3] = 0.5
    world_from_camera = np.eye(4)
    world_from_camera[0, 3] = 1.0
    with pytest.raises(ValueError, match="calibration"):
        pose_in_world(camera_from_object, "camera", None, np.eye(4))
    world = pose_in_world(camera_from_object, "camera", world_from_camera, np.eye(4))
    np.testing.assert_allclose(world[:3, 3], (1.0, 0.0, 0.5))
