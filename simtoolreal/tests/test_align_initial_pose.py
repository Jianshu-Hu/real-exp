from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simtoolreal"))
from align_initial_pose import DEFAULT_WORLD_FROM_ROBOT, DEFAULT_URDF, SIM_INITIAL_ARM, bridge_state, interpolate
from kinematics import PolicyKinematics


def test_training_target_has_seven_joints() -> None:
    assert SIM_INITIAL_ARM.shape == (7,)
    assert np.all(np.isfinite(SIM_INITIAL_ARM))


def test_shifted_target_has_expected_policy_palm_pose() -> None:
    q = np.zeros(27)
    q[:7] = SIM_INITIAL_ARM
    palm, _, _ = PolicyKinematics(DEFAULT_URDF).evaluate(q, DEFAULT_WORLD_FROM_ROBOT)
    np.testing.assert_allclose(palm, [0.0651, -0.1222, 0.7069], atol=2e-5)


def test_interpolation_starts_and_ends_exactly() -> None:
    start = np.zeros(7)
    goal = np.ones(7)
    np.testing.assert_allclose(interpolate(start, goal, 0.0, 30.0), start)
    np.testing.assert_allclose(interpolate(start, goal, 30.0, 30.0), goal)


def test_bridge_state_requires_right_27d_packet() -> None:
    packet = {"arm_mode": "right", "include_hand": True, "joint_state": [0.0] * 27}
    assert bridge_state(packet).shape == (27,)
    with pytest.raises(ValueError):
        bridge_state({"arm_mode": "left", "include_hand": True, "joint_state": [0.0] * 27})
