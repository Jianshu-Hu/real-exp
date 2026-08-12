from __future__ import annotations

import numpy as np

from data_collection.validate_dataset import check_joint_safety_constraints
from utils.limit import (
    FR3_SAFE_POSITION_LOWER_RAD,
    FR3_SAFE_POSITION_UPPER_RAD,
)


def dual_arm_vector(left: np.ndarray, right: np.ndarray) -> list[float]:
    return [*left, 0.5, *right, 0.5]


def test_dataset_warning_reports_violation_frame_indices() -> None:
    midpoint = 0.5 * (FR3_SAFE_POSITION_LOWER_RAD + FR3_SAFE_POSITION_UPPER_RAD)
    states = [dual_arm_vector(midpoint, midpoint) for _ in range(4)]
    actions = [dual_arm_vector(midpoint, midpoint) for _ in range(4)]
    states[2] = dual_arm_vector(
        np.array(
            [FR3_SAFE_POSITION_UPPER_RAD[0] + 0.01, *midpoint[1:]],
        ),
        midpoint,
    )
    rows = [
        {
            "frame_index": frame_index + 10,
            "timestamp": frame_index * 0.1,
            "observation.state": state,
            "action": action,
        }
        for frame_index, (state, action) in enumerate(zip(states, actions))
    ]

    issues, warnings, metrics = check_joint_safety_constraints(
        rows,
        "absolute_joint_position",
    )

    assert issues == []
    assert len(warnings) == 1
    assert warnings[0].startswith("left state safety violations:")
    assert "position=1 frames=[12]" in warnings[0]
    assert "velocity=2 frames=[12, 13]" in warnings[0]
    assert "acceleration=2 frames=[12, 13]" in warnings[0]
    assert metrics["state_violation_steps"] == 2
    assert metrics["action_violation_steps"] == 0


def test_dataset_checks_absolute_action_at_each_timestep() -> None:
    midpoint = 0.5 * (FR3_SAFE_POSITION_LOWER_RAD + FR3_SAFE_POSITION_UPPER_RAD)
    states = [dual_arm_vector(midpoint, midpoint) for _ in range(4)]
    actions = [dual_arm_vector(midpoint, midpoint) for _ in range(4)]
    unsafe_action = midpoint.copy()
    unsafe_action[0] += 0.3
    actions[1] = dual_arm_vector(unsafe_action, midpoint)
    rows = [
        {
            "frame_index": frame_index,
            "timestamp": frame_index * 0.1,
            "observation.state": state,
            "action": action,
        }
        for frame_index, (state, action) in enumerate(zip(states, actions))
    ]

    issues, warnings, metrics = check_joint_safety_constraints(
        rows,
        "absolute_joint_position",
    )

    assert issues == []
    assert len(warnings) == 1
    assert warnings[0].startswith("left action safety violations:")
    assert "position=0 frames=[none]" in warnings[0]
    assert "velocity=2 frames=[1, 2]" in warnings[0]
    assert "acceleration=2 frames=[2, 3]" in warnings[0]
    assert metrics["state_violation_steps"] == 0
    assert metrics["action_violation_steps"] == 3
