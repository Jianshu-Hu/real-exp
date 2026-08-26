from __future__ import annotations

import numpy as np
import pytest

from deploy.franka_act_policy_executor import FrankaPolicyExecutor as ActExecutor
from deploy.franka_diffusion_policy_executor import (
    FrankaPolicyExecutor as DiffusionExecutor,
    summarize_action_deltas,
)
from utils.trajectory_metadata import (
    split_absolute_transport_action,
    validate_trajectory_config,
)


class _Action:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def get_action(self) -> np.ndarray:
        return self._values


def test_schema_v1_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        validate_trajectory_config(
            {
                "schema_version": 1,
                "arm_mode": "left",
                "arms": ["left"],
                "end_effector": "arm",
                "state_action_mode": "joint",
                "robot_state_dim": 7,
                "action_dim": 7,
            },
            7,
            7,
            source="test",
        )


def _ee_gripper_config() -> dict[str, object]:
    return validate_trajectory_config(
        {
            "schema_version": 2,
            "arm_mode": "duo",
            "arms": ["left", "right"],
            "end_effector": "gripper",
            "include_gripper": True,
            "include_hand": False,
            "state_action_mode": "end_effector",
            "state_representation": "end_effector_position_rotation_6d",
            "action_representation": "delta_end_effector_position_rotation_vector",
            "robot_state_dim": 20,
            "action_dim": 14,
        },
        20,
        14,
        source="test",
    )


def test_absolute_ee_transport_layout_is_20d_for_dual_gripper() -> None:
    config = _ee_gripper_config()
    values = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5]), 2)
    split = split_absolute_transport_action(values, config)
    assert split["left_ee_pose"].shape == (9,)
    assert split["right_ee_pose"].shape == (9,)
    assert split["left_gripper"] == 0.5
    assert split["right_gripper"] == 0.5
    assert summarize_action_deltas([values, values], config)["count"] == 1


def test_both_executors_accept_absolute_ee_transport_targets() -> None:
    config = _ee_gripper_config()
    values = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5]), 2)
    for executor_type in (ActExecutor, DiffusionExecutor):
        executor = executor_type.__new__(executor_type)
        executor.trajectory_config = config
        executor.action_config = {
            "arm_action_representation": "delta_end_effector_position_rotation_vector",
            "gripper_action_representation": "absolute_width",
            "transport_action_representation": "absolute_target",
        }
        payload = executor._command_payload_from_action(_Action(values))
        assert len(payload["left_ee_pose_target"]) == 9
        assert len(payload["right_ee_pose_target"]) == 9
        assert payload["left_gripper_command"] == 0.5
        assert payload["right_gripper_command"] == 0.5
