from __future__ import annotations

from data_collection.trajectory_metadata import legacy_trajectory_config


def test_legacy_metadata_uses_explicit_right_arm_mode() -> None:
    config = legacy_trajectory_config(
        {
            "arm_mode": "right",
            "include_right_arm": False,
            "include_gripper": False,
            "include_hand": True,
        },
        state_dim=27,
        action_dim=27,
    )

    assert config["arm_mode"] == "right"
    assert config["arms"] == ["right"]
    assert config["end_effector"] == "hand"


def test_legacy_metadata_without_arm_mode_keeps_historical_left_default() -> None:
    config = legacy_trajectory_config(
        {"include_right_arm": False, "include_gripper": False, "include_hand": True},
        state_dim=27,
        action_dim=27,
    )

    assert config["arm_mode"] == "left"
    assert config["arms"] == ["left"]
