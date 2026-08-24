from __future__ import annotations

import numpy as np
import json
import pytest

from pathlib import Path

from utils.trajectory_metadata import (
    require_dataset_trajectory_config,
    split_trajectory_vector,
    validate_action_trajectory_contract,
    validate_live_packet,
)


def test_explicit_trajectory_metadata_is_required(tmp_path: Path) -> None:
    info_path = tmp_path / "meta/info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {"shape": [8]},
                    "action": {"shape": [8]},
                }
            }
        )
    )

    with pytest.raises(FileNotFoundError, match="real_exp_trajectory_config.json"):
        require_dataset_trajectory_config(tmp_path)


def test_action_metadata_must_agree_with_hand_trajectory_contract() -> None:
    config = require_dataset_trajectory_config(Path("data/test-right-hand"))
    action_config = json.loads(
        Path("data/test-right-hand/meta/real_exp_action_config.json").read_text()
    )

    validate_action_trajectory_contract(action_config, config)
    with pytest.raises(ValueError, match="include_hand"):
        validate_action_trajectory_contract(action_config | {"include_hand": False}, config)
    with pytest.raises(ValueError, match="hand_action_representation"):
        validate_action_trajectory_contract(
            action_config | {"hand_action_representation": "delta_joint_position"}, config
        )


def test_dataset_contracts_describe_distinct_vector_layouts() -> None:
    left = require_dataset_trajectory_config(Path("data/test-left-gripper"))
    right = require_dataset_trajectory_config(Path("data/test-right-hand"))

    left_parts = split_trajectory_vector(np.arange(8), left)
    right_parts = split_trajectory_vector(np.arange(27), right)

    np.testing.assert_array_equal(left_parts["left_arm"], np.arange(7))
    assert left_parts["left_gripper"] == 7.0
    assert left_parts["right_arm"] is None
    np.testing.assert_array_equal(right_parts["right_arm"], np.arange(7))
    np.testing.assert_array_equal(right_parts["right_hand"], np.arange(7, 27))
    assert right_parts["left_arm"] is None


def test_live_packet_must_match_explicit_dataset_contract() -> None:
    config = require_dataset_trajectory_config(Path("data/test-left-gripper"))
    valid_packet = {
        "robot_state_dim": 8,
        "action_dim": 8,
        "arm_mode": "left",
        "include_right_arm": False,
        "include_gripper": True,
        "include_hand": False,
    }
    validate_live_packet(config, valid_packet)

    invalid_packet = valid_packet | {"arm_mode": "right"}
    try:
        validate_live_packet(config, invalid_packet)
    except ValueError as exc:
        assert "arm_mode" in str(exc)
    else:  # pragma: no cover - explicit assertion improves failure output
        raise AssertionError("live arm mode mismatch was accepted")
