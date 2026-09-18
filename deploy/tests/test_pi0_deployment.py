from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.franka_pi0_policy_executor import (
    FrankaPi0PolicyExecutor,
    validate_live_packet,
    validate_server_metadata,
)
from deploy.pi0_deployment import (
    PI0_DEFAULT_ACTIONS_PER_CHUNK,
    PI0_HORIZON,
    deployment_lines,
    is_pi0_checkpoint,
    pi0_deployment_contract,
)
from utils.deployment_metadata import validate_deployment_contract


def test_pi0_contract_matches_trained_real_policy() -> None:
    contract = validate_deployment_contract(pi0_deployment_contract())

    assert contract["policy_type"] == "pi0"
    assert contract["fps"] == 15.0
    assert contract["max_actions_per_chunk"] == 50
    assert contract["actions_per_chunk"] == 8
    assert contract["camera_names"] == ["cam_front", "cam_left"]
    assert contract["trajectory_config"]["robot_state_dim"] == 8
    assert contract["trajectory_config"]["action_dim"] == 8
    assert contract["action_config"]["transport_action_representation"] == "absolute_target"
    assert deployment_lines(contract) == [
        "left",
        "gripper",
        "15",
        "8",
        "8",
        "joint",
        "cam_front,cam_left",
        "pi0",
        "8",
    ]


def test_pi0_checkpoint_recognition_requires_norm_stats(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    assert not is_pi0_checkpoint(checkpoint)

    stats = checkpoint / "assets/memory_260915-franka-left-2view-v1/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}")
    assert is_pi0_checkpoint(checkpoint)


def live_packet() -> dict:
    return {
        "arm_mode": "left",
        "include_right_arm": False,
        "include_gripper": True,
        "include_hand": False,
        "state_action_mode": "joint",
        "robot_state_dim": 8,
        "action_dim": 8,
        "camera_names": ["cam_front", "cam_left"],
        "camera_bundle_sequence": 12,
        "robot_state_stamp_s": 1.0,
        "state": np.zeros(8, dtype=np.float32),
    }


def test_pi0_live_packet_contract() -> None:
    validate_live_packet(live_packet())

    wrong = live_packet()
    wrong["camera_names"] = ["cam_front", "cam_right"]
    with pytest.raises(ValueError, match="cam_left"):
        validate_live_packet(wrong)


def test_pi0_server_metadata_contract() -> None:
    metadata = validate_server_metadata(pi0_deployment_contract())
    assert metadata["max_actions_per_chunk"] == PI0_HORIZON
    assert metadata["actions_per_chunk"] == PI0_DEFAULT_ACTIONS_PER_CHUNK

    wrong = pi0_deployment_contract()
    wrong["protocol_version"] = 2
    with pytest.raises(ValueError, match="protocol_version"):
        validate_server_metadata(wrong)


def test_pi0_payload_is_absolute_and_clips_gripper_only() -> None:
    action = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, 1.4])
    payload = FrankaPi0PolicyExecutor._command_payload_from_action(action)

    assert payload["left_joint_target"] == pytest.approx(action[:7])
    assert payload["left_gripper_command"] == 1.0
    assert set(payload) == {"timestamp", "left_joint_target", "left_gripper_command"}


def test_pi0_temporal_proposals_blend_overlapping_chunks() -> None:
    executor = FrankaPi0PolicyExecutor(
        SimpleNamespace(actions_per_chunk=8, fps=None, temporal_proposal_decay=0.5)
    )
    first = np.zeros((PI0_HORIZON, 8), dtype=np.float32)
    second = np.ones((PI0_HORIZON, 8), dtype=np.float32)

    first_stats = executor._merge_action_chunk(first, first_timestep=0)
    second_stats = executor._merge_action_chunk(second, first_timestep=8)

    assert first_stats["added"] == PI0_HORIZON
    assert first_stats["blended"] == 0
    assert second_stats["added"] == 8
    assert second_stats["blended"] == PI0_HORIZON - 8
    assert np.allclose(executor.action_queue[8], 2.0 / 3.0)
    assert np.allclose(executor.action_queue[PI0_HORIZON + 7], 1.0)
