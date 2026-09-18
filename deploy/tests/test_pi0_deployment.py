from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from deploy.franka_pi0_policy_executor import (
    FrankaPi0PolicyExecutor,
    validate_live_packet,
    validate_server_metadata,
)
from deploy.build_deployment_camera_config import build_config as build_camera_config
from deploy.pi0_deployment import (
    PI0_DEFAULT_ACTIONS_PER_CHUNK,
    PI0_HORIZON,
    PI0_PRESS_BUTTON_TRAIN_CONFIG,
    deployment_lines,
    is_pi0_checkpoint,
    load_pi0_deployment_contract,
    pi0_deployment_contract,
)
from deploy.serve_pi0_policy import ValidatedPi0Policy
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
    assert (
        contract["action_config"]["transport_action_representation"]
        == "absolute_target"
    )
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


def test_pi0_camera_publisher_config_matches_image_contract() -> None:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "gello_software/ros2/src/franka_realsense_camera_publisher/config/example_three_cameras.yaml"
    )
    parameters = yaml.safe_load(config_path.read_text())["realsense_camera_publisher"][
        "ros__parameters"
    ]
    contract = pi0_deployment_contract()["features"]

    assert (parameters["height"], parameters["width"], 3) == tuple(
        contract["observation.images.cam_left"]["shape"]
    )
    assert (parameters["camera_3_height"], parameters["camera_3_width"], 3) == tuple(
        contract["observation.images.cam_front"]["shape"]
    )

    runtime = build_camera_config(
        yaml.safe_load(config_path.read_text()),
        SimpleNamespace(
            camera_1_enabled=True, camera_2_enabled=False, camera_3_enabled=True
        ),
    )["realsense_camera_publisher"]["ros__parameters"]
    assert [runtime[f"camera_{index}_enabled"] for index in range(1, 4)] == [
        True,
        False,
        True,
    ]


def test_pi0_checkpoint_recognition_requires_norm_stats(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    assert not is_pi0_checkpoint(checkpoint)

    stats = checkpoint / "assets/memory_260915-franka-left-2view-v1/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"std": [1.0] * 8 + [0.0] * 24},
                    "actions": {"std": [1.0] * 8 + [0.0] * 24},
                }
            }
        )
    )
    assert is_pi0_checkpoint(checkpoint)


@pytest.mark.parametrize(
    ("asset_id", "active_dim", "arm_mode", "cameras"),
    [
        ("task-franka-left-2view-v1", 8, "left", ["cam_front", "cam_left"]),
        ("task-franka-right-2view-v1", 8, "right", ["cam_front", "cam_right"]),
        (
            "task-franka-3view-v1",
            16,
            "duo",
            ["cam_front", "cam_left", "cam_right"],
        ),
    ],
)
def test_pi0_checkpoint_profile_detection(
    tmp_path: Path,
    asset_id: str,
    active_dim: int,
    arm_mode: str,
    cameras: list[str],
) -> None:
    checkpoint = tmp_path / "step"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    stats = checkpoint / "assets" / asset_id / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"std": [1.0] * active_dim + [0.0] * (32 - active_dim)},
                    "actions": {"std": [1.0] * active_dim + [0.0] * (32 - active_dim)},
                }
            }
        )
    )

    contract = load_pi0_deployment_contract(checkpoint)
    assert is_pi0_checkpoint(checkpoint)
    assert contract["trajectory_config"]["arm_mode"] == arm_mode
    assert contract["trajectory_config"]["action_dim"] == active_dim
    assert contract["camera_names"] == cameras


def test_press_button_checkpoint_uses_its_registered_training_config(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "press_button"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    stats = (
        checkpoint / "assets/press_button-memory-260917-franka-3view-v1/norm_stats.json"
    )
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"std": [1.0] * 16 + [0.0] * 16},
                    "actions": {"std": [1.0] * 16 + [0.0] * 16},
                }
            }
        )
    )
    contract = load_pi0_deployment_contract(checkpoint)

    assert contract["train_config"] == PI0_PRESS_BUTTON_TRAIN_CONFIG
    assert (
        contract["checkpoint_asset_id"] == "press_button-memory-260917-franka-3view-v1"
    )
    assert contract["trajectory_config"]["arm_mode"] == "duo"
    assert contract["camera_names"] == ["cam_front", "cam_left", "cam_right"]


def test_server_metadata_allows_checkpoint_specific_chunk_sizes() -> None:
    metadata = pi0_deployment_contract()
    metadata["max_actions_per_chunk"] = 24
    metadata["actions_per_chunk"] = 6

    assert validate_server_metadata(metadata)["max_actions_per_chunk"] == 24


def test_executor_defaults_to_checkpoint_action_chunk_size() -> None:
    executor = FrankaPi0PolicyExecutor(
        SimpleNamespace(
            actions_per_chunk=None,
            fps=None,
            task=None,
            temporal_proposal_decay=0.5,
        )
    )
    metadata = executor._configure_from_metadata(
        {
            **pi0_deployment_contract(),
            "max_actions_per_chunk": 24,
            "actions_per_chunk": 6,
        }
    )

    assert metadata["max_actions_per_chunk"] == 24
    assert executor.horizon == 24
    assert executor.actions_per_chunk == 6


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


def test_pi0_right_payload_routes_only_right_arm() -> None:
    trajectory = {
        **pi0_deployment_contract()["trajectory_config"],
        "arm_mode": "right",
        "arms": ["right"],
    }
    action = np.arange(8, dtype=float)

    payload = FrankaPi0PolicyExecutor._command_payload_from_action(action, trajectory)

    assert payload["right_joint_target"] == pytest.approx(action[:7])
    assert payload["right_gripper_command"] == 1.0
    assert set(payload) == {"timestamp", "right_joint_target", "right_gripper_command"}


def test_pi0_duo_payload_routes_both_arm_blocks(tmp_path: Path) -> None:
    checkpoint = tmp_path / "duo"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    stats = checkpoint / "assets/task-franka-3view-v1/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"std": [1.0] * 16 + [0.0] * 16},
                    "actions": {"std": [1.0] * 16 + [0.0] * 16},
                }
            }
        )
    )
    contract = load_pi0_deployment_contract(checkpoint)
    action = np.arange(16, dtype=float)
    action[7] = -1.0
    action[15] = 2.0

    payload = FrankaPi0PolicyExecutor._command_payload_from_action(
        action, contract["trajectory_config"]
    )

    assert payload["left_joint_target"] == pytest.approx(action[:7])
    assert payload["left_gripper_command"] == 0.0
    assert payload["right_joint_target"] == pytest.approx(action[8:15])
    assert payload["right_gripper_command"] == 1.0


def test_pi0_duo_server_maps_three_camera_bundle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "duo"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    stats = checkpoint / "assets/task-franka-3view-v1/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"std": [1.0] * 16 + [0.0] * 16},
                    "actions": {"std": [1.0] * 16 + [0.0] * 16},
                }
            }
        )
    )
    contract = load_pi0_deployment_contract(checkpoint)
    now = time.time()
    bundle = {
        "camera_sync": {
            "bundle_ready": True,
            "max_skew_s": 0.01,
            "reference_stamp_s": now,
        },
        "cameras": {
            "cam_front": {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)},
            "cam_left": {"rgb": np.zeros((240, 424, 3), dtype=np.uint8)},
            "cam_right": {"rgb": np.zeros((240, 424, 3), dtype=np.uint8)},
        },
    }

    class Policy:
        observation = None

        def infer(self, observation):
            self.observation = observation
            return {"actions": np.zeros((PI0_HORIZON, 16), dtype=np.float32)}

        def reset_history(self):
            return None

    policy = Policy()
    validated = ValidatedPi0Policy(
        policy,
        SimpleNamespace(get=lambda sequence: bundle if sequence == 7 else None),
        max_observation_age=0.25,
        max_camera_skew=0.067,
        contract=contract,
    )

    result = validated.infer(
        {
            "state": np.zeros(16, dtype=np.float32),
            "camera_bundle_sequence": 7,
            "robot_state_stamp_s": now,
        }
    )

    assert result["actions"].shape == (PI0_HORIZON, 16)
    assert set(policy.observation["images"]) == {
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    }


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
