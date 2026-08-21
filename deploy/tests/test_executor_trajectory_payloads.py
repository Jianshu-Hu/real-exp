from __future__ import annotations

from pathlib import Path
import subprocess
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from typing import Any

import numpy as np
import pytest

from deploy.franka_act_policy_executor import FrankaPolicyExecutor as ActExecutor
from deploy.deploy_lerobot_policy import (
    trajectory_contract_mismatches,
    validate_checkpoint_trajectory_contract,
)
from deploy.build_deployment_bridge_config import build_config, parse_args as parse_bridge_args
from deploy.franka_diffusion_policy_executor import FrankaPolicyExecutor as DiffusionExecutor
from deploy.wuji_hand_command_server import hand_target, telemetry_payload
from utils.trajectory_metadata import require_dataset_trajectory_config
from utils.deployment_metadata import (
    build_deployment_contract,
    load_checkpoint_deployment_contract,
    write_checkpoint_deployment_metadata,
)


class FakeTimedAction:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def get_action(self) -> np.ndarray:
        return self.values


def make_executor(
    executor_type: type[ActExecutor] | type[DiffusionExecutor], trajectory_config: dict[str, Any]
) -> ActExecutor | DiffusionExecutor:
    # Payload generation is pure once the two metadata contracts are resolved.
    executor = executor_type.__new__(executor_type)
    executor.trajectory_config = trajectory_config
    executor.action_config = {"gripper_action_representation": "absolute_width"}
    return executor


@pytest.mark.parametrize("executor_type", [ActExecutor, DiffusionExecutor])
def test_live_camera_set_must_match_dataset_exactly(executor_type: type[Any]) -> None:
    executor = executor_type.__new__(executor_type)
    executor.dataset_info = {
        "features": {
            "observation.images.cam_left": {"dtype": "video"},
            "observation.images.cam_front": {"dtype": "video"},
        }
    }

    executor._validate_live_cameras({"camera_names": ["cam_left", "cam_front"]})
    with pytest.raises(ValueError, match="missing: cam_front"):
        executor._validate_live_cameras({"camera_names": ["cam_left"]})
    with pytest.raises(ValueError, match="unexpected: cam_right"):
        executor._validate_live_cameras(
            {"camera_names": ["cam_left", "cam_front", "cam_right"]}
        )


@pytest.mark.parametrize("executor_type", [ActExecutor, DiffusionExecutor])
def test_left_gripper_payload_targets_only_the_left_arm(executor_type: type[Any]) -> None:
    config = require_dataset_trajectory_config(Path("data/test-left-gripper"))
    executor = make_executor(executor_type, config)

    payload = executor._command_payload_from_action(FakeTimedAction(np.arange(8) / 10.0))

    assert set(payload) == {"timestamp", "left_joint_target", "left_gripper_command"}
    assert payload["left_joint_target"] == pytest.approx(np.arange(7) / 10.0)
    assert payload["left_gripper_command"] == pytest.approx(0.7)


@pytest.mark.parametrize("executor_type", [ActExecutor, DiffusionExecutor])
def test_right_hand_payload_targets_only_the_right_arm(executor_type: type[Any]) -> None:
    config = require_dataset_trajectory_config(Path("data/test-right-hand"))
    executor = make_executor(executor_type, config)

    payload = executor._command_payload_from_action(FakeTimedAction(np.arange(27, dtype=float)))

    assert set(payload) == {"timestamp", "right_joint_target", "right_hand_target"}
    assert payload["right_joint_target"] == pytest.approx(np.arange(7, dtype=float))
    assert payload["right_hand_target"] == pytest.approx(np.arange(7, 27, dtype=float))


@pytest.mark.parametrize("executor_type", [ActExecutor, DiffusionExecutor])
def test_duo_gripper_payload_targets_both_arms(executor_type: type[Any]) -> None:
    config = require_dataset_trajectory_config(Path("data/test-traj-gen-pick-and-place"))
    executor = make_executor(executor_type, config)

    payload = executor._command_payload_from_action(FakeTimedAction(np.arange(16) / 20.0))

    assert set(payload) == {
        "timestamp",
        "left_joint_target",
        "left_gripper_command",
        "right_joint_target",
        "right_gripper_command",
    }
    assert payload["left_joint_target"] == pytest.approx(np.arange(7) / 20.0)
    assert payload["left_gripper_command"] == pytest.approx(0.35)
    assert payload["right_joint_target"] == pytest.approx(np.arange(8, 15) / 20.0)
    assert payload["right_gripper_command"] == pytest.approx(0.75)


def test_checkpoint_contract_must_match_model_and_live_feature_dimensions() -> None:
    trajectory_config = require_dataset_trajectory_config(Path("data/test-left-gripper"))
    policy_config = {
        "input_features": {"observation.state": {"shape": [8]}},
        "output_features": {"action": {"shape": [8]}},
    }

    validated = validate_checkpoint_trajectory_contract(
        policy_config,
        trajectory_config,
        source="test checkpoint",
        live_state_dim=8,
    )

    assert validated["arm_mode"] == "left"
    with pytest.raises(ValueError, match="live=27, checkpoint=8"):
        validate_checkpoint_trajectory_contract(
            policy_config,
            trajectory_config,
            source="test checkpoint",
            live_state_dim=27,
        )


def test_checkpoint_manifest_is_self_contained(tmp_path: Path) -> None:
    dataset = Path("data/test-left-gripper")
    info = json.loads((dataset / "meta/info.json").read_text())
    action = json.loads((dataset / "meta/real_exp_action_config.json").read_text())
    trajectory = require_dataset_trajectory_config(dataset)
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "type": "act",
                "n_action_steps": 32,
                "input_features": {
                    "observation.state": {"shape": [8]},
                    "observation.images.cam_left": {"shape": [3, 224, 224]},
                    "observation.images.cam_front": {"shape": [3, 224, 224]},
                },
                "output_features": {"action": {"shape": [8]}},
            }
        )
    )
    write_checkpoint_deployment_metadata(checkpoint, info, action, trajectory)

    manifest = load_checkpoint_deployment_contract(checkpoint)
    assert manifest["trajectory_config"]["arm_mode"] == "left"
    assert manifest["fps"] == 15.0
    assert manifest["camera_names"] == ["cam_front", "cam_left"]
    assert manifest["max_actions_per_chunk"] == 32


def test_dataset_checkpoint_contract_mismatch_identifies_the_changed_hardware() -> None:
    left = require_dataset_trajectory_config(Path("data/test-left-gripper"))
    right = require_dataset_trajectory_config(Path("data/test-right-hand"))

    mismatches = trajectory_contract_mismatches(left, right)

    assert any(mismatch.startswith("arm_mode:") for mismatch in mismatches)
    assert any(mismatch.startswith("end_effector:") for mismatch in mismatches)
    assert any(mismatch.startswith("action_dim:") for mismatch in mismatches)


def parse_config_output(output: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in output.splitlines())


@pytest.mark.parametrize(
    ("dataset", "expected_client", "expected_server"),
    [
        (
            "test-left-gripper",
            {
                "arm_mode": "left",
                "end_effector": "gripper",
                "state_dim": "8",
                "action_dim": "8",
                "cameras": "cam_left,cam_front",
                "robot_config": "example_fr3_config.yaml",
                "gripper_config": "example_fr3_config_franka_hand.yaml",
            },
            {
                "include_right_arm": "false",
                "include_gripper": "true",
                "include_hand": "false",
                "camera_1_enabled": "true",
                "camera_2_enabled": "true",
                "camera_3_enabled": "false",
            },
        ),
        (
            "test-right-hand",
            {
                "arm_mode": "right",
                "end_effector": "hand",
                "state_dim": "27",
                "action_dim": "27",
                "cameras": "cam_front,cam_right",
                "robot_config": "example_fr3_right_config_no_gripper.yaml",
                "gripper_config": "",
            },
            {
                "include_right_arm": "false",
                "include_gripper": "false",
                "include_hand": "true",
                "camera_1_enabled": "false",
                "camera_2_enabled": "true",
                "camera_3_enabled": "true",
            },
        ),
        (
            "test-traj-gen-pick-and-place",
            {
                "arm_mode": "duo",
                "end_effector": "gripper",
                "state_dim": "16",
                "action_dim": "16",
                "cameras": "cam_left,cam_front,cam_right",
                "robot_config": "example_fr3_duo_config.yaml",
                "gripper_config": "example_fr3_duo_config_franka_hand.yaml",
            },
            {
                "include_right_arm": "true",
                "include_gripper": "true",
                "include_hand": "false",
                "camera_1_enabled": "true",
                "camera_2_enabled": "true",
                "camera_3_enabled": "true",
            },
        ),
    ],
)
def test_deployment_launchers_resolve_dataset_contract(
    tmp_path: Path,
    dataset: str, expected_client: dict[str, str], expected_server: dict[str, str]
) -> None:
    dataset_root = f"data/{dataset}"
    dataset_path = Path(dataset_root)
    info = json.loads((dataset_path / "meta/info.json").read_text())
    action = json.loads((dataset_path / "meta/real_exp_action_config.json").read_text())
    trajectory = require_dataset_trajectory_config(dataset_path)
    policy_config = {
        "type": "act",
        "n_action_steps": 32,
        "input_features": {
            "observation.state": {"shape": [trajectory["robot_state_dim"]]},
            **{
                key: {"shape": [3, 224, 224]}
                for key, feature in info["features"].items()
                if key.startswith("observation.images.")
            },
        },
        "output_features": {"action": {"shape": [trajectory["action_dim"]]}},
    }
    contract = build_deployment_contract(policy_config, info, action, trajectory)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/deployment-metadata":
                self.send_error(404)
                return
            payload = json.dumps(contract).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    try:
        metadata_server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    except PermissionError:
        pytest.skip("network sockets are unavailable in this test sandbox")
    threading.Thread(target=metadata_server.serve_forever, daemon=True).start()
    client = subprocess.run(
        [
            "bash", "scripts/start_deployment_client.sh", "--metadata-address",
            f"127.0.0.1:{metadata_server.server_port}", "--print-config",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata_server.shutdown()
    metadata_server.server_close()
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "meta").mkdir(parents=True)
    (checkpoint / "config.json").write_text(json.dumps(policy_config))
    write_checkpoint_deployment_metadata(checkpoint, info, action, trajectory)
    server = subprocess.run(
        ["bash", "scripts/start_deployment_server.sh", "--policy-path", str(checkpoint), "--print-config"],
        check=True,
        capture_output=True,
        text=True,
    )
    client_config = parse_config_output(client.stdout)
    server_config = parse_config_output(server.stdout)
    for key, value in expected_client.items():
        assert client_config[key] == value
    for key, value in expected_server.items():
        assert server_config[key] == value


def test_wuji_worker_validates_targets_and_builds_bridge_telemetry() -> None:
    values = np.arange(20, dtype=float)

    np.testing.assert_array_equal(hand_target(values.tolist()), values)
    np.testing.assert_array_equal(hand_target({"target": values.tolist()}), values)
    assert hand_target(values[:19]) is None
    assert hand_target([*values[:19], np.nan]) is None
    assert telemetry_payload("right", values, values + 1, 123.0) == {
        "side": "right",
        "current": values.tolist(),
        "target": (values + 1).tolist(),
        "stamp_s": 123.0,
    }


def test_runtime_bridge_config_rejects_inconsistent_arm_and_end_effector_flags() -> None:
    base = {"lerobot_data_bridge": {"ros__parameters": {}}}
    common = [
        "--base-config", "unused.yaml",
        "--output", "unused-output.yaml",
        "--sample-rate-hz", "15",
        "--publish-host", "192.168.50.13",
        "--publish-port", "5555",
        "--command-host", "192.168.50.13",
        "--command-port", "5556",
        "--camera-cache-host", "127.0.0.1",
        "--camera-cache-port", "5557",
        "--hand-telemetry-host", "192.168.50.13",
        "--hand-telemetry-port", "5558",
        "--camera-1-enabled", "false",
        "--camera-2-enabled", "true",
        "--camera-3-enabled", "true",
    ]
    inconsistent_arm = parse_bridge_args(
        common
        + [
            "--include-right-arm", "true",
            "--arm-mode", "right",
            "--include-gripper", "false",
            "--include-hand", "true",
        ]
    )
    inconsistent_effector = parse_bridge_args(
        common
        + [
            "--include-right-arm", "false",
            "--arm-mode", "right",
            "--include-gripper", "true",
            "--include-hand", "true",
        ]
    )

    with pytest.raises(ValueError, match="include_right_arm"):
        build_config(base, inconsistent_arm)
    with pytest.raises(ValueError, match="both a gripper and a Wuji hand"):
        build_config(base, inconsistent_effector)
