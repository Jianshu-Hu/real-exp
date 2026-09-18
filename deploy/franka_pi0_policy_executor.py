"""Robot-side executor for the RMBench Franka Pi0 policy.

The Pi0/JAX process and camera pixels stay on the inference computer. The
server metadata selects a left-only, right-only, or dual-arm Franka contract;
this executor validates the live bridge and routes absolute targets to only
the selected arm(s). Passing ``--execute`` is the only operation that enables
real robot commands.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess  # nosec
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.action_aggregation import TemporalProposalAggregator  # noqa: E402
from deploy.pi0_deployment import (  # noqa: E402
    PI0_DEFAULT_ACTIONS_PER_CHUNK,
    PI0_HORIZON,
    pi0_deployment_contract,
)
from utils.trajectory_metadata import split_trajectory_vector  # noqa: E402


DEFAULT_SERVER_IP = os.environ.get("DEPLOYMENT_SERVER_IP", "192.168.50.13")
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs" / "deployment_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-address", default=f"{DEFAULT_SERVER_IP}:8080")
    parser.add_argument("--zmq-host", default=DEFAULT_SERVER_IP)
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--command-zmq-host", default=DEFAULT_SERVER_IP)
    parser.add_argument("--command-zmq-port", type=int, default=5556)
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=None,
        help="Execution prefix length; defaults to the checkpoint metadata.",
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--task",
        default=None,
        help="Optional language prompt override; defaults to checkpoint metadata.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print action-chunk intervals and inference request/response times.",
    )
    parser.add_argument(
        "--temporal-proposal-decay",
        type=float,
        default=0.5,
        help=(
            "Exponential generation-age decay for overlapping action proposals. The newest "
            "proposal has weight 1 and each older generation has weight decay**age. "
            "Defaults to 0.5."
        ),
    )
    parser.add_argument("--bridge-activation-service", default="/set_deployment_active")
    parser.add_argument("--no-auto-activate-bridge", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def import_zmq_runtime():
    try:
        import zmq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyzmq is required in the robot-side environment."
        ) from exc
    return zmq


def validate_live_packet(
    packet: dict[str, Any], metadata: dict[str, Any] | None = None
) -> None:
    metadata = metadata or pi0_deployment_contract()
    trajectory = metadata["trajectory_config"]
    expected = {
        "arm_mode": trajectory["arm_mode"],
        "include_right_arm": trajectory["arm_mode"] == "duo",
        "include_gripper": True,
        "include_hand": False,
        "state_action_mode": "joint",
        "robot_state_dim": trajectory["robot_state_dim"],
        "action_dim": trajectory["action_dim"],
    }
    mismatches = [
        f"{key}: live={packet.get(key)!r}, Pi0={value!r}"
        for key, value in expected.items()
        if packet.get(key) != value
    ]
    live_cameras = set(packet.get("camera_names", ()))
    expected_cameras = set(metadata["camera_names"])
    if live_cameras != expected_cameras:
        mismatches.append(
            f"cameras: live={sorted(live_cameras)!r}, Pi0={sorted(expected_cameras)!r}"
        )
    state_dim = int(trajectory["robot_state_dim"])
    state = np.asarray(packet.get("state"), dtype=np.float32)
    if state.shape != (state_dim,) or not np.isfinite(state).all():
        mismatches.append(
            f"state must be finite shape ({state_dim},), got {state.shape}"
        )
    try:
        state_stamp = float(packet["robot_state_stamp_s"])
    except (KeyError, TypeError, ValueError):
        mismatches.append("robot_state_stamp_s is missing or invalid")
    else:
        if not np.isfinite(state_stamp):
            mismatches.append("robot_state_stamp_s must be finite")
    if mismatches:
        raise ValueError(
            "Live bridge does not match the Pi0 contract: " + "; ".join(mismatches)
        )


def validate_server_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    required = {
        "protocol": "real-exp-pi0-websocket",
        "protocol_version": 1,
        "policy_type": "pi0",
    }
    mismatches = [
        f"{key}: server={metadata.get(key)!r}, executor={value!r}"
        for key, value in required.items()
        if metadata.get(key) != value
    ]
    try:
        max_actions_per_chunk = int(metadata["max_actions_per_chunk"])
        actions_per_chunk = int(metadata["actions_per_chunk"])
    except (KeyError, TypeError, ValueError):
        mismatches.append("invalid action chunk metadata")
    else:
        if (
            max_actions_per_chunk <= 0
            or not 1 <= actions_per_chunk <= max_actions_per_chunk
        ):
            mismatches.append(
                "actions_per_chunk/max_actions_per_chunk="
                f"{actions_per_chunk}/{max_actions_per_chunk}"
            )
    trajectory = metadata.get("trajectory_config") or {}
    arm_mode = trajectory.get("arm_mode")
    expected_arms = ["left", "right"] if arm_mode == "duo" else [arm_mode]
    if (
        arm_mode not in {"left", "right", "duo"}
        or trajectory.get("arms") != expected_arms
    ):
        mismatches.append(f"arm layout={arm_mode!r}/{trajectory.get('arms')!r}")
    expected_dim = 16 if arm_mode == "duo" else 8
    if (
        trajectory.get("robot_state_dim") != expected_dim
        or trajectory.get("action_dim") != expected_dim
    ):
        mismatches.append(
            "state/action dimensions="
            f"{trajectory.get('robot_state_dim')!r}/{trajectory.get('action_dim')!r}"
        )
    camera_names = metadata.get("camera_names")
    allowed_cameras = {"cam_front", *(f"cam_{side}" for side in expected_arms)}
    if (
        not isinstance(camera_names, list)
        or not camera_names
        or len(camera_names) != len(set(camera_names))
        or "cam_front" not in camera_names
        or set(camera_names) - allowed_cameras
    ):
        mismatches.append(f"camera_names={camera_names!r}")
    if mismatches:
        raise ValueError("Pi0 server metadata mismatch: " + "; ".join(mismatches))
    return metadata


class FrankaPi0PolicyExecutor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if args.actions_per_chunk is not None and args.actions_per_chunk <= 0:
            raise ValueError("--actions-per-chunk must be positive.")
        if args.fps is not None and args.fps <= 0:
            raise ValueError("--fps must be positive.")
        if not 0.0 <= args.temporal_proposal_decay <= 1.0:
            raise ValueError(
                "--temporal-proposal-decay must be between 0 and 1, "
                f"got {args.temporal_proposal_decay}"
            )
        self.websocket = None
        self.command_socket = None
        self.bridge_active = False
        self.fps = 15.0
        self.horizon = PI0_HORIZON
        self.requested_actions_per_chunk = args.actions_per_chunk
        self.actions_per_chunk = (
            PI0_DEFAULT_ACTIONS_PER_CHUNK
            if args.actions_per_chunk is None
            else args.actions_per_chunk
        )
        self.metadata = pi0_deployment_contract()
        self.trajectory_config = self.metadata["trajectory_config"]
        self.action_dim = int(self.trajectory_config["action_dim"])
        self.prompt = str(self.metadata["prompt"])
        self.log_file = None
        self.log_path: Path | None = None
        self._last_chunk_received_at: float | None = None
        self.action_aggregator = TemporalProposalAggregator(
            args.temporal_proposal_decay
        )
        self.action_queue: dict[int, np.ndarray] = {}

    @staticmethod
    def _command_payload_from_action(
        action: Any, trajectory_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        trajectory_config = (
            trajectory_config or pi0_deployment_contract()["trajectory_config"]
        )
        values = np.asarray(action, dtype=float)
        action_dim = int(trajectory_config["action_dim"])
        if values.shape != (action_dim,) or not np.isfinite(values).all():
            raise ValueError(
                f"Pi0 action must be finite with shape ({action_dim},), got {values.shape}."
            )
        split = split_trajectory_vector(values, trajectory_config)
        payload: dict[str, Any] = {"timestamp": time.time()}
        for side in trajectory_config["arms"]:
            payload[f"{side}_joint_target"] = np.asarray(
                split[f"{side}_arm"], dtype=float
            ).tolist()
            payload[f"{side}_gripper_command"] = float(
                np.clip(split[f"{side}_gripper"], 0.0, 1.0)
            )
        return payload

    def _set_bridge_active(self, active: bool) -> None:
        if self.args.no_auto_activate_bridge or self.bridge_active == active:
            return
        command = [
            "ros2",
            "service",
            "call",
            self.args.bridge_activation_service,
            "std_srvs/srv/SetBool",
            f"{{data: {'true' if active else 'false'}}}",
        ]
        environment = os.environ.copy()
        environment.setdefault("ROS_DOMAIN_ID", "0")
        environment["ROS_LOCALHOST_ONLY"] = "0"
        environment["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "SUBNET"
        result = subprocess.run(  # nosec B603
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
            env=environment,
        )
        if result.returncode != 0 or "success=True" not in result.stdout:
            raise RuntimeError(
                f"Could not {'activate' if active else 'deactivate'} deployment bridge.\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        self.bridge_active = active

    def _connect_policy(self) -> dict[str, Any]:
        try:
            import websocket
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "websocket-client is required in the robot-side environment."
            ) from exc
        uri = f"ws://{self.args.server_address}"
        self.websocket = websocket.create_connection(
            uri, timeout=300, enable_multithread=False
        )
        return self._configure_from_metadata(json.loads(self.websocket.recv()))

    def _configure_from_metadata(self, value: dict[str, Any]) -> dict[str, Any]:
        metadata = validate_server_metadata(value)
        self.metadata = metadata
        self.trajectory_config = metadata["trajectory_config"]
        self.action_dim = int(self.trajectory_config["action_dim"])
        self.prompt = self.args.task or str(metadata["prompt"])
        self.fps = float(metadata["fps"])
        self.horizon = int(metadata["max_actions_per_chunk"])
        self.actions_per_chunk = (
            int(metadata["actions_per_chunk"])
            if self.requested_actions_per_chunk is None
            else int(self.requested_actions_per_chunk)
        )
        if not 1 <= self.actions_per_chunk <= self.horizon:
            raise ValueError(
                f"--actions-per-chunk must be in [1, {self.horizon}], "
                f"got {self.actions_per_chunk}."
            )
        if self.args.fps is not None and not np.isclose(self.args.fps, self.fps):
            raise ValueError(
                f"--fps={self.args.fps:g} does not match Pi0 server fps={self.fps:g}."
            )
        return metadata

    def _infer(self, packet: dict[str, Any]) -> np.ndarray:
        if self.websocket is None:
            raise RuntimeError("Pi0 websocket is not connected.")
        request = {
            "state": np.asarray(packet["state"], dtype=float).tolist(),
            "camera_bundle_sequence": int(packet["camera_bundle_sequence"]),
            "robot_state_stamp_s": float(packet["robot_state_stamp_s"]),
            "prompt": self.prompt,
        }
        started = time.perf_counter()
        self.websocket.send(json.dumps(request, separators=(",", ":")))
        response = json.loads(self.websocket.recv())
        chunk_received_at = time.perf_counter()
        if "error" in response:
            raise RuntimeError(f"Pi0 server rejected inference: {response['error']}")
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if (
            actions.shape != (self.horizon, self.action_dim)
            or not np.isfinite(actions).all()
        ):
            raise RuntimeError(f"Pi0 server returned invalid actions {actions.shape}.")
        inference_s = chunk_received_at - started
        if self.args.debug:
            if self._last_chunk_received_at is None:
                print(
                    f"[debug] first action chunk: inference={inference_s:.3f}s",
                    flush=True,
                )
            else:
                chunk_interval_s = chunk_received_at - self._last_chunk_received_at
                print(
                    f"[debug] action chunk interval={chunk_interval_s:.3f}s "
                    f"inference={inference_s:.3f}s",
                    flush=True,
                )
        self._last_chunk_received_at = chunk_received_at
        self._log(
            {
                "event": "action_chunk_received",
                "inference_s": inference_s,
                "camera_bundle_sequence": request["camera_bundle_sequence"],
                "state": request["state"],
                "actions": actions[: self.actions_per_chunk].tolist(),
            }
        )
        return actions

    def _merge_action_chunk(
        self, actions: np.ndarray, first_timestep: int
    ) -> dict[str, Any]:
        """Merge a chunk into the future action queue by execution timestep."""
        generation = self.action_aggregator.begin_chunk()
        blended = 0
        added = 0
        for offset, action in enumerate(actions):
            timestep = first_timestep + offset
            merged_action = self.action_aggregator.add(timestep, generation, action)
            if timestep in self.action_queue:
                blended += 1
            else:
                added += 1
            self.action_queue[timestep] = merged_action
        return {
            "chunk_generation": generation,
            "added": added,
            "blended": blended,
            "queue_size": len(self.action_queue),
            "first_timestep": first_timestep,
            "last_timestep": first_timestep + len(actions) - 1,
        }

    def _pop_action(self, timestep: int) -> np.ndarray:
        try:
            action = self.action_queue.pop(timestep)
        except KeyError as exc:
            raise RuntimeError(
                f"No merged Pi0 action is available for execution timestep {timestep}."
            ) from exc
        self.action_aggregator.discard(timestep)
        return action

    def _init_log(self) -> None:
        name = self.args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S_pi0")
        log_dir = self.args.log_dir.expanduser().resolve() / name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / "samples.jsonl"
        self.log_file = self.log_path.open("a", buffering=1)
        (log_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "server_address": self.args.server_address,
                    "fps": self.fps,
                    "actions_per_chunk": self.actions_per_chunk,
                    "temporal_proposal_decay": self.args.temporal_proposal_decay,
                    "prompt": self.prompt,
                    "arm_mode": self.trajectory_config["arm_mode"],
                    "state_action_dim": self.action_dim,
                    "execute": self.args.execute,
                    "action_representation": "absolute_target",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Deployment log: {self.log_path}")

    def _log(self, record: dict[str, Any]) -> None:
        if self.log_file is not None:
            self.log_file.write(json.dumps({"wall_time": time.time(), **record}) + "\n")

    @staticmethod
    def _recv_latest(socket: Any, zmq: Any) -> dict[str, Any]:
        packet = socket.recv_pyobj()
        while True:
            try:
                packet = socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                return packet

    def run(self) -> None:
        zmq = import_zmq_runtime()
        metadata = self._connect_policy()
        print("Franka Pi0 policy executor")
        print("--------------------------")
        print(f"server_address: {self.args.server_address}")
        print(f"train_config: {metadata['train_config']}")
        print(f"arm_mode: {self.trajectory_config['arm_mode']}")
        print(f"fps: {self.fps:g}")
        print(f"actions_per_chunk: {self.actions_per_chunk} / {self.horizon}")
        print(f"temporal_proposal_decay: {self.args.temporal_proposal_decay:g}")
        print(f"execute: {self.args.execute}")

        context = zmq.Context()
        observation_socket = context.socket(zmq.SUB)
        observation_socket.setsockopt(zmq.RCVHWM, 1)
        observation_socket.setsockopt(zmq.CONFLATE, 1)
        observation_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        observation_socket.connect(f"tcp://{self.args.zmq_host}:{self.args.zmq_port}")
        command_socket = context.socket(zmq.PUSH)
        command_socket.setsockopt(zmq.SNDHWM, 1)
        command_socket.connect(
            f"tcp://{self.args.command_zmq_host}:{self.args.command_zmq_port}"
        )
        self.command_socket = command_socket
        self._set_bridge_active(True)
        print("Waiting for the live bridge...")
        try:
            first_packet = observation_socket.recv_pyobj()
            validate_live_packet(first_packet, metadata)
            print(
                f"Live contract: arm_mode={self.trajectory_config['arm_mode']}, "
                f"state/action={self.action_dim}/{self.action_dim}, "
                f"cameras={metadata['camera_names']}, absolute targets"
            )
            self._init_log()
            current_packet = first_packet
            next_timestep = 0
            while True:
                validate_live_packet(current_packet, metadata)
                chunk = self._infer(current_packet)
                merge_stats = self._merge_action_chunk(chunk, next_timestep)
                self._log({"event": "action_chunk_merged", **merge_stats})
                for chunk_index in range(self.actions_per_chunk):
                    loop_started = time.perf_counter()
                    current_packet = self._recv_latest(observation_socket, zmq)
                    validate_live_packet(current_packet, metadata)
                    action = self._pop_action(next_timestep)
                    payload = self._command_payload_from_action(
                        action, self.trajectory_config
                    )
                    if self.args.execute:
                        command_socket.send_pyobj(payload)
                    self._log(
                        {
                            "event": "action_executed"
                            if self.args.execute
                            else "action_predicted",
                            "chunk_index": chunk_index,
                            "timestep": next_timestep,
                            "action": np.asarray(action, dtype=float).tolist(),
                            "current_state": list(current_packet["state"]),
                            "command_payload": payload if self.args.execute else None,
                        }
                    )
                    next_timestep += 1
                    time.sleep(
                        max(0.0, 1.0 / self.fps - (time.perf_counter() - loop_started))
                    )
                current_packet = self._recv_latest(observation_socket, zmq)
        except KeyboardInterrupt:
            print("\nStopping Franka Pi0 policy executor...")
        finally:
            try:
                self._set_bridge_active(False)
            except Exception as exc:
                print(f"Warning: could not deactivate bridge: {exc}", file=sys.stderr)
            if self.websocket is not None:
                self.websocket.close()
            if self.log_file is not None:
                self.log_file.close()
            observation_socket.close(0)
            command_socket.close(0)
            context.term()


def main() -> None:
    FrankaPi0PolicyExecutor(parse_args()).run()


if __name__ == "__main__":
    main()
