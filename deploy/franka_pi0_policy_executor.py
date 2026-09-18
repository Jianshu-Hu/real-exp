"""Robot-side executor for the RMBench Franka Pi0 policy.

The Pi0/JAX process and camera pixels stay on the inference computer. This
executor sends only the current 8-D robot state plus the synchronized camera
bundle sequence, receives a 50x8 absolute-target action chunk, and executes a
short receding-horizon prefix. Overlapping future proposals from successive
chunks are temporally aggregated. Passing ``--execute`` is the only operation
that enables real robot commands.
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

from deploy.action_aggregation import TemporalProposalAggregator
from deploy.pi0_deployment import PI0_DEFAULT_ACTIONS_PER_CHUNK, PI0_HORIZON, PI0_PROMPT


DEFAULT_SERVER_IP = os.environ.get("DEPLOYMENT_SERVER_IP", "192.168.50.13")
DEFAULT_LOG_ROOT = REPO_ROOT / "outputs" / "deployment_logs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-address", default=f"{DEFAULT_SERVER_IP}:8080")
    parser.add_argument("--zmq-host", default=DEFAULT_SERVER_IP)
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--command-zmq-host", default=DEFAULT_SERVER_IP)
    parser.add_argument("--command-zmq-port", type=int, default=5556)
    parser.add_argument("--actions-per-chunk", type=int, default=PI0_DEFAULT_ACTIONS_PER_CHUNK)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--task", default=PI0_PROMPT, help="Language prompt sent to Pi0.")
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
        raise ModuleNotFoundError("pyzmq is required in the robot-side environment.") from exc
    return zmq


def validate_live_packet(packet: dict[str, Any]) -> None:
    expected = {
        "arm_mode": "left",
        "include_right_arm": False,
        "include_gripper": True,
        "include_hand": False,
        "state_action_mode": "joint",
        "robot_state_dim": 8,
        "action_dim": 8,
    }
    mismatches = [
        f"{key}: live={packet.get(key)!r}, Pi0={value!r}"
        for key, value in expected.items()
        if packet.get(key) != value
    ]
    live_cameras = set(packet.get("camera_names", ()))
    expected_cameras = {"cam_front", "cam_left"}
    if live_cameras != expected_cameras:
        mismatches.append(
            f"cameras: live={sorted(live_cameras)!r}, Pi0={sorted(expected_cameras)!r}"
        )
    state = np.asarray(packet.get("state"), dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        mismatches.append(f"state must be finite shape (8,), got {state.shape}")
    try:
        state_stamp = float(packet["robot_state_stamp_s"])
    except (KeyError, TypeError, ValueError):
        mismatches.append("robot_state_stamp_s is missing or invalid")
    else:
        if not np.isfinite(state_stamp):
            mismatches.append("robot_state_stamp_s must be finite")
    if mismatches:
        raise ValueError("Live bridge does not match the Pi0 contract: " + "; ".join(mismatches))


def validate_server_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    required = {
        "protocol": "real-exp-pi0-websocket",
        "protocol_version": 1,
        "policy_type": "pi0",
        "max_actions_per_chunk": PI0_HORIZON,
    }
    mismatches = [
        f"{key}: server={metadata.get(key)!r}, executor={value!r}"
        for key, value in required.items()
        if metadata.get(key) != value
    ]
    if set(metadata.get("camera_names", ())) != {"cam_front", "cam_left"}:
        mismatches.append(f"camera_names={metadata.get('camera_names')!r}")
    if mismatches:
        raise ValueError("Pi0 server metadata mismatch: " + "; ".join(mismatches))
    return metadata


class FrankaPi0PolicyExecutor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        if not 1 <= args.actions_per_chunk <= PI0_HORIZON:
            raise ValueError(
                f"--actions-per-chunk must be in [1, {PI0_HORIZON}], got {args.actions_per_chunk}."
            )
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
        self.log_file = None
        self.log_path: Path | None = None
        self._last_chunk_received_at: float | None = None
        self.action_aggregator = TemporalProposalAggregator(args.temporal_proposal_decay)
        self.action_queue: dict[int, np.ndarray] = {}

    @staticmethod
    def _command_payload_from_action(action: Any) -> dict[str, Any]:
        values = np.asarray(action, dtype=float)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError(f"Pi0 action must be finite with shape (8,), got {values.shape}.")
        return {
            "timestamp": time.time(),
            "left_joint_target": values[:7].tolist(),
            "left_gripper_command": float(np.clip(values[7], 0.0, 1.0)),
        }

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
            command, check=False, capture_output=True, text=True, timeout=15.0, env=environment
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
        self.websocket = websocket.create_connection(uri, timeout=300, enable_multithread=False)
        metadata = validate_server_metadata(json.loads(self.websocket.recv()))
        self.fps = float(metadata["fps"])
        if self.args.fps is not None and not np.isclose(self.args.fps, self.fps):
            raise ValueError(f"--fps={self.args.fps:g} does not match Pi0 server fps={self.fps:g}.")
        return metadata

    def _infer(self, packet: dict[str, Any]) -> np.ndarray:
        if self.websocket is None:
            raise RuntimeError("Pi0 websocket is not connected.")
        request = {
            "state": np.asarray(packet["state"], dtype=float).tolist(),
            "camera_bundle_sequence": int(packet["camera_bundle_sequence"]),
            "robot_state_stamp_s": float(packet["robot_state_stamp_s"]),
            "prompt": self.args.task,
        }
        started = time.perf_counter()
        self.websocket.send(json.dumps(request, separators=(",", ":")))
        response = json.loads(self.websocket.recv())
        chunk_received_at = time.perf_counter()
        if "error" in response:
            raise RuntimeError(f"Pi0 server rejected inference: {response['error']}")
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if actions.shape != (PI0_HORIZON, 8) or not np.isfinite(actions).all():
            raise RuntimeError(f"Pi0 server returned invalid actions {actions.shape}.")
        inference_s = chunk_received_at - started
        if self.args.debug:
            if self._last_chunk_received_at is None:
                print(f"[debug] first action chunk: inference={inference_s:.3f}s", flush=True)
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
                "actions": actions[: self.args.actions_per_chunk].tolist(),
            }
        )
        return actions

    def _merge_action_chunk(self, actions: np.ndarray, first_timestep: int) -> dict[str, Any]:
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
                    "actions_per_chunk": self.args.actions_per_chunk,
                    "temporal_proposal_decay": self.args.temporal_proposal_decay,
                    "prompt": self.args.task,
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
        print(f"fps: {self.fps:g}")
        print(f"actions_per_chunk: {self.args.actions_per_chunk} / {PI0_HORIZON}")
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
            validate_live_packet(first_packet)
            print(
                "Live contract: state/action=8/8, cameras=['cam_front', 'cam_left'], "
                "absolute targets"
            )
            self._init_log()
            current_packet = first_packet
            next_timestep = 0
            while True:
                validate_live_packet(current_packet)
                chunk = self._infer(current_packet)
                merge_stats = self._merge_action_chunk(chunk, next_timestep)
                self._log({"event": "action_chunk_merged", **merge_stats})
                for chunk_index in range(self.args.actions_per_chunk):
                    loop_started = time.perf_counter()
                    current_packet = self._recv_latest(observation_socket, zmq)
                    validate_live_packet(current_packet)
                    action = self._pop_action(next_timestep)
                    payload = self._command_payload_from_action(action)
                    if self.args.execute:
                        command_socket.send_pyobj(payload)
                    self._log(
                        {
                            "event": "action_executed" if self.args.execute else "action_predicted",
                            "chunk_index": chunk_index,
                            "timestep": next_timestep,
                            "action": np.asarray(action, dtype=float).tolist(),
                            "current_state": list(current_packet["state"]),
                            "command_payload": payload if self.args.execute else None,
                        }
                    )
                    next_timestep += 1
                    time.sleep(max(0.0, 1.0 / self.fps - (time.perf_counter() - loop_started)))
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
