from __future__ import annotations

import argparse
import json
import pickle  # nosec
import subprocess  # nosec
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_DEPLOYMENT_LOG_ROOT = DEFAULT_OUTPUT_ROOT / "deployment_logs"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "pick_and_place_test"
ACTION_CONFIG_REL_PATH = Path("meta/real_exp_action_config.json")
INFO_REL_PATH = Path("meta/info.json")


def import_grpc_runtime():
    try:
        import grpc
        from lerobot.transport import services_pb2, services_pb2_grpc
        from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "gRPC runtime is missing. Install grpcio in the lerobot environment before running the executor."
        ) from exc

    return grpc, services_pb2, services_pb2_grpc, grpc_channel_options, send_bytes_in_chunks


def import_zmq_runtime():
    try:
        import zmq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyzmq is missing. Install pyzmq in the lerobot environment before running the executor."
        ) from exc

    return zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robot-side Franka executor for a remote LeRobot policy server."
    )
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--policy-type", default=None)
    parser.add_argument("--actions-per-chunk", type=int, default=None)
    parser.add_argument("--zmq-host", default="127.0.0.1")
    parser.add_argument("--zmq-port", type=int, default=5555)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--task", default="pick and place")
    parser.add_argument(
        "--chunk-size-threshold",
        type=float,
        default=0.9,
        help=(
            "Send a new observation when queue_size / actions_per_chunk is at or below this value. "
            "Default 0.9 was validated to give smooth overlapping ACT chunks."
        ),
    )
    parser.add_argument(
        "--aggregate-fn",
        default="conservative",
        help=(
            "Aggregation function for overlapping timesteps. "
            "Default conservative keeps more weight on the queued plan for smoother Franka motion."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--command-zmq-host", default="127.0.0.1")
    parser.add_argument("--command-zmq-port", type=int, default=5556)
    parser.add_argument("--bridge-activation-service", default="/set_deployment_active")
    parser.add_argument("--no-auto-activate-bridge", action="store_true")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_DEPLOYMENT_LOG_ROOT,
        help="Directory for deployment comparison logs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional deployment log run name. Defaults to a timestamp plus policy/chunk settings.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def infer_policy_type(policy_path: Path) -> str:
    config = load_json(policy_path / "config.json")
    policy_type = config.get("type")
    if not isinstance(policy_type, str) or not policy_type:
        raise ValueError(f"Could not infer policy type from {policy_path / 'config.json'}")
    return policy_type


def infer_actions_per_chunk(policy_path: Path, policy_type: str) -> int:
    config = load_json(policy_path / "config.json")
    if policy_type == "act":
        return int(config.get("n_action_steps", config.get("chunk_size", 1)))
    if policy_type == "diffusion":
        return int(config.get("n_action_steps", 1))
    if policy_type == "vqbet":
        return int(config.get("n_action_pred_token", 1))
    return 1


def maybe_load_policy_config(policy_path: Path) -> dict[str, Any] | None:
    config_path = policy_path / "config.json"
    if not config_path.exists():
        return None
    return load_json(config_path)


def build_live_lerobot_features(first_packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    state_dim = int(first_packet["robot_state_dim"])
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"state_{idx}" for idx in range(state_dim)],
        }
    }
    for camera_name in first_packet["camera_names"]:
        camera = first_packet["cameras"][camera_name]
        height, width, channels = camera["shape"]
        features[f"observation.images.{camera_name}"] = {
            "dtype": "image",
            "shape": (int(height), int(width), int(channels)),
            "names": ["height", "width", "channels"],
        }
    return features


def packet_to_raw_observation(packet: dict[str, Any], task: str) -> dict[str, Any]:
    state = np.asarray(packet["state"], dtype=np.float32)
    raw_observation: dict[str, Any] = {f"state_{idx}": float(value) for idx, value in enumerate(state)}
    for camera_name in packet["camera_names"]:
        raw_observation[camera_name] = np.asarray(packet["cameras"][camera_name]["rgb"], dtype=np.uint8)
    raw_observation["task"] = task
    return raw_observation


def split_action(action: np.ndarray | torch.Tensor) -> dict[str, Any]:
    action = np.asarray(action, dtype=float)
    if action.shape[0] == 16:
        return {
            "left_arm": action[0:7],
            "left_gripper": float(action[7]),
            "right_arm": action[8:15],
            "right_gripper": float(action[15]),
        }
    if action.shape[0] == 14:
        return {
            "left_arm": action[0:7],
            "left_gripper": None,
            "right_arm": action[7:14],
            "right_gripper": None,
        }
    raise ValueError(f"Unsupported action dimension: {action.shape[0]}")


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def default_run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    policy_name = args.policy_path.expanduser().name
    return (
        f"{timestamp}_{policy_name}_{args.policy_type or 'auto'}"
        f"_apc{args.actions_per_chunk or 'auto'}"
        f"_thr{args.chunk_size_threshold:g}_{args.aggregate_fn}"
    )


def format_optional_seconds(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}s"


class FrankaPolicyExecutor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.policy_path = args.policy_path.expanduser()
        self.dataset_root = args.dataset_root.resolve()
        local_policy_config = maybe_load_policy_config(self.policy_path)
        if args.policy_type is not None:
            self.policy_type = args.policy_type
        elif local_policy_config is not None:
            self.policy_type = infer_policy_type(self.policy_path)
        else:
            raise FileNotFoundError(
                "Could not find a local policy config.json for --policy-path. "
                "Pass --policy-type explicitly when the checkpoint only exists on the server."
            )

        if args.actions_per_chunk is not None:
            self.actions_per_chunk = args.actions_per_chunk
        elif local_policy_config is not None:
            self.actions_per_chunk = infer_actions_per_chunk(self.policy_path, self.policy_type)
        else:
            raise FileNotFoundError(
                "Could not find a local policy config.json for --policy-path. "
                "Pass --actions-per-chunk explicitly when the checkpoint only exists on the server."
            )

        if not 0.0 <= args.chunk_size_threshold <= 1.0:
            raise ValueError(f"--chunk-size-threshold must be between 0 and 1, got {args.chunk_size_threshold}")

        self.aggregate_fn = get_aggregate_function(args.aggregate_fn)
        self.dataset_info = load_json(self.dataset_root / INFO_REL_PATH)
        action_config_path = self.dataset_root / ACTION_CONFIG_REL_PATH
        if not action_config_path.exists():
            raise FileNotFoundError(
                f"Missing action metadata: {action_config_path}. "
                "Deployment requires meta/real_exp_action_config.json."
            )
        self.action_config = load_json(action_config_path)
        if self._arm_action_representation() != "absolute_joint_position":
            raise ValueError(
                "Deployment expects absolute_joint_position arm actions. "
                "Record and train a new absolute-target dataset before executing this policy."
            )

        grpc, services_pb2, services_pb2_grpc, grpc_channel_options, send_bytes_in_chunks = import_grpc_runtime()
        self.grpc = grpc
        self.services_pb2 = services_pb2
        self.send_bytes_in_chunks = send_bytes_in_chunks
        self.channel = grpc.insecure_channel(
            args.server_address,
            grpc_channel_options(initial_backoff=f"{1.0 / args.fps:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.action_queue: deque[TimedAction] = deque()
        self.action_queue_lock = threading.Lock()
        self.inflight_observation_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.shutdown_event = threading.Event()
        self.command_socket = None
        self.bridge_active = False
        self.latest_executed_timestep = -1
        self.must_go_pending = True
        self.action_chunk_size = max(1, self.actions_per_chunk)
        self.inflight_observation_timestep: int | None = None
        self.log_file = None
        self.log_path: Path | None = None
        self.log_started_monotonic = time.perf_counter()
        self.inflight_observation_sent_monotonic: float | None = None

    def _arm_action_representation(self) -> str:
        return str(
            self.action_config.get("arm_action_representation", "absolute_joint_position")
        ).strip().lower()

    def _gripper_action_representation(self) -> str:
        return str(
            self.action_config.get("gripper_action_representation", "binary_open_close")
        ).strip().lower()

    def _joint_targets_from_action(self, split: dict[str, Any]) -> dict[str, list[float]]:
        return {
            "left": np.asarray(split["left_arm"], dtype=float).tolist(),
            "right": np.asarray(split["right_arm"], dtype=float).tolist(),
        }

    def _queue_snapshot(self) -> dict[str, Any]:
        with self.action_queue_lock:
            timesteps = [action.get_timestep() for action in self.action_queue]
        return {
            "size": len(timesteps),
            "first_timestep": timesteps[0] if timesteps else None,
            "last_timestep": timesteps[-1] if timesteps else None,
        }

    def _next_observation_timestep(self) -> tuple[int, dict[str, Any]]:
        with self.action_queue_lock:
            queue_snapshot = self._queue_snapshot_unlocked()
            if self.action_queue:
                return self.action_queue[0].get_timestep(), queue_snapshot

        return max(self.latest_executed_timestep + 1, 0), queue_snapshot

    def _mark_observation_inflight(self, timestep: int) -> None:
        with self.inflight_observation_lock:
            self.inflight_observation_timestep = timestep
            self.inflight_observation_sent_monotonic = time.perf_counter()

    def _clear_observation_inflight(self) -> tuple[int | None, float | None]:
        with self.inflight_observation_lock:
            timestep = self.inflight_observation_timestep
            sent_monotonic = self.inflight_observation_sent_monotonic
            self.inflight_observation_timestep = None
            self.inflight_observation_sent_monotonic = None
        latency_s = time.perf_counter() - sent_monotonic if sent_monotonic is not None else None
        return timestep, latency_s

    def _get_inflight_observation_timestep(self) -> int | None:
        with self.inflight_observation_lock:
            return self.inflight_observation_timestep

    def _init_log(self, first_packet: dict[str, Any], dataset_action_dim: int) -> None:
        run_name = self.args.run_name or default_run_name(self.args)
        log_dir = self.args.log_dir.expanduser().resolve() / run_name
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = log_dir / "samples.jsonl"
        metadata_path = log_dir / "metadata.json"
        metadata = {
            "created_wall_time": datetime.now().isoformat(timespec="seconds"),
            "policy_path": str(self.policy_path),
            "policy_type": self.policy_type,
            "server_address": self.args.server_address,
            "dataset_root": str(self.dataset_root),
            "dataset_action_dim": dataset_action_dim,
            "live_robot_state_dim": int(first_packet["robot_state_dim"]),
            "live_action_dim": int(first_packet["action_dim"]),
            "camera_names": list(first_packet["camera_names"]),
            "actions_per_chunk": self.actions_per_chunk,
            "chunk_size_threshold": self.args.chunk_size_threshold,
            "aggregate_fn": self.args.aggregate_fn,
            "fps": self.args.fps,
            "task": self.args.task,
            "execute": self.args.execute,
            "command_zmq": f"tcp://{self.args.command_zmq_host}:{self.args.command_zmq_port}",
            "observation_zmq": f"tcp://{self.args.zmq_host}:{self.args.zmq_port}",
            "arm_action_representation": self._arm_action_representation(),
            "gripper_action_representation": self._gripper_action_representation(),
        }
        metadata_path.write_text(json.dumps(json_safe(metadata), indent=2) + "\n")
        self.log_file = self.log_path.open("a", buffering=1)
        print(f"Deployment log: {self.log_path}")

    def _write_log_record(self, record: dict[str, Any]) -> None:
        if self.log_file is None:
            return
        enriched = {
            "wall_time": time.time(),
            "elapsed_s": time.perf_counter() - self.log_started_monotonic,
            **record,
        }
        with self.log_lock:
            if self.log_file is not None:
                self.log_file.write(json.dumps(json_safe(enriched), separators=(",", ":")) + "\n")

    def _close_log(self) -> None:
        with self.log_lock:
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None

    def _log_observation_sent(
        self,
        observation: TimedObservation,
        current_packet: dict[str, Any],
        queue_snapshot: dict[str, Any],
    ) -> None:
        self._write_log_record(
            {
                "event": "observation_sent",
                "observation_timestep": observation.get_timestep(),
                "must_go": observation.must_go,
                "queue": queue_snapshot,
                "inflight_observation_timestep": self._get_inflight_observation_timestep(),
                "latest_executed_timestep": self.latest_executed_timestep,
                "robot_state": np.asarray(current_packet["state"], dtype=float).tolist(),
                "bridge_action": np.asarray(current_packet.get("action", []), dtype=float).tolist(),
                "camera_stamps_s": {
                    camera_name: current_packet["cameras"][camera_name].get("stamp_s")
                    for camera_name in current_packet["camera_names"]
                },
            }
        )

    def _log_action_received(
        self,
        incoming_actions: list[TimedAction],
        filtered_actions: list[TimedAction],
        aggregate_stats: dict[str, Any],
        cleared_inflight_timestep: int | None,
        inflight_latency_s: float | None,
    ) -> None:
        self._write_log_record(
            {
                "event": "action_chunk_received",
                "incoming_count": len(incoming_actions),
                "filtered_count": len(filtered_actions),
                "incoming_timesteps": [action.get_timestep() for action in incoming_actions],
                "filtered_timesteps": [action.get_timestep() for action in filtered_actions],
                "first_action": (
                    np.asarray(filtered_actions[0].get_action(), dtype=float).tolist()
                    if filtered_actions
                    else None
                ),
                "cleared_inflight_observation_timestep": cleared_inflight_timestep,
                "inflight_latency_s": inflight_latency_s,
                "overlap_count": aggregate_stats["blended"],
                "new_action_count": aggregate_stats["added"],
                "dropped_action_count": aggregate_stats["dropped"],
                "aggregate": aggregate_stats,
                "latest_executed_timestep": self.latest_executed_timestep,
            }
        )

    def _log_action_chunk_filtered(
        self,
        incoming_actions: list[TimedAction],
        cleared_inflight_timestep: int | None,
        inflight_latency_s: float | None,
    ) -> None:
        incoming_timesteps = [action.get_timestep() for action in incoming_actions]
        self._write_log_record(
            {
                "event": "action_chunk_filtered",
                "incoming_count": len(incoming_actions),
                "incoming_timesteps": incoming_timesteps,
                "cleared_inflight_observation_timestep": cleared_inflight_timestep,
                "inflight_latency_s": inflight_latency_s,
                "latest_executed_timestep": self.latest_executed_timestep,
                "reason": "all incoming actions were already executed",
            }
        )
        print(
            "[TIMING] chunk filtered "
            f"inflight={cleared_inflight_timestep} "
            f"latency={format_optional_seconds(inflight_latency_s)} "
            f"incoming={incoming_timesteps[:1]}..{incoming_timesteps[-1:] if incoming_timesteps else []} "
            f"latest_executed={self.latest_executed_timestep}"
        )

    def _log_action_executed(
        self,
        action: TimedAction,
        current_packet: dict[str, Any],
        payload: dict[str, Any] | None,
        queue_snapshot_before_pop: dict[str, Any],
        queue_snapshot_after_pop: dict[str, Any],
    ) -> None:
        predicted_action = np.asarray(action.get_action(), dtype=float)
        split = split_action(predicted_action)
        current_state = np.asarray(current_packet["state"], dtype=float)
        state_split = split_action(current_state) if current_state.shape[0] in {14, 16} else None
        joint_targets = self._joint_targets_from_action(split)
        left_delta_from_state = None
        right_delta_from_state = None
        if state_split is not None:
            left_delta_from_state = (
                np.asarray(joint_targets["left"], dtype=float)
                - np.asarray(state_split["left_arm"], dtype=float)
            ).tolist()
            right_delta_from_state = (
                np.asarray(joint_targets["right"], dtype=float)
                - np.asarray(state_split["right_arm"], dtype=float)
            ).tolist()

        self._write_log_record(
            {
                "event": "action_executed" if self.args.execute else "action_predicted",
                "action_timestep": action.get_timestep(),
                "action_timestamp": action.get_timestamp(),
                "latest_executed_timestep": self.latest_executed_timestep,
                "queue_before_pop": queue_snapshot_before_pop,
                "queue_after_pop": queue_snapshot_after_pop,
                "robot_state": current_state.tolist(),
                "robot_state_split": state_split,
                "predicted_action": predicted_action.tolist(),
                "predicted_split": split,
                "command_payload": payload,
                "left_target_minus_state": left_delta_from_state,
                "right_target_minus_state": right_delta_from_state,
            }
        )

    def _gripper_command_from_action(self, value: float | None) -> float | None:
        if value is None:
            return None
        representation = self._gripper_action_representation()
        if representation == "binary_open_close":
            return 1.0 if float(value) >= 0.5 else 0.0
        if representation == "absolute_width":
            return max(0.0, min(1.0, float(value)))
        raise ValueError(f"Unsupported gripper action representation '{representation}' for bridge execution.")

    def _set_bridge_active(self, active: bool) -> None:
        if self.args.no_auto_activate_bridge:
            return
        if self.bridge_active == active:
            return

        state = "true" if active else "false"
        command = [
            "ros2",
            "service",
            "call",
            self.args.bridge_activation_service,
            "std_srvs/srv/SetBool",
            f"{{data: {state}}}",
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5.0)  # nosec B603
        if result.returncode != 0 or "success=True" not in result.stdout:
            raise RuntimeError(
                f"Failed to {'activate' if active else 'deactivate'} bridge via "
                f"{self.args.bridge_activation_service}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )
        self.bridge_active = active
        print(f"Deployment bridge {'activated' if active else 'deactivated'} via {self.args.bridge_activation_service}.")

    def connect_policy_server(self, first_packet: dict[str, Any]) -> None:
        lerobot_features = build_live_lerobot_features(first_packet)
        policy_config = RemotePolicyConfig(
            policy_type=self.policy_type,
            pretrained_name_or_path=str(self.policy_path),
            lerobot_features=lerobot_features,
            actions_per_chunk=self.actions_per_chunk,
            device=self.args.policy_device,
        )
        self.stub.Ready(self.services_pb2.Empty())
        self.stub.SendPolicyInstructions(self.services_pb2.PolicySetup(data=pickle.dumps(policy_config)))  # nosec

    def send_observation(self, observation: TimedObservation) -> None:
        observation_bytes = pickle.dumps(observation)  # nosec
        request_iterator = self.send_bytes_in_chunks(
            observation_bytes,
            self.services_pb2.Observation,
            log_prefix="[FRANKA_EXECUTOR] Observation",
            silent=True,
        )
        self.stub.SendObservations(request_iterator)

    def _ready_to_send_observation(self) -> bool:
        if self._get_inflight_observation_timestep() is not None:
            return False

        with self.action_queue_lock:
            queue_fraction = len(self.action_queue) / float(max(self.action_chunk_size, 1))
        return queue_fraction <= self.args.chunk_size_threshold

    def _aggregate_action_queue(self, incoming_actions: list[TimedAction]) -> dict[str, Any]:
        with self.action_queue_lock:
            current_queue = {action.get_timestep(): action for action in self.action_queue}

        added = 0
        blended = 0
        dropped = 0
        for new_action in incoming_actions:
            timestep = new_action.get_timestep()
            if timestep <= self.latest_executed_timestep:
                dropped += 1
                continue

            if timestep not in current_queue:
                current_queue[timestep] = new_action
                added += 1
                continue

            old_action = current_queue[timestep]
            old_tensor = torch.as_tensor(np.asarray(old_action.get_action(), dtype=np.float32))
            new_tensor = torch.as_tensor(np.asarray(new_action.get_action(), dtype=np.float32))
            blended_tensor = self.aggregate_fn(old_tensor, new_tensor)
            new_action.action = blended_tensor.detach().cpu().numpy()
            current_queue[timestep] = new_action
            blended += 1

        ordered_timesteps = sorted(current_queue)
        with self.action_queue_lock:
            self.action_queue = deque(current_queue[timestep] for timestep in ordered_timesteps)

        return {
            "added": added,
            "blended": blended,
            "dropped": dropped,
            "queue_size": len(ordered_timesteps),
            "first_timestep": ordered_timesteps[0] if ordered_timesteps else None,
            "last_timestep": ordered_timesteps[-1] if ordered_timesteps else None,
        }

    def receive_actions(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                response = self.stub.GetActions(self.services_pb2.Empty())
                if len(response.data) == 0:
                    cleared_timestep, latency_s = self._clear_observation_inflight()
                    if cleared_timestep is not None:
                        self._write_log_record(
                            {
                                "event": "action_request_timeout",
                                "cleared_inflight_observation_timestep": cleared_timestep,
                                "inflight_latency_s": latency_s,
                                "latest_executed_timestep": self.latest_executed_timestep,
                            }
                        )
                        print(
                            "[TIMING] action request timeout "
                            f"inflight={cleared_timestep} latency={format_optional_seconds(latency_s)}"
                        )
                    continue

                incoming_actions: list[TimedAction] = pickle.loads(response.data)  # nosec
                if not incoming_actions:
                    self._clear_observation_inflight()
                    continue

                self.action_chunk_size = max(self.action_chunk_size, len(incoming_actions))
                cleared_inflight_timestep, inflight_latency_s = self._clear_observation_inflight()
                filtered = [
                    action
                    for action in incoming_actions
                    if action.get_timestep() > self.latest_executed_timestep
                ]
                if not filtered:
                    self._log_action_chunk_filtered(
                        incoming_actions,
                        cleared_inflight_timestep,
                        inflight_latency_s,
                    )
                    continue

                aggregate_stats = self._aggregate_action_queue(filtered)
                self._log_action_received(
                    incoming_actions,
                    filtered,
                    aggregate_stats,
                    cleared_inflight_timestep,
                    inflight_latency_s,
                )
                incoming_timesteps = [action.get_timestep() for action in incoming_actions]
                filtered_timesteps = [action.get_timestep() for action in filtered]
                print(
                    "[TIMING] chunk received "
                    f"inflight={cleared_inflight_timestep} "
                    f"latency={format_optional_seconds(inflight_latency_s)} "
                    f"incoming={incoming_timesteps[0]}..{incoming_timesteps[-1]} "
                    f"filtered={filtered_timesteps[0]}..{filtered_timesteps[-1]} "
                    f"added={aggregate_stats['added']} "
                    f"blended={aggregate_stats['blended']} "
                    f"dropped={aggregate_stats['dropped']} "
                    f"queue={aggregate_stats['first_timestep']}..{aggregate_stats['last_timestep']} "
                    f"latest_executed={self.latest_executed_timestep}"
                )
                self.must_go_pending = True
            except self.grpc.RpcError:
                cleared_timestep, latency_s = self._clear_observation_inflight()
                if cleared_timestep is not None:
                    self._write_log_record(
                        {
                            "event": "action_request_rpc_error",
                            "cleared_inflight_observation_timestep": cleared_timestep,
                            "inflight_latency_s": latency_s,
                            "latest_executed_timestep": self.latest_executed_timestep,
                        }
                    )
                    print(
                        "[TIMING] action request RPC error "
                        f"inflight={cleared_timestep} latency={format_optional_seconds(latency_s)}"
                    )
                time.sleep(0.05)
            except Exception as exc:  # pragma: no cover
                print(f"Action receiver error: {exc}")
                time.sleep(0.05)

    def setup_command_bridge(self, zmq_context: Any) -> None:
        if not self.args.execute:
            return
        self.command_socket = zmq_context.socket(import_zmq_runtime().PUSH)
        self.command_socket.setsockopt(import_zmq_runtime().SNDHWM, 1)
        self.command_socket.connect(f"tcp://{self.args.command_zmq_host}:{self.args.command_zmq_port}")
        print("Bridge command backend:", f"tcp://{self.args.command_zmq_host}:{self.args.command_zmq_port}")

    def _command_payload_from_action(self, action: TimedAction) -> dict[str, Any]:
        split = split_action(np.asarray(action.get_action(), dtype=float))
        joint_targets = self._joint_targets_from_action(split)
        left_gripper_command = self._gripper_command_from_action(split["left_gripper"])
        right_gripper_command = self._gripper_command_from_action(split["right_gripper"])
        return {
            "timestamp": time.time(),
            "left_joint_target": joint_targets["left"],
            "right_joint_target": joint_targets["right"],
            "left_gripper_command": left_gripper_command,
            "right_gripper_command": right_gripper_command,
        }

    def _send_bridge_command(self, payload: dict[str, Any]) -> None:
        if self.command_socket is None:
            raise RuntimeError("Bridge command socket is not initialized.")

        self._set_bridge_active(True)
        self.command_socket.send_pyobj(payload)

    def maybe_pop_action(self) -> tuple[TimedAction | None, dict[str, Any], dict[str, Any]]:
        with self.action_queue_lock:
            queue_before = self._queue_snapshot_unlocked()
            if not self.action_queue:
                return None, queue_before, queue_before
            action = self.action_queue.popleft()
            queue_after = self._queue_snapshot_unlocked()
        self.latest_executed_timestep = action.get_timestep()
        return action, queue_before, queue_after

    def _queue_snapshot_unlocked(self) -> dict[str, Any]:
        timesteps = [action.get_timestep() for action in self.action_queue]
        return {
            "size": len(timesteps),
            "first_timestep": timesteps[0] if timesteps else None,
            "last_timestep": timesteps[-1] if timesteps else None,
        }

    def run(self) -> None:
        zmq = import_zmq_runtime()
        dataset_action_dim = int(self.dataset_info["features"]["action"]["shape"][0])
        print("Franka policy executor")
        print("----------------------")
        print(f"policy_path: {self.policy_path}")
        print(f"policy_type: {self.policy_type}")
        print(f"server_address: {self.args.server_address}")
        print(f"dataset_action_dim: {dataset_action_dim}")
        print(f"actions_per_chunk: {self.actions_per_chunk}")
        print(f"chunk_size_threshold: {self.args.chunk_size_threshold}")
        print(f"aggregate_fn: {self.args.aggregate_fn}")
        print(f"execute: {self.args.execute}")

        context = zmq.Context()
        self._set_bridge_active(True)
        print("Waiting for the first ZMQ packet to infer live observation features...")

        socket = context.socket(zmq.SUB)
        socket.connect(f"tcp://{self.args.zmq_host}:{self.args.zmq_port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.setsockopt(zmq.RCVTIMEO, 100)

        first_packet = None
        while first_packet is None:
            try:
                first_packet = socket.recv_pyobj()
            except zmq.Again:
                continue

        print("Received first packet from live bridge.")
        print(
            f"state_dim={first_packet['robot_state_dim']}, action_dim={first_packet['action_dim']}, "
            f"cameras={list(first_packet['camera_names'])}"
        )
        if int(first_packet["action_dim"]) != dataset_action_dim:
            raise ValueError(
                f"Live bridge action_dim={first_packet['action_dim']} does not match dataset action_dim={dataset_action_dim}."
            )

        self._init_log(first_packet, dataset_action_dim)
        self.connect_policy_server(first_packet)
        self.setup_command_bridge(context)
        receiver_thread = threading.Thread(target=self.receive_actions, daemon=True)
        receiver_thread.start()

        current_packet = first_packet
        try:
            while not self.shutdown_event.is_set():
                loop_start = time.perf_counter()
                try:
                    current_packet = socket.recv_pyobj()
                except zmq.Again:
                    continue

                if self._ready_to_send_observation():
                    observation_timestep, queue_snapshot = self._next_observation_timestep()
                    self._mark_observation_inflight(observation_timestep)
                    observation = TimedObservation(
                        timestamp=time.time(),
                        observation=packet_to_raw_observation(current_packet, self.args.task),
                        timestep=observation_timestep,
                        must_go=True,
                    )
                    try:
                        self.send_observation(observation)
                        self._log_observation_sent(observation, current_packet, queue_snapshot)
                    except Exception:
                        self._clear_observation_inflight()
                        raise
                    print(
                        "[TIMING] observation sent "
                        f"timestep={observation_timestep} "
                        f"queue_size={queue_snapshot['size']} "
                        f"queue={queue_snapshot['first_timestep']}..{queue_snapshot['last_timestep']} "
                        f"latest_executed={self.latest_executed_timestep} "
                        f"must_go={observation.must_go}"
                    )
                    self.must_go_pending = False

                action, queue_before_pop, queue_after_pop = self.maybe_pop_action()
                if action is not None:
                    payload = self._command_payload_from_action(action)
                    if self.args.execute:
                        self._send_bridge_command(payload)
                    else:
                        print(
                            "Predicted action:",
                            {
                                "timestep": action.get_timestep(),
                                "left_arm": np.round(np.asarray(split_action(action.get_action())["left_arm"]), 4).tolist(),
                                "right_arm": np.round(np.asarray(split_action(action.get_action())["right_arm"]), 4).tolist(),
                            },
                        )
                    self._log_action_executed(
                        action,
                        current_packet,
                        payload if self.args.execute else None,
                        queue_before_pop,
                        queue_after_pop,
                    )

                sleep_time = max(0.0, (1.0 / self.args.fps) - (time.perf_counter() - loop_start))
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nStopping Franka policy executor...")
        finally:
            self.shutdown_event.set()
            self.channel.close()
            receiver_thread.join(timeout=1.0)
            self._set_bridge_active(False)
            if self.command_socket is not None:
                self.command_socket.close(0)
            self._close_log()
            socket.close(0)
            context.term()


def main() -> None:
    args = parse_args()
    executor = FrankaPolicyExecutor(args)
    executor.run()


if __name__ == "__main__":
    main()
