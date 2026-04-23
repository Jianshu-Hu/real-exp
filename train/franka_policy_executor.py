from __future__ import annotations

import argparse
import json
import pickle  # nosec
import subprocess  # nosec
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation


REPO_ROOT = Path(__file__).resolve().parents[1]
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
        default=0.5,
        help="Send a new observation when queue_size / actions_per_chunk is at or below this value.",
    )
    parser.add_argument(
        "--aggregate-fn",
        default="weighted_average",
        help="Aggregation function for overlapping timesteps. See lerobot.async_inference.configs.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--command-zmq-host", default="127.0.0.1")
    parser.add_argument("--command-zmq-port", type=int, default=5556)
    parser.add_argument("--bridge-activation-service", default="/set_deployment_active")
    parser.add_argument("--no-auto-activate-bridge", action="store_true")
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
        self.shutdown_event = threading.Event()
        self.command_socket = None
        self.bridge_active = False
        self.latest_executed_timestep = -1
        self.must_go_pending = True
        self.action_chunk_size = max(1, self.actions_per_chunk)

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
                    continue

                incoming_actions: list[TimedAction] = pickle.loads(response.data)  # nosec
                if not incoming_actions:
                    continue

                self.action_chunk_size = max(self.action_chunk_size, len(incoming_actions))
                filtered = [
                    action
                    for action in incoming_actions
                    if action.get_timestep() > self.latest_executed_timestep
                ]
                if not filtered:
                    continue

                self._aggregate_action_queue(filtered)
                self.must_go_pending = True
            except self.grpc.RpcError:
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

    def _send_bridge_command(self, action: TimedAction) -> None:
        if self.command_socket is None:
            raise RuntimeError("Bridge command socket is not initialized.")

        split = split_action(np.asarray(action.get_action(), dtype=float))
        joint_targets = self._joint_targets_from_action(split)
        left_gripper_command = self._gripper_command_from_action(split["left_gripper"])
        right_gripper_command = self._gripper_command_from_action(split["right_gripper"])
        payload = {
            "timestamp": time.time(),
            "left_joint_target": joint_targets["left"],
            "right_joint_target": joint_targets["right"],
            "left_gripper_command": left_gripper_command,
            "right_gripper_command": right_gripper_command,
        }
        self._set_bridge_active(True)
        self.command_socket.send_pyobj(payload)

    def maybe_pop_action(self) -> TimedAction | None:
        with self.action_queue_lock:
            if not self.action_queue:
                return None
            action = self.action_queue.popleft()
        self.latest_executed_timestep = action.get_timestep()
        return action

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
                    with self.action_queue_lock:
                        queue_empty = not self.action_queue
                    observation = TimedObservation(
                        timestamp=time.time(),
                        observation=packet_to_raw_observation(current_packet, self.args.task),
                        timestep=max(self.latest_executed_timestep, 0),
                        must_go=self.must_go_pending and queue_empty,
                    )
                    self.send_observation(observation)
                    if observation.must_go:
                        self.must_go_pending = False

                action = self.maybe_pop_action()
                if action is not None:
                    if self.args.execute:
                        self._send_bridge_command(action)
                    else:
                        print(
                            "Predicted action:",
                            {
                                "timestep": action.get_timestep(),
                                "left_arm": np.round(np.asarray(split_action(action.get_action())["left_arm"]), 4).tolist(),
                                "right_arm": np.round(np.asarray(split_action(action.get_action())["right_arm"]), 4).tolist(),
                            },
                        )

                sleep_time = max(0.0, (1.0 / self.args.fps) - (time.perf_counter() - loop_start))
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nStopping Franka policy executor...")
        finally:
            self.shutdown_event.set()
            receiver_thread.join(timeout=1.0)
            self._set_bridge_active(False)
            if self.command_socket is not None:
                self.command_socket.close(0)
            socket.close(0)
            context.term()
            self.channel.close()


def main() -> None:
    args = parse_args()
    executor = FrankaPolicyExecutor(args)
    executor.run()


if __name__ == "__main__":
    main()
