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

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation

try:
    import pylibfranka
except ModuleNotFoundError:  # pragma: no cover - depends on robot computer env
    pylibfranka = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "pick_and_place_test"
ACTION_CONFIG_REL_PATH = Path("meta/real_exp_action_config.json")
INFO_REL_PATH = Path("meta/info.json")
MAX_JOINT_VELOCITIES_RAD_PER_S = np.array([0.35, 0.35, 0.35, 0.35, 0.50, 0.50, 0.50], dtype=float)
MAX_JOINT_ACCELERATIONS_RAD_PER_S2 = np.array([0.80, 0.80, 0.80, 0.80, 1.20, 1.20, 1.20], dtype=float)


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
    parser.add_argument(
        "--policy-path",
        type=Path,
        required=True,
        help=(
            "Path to the trained LeRobot checkpoint directory. "
            "This path is forwarded to the policy server and must be valid on the server machine. "
            "It does not need to exist locally if --policy-type and --actions-per-chunk are provided."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the local dataset root used to define the action contract.",
    )
    parser.add_argument(
        "--server-address",
        default="127.0.0.1:8080",
        help="gRPC address of the policy server, for example 192.168.1.10:8080.",
    )
    parser.add_argument(
        "--policy-device",
        default="cuda",
        help="Device string that the remote policy server should use.",
    )
    parser.add_argument(
        "--policy-type",
        default=None,
        help="Policy type override. If omitted, inferred from policy config.json.",
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=None,
        help="Override number of actions to request per chunk from the server.",
    )
    parser.add_argument("--zmq-host", default="127.0.0.1", help="ZMQ host for live robot observations.")
    parser.add_argument("--zmq-port", type=int, default=5555, help="ZMQ port for live robot observations.")
    parser.add_argument("--fps", type=int, default=15, help="Executor control frequency.")
    parser.add_argument("--task", default="pick and place", help="Task string passed to the policy.")
    parser.add_argument("--ip-left", default="172.16.0.3", help="IP address of the left Franka robot.")
    parser.add_argument("--ip-right", default="172.16.0.2", help="IP address of the right Franka robot.")
    parser.add_argument(
        "--velocity-filter-tau",
        type=float,
        default=0.005,
        help="Low-pass filter time constant for arm velocity commands in seconds.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute policy commands. By default this forwards targets to the ROS2 deployment bridge.",
    )
    parser.add_argument(
        "--execute-backend",
        choices=("bridge", "pylibfranka"),
        default="bridge",
        help="Execution backend to use when --execute is set.",
    )
    parser.add_argument(
        "--command-zmq-host",
        default="127.0.0.1",
        help="Host of the bridge command socket used by the bridge execution backend.",
    )
    parser.add_argument(
        "--command-zmq-port",
        type=int,
        default=5556,
        help="Port of the bridge command socket used by the bridge execution backend.",
    )
    parser.add_argument(
        "--bridge-activation-service",
        default="/set_deployment_active",
        help="ROS 2 SetBool service used to activate/deactivate the deployment bridge.",
    )
    parser.add_argument(
        "--no-auto-activate-bridge",
        action="store_true",
        help="Do not automatically activate the deployment bridge before sending the first bridge command.",
    )
    parser.add_argument(
        "--enable-gripper",
        dest="enable_gripper",
        action="store_true",
        default=True,
        help="Enable best-effort gripper commands when gripper actions are present (default).",
    )
    parser.add_argument(
        "--disable-gripper",
        dest="enable_gripper",
        action="store_false",
        help="Disable best-effort gripper commands even when gripper actions are present.",
    )
    parser.add_argument(
        "--gripper-open-width",
        type=float,
        default=0.08,
        help="Open width in meters for binary gripper commands.",
    )
    parser.add_argument(
        "--gripper-closed-width",
        type=float,
        default=0.0,
        help="Closed width in meters for binary gripper commands.",
    )
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=0.05,
        help="Best-effort gripper speed for move commands.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def infer_policy_type(policy_path: Path) -> str | None:
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


def split_action(action: np.ndarray) -> dict[str, Any]:
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


def split_state(state: np.ndarray) -> dict[str, Any]:
    state = np.asarray(state, dtype=float)
    if state.shape[0] == 16:
        return {
            "left_arm": state[0:7],
            "left_gripper": float(state[7]),
            "right_arm": state[8:15],
            "right_gripper": float(state[15]),
        }
    if state.shape[0] == 14:
        return {
            "left_arm": state[0:7],
            "left_gripper": None,
            "right_arm": state[7:14],
            "right_gripper": None,
        }
    raise ValueError(f"Unsupported state dimension: {state.shape[0]}")


def controller_mode() -> object:
    if pylibfranka is None:
        raise RuntimeError("pylibfranka is required for robot execution mode.")
    try:
        return pylibfranka.ControllerMode.kJointImpedance
    except AttributeError:
        return pylibfranka.ControllerMode.JointImpedance


def limit_velocity_command(current_velocity: np.ndarray, target_velocity: np.ndarray, dt: float) -> np.ndarray:
    clipped_target_velocity = np.clip(
        np.asarray(target_velocity, dtype=float),
        -MAX_JOINT_VELOCITIES_RAD_PER_S,
        MAX_JOINT_VELOCITIES_RAD_PER_S,
    )
    max_delta_velocity = MAX_JOINT_ACCELERATIONS_RAD_PER_S2 * max(dt, 1e-6)
    velocity_step = np.clip(
        clipped_target_velocity - current_velocity,
        -max_delta_velocity,
        max_delta_velocity,
    )
    return np.clip(
        current_velocity + velocity_step,
        -MAX_JOINT_VELOCITIES_RAD_PER_S,
        MAX_JOINT_VELOCITIES_RAD_PER_S,
    )


def read_once(control: Any) -> float:
    _, time_step = control.readOnce()
    if hasattr(time_step, "to_sec"):
        return float(time_step.to_sec())
    if hasattr(time_step, "toSec"):
        return float(time_step.toSec())
    return float(time_step)


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
                "When the checkpoint exists only on the policy server machine, pass "
                "--policy-type explicitly."
            )

        if args.actions_per_chunk is not None:
            self.actions_per_chunk = args.actions_per_chunk
        elif local_policy_config is not None:
            self.actions_per_chunk = infer_actions_per_chunk(self.policy_path, self.policy_type)
        else:
            raise FileNotFoundError(
                "Could not find a local policy config.json for --policy-path. "
                "When the checkpoint exists only on the policy server machine, pass "
                "--actions-per-chunk explicitly."
            )
        self.dataset_info = load_json(self.dataset_root / INFO_REL_PATH)
        action_config_path = self.dataset_root / ACTION_CONFIG_REL_PATH
        if not action_config_path.exists():
            raise FileNotFoundError(
                f"Missing action metadata: {action_config_path}. "
                "Deployment now requires an absolute-target dataset recorded with "
                "meta/real_exp_action_config.json."
            )
        self.action_config = load_json(action_config_path)
        arm_action_representation = self._arm_action_representation()
        if arm_action_representation != "absolute_joint_position":
            raise ValueError(
                f"Dataset arm_action_representation is '{arm_action_representation}'. "
                "Deployment now expects absolute_joint_position arm actions. "
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
        self.action_lock = threading.Lock()
        self.latest_executed_timestep = -1
        self.shutdown_event = threading.Event()
        self.left_control = None
        self.right_control = None
        self.left_gripper = None
        self.right_gripper = None
        self.left_commanded_velocity = np.zeros(7, dtype=float)
        self.right_commanded_velocity = np.zeros(7, dtype=float)
        self.last_left_gripper_command: float | None = None
        self.last_right_gripper_command: float | None = None
        self.command_socket = None
        self.bridge_active = False

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
        payload = pickle.dumps(policy_config)  # nosec
        self.stub.SendPolicyInstructions(self.services_pb2.PolicySetup(data=payload))

    def start_action_receiver(self) -> threading.Thread:
        receiver = threading.Thread(target=self.receive_actions, daemon=True)
        receiver.start()
        return receiver

    def receive_actions(self) -> None:
        while not self.shutdown_event.is_set():
            try:
                response = self.stub.GetActions(self.services_pb2.Empty())
                if len(response.data) == 0:
                    continue
                incoming_actions = pickle.loads(response.data)  # nosec
                with self.action_lock:
                    filtered = [
                        action
                        for action in incoming_actions
                        if action.get_timestep() > self.latest_executed_timestep
                    ]
                    if filtered:
                        self.action_queue = deque(filtered)
            except self.grpc.RpcError as exc:
                print(f"Action receiver RPC error: {exc}")
                time.sleep(0.2)
            except Exception as exc:  # pragma: no cover - runtime path
                print(f"Action receiver error: {exc}")
                time.sleep(0.2)

    def send_observation(self, observation: TimedObservation) -> None:
        observation_bytes = pickle.dumps(observation)  # nosec
        request_iterator = self.send_bytes_in_chunks(
            observation_bytes,
            self.services_pb2.Observation,
            log_prefix="[FRANKA_EXECUTOR] Observation",
            silent=True,
        )
        self.stub.SendObservations(request_iterator)

    def maybe_pop_action(self) -> TimedAction | None:
        with self.action_lock:
            if not self.action_queue:
                return None
            action = self.action_queue.popleft()
        self.latest_executed_timestep = action.get_timestep()
        return action

    def setup_robot_controls(self) -> None:
        if not self.args.execute:
            return
        if self.args.execute_backend != "pylibfranka":
            return
        if pylibfranka is None:
            raise RuntimeError("pylibfranka is not installed in this environment.")

        left_robot = pylibfranka.Robot(self.args.ip_left)
        right_robot = pylibfranka.Robot(self.args.ip_right)
        self.left_control = left_robot.start_joint_velocity_control(controller_mode())
        self.right_control = right_robot.start_joint_velocity_control(controller_mode())

        if self.args.enable_gripper:
            self.left_gripper = pylibfranka.Gripper(self.args.ip_left)
            self.right_gripper = pylibfranka.Gripper(self.args.ip_right)

    def stop_robot_controls(self) -> None:
        if (
            not self.args.execute
            or self.args.execute_backend != "pylibfranka"
            or pylibfranka is None
        ):
            return
        zero_velocity = pylibfranka.JointVelocities([0.0] * 7)
        zero_velocity.motion_finished = True
        for control in (self.left_control, self.right_control):
            if control is None:
                continue
            try:
                control.writeOnce(zero_velocity)
            except Exception:
                pass

    def apply_gripper_command(self, gripper: Any, last_value: float | None, value: float | None) -> float | None:
        if gripper is None or value is None:
            return last_value

        representation = str(
            self.action_config.get("gripper_action_representation", "binary_open_close")
        ).strip().lower()

        if representation == "binary_open_close":
            latched_value = 1.0 if value >= 0.5 else 0.0
            if last_value is None or latched_value != last_value:
                target_width = (
                    self.args.gripper_open_width if latched_value >= 0.5 else self.args.gripper_closed_width
                )
                gripper.move(float(target_width), float(self.args.gripper_speed))
            return latched_value

        if representation == "absolute_width":
            if last_value is None or abs(value - last_value) > 1e-3:
                gripper.move(float(value), float(self.args.gripper_speed))
            return float(value)

        print(f"Unsupported gripper representation '{representation}', skipping gripper command.")
        return last_value

    def setup_command_bridge(self, zmq_context: Any) -> None:
        if not self.args.execute or self.args.execute_backend != "bridge":
            return
        self.command_socket = zmq_context.socket(import_zmq_runtime().PUSH)
        self.command_socket.setsockopt(import_zmq_runtime().SNDHWM, 1)
        self.command_socket.connect(f"tcp://{self.args.command_zmq_host}:{self.args.command_zmq_port}")
        print(
            "Bridge command backend:",
            f"tcp://{self.args.command_zmq_host}:{self.args.command_zmq_port}",
        )

    def _arm_action_representation(self) -> str:
        return str(
            self.action_config.get("arm_action_representation", "absolute_joint_position")
        ).strip().lower()

    def _gripper_action_representation(self) -> str:
        return str(
            self.action_config.get("gripper_action_representation", "binary_open_close")
        ).strip().lower()

    def _joint_targets_from_action(self, current_state: np.ndarray, split: dict[str, Any]) -> dict[str, list[float]]:
        representation = self._arm_action_representation()
        if representation == "absolute_joint_position":
            return {
                "left": np.asarray(split["left_arm"], dtype=float).tolist(),
                "right": np.asarray(split["right_arm"], dtype=float).tolist(),
            }
        raise ValueError(
            f"Unsupported arm action representation '{representation}' for bridge execution. "
            "Record and train datasets with arm_action_representation=absolute_joint_position."
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
        try:
            result = subprocess.run(  # nosec B603
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Could not find the 'ros2' CLI. Source the ROS 2 environment before running "
                "the executor, or pass --no-auto-activate-bridge and activate manually."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out calling bridge activation service {self.args.bridge_activation_service}."
            ) from exc

        if result.returncode != 0 or "success=True" not in result.stdout:
            raise RuntimeError(
                f"Failed to {'activate' if active else 'deactivate'} bridge via "
                f"{self.args.bridge_activation_service}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

        self.bridge_active = active
        print(f"Deployment bridge {'activated' if active else 'deactivated'} via {self.args.bridge_activation_service}.")

    def _send_bridge_command(self, split: dict[str, Any], current_packet: dict[str, Any]) -> None:
        if self.command_socket is None:
            raise RuntimeError("Bridge command socket is not initialized.")

        current_state = np.asarray(current_packet["state"], dtype=float)
        joint_targets = self._joint_targets_from_action(current_state, split)
        payload = {
            "timestamp": time.time(),
            "left_joint_target": joint_targets["left"],
            "right_joint_target": joint_targets["right"],
            "left_gripper_command": self._gripper_command_from_action(split["left_gripper"]),
            "right_gripper_command": self._gripper_command_from_action(split["right_gripper"]),
        }
        self._set_bridge_active(True)
        self.command_socket.send_pyobj(payload)

    def apply_action(self, action_tensor: Any, current_packet: dict[str, Any]) -> None:
        action = np.asarray(action_tensor, dtype=float)
        split = split_action(action)

        if not self.args.execute:
            print(
                "Predicted action:",
                {
                    "left_arm": np.round(split["left_arm"], 4).tolist(),
                    "right_arm": np.round(split["right_arm"], 4).tolist(),
                    "left_gripper": split["left_gripper"],
                    "right_gripper": split["right_gripper"],
                },
            )
            return

        if self.args.execute_backend == "bridge":
            self._send_bridge_command(split, current_packet)
            return

        if self._arm_action_representation() != "absolute_joint_position":
            raise ValueError(
                "The direct pylibfranka executor now expects absolute_joint_position arm actions."
            )

        dt = 1.0 / self.args.fps
        if self.left_control is not None:
            dt = read_once(self.left_control)
        if self.right_control is not None:
            dt = min(dt, read_once(self.right_control))

        current_split = split_state(np.asarray(current_packet["state"], dtype=float))
        target_left_velocity = (
            np.asarray(split["left_arm"], dtype=float)
            - np.asarray(current_split["left_arm"], dtype=float)
        ) / max(1.0 / self.args.fps, 1e-6)
        target_right_velocity = (
            np.asarray(split["right_arm"], dtype=float)
            - np.asarray(current_split["right_arm"], dtype=float)
        ) / max(1.0 / self.args.fps, 1e-6)

        if self.args.velocity_filter_tau > 0:
            alpha = dt / max(self.args.velocity_filter_tau + dt, 1e-6)
            target_left_velocity = self.left_commanded_velocity + alpha * (
                target_left_velocity - self.left_commanded_velocity
            )
            target_right_velocity = self.right_commanded_velocity + alpha * (
                target_right_velocity - self.right_commanded_velocity
            )

        self.left_commanded_velocity = limit_velocity_command(
            self.left_commanded_velocity, target_left_velocity, dt
        )
        self.right_commanded_velocity = limit_velocity_command(
            self.right_commanded_velocity, target_right_velocity, dt
        )

        self.left_control.writeOnce(pylibfranka.JointVelocities(self.left_commanded_velocity.tolist()))
        self.right_control.writeOnce(pylibfranka.JointVelocities(self.right_commanded_velocity.tolist()))

        if self.args.enable_gripper:
            self.last_left_gripper_command = self.apply_gripper_command(
                self.left_gripper, self.last_left_gripper_command, split["left_gripper"]
            )
            self.last_right_gripper_command = self.apply_gripper_command(
                self.right_gripper, self.last_right_gripper_command, split["right_gripper"]
            )

    def run(self) -> None:
        zmq = import_zmq_runtime()
        dataset_action_dim = int(self.dataset_info["features"]["action"]["shape"][0])
        print("Franka policy executor")
        print("----------------------")
        print(f"policy_path: {self.policy_path}")
        print(f"policy_type: {self.policy_type}")
        print(f"server_address: {self.args.server_address}")
        print(f"dataset_action_dim: {dataset_action_dim}")
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
        receiver_thread = self.start_action_receiver()
        self.setup_robot_controls()
        self.setup_command_bridge(context)

        timestep = 0
        try:
            while not self.shutdown_event.is_set():
                loop_start = time.perf_counter()
                if timestep == 0:
                    current_packet = first_packet
                else:
                    try:
                        current_packet = socket.recv_pyobj()
                    except zmq.Again:
                        time.sleep(0.001)
                        continue
                raw_observation = packet_to_raw_observation(current_packet, self.args.task)
                must_go = False
                with self.action_lock:
                    must_go = len(self.action_queue) == 0

                observation = TimedObservation(
                    timestamp=time.time(),
                    observation=raw_observation,
                    timestep=timestep,
                    must_go=must_go,
                )
                self.send_observation(observation)

                action = self.maybe_pop_action()
                if action is not None:
                    self.apply_action(action.get_action(), current_packet)

                timestep += 1
                sleep_time = max(0.0, (1.0 / self.args.fps) - (time.perf_counter() - loop_start))
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping Franka policy executor...")
        finally:
            self.shutdown_event.set()
            receiver_thread.join(timeout=1.0)
            self.stop_robot_controls()
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
