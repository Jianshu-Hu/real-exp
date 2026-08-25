from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from utils.dataset_stats import ensure_dataset_stats, normalize_episode_metadata
from utils.fr3_kinematics import pose_error, pose_vector_to_matrix, wrapped_pose_delta
from utils.trajectory_metadata import (
    TRAJECTORY_CONFIG_PATH,
    trajectory_config_from_packet,
    write_trajectory_config,
)

LEROBOT_INFO_PATH = Path("meta/info.json")
ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
SYSTEM_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
DEFAULT_BRIDGE_READY_TIMEOUT_SEC = 2.0


def clamp_gripper_values(values: np.ndarray) -> np.ndarray:
    """Clamp gripper widths while leaving arm and Wuji hand layouts unchanged."""
    result = np.asarray(values, dtype=np.float32).copy()
    if result.ndim != 1:
        raise ValueError(f"Expected a one-dimensional action/state vector, got shape {result.shape}.")
    if result.size in {7, 14, 27, 54}:
        return result
    if result.size not in {8, 16}:
        raise ValueError(
            "Expected a 7/14-D arm-only, 8/16-D gripper, 27-D single-arm hand, or "
            "54-D dual-arm hand vector for normalization; "
            f"got {result.size} dimensions."
        )
    gripper_indices = [7] if result.size == 8 else [7, 15]
    result[gripper_indices] = np.clip(result[gripper_indices], 0.0, 1.0)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record robot state, teleop actions, and three RGB camera streams into LeRobot format."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("DATA_COLLECTION_SERVER_IP", "192.168.50.13"),
        help=(
            "ZMQ host used by the ROS 2 bridge (default: DATA_COLLECTION_SERVER_IP "
            "or 192.168.50.13)."
        ),
    )
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port used by the ROS 2 bridge.")
    parser.add_argument(
        "--repo-id",
        default="local/franka_gello_teleop",
        help="Dataset repo id stored in LeRobot metadata.",
    )
    parser.add_argument(
        "--local-dir",
        default="./lerobot_data",
        help="Directory where the LeRobot dataset is written.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Expected recording rate. This should match the ROS 2 bridge sample rate.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional task string to override the task name coming from the ROS 2 bridge.",
    )
    parser.add_argument(
        "--bridge-ready-timeout",
        type=float,
        default=DEFAULT_BRIDGE_READY_TIMEOUT_SEC,
        help=(
            "Maximum age in seconds of the latest valid bridge sample when starting an episode "
            f"(default: {DEFAULT_BRIDGE_READY_TIMEOUT_SEC:g})."
        ),
    )
    return parser.parse_args()


def bridge_packet_readiness(packet: Any) -> tuple[bool, str]:
    """Return whether a bridge packet contains enough data to record safely.

    The bridge only publishes after all required arm, end-effector, and camera
    inputs are ready. Validate the packet shape here as well so a malformed or
    deployment-only packet cannot arm recording and fail later on the first frame.
    """
    if not isinstance(packet, dict):
        return False, "received a non-dictionary bridge packet"

    try:
        state = np.asarray(packet["state"], dtype=np.float32)
        action = np.asarray(packet["action"], dtype=np.float32)
        joint_state = np.asarray(packet["joint_state"], dtype=np.float32)
        target_joint = np.asarray(packet["target_joint"], dtype=np.float32)
        robot_state_dim = int(packet["robot_state_dim"])
        action_dim = int(packet["action_dim"])
        joint_state_dim = int(packet["joint_state_dim"])
        target_joint_dim = int(packet["target_joint_dim"])
        camera_names = list(packet["camera_names"])
        cameras = packet["cameras"]
        state_action_mode = str(packet.get("state_action_mode", "joint")).strip().lower()
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"bridge packet is missing required fields ({exc})"

    if state.ndim != 1 or state.size != robot_state_dim:
        return False, f"bridge state has shape {state.shape}, expected ({robot_state_dim},)"
    if action.ndim != 1 or action.size != action_dim:
        return False, f"bridge action has shape {action.shape}, expected ({action_dim},)"
    if joint_state.ndim != 1 or joint_state.size != joint_state_dim:
        return False, f"bridge joint_state has shape {joint_state.shape}, expected ({joint_state_dim},)"
    if target_joint.ndim != 1 or target_joint.size != target_joint_dim:
        return False, f"bridge target_joint has shape {target_joint.shape}, expected ({target_joint_dim},)"
    try:
        trajectory_for_dims = trajectory_config_from_packet(packet)
        expected_joint_dim = 7 * len(trajectory_for_dims["arms"])
        if trajectory_for_dims["end_effector"] == "gripper":
            expected_joint_dim += len(trajectory_for_dims["arms"])
        elif trajectory_for_dims["end_effector"] == "hand":
            expected_joint_dim += 20 * len(trajectory_for_dims["arms"])
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"bridge joint metadata is invalid ({exc})"
    if joint_state_dim != expected_joint_dim or target_joint_dim != expected_joint_dim:
        return False, (
            f"bridge joint dimensions {joint_state_dim}/{target_joint_dim} "
            f"do not match expected {expected_joint_dim}"
        )
    if not np.all(np.isfinite(joint_state)) or not np.all(np.isfinite(target_joint)):
        return False, "bridge joint_state/target_joint contains non-finite values"
    if not isinstance(cameras, dict):
        return False, "bridge packet cameras field is not a dictionary"

    for camera_name in camera_names:
        camera = cameras.get(camera_name)
        if not isinstance(camera, dict) or "rgb" not in camera or "shape" not in camera:
            return False, f"bridge packet is missing camera data for {camera_name!r}"
        rgb = np.asarray(camera["rgb"])
        shape = camera["shape"]
        if rgb.ndim != 3 or tuple(rgb.shape) != tuple(shape) or rgb.shape[-1] != 3:
            return False, f"bridge camera {camera_name!r} has invalid RGB shape {rgb.shape}"
    try:
        arms = trajectory_config_from_packet(packet)["arms"]
        expected = 6 * len(arms)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"bridge end-effector metadata is invalid ({exc})"
    ee_values: dict[str, np.ndarray] = {}
    for key in ("ee_pose", "target_ee_pose", "delta_ee_pose"):
        values = np.asarray(packet.get(key, []), dtype=np.float32)
        if values.shape != (expected,) or not np.all(np.isfinite(values)):
            return False, f"bridge {key} has shape {values.shape}, expected ({expected},)"
        ee_values[key] = values
    for arm_index, arm_name in enumerate(arms):
        pose_slice = slice(arm_index * 6, arm_index * 6 + 6)
        reconstructed = (
            ee_values["ee_pose"][pose_slice]
            + ee_values["delta_ee_pose"][pose_slice]
        )
        position_error, orientation_error = pose_error(
            pose_vector_to_matrix(reconstructed),
            pose_vector_to_matrix(ee_values["target_ee_pose"][pose_slice]),
        )
        if position_error > 1e-5 or orientation_error > 1e-5:
            return False, (
                f"bridge {arm_name} EE target/delta disagree "
                f"({position_error:.6g} m, {orientation_error:.6g} rad)"
            )
    if state_action_mode not in {"joint", "end_effector"}:
        return False, f"unsupported state_action_mode={state_action_mode!r}"

    try:
        action_config_from_packet(packet)
        trajectory_config_from_packet(packet)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"bridge packet metadata is invalid ({exc})"
    return True, ""


def bridge_is_ready(
    latest_packet: Any,
    received_at_monotonic: float | None,
    now_monotonic: float,
    max_age_sec: float,
) -> tuple[bool, str]:
    """Check that a valid bridge sample has arrived recently enough to record."""
    if latest_packet is None or received_at_monotonic is None:
        return False, "no valid bridge sample has been received yet"
    if max_age_sec <= 0:
        raise ValueError("bridge-ready timeout must be positive")
    age_sec = max(0.0, now_monotonic - received_at_monotonic)
    if age_sec > max_age_sec:
        return False, f"latest bridge sample is {age_sec:.1f}s old (limit {max_age_sec:.1f}s)"
    return True, ""


def start_command_listener() -> Queue[str]:
    commands: Queue[str] = Queue()

    def _read_commands() -> None:
        while True:
            try:
                command = input().strip().lower()
            except EOFError:
                commands.put("q")
                return

            if command:
                commands.put(command)
            if command == "q":
                return

    listener = threading.Thread(target=_read_commands, daemon=True)
    listener.start()
    return commands


def is_lerobot_dataset_root(root: Path) -> bool:
    return (root / LEROBOT_INFO_PATH).exists()


def action_config_from_packet(packet: dict[str, Any]) -> dict[str, Any]:
    packet_arm_representation = str(packet.get("arm_action_representation", "absolute_joint_position")).strip().lower()
    state_action_mode = trajectory_config_from_packet(packet)["state_action_mode"]
    expected_arm_representation = (
        "absolute_joint_position" if state_action_mode == "joint" else "delta_end_effector_pose"
    )
    if packet_arm_representation != expected_arm_representation:
        raise ValueError(
            f"ROS 2 bridge published arm_action_representation={packet_arm_representation!r}. "
            f"Expected {expected_arm_representation!r} for state_action_mode={state_action_mode!r}."
        )
    arm_action_representation = expected_arm_representation
    gripper_action_representation = str(packet.get("gripper_action_representation", "absolute_width"))
    arm_action_definition = "q_target[t+1]" if state_action_mode == "joint" else "ee_target[t+1]-ee_current[t]"
    gripper_action_definition = {
        "absolute_width": "open_width_percent",
        "binary_open_close": "latched_binary_command (0=close, 1=open)",
    }.get(gripper_action_representation, gripper_action_representation)
    arm_mode = trajectory_config_from_packet(packet)["arm_mode"]
    return {
        "arm_action_representation": arm_action_representation,
        "arm_action_definition": arm_action_definition,
        "gripper_action_representation": gripper_action_representation,
        "gripper_action_definition": gripper_action_definition,
        "hand_action_representation": "absolute_joint_position",
        "hand_action_definition": "hand_q_target[t+1]",
        "arm_mode": arm_mode,
        "include_right_arm": bool(packet.get("include_right_arm", True)),
        "include_gripper": bool(packet.get("include_gripper", True)),
        "include_hand": bool(packet.get("include_hand", False)),
        "action_dim": int(packet["action_dim"]),
        "state_action_mode": trajectory_config_from_packet(packet)["state_action_mode"],
        "state_representation": trajectory_config_from_packet(packet)["state_representation"],
        "action_representation": trajectory_config_from_packet(packet)["action_representation"],
    }


def load_action_config(dataset_root: Path) -> dict[str, Any] | None:
    action_config_path = dataset_root / ACTION_CONFIG_PATH
    if not action_config_path.exists():
        return None
    return json.loads(action_config_path.read_text())


def write_action_config(dataset_root: Path, action_config: dict[str, Any]) -> None:
    action_config_path = dataset_root / ACTION_CONFIG_PATH
    action_config_path.parent.mkdir(parents=True, exist_ok=True)
    action_config_path.write_text(json.dumps(action_config, indent=2, sort_keys=True) + "\n")


def finalize_dataset(dataset: LeRobotDataset, repo_id: str) -> None:
    dataset.finalize()
    ensure_dataset_stats(repo_id, Path(dataset.root), force_recompute=True)


def assumed_legacy_action_config(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "arm_action_representation": "delta_joint_position",
        "arm_action_definition": "q[t+1]-q[t]",
        "gripper_action_representation": "absolute_width",
        "gripper_action_definition": "open_width_percent",
        "hand_action_representation": "absolute_joint_position",
        "hand_action_definition": "hand_q_target[t+1]",
        "arm_mode": trajectory_config_from_packet(packet)["arm_mode"],
        "include_right_arm": bool(packet.get("include_right_arm", True)),
        "include_gripper": bool(packet.get("include_gripper", True)),
        "include_hand": bool(packet.get("include_hand", False)),
        "action_dim": int(packet["action_dim"]),
        "state_action_mode": "joint",
        "state_representation": "joint",
        "action_representation": "target_joint",
    }


def build_features(first_packet: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    camera_names: list[str] = list(first_packet["camera_names"])
    trajectory_config = trajectory_config_from_packet(first_packet)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (int(first_packet["robot_state_dim"]),),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (int(first_packet["action_dim"]),),
            "names": ["action"],
        },
    }
    pose_dim = 6 * len(trajectory_config["arms"])
    joint_dim = 7 * len(trajectory_config["arms"])
    if trajectory_config["end_effector"] == "gripper":
        joint_dim += len(trajectory_config["arms"])
    elif trajectory_config["end_effector"] == "hand":
        joint_dim += 20 * len(trajectory_config["arms"])
    features["observation.joint_state"] = {
        "dtype": "float32", "shape": (joint_dim,), "names": ["joint_state"]
    }
    features["action.target_joint"] = {
        "dtype": "float32", "shape": (joint_dim,), "names": ["target_joint"]
    }
    features["observation.ee_pose"] = {
        "dtype": "float32", "shape": (pose_dim,), "names": ["ee_pose"]
    }
    features["action.delta_ee_pose"] = {
        "dtype": "float32", "shape": (pose_dim,), "names": ["delta_ee_pose"]
    }
    features["action.target_ee_pose"] = {
        "dtype": "float32", "shape": (pose_dim,), "names": ["target_ee_pose"]
    }

    for camera_name in camera_names:
        camera = first_packet["cameras"][camera_name]
        height, width, channels = camera["shape"]
        if channels != 3:
            raise ValueError(
                f"Camera '{camera_name}' must provide RGB frames with 3 channels, received {channels}."
            )
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": (3, int(height), int(width)),
            "names": ["c", "h", "w"],
        }

    return features, camera_names


def normalize_feature_specs(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for name, feature in features.items():
        if name in SYSTEM_FEATURES:
            continue
        normalized[name] = {
            "dtype": feature["dtype"],
            "shape": tuple(feature["shape"]),
            "names": None if feature.get("names") is None else tuple(feature["names"]),
        }
    return normalized


def derive_compatible_dataset_root(dataset_root: Path, suffix_parts: list[str]) -> Path:
    suffix = "_".join(suffix_parts)
    candidate = dataset_root.parent / f"{dataset_root.name}_{suffix}"
    index = 1
    while candidate.exists():
        candidate = dataset_root.parent / f"{dataset_root.name}_{suffix}_{index}"
        index += 1
    return candidate


def make_dataset(
    first_packet: dict[str, Any], repo_id: str, fps: int, dataset_root: Path
) -> tuple[LeRobotDataset, list[str], bool]:
    features, camera_names = build_features(first_packet)
    action_config = action_config_from_packet(first_packet)
    trajectory_config = trajectory_config_from_packet(first_packet)
    if is_lerobot_dataset_root(dataset_root):
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=dataset_root,
        )
        existing_action_config = load_action_config(dataset_root)
        trajectory_config_path = dataset_root / TRAJECTORY_CONFIG_PATH
        existing_trajectory_config = (
            json.loads(trajectory_config_path.read_text()) if trajectory_config_path.exists() else None
        )
        resolved_existing_action_config = (
            existing_action_config if existing_action_config is not None else assumed_legacy_action_config(first_packet)
        )
        if (
            normalize_feature_specs(dataset.features) == normalize_feature_specs(features)
            and resolved_existing_action_config == action_config
            and (existing_trajectory_config is None or existing_trajectory_config == trajectory_config)
        ):
            write_action_config(dataset_root, action_config)
            write_trajectory_config(dataset_root, trajectory_config)
            return dataset, camera_names, True

        existing_features = sorted(normalize_feature_specs(dataset.features))
        incoming_features = sorted(normalize_feature_specs(features))
        compatible_root = derive_compatible_dataset_root(
            dataset_root,
            camera_names + [action_config["arm_action_representation"]],
        )
        print(
            "Existing dataset metadata does not match the current ROS 2 stream. "
            f"Creating a new dataset at {compatible_root} instead of appending to {dataset_root}."
        )
        print(f"  existing features: {', '.join(existing_features)}")
        print(f"  incoming features: {', '.join(incoming_features)}")
        if existing_action_config is not None:
            print(f"  existing action config: {existing_action_config}")
        else:
            print(f"  existing action config: assumed legacy {resolved_existing_action_config}")
        print(f"  incoming action config: {action_config}")
        if existing_trajectory_config is not None:
            print(f"  existing trajectory config: {existing_trajectory_config}")
        print(f"  incoming trajectory config: {trajectory_config}")
        dataset_root = compatible_root

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
        root=dataset_root,
    )
    write_action_config(dataset_root, action_config)
    write_trajectory_config(dataset_root, trajectory_config)
    return dataset, camera_names, False


def packet_to_frame(packet: dict[str, Any], camera_names: list[str], task_name: str) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "observation.state": clamp_gripper_values(np.asarray(packet["state"], dtype=np.float32)),
        "action": clamp_gripper_values(np.asarray(packet["action"], dtype=np.float32)),
        "task": task_name,
    }
    frame["observation.ee_pose"] = np.asarray(packet["ee_pose"], dtype=np.float32)
    frame["action.delta_ee_pose"] = np.asarray(packet["delta_ee_pose"], dtype=np.float32)
    frame["action.target_ee_pose"] = np.asarray(packet["target_ee_pose"], dtype=np.float32)
    frame["observation.joint_state"] = np.asarray(packet["joint_state"], dtype=np.float32)
    frame["action.target_joint"] = np.asarray(packet["target_joint"], dtype=np.float32)
    for camera_name in camera_names:
        rgb = np.asarray(packet["cameras"][camera_name]["rgb"], dtype=np.uint8)
        frame[f"observation.images.{camera_name}"] = np.transpose(rgb, (2, 0, 1))
    return frame


def compute_recorded_action(
    current_packet: dict[str, Any],
    next_packet: dict[str, Any],
) -> np.ndarray:
    current_action = np.asarray(current_packet["action"], dtype=np.float32)
    next_action = np.asarray(next_packet["action"], dtype=np.float32)
    action_dim = int(current_packet["action_dim"])

    trajectory_config = trajectory_config_from_packet(current_packet)
    if trajectory_config["state_action_mode"] == "end_effector":
        current_pose = np.asarray(current_packet["ee_pose"], dtype=np.float32)
        next_target_pose = np.asarray(next_packet["target_ee_pose"], dtype=np.float32)
        if current_pose.shape != next_target_pose.shape:
            raise ValueError("End-effector pose and target pose dimensions do not match.")
        recorded_action = np.concatenate(
            [
                wrapped_pose_delta(current_pose[offset : offset + 6], next_target_pose[offset : offset + 6])
                for offset in range(0, len(current_pose), 6)
            ]
        ).astype(np.float32)
        if trajectory_config["end_effector"] == "gripper":
            for index, _ in enumerate(trajectory_config["arms"]):
                offset = index * 7
                recorded_action = np.insert(recorded_action, offset + 6, float(next_action[offset + 6]))
        elif trajectory_config["end_effector"] == "hand":
            values: list[float] = []
            for index, _ in enumerate(trajectory_config["arms"]):
                pose_offset = index * 6
                values.extend(recorded_action[pose_offset : pose_offset + 6])
                values.extend(next_action[index * 26 + 6 : index * 26 + 26])
            recorded_action = np.asarray(values, dtype=np.float32)
        return recorded_action
    end_effector = trajectory_config["end_effector"]
    arms = trajectory_config["arms"]
    block_size = 7 + (1 if end_effector == "gripper" else 20 if end_effector == "hand" else 0)
    expected_dim = block_size * len(arms)
    if action_dim != expected_dim:
        raise ValueError(f"Trajectory metadata expects {expected_dim} action values, got {action_dim}.")
    recorded_action = np.empty(action_dim, dtype=np.float32)
    for arm_index in range(len(arms)):
        offset = arm_index * block_size
        recorded_action[offset : offset + 7] = next_action[offset : offset + 7]
        if end_effector == "gripper":
            recorded_action[offset + 7] = current_action[offset + 7]
        elif end_effector == "hand":
            recorded_action[offset + 7 : offset + 27] = next_action[offset + 7 : offset + 27]
    return recorded_action


def packet_pair_to_frame(
    current_packet: dict[str, Any],
    next_packet: dict[str, Any],
    camera_names: list[str],
    task_name: str,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "observation.state": clamp_gripper_values(np.asarray(current_packet["state"], dtype=np.float32)),
        "action": clamp_gripper_values(compute_recorded_action(current_packet, next_packet)),
        "task": task_name,
    }
    frame["observation.ee_pose"] = np.asarray(current_packet["ee_pose"], dtype=np.float32)
    current_ee_pose = np.asarray(current_packet["ee_pose"], dtype=np.float32)
    target_ee_pose = np.asarray(next_packet["target_ee_pose"], dtype=np.float32)
    frame["action.delta_ee_pose"] = np.concatenate(
        [
            wrapped_pose_delta(current_ee_pose[offset : offset + 6], target_ee_pose[offset : offset + 6])
            for offset in range(0, len(current_ee_pose), 6)
        ]
    ).astype(np.float32)
    frame["action.target_ee_pose"] = target_ee_pose
    frame["observation.joint_state"] = np.asarray(current_packet["joint_state"], dtype=np.float32)
    frame["action.target_joint"] = np.asarray(next_packet["target_joint"], dtype=np.float32)
    for camera_name in camera_names:
        rgb = np.asarray(current_packet["cameras"][camera_name]["rgb"], dtype=np.uint8)
        frame[f"observation.images.{camera_name}"] = np.transpose(rgb, (2, 0, 1))
    return frame


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.local_dir).expanduser()
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    if dataset_root.exists() and not is_lerobot_dataset_root(dataset_root):
        if any(dataset_root.iterdir()):
            raise FileExistsError(
                f"Dataset directory '{dataset_root}' already exists and is not a LeRobot dataset. "
                "Choose a new --local-dir or remove the existing directory."
            )
        dataset_root.rmdir()

    if normalize_episode_metadata(dataset_root):
        print(f"Normalized episode metadata schemas in {dataset_root}")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://{args.host}:{args.port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 100)

    dataset: LeRobotDataset | None = None
    camera_names: list[str] | None = None
    task_name = args.task
    recording_active = False
    episode_count = 0
    pending_packet: dict[str, Any] | None = None
    latest_packet: dict[str, Any] | None = None
    latest_packet_received_at: float | None = None
    last_invalid_packet_warning_at = 0.0
    commands = start_command_listener()

    print(f"Listening for ROS 2 bridge samples on tcp://{args.host}:{args.port}")
    print("Episode controls:")
    print("  s + Enter: start recording a new episode (requires a fresh bridge sample)")
    print("  e + Enter: end and save the current episode")
    print("  d + Enter: discard the current episode")
    print("  q + Enter: quit the recorder")

    try:
        while True:
            packet: dict[str, Any] | None = None
            try:
                received_packet = socket.recv_pyobj()
            except zmq.Again:
                pass
            else:
                packet_ready, packet_reason = bridge_packet_readiness(received_packet)
                if packet_ready:
                    packet = received_packet
                    latest_packet = received_packet
                    latest_packet_received_at = time.monotonic()
                else:
                    now = time.monotonic()
                    if now - last_invalid_packet_warning_at >= 5.0:
                        print(f"Ignoring invalid bridge packet: {packet_reason}", file=sys.stderr)
                        last_invalid_packet_warning_at = now

            try:
                while True:
                    command = commands.get_nowait()
                    if command == "s":
                        if recording_active:
                            print("Already recording.")
                            continue
                        ready, reason = bridge_is_ready(
                            latest_packet,
                            latest_packet_received_at,
                            time.monotonic(),
                            args.bridge_ready_timeout,
                        )
                        if not ready:
                            print(
                                "Cannot start recording: "
                                f"{reason}. Check that the LeRobot bridge is publishing "
                                "the selected arm/end-effector data."
                            )
                            continue
                        if dataset is not None and dataset.has_pending_frames():
                            dataset.clear_episode_buffer()
                        pending_packet = None
                        recording_active = True
                        print("Recording started.")
                    elif command == "e":
                        if dataset is None or not dataset.has_pending_frames():
                            recording_active = False
                            pending_packet = None
                            print("No recorded frames to save.")
                            continue
                        dataset.save_episode()
                        episode_count += 1
                        recording_active = False
                        pending_packet = None
                        print(f"Episode {episode_count} saved to {dataset.root}")
                    elif command == "d":
                        recording_active = False
                        pending_packet = None
                        if dataset is not None and dataset.has_pending_frames():
                            dataset.clear_episode_buffer()
                            print("Current episode discarded.")
                        else:
                            print("No buffered episode to discard.")
                    elif command == "q":
                        raise KeyboardInterrupt
                    else:
                        print("Unknown command. Use: s, e, d, q")
            except Empty:
                pass

            if packet is None:
                continue

            if dataset is None:
                dataset, camera_names, resumed_dataset = make_dataset(
                    first_packet=packet,
                    repo_id=args.repo_id,
                    fps=args.fps,
                    dataset_root=dataset_root,
                )
                if task_name is None:
                    task_name = str(packet.get("task", "franka_gello_teleop"))

                print(f"LeRobot dataset {'resumed' if resumed_dataset else 'initialized'} with:")
                print(f"  root: {dataset.root}")
                print(f"  robot state dim: {packet['robot_state_dim']}")
                print(f"  action dim: {packet['action_dim']}")
                print(
                    "  action config: "
                    f"arm={action_config_from_packet(packet)['arm_action_representation']} "
                    f"({action_config_from_packet(packet)['arm_action_definition']}), "
                    f"gripper={action_config_from_packet(packet)['gripper_action_representation']}"
                )
                trajectory_config = trajectory_config_from_packet(packet)
                print(
                    "  trajectory setting: "
                    f"{trajectory_config['end_effector']} / {trajectory_config['arm_mode']}"
                )
                print(f"  cameras: {', '.join(camera_names)}")

            if camera_names is None:
                raise RuntimeError("Camera names were not initialized.")

            if recording_active:
                active_task_name = task_name or str(packet.get("task", "franka_gello_teleop"))
                if pending_packet is None:
                    pending_packet = packet
                else:
                    frame = packet_pair_to_frame(pending_packet, packet, camera_names, active_task_name)
                    dataset.add_frame(frame)
                    pending_packet = packet

    except KeyboardInterrupt:
        print("\nStopping collection...")
        if dataset is None or not dataset.has_pending_frames():
            if dataset is None:
                print("No samples received. Nothing was saved.")
            else:
                print("No unsaved episode in the recording buffer.")
        else:
            dataset.save_episode()
            episode_count += 1
            finalize_dataset(dataset, args.repo_id)
            print(f"Episode {episode_count} saved to {dataset.root}")
            return
        if dataset is not None:
            finalize_dataset(dataset, args.repo_id)
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
