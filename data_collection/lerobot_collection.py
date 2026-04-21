from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dataset_stats import ensure_dataset_stats

LEROBOT_INFO_PATH = Path("meta/info.json")
ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
SYSTEM_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record robot state, teleop actions, and three RGB camera streams into LeRobot format."
    )
    parser.add_argument("--host", default="127.0.0.1", help="ZMQ host used by the ROS 2 bridge.")
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
    return parser.parse_args()


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
    if packet_arm_representation != "absolute_joint_position":
        raise ValueError(
            f"ROS 2 bridge published arm_action_representation={packet_arm_representation!r}. "
            "Data collection now requires absolute_joint_position arm actions."
        )
    arm_action_representation = "absolute_joint_position"
    gripper_action_representation = str(packet.get("gripper_action_representation", "absolute_width"))
    arm_action_definition = "q_target[t+1]"
    gripper_action_definition = {
        "absolute_width": "open_width_percent",
        "binary_open_close": "latched_binary_command (0=close, 1=open)",
    }.get(gripper_action_representation, gripper_action_representation)
    return {
        "arm_action_representation": arm_action_representation,
        "arm_action_definition": arm_action_definition,
        "gripper_action_representation": gripper_action_representation,
        "gripper_action_definition": gripper_action_definition,
        "include_right_arm": bool(packet.get("include_right_arm", True)),
        "include_gripper": bool(packet.get("include_gripper", True)),
        "action_dim": int(packet["action_dim"]),
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
        "include_right_arm": bool(packet.get("include_right_arm", True)),
        "include_gripper": bool(packet.get("include_gripper", True)),
        "action_dim": int(packet["action_dim"]),
    }


def build_features(first_packet: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    camera_names: list[str] = list(first_packet["camera_names"])
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
    if is_lerobot_dataset_root(dataset_root):
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=dataset_root,
        )
        existing_action_config = load_action_config(dataset_root)
        resolved_existing_action_config = (
            existing_action_config if existing_action_config is not None else assumed_legacy_action_config(first_packet)
        )
        if (
            normalize_feature_specs(dataset.features) == normalize_feature_specs(features)
            and resolved_existing_action_config == action_config
        ):
            write_action_config(dataset_root, action_config)
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
        dataset_root = compatible_root

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
        root=dataset_root,
    )
    write_action_config(dataset_root, action_config)
    return dataset, camera_names, False


def packet_to_frame(packet: dict[str, Any], camera_names: list[str], task_name: str) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "observation.state": np.asarray(packet["state"], dtype=np.float32),
        "action": np.asarray(packet["action"], dtype=np.float32),
        "task": task_name,
    }
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

    if action_dim == 16:
        recorded_action = np.empty(16, dtype=np.float32)
        recorded_action[0:7] = next_action[0:7]
        recorded_action[7] = current_action[7]
        recorded_action[8:15] = next_action[8:15]
        recorded_action[15] = current_action[15]
        return recorded_action

    if action_dim == 14:
        recorded_action = np.empty(14, dtype=np.float32)
        recorded_action[0:7] = next_action[0:7]
        recorded_action[7:14] = next_action[7:14]
        return recorded_action

    raise ValueError(f"Unsupported action dimension for recorder absolute target transform: {action_dim}")


def packet_pair_to_frame(
    current_packet: dict[str, Any],
    next_packet: dict[str, Any],
    camera_names: list[str],
    task_name: str,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "observation.state": np.asarray(current_packet["state"], dtype=np.float32),
        "action": compute_recorded_action(current_packet, next_packet),
        "task": task_name,
    }
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
    commands = start_command_listener()

    print(f"Listening for ROS 2 bridge samples on tcp://{args.host}:{args.port}")
    print("Episode controls:")
    print("  s + Enter: start recording a new episode")
    print("  e + Enter: end and save the current episode")
    print("  d + Enter: discard the current episode")
    print("  q + Enter: quit the recorder")

    try:
        while True:
            try:
                while True:
                    command = commands.get_nowait()
                    if command == "s":
                        if recording_active:
                            print("Already recording.")
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

            try:
                packet = socket.recv_pyobj()
            except zmq.Again:
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
            print("No samples received. Nothing was saved.")
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
