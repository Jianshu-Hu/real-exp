from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LEROBOT_INFO_PATH = Path("meta/info.json")
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


def derive_compatible_dataset_root(dataset_root: Path, camera_names: list[str]) -> Path:
    suffix = "_".join(camera_names)
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
    if is_lerobot_dataset_root(dataset_root):
        dataset = LeRobotDataset.resume(
            repo_id=repo_id,
            root=dataset_root,
        )
        if normalize_feature_specs(dataset.features) == normalize_feature_specs(features):
            return dataset, camera_names, True

        existing_features = sorted(normalize_feature_specs(dataset.features))
        incoming_features = sorted(normalize_feature_specs(features))
        compatible_root = derive_compatible_dataset_root(dataset_root, camera_names)
        print(
            "Existing dataset schema does not match the current ROS 2 stream. "
            f"Creating a new dataset at {compatible_root} instead of appending to {dataset_root}."
        )
        print(f"  existing features: {', '.join(existing_features)}")
        print(f"  incoming features: {', '.join(incoming_features)}")
        dataset_root = compatible_root

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        use_videos=True,
        root=dataset_root,
    )
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
                        recording_active = True
                        print("Recording started.")
                    elif command == "e":
                        if dataset is None or not dataset.has_pending_frames():
                            recording_active = False
                            print("No recorded frames to save.")
                            continue
                        dataset.save_episode()
                        episode_count += 1
                        recording_active = False
                        print(f"Episode {episode_count} saved to {dataset.root}")
                    elif command == "d":
                        recording_active = False
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
                print(f"  cameras: {', '.join(camera_names)}")

            if camera_names is None:
                raise RuntimeError("Camera names were not initialized.")

            if recording_active:
                active_task_name = task_name or str(packet.get("task", "franka_gello_teleop"))
                frame = packet_to_frame(packet, camera_names, active_task_name)
                dataset.add_frame(frame)

    except KeyboardInterrupt:
        print("\nStopping collection...")
        if dataset is None or not dataset.has_pending_frames():
            print("No samples received. Nothing was saved.")
        else:
            dataset.save_episode()
            episode_count += 1
            dataset.finalize()
            print(f"Episode {episode_count} saved to {dataset.root}")
            return
        if dataset is not None:
            dataset.finalize()
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
