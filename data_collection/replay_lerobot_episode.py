"""Replay a LeRobot episode through the ROS 2 collection controller.

Usage:
    python data_collection/replay_lerobot_episode.py \\
        --dataset-root data/my_dataset --episode 0 --dry-run

The tool publishes the same arm and gripper command topics used during data
collection. Running without ``--dry-run`` commands real hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
TRACE_FILENAME = "trace.csv"
SUMMARY_FILENAME = "summary.json"
RUN_CONFIG_FILENAME = "run_config.json"
JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]


@dataclass
class EpisodeData:
    states: np.ndarray
    actions: np.ndarray
    frame_indices: np.ndarray
    timestamps: np.ndarray
    fps: float
    action_config: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a local LeRobot episode through the same ROS 2 collection-controller "
            "topics used during data collection."
        )
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument(
        "--mode",
        choices=("action", "state", "policy"),
        default="action",
        help="Target source: dataset action, observation.state, or policy (reserved).",
    )
    parser.add_argument("--fps", type=float, default=None, help="Replay FPS. Defaults to dataset metadata fps.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("replay_trace"),
        help="Directory where trace/report inputs are written. Defaults to replay_trace.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive dataset frame_index to start from.")
    parser.add_argument("--end-frame", type=int, default=None, help="Exclusive dataset frame_index to stop at.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of selected frames to replay.")
    parser.add_argument("--dry-run", action="store_true", help="Load and summarize targets without publishing.")
    parser.add_argument("--no-gripper", action="store_true", help="Skip gripper command publishing and trace targets.")
    parser.add_argument("--left-target-topic", default="/left/gello/joint_states")
    parser.add_argument("--right-target-topic", default="/right/gello/joint_states")
    parser.add_argument("--left-gripper-topic", default="/left/gripper/gripper_client/target_gripper_width_percent")
    parser.add_argument("--right-gripper-topic", default="/right/gripper/gripper_client/target_gripper_width_percent")
    parser.add_argument("--left-state-topic", default="/left/franka/joint_states")
    parser.add_argument("--right-state-topic", default="/right/franka/joint_states")
    parser.add_argument("--left-gripper-state-topic", default="/left/franka_gripper/joint_states")
    parser.add_argument("--right-gripper-state-topic", default="/right/franka_gripper/joint_states")
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

    threading.Thread(target=_read_commands, daemon=True).start()
    return commands


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_episode_data(dataset_root: Path, episode_index: int) -> EpisodeData:
    dataset_root = dataset_root.expanduser()
    info = load_json(dataset_root / "meta" / "info.json")
    action_config = load_json(dataset_root / ACTION_CONFIG_PATH)

    arm_action_representation = str(action_config.get("arm_action_representation", "")).strip().lower()
    if arm_action_representation != "absolute_joint_position":
        raise ValueError(
            "LeRobot episode replay currently requires arm_action_representation="
            f"absolute_joint_position, got {arm_action_representation!r}."
        )
    gripper_action_representation = str(
        action_config.get("gripper_action_representation", "absolute_width")
    ).strip().lower()
    if gripper_action_representation != "absolute_width":
        raise ValueError(
            "LeRobot episode replay requires continuous absolute_width gripper actions, "
            f"got {gripper_action_representation!r}."
        )

    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_root / 'data'}")

    rows: list[tuple[int, float, list[float], list[float]]] = []
    available_episodes: set[int] = set()
    for parquet_file in parquet_files:
        table = pq.read_table(
            parquet_file,
            columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
        )
        data = table.to_pydict()
        for row_episode, frame_index, timestamp, state, action in zip(
            data["episode_index"],
            data["frame_index"],
            data["timestamp"],
            data["observation.state"],
            data["action"],
            strict=True,
        ):
            available_episodes.add(int(row_episode))
            if int(row_episode) != episode_index:
                continue
            rows.append((int(frame_index), float(timestamp), state, action))

    if not rows:
        raise ValueError(f"Episode {episode_index} not found. Available episodes: {sorted(available_episodes)}")

    rows.sort(key=lambda item: item[0])
    states = np.asarray([row[2] for row in rows], dtype=float)
    actions = np.asarray([row[3] for row in rows], dtype=float)
    if states.ndim != 2 or actions.ndim != 2 or states.shape[1] != 16 or actions.shape[1] != 16:
        raise ValueError(
            "Controller-matched replay currently supports 16-dim bimanual datasets only. "
            f"Got state shape {states.shape}, action shape {actions.shape}."
        )

    return EpisodeData(
        states=states,
        actions=actions,
        frame_indices=np.asarray([row[0] for row in rows], dtype=int),
        timestamps=np.asarray([row[1] for row in rows], dtype=float),
        fps=float(info["fps"]),
        action_config=action_config,
    )


def select_frame_range(data: EpisodeData, start_frame: int, end_frame: int | None, max_frames: int | None) -> EpisodeData:
    mask = data.frame_indices >= start_frame
    if end_frame is not None:
        mask &= data.frame_indices < end_frame
    indices = np.nonzero(mask)[0]
    if max_frames is not None:
        indices = indices[:max_frames]
    if len(indices) == 0:
        raise ValueError("Selected frame range is empty.")

    return EpisodeData(
        states=data.states[indices],
        actions=data.actions[indices],
        frame_indices=data.frame_indices[indices],
        timestamps=data.timestamps[indices],
        fps=data.fps,
        action_config=data.action_config,
    )


def split_targets(data: EpisodeData, mode: str) -> dict[str, np.ndarray]:
    if mode == "policy":
        raise NotImplementedError(
            "policy mode is reserved for closed-loop smoke-test integration and is not implemented in v1."
        )

    source = data.actions if mode == "action" else data.states
    return {
        "left_arm": source[:, 0:7],
        "right_arm": source[:, 8:15],
        "left_gripper_raw": source[:, 7],
        "right_gripper_raw": source[:, 15],
    }


def continuous_gripper_targets(values: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Gripper targets must contain only finite values.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Normalized continuous gripper targets must be within [0, 1].")
    return values.copy()


def print_dry_run_summary(data: EpisodeData, mode: str, fps: float, no_gripper: bool) -> None:
    targets = split_targets(data, mode)
    print("Controller-matched replay dry run")
    print("----------------------------------")
    print(f"frames: {len(data.frame_indices)}")
    print(f"frame range: {int(data.frame_indices[0])}..{int(data.frame_indices[-1])}")
    print(f"dataset fps: {data.fps:g}")
    print(f"replay fps: {fps:g}")
    print(f"mode: {mode}")
    print(f"arm action config: {data.action_config.get('arm_action_representation')} / {data.action_config.get('arm_action_definition')}")
    for arm_name in ("left_arm", "right_arm"):
        values = np.asarray(targets[arm_name], dtype=float)
        print(f"{arm_name} target min: {np.min(values, axis=0).round(6).tolist()}")
        print(f"{arm_name} target max: {np.max(values, axis=0).round(6).tolist()}")
    if not no_gripper:
        left = continuous_gripper_targets(targets["left_gripper_raw"], mode)
        right = continuous_gripper_targets(targets["right_gripper_raw"], mode)
        print(f"left gripper target counts: {value_counts(left)}")
        print(f"right gripper target counts: {value_counts(right)}")


def value_counts(values: np.ndarray) -> dict[str, int]:
    return {str(float(value)): int(np.sum(values == value)) for value in sorted(set(values.tolist()))}


def flatten_joint(prefix: str, values: np.ndarray | None) -> dict[str, float | str]:
    row: dict[str, float | str] = {}
    for index in range(7):
        row[f"{prefix}_{index + 1}"] = "" if values is None else float(values[index])
    return row


def trace_fieldnames() -> list[str]:
    fields = ["time_s", "frame_index", "dataset_timestamp", "mode", "target_source"]
    for prefix in ("left_target_q", "right_target_q", "left_actual_q", "right_actual_q", "left_error_q", "right_error_q"):
        fields.extend(f"{prefix}_{index}" for index in range(1, 8))
    fields.extend(
        [
            "left_gripper_target",
            "right_gripper_target",
            "left_gripper_actual",
            "right_gripper_actual",
            "abort_requested",
            "controller_ready",
        ]
    )
    return fields


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def import_ros_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float32
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ROS 2 Python dependencies are required for LeRobot episode replay. "
            "Run this script in the robot_control Docker/devcontainer environment."
        ) from exc
    return rclpy, Node, JointState, Float32


def build_replay_node_class(Node: Any, JointState: Any, Float32: Any) -> type:
    class LerobotEpisodeReplayNode(Node):  # type: ignore[misc, valid-type]
        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__("lerobot_episode_replay")
            self.left_target_publisher = self.create_publisher(JointState, args.left_target_topic, 10)
            self.right_target_publisher = self.create_publisher(JointState, args.right_target_topic, 10)
            self.left_gripper_publisher = self.create_publisher(Float32, args.left_gripper_topic, 10)
            self.right_gripper_publisher = self.create_publisher(Float32, args.right_gripper_topic, 10)
            self.left_actual_q: np.ndarray | None = None
            self.right_actual_q: np.ndarray | None = None
            self.left_gripper_actual: float | None = None
            self.right_gripper_actual: float | None = None
            self.create_subscription(
                JointState,
                args.left_state_topic,
                lambda msg: self._store_arm_state("left", msg),
                10,
            )
            self.create_subscription(
                JointState,
                args.right_state_topic,
                lambda msg: self._store_arm_state("right", msg),
                10,
            )
            self.create_subscription(
                JointState,
                args.left_gripper_state_topic,
                lambda msg: self._store_gripper_state("left", msg),
                10,
            )
            self.create_subscription(
                JointState,
                args.right_gripper_state_topic,
                lambda msg: self._store_gripper_state("right", msg),
                10,
            )

        def _ordered_arm_values(self, msg: Any) -> np.ndarray | None:
            if len(msg.position) < 7:
                return None
            if len(msg.name) >= 7:
                values: list[float | None] = [None] * 7
                for name, position in zip(msg.name, msg.position, strict=False):
                    for joint_index in range(1, 8):
                        if name.endswith(f"joint{joint_index}"):
                            values[joint_index - 1] = float(position)
                if all(value is not None for value in values):
                    return np.asarray([float(value) for value in values], dtype=float)
            return np.asarray(msg.position[:7], dtype=float)

        def _store_arm_state(self, arm_name: str, msg: Any) -> None:
            values = self._ordered_arm_values(msg)
            if values is None:
                return
            if arm_name == "left":
                self.left_actual_q = values
            else:
                self.right_actual_q = values

        def _store_gripper_state(self, arm_name: str, msg: Any) -> None:
            if not msg.position:
                return
            width = float(sum(msg.position))
            if arm_name == "left":
                self.left_gripper_actual = width
            else:
                self.right_gripper_actual = width

        def publish_targets(
            self,
            left_target: np.ndarray,
            right_target: np.ndarray,
            left_gripper: float | None,
            right_gripper: float | None,
        ) -> None:
            now_msg = self.get_clock().now().to_msg()
            left_msg = JointState()
            left_msg.header.stamp = now_msg
            left_msg.name = JOINT_NAMES
            left_msg.position = [float(value) for value in left_target]
            right_msg = JointState()
            right_msg.header.stamp = now_msg
            right_msg.name = JOINT_NAMES
            right_msg.position = [float(value) for value in right_target]
            self.left_target_publisher.publish(left_msg)
            self.right_target_publisher.publish(right_msg)

            if left_gripper is not None:
                msg = Float32()
                msg.data = float(left_gripper)
                self.left_gripper_publisher.publish(msg)
            if right_gripper is not None:
                msg = Float32()
                msg.data = float(right_gripper)
                self.right_gripper_publisher.publish(msg)

        def controller_ready(self, no_gripper: bool) -> bool:
            arms_ready = self.left_actual_q is not None and self.right_actual_q is not None
            if no_gripper:
                return arms_ready
            return arms_ready and self.left_gripper_actual is not None and self.right_gripper_actual is not None

    return LerobotEpisodeReplayNode


def build_trace_row(
    *,
    elapsed_s: float,
    frame_index: int,
    dataset_timestamp: float,
    mode: str,
    target_source: str,
    left_target: np.ndarray,
    right_target: np.ndarray,
    left_actual: np.ndarray | None,
    right_actual: np.ndarray | None,
    left_gripper_target: float | None,
    right_gripper_target: float | None,
    left_gripper_actual: float | None,
    right_gripper_actual: float | None,
    abort_requested: bool,
    controller_ready: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "time_s": elapsed_s,
        "frame_index": frame_index,
        "dataset_timestamp": dataset_timestamp,
        "mode": mode,
        "target_source": target_source,
        "left_gripper_target": "" if left_gripper_target is None else float(left_gripper_target),
        "right_gripper_target": "" if right_gripper_target is None else float(right_gripper_target),
        "left_gripper_actual": "" if left_gripper_actual is None else float(left_gripper_actual),
        "right_gripper_actual": "" if right_gripper_actual is None else float(right_gripper_actual),
        "abort_requested": bool(abort_requested),
        "controller_ready": bool(controller_ready),
    }
    row.update(flatten_joint("left_target_q", left_target))
    row.update(flatten_joint("right_target_q", right_target))
    row.update(flatten_joint("left_actual_q", left_actual))
    row.update(flatten_joint("right_actual_q", right_actual))
    row.update(flatten_joint("left_error_q", None if left_actual is None else left_target - left_actual))
    row.update(flatten_joint("right_error_q", None if right_actual is None else right_target - right_actual))
    return row


def consume_commands(commands: Queue[str]) -> str | None:
    command_seen: str | None = None
    while True:
        try:
            command = commands.get_nowait()
        except Empty:
            return command_seen
        if command == "q":
            return "q"
        command_seen = command


def wait_for_start(rclpy: Any, node: Any, commands: Queue[str], no_gripper: bool) -> bool:
    print("Waiting for ROS state samples. Type `s` + Enter to start, or `q` + Enter to abort.")
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        command = consume_commands(commands)
        if command == "q":
            return False
        if command == "s":
            if not node.controller_ready(no_gripper):
                print("Warning: starting before all actual state samples are available.")
            return True
    return False


def sleep_with_spin_and_abort(
    rclpy: Any,
    node: Any,
    commands: Queue[str],
    deadline_s: float,
    *,
    spin_step_s: float = 0.01,
) -> bool:
    """Sleep until deadline while keeping ROS subscriptions and keyboard abort live.

    Returns True when the user requested abort.
    """
    while rclpy.ok():
        remaining_s = deadline_s - time.perf_counter()
        if remaining_s <= 0:
            return False
        command = consume_commands(commands)
        if command == "q":
            return True
        rclpy.spin_once(node, timeout_sec=min(spin_step_s, remaining_s))
    return True


def run_replay(args: argparse.Namespace, data: EpisodeData, fps: float) -> None:
    rclpy, Node, JointState, Float32 = import_ros_dependencies()
    ReplayNode = build_replay_node_class(Node, JointState, Float32)

    targets = split_targets(data, args.mode)
    left_gripper_targets = None if args.no_gripper else continuous_gripper_targets(targets["left_gripper_raw"], args.mode)
    right_gripper_targets = None if args.no_gripper else continuous_gripper_targets(targets["right_gripper_raw"], args.mode)
    args.output.mkdir(parents=True, exist_ok=True)

    write_json(
        args.output / RUN_CONFIG_FILENAME,
        {
            "dataset_root": str(args.dataset_root),
            "episode": args.episode,
            "mode": args.mode,
            "fps": fps,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "max_frames": args.max_frames,
            "no_gripper": args.no_gripper,
            "trace_file": TRACE_FILENAME,
        },
    )

    rclpy.init()
    node = ReplayNode(args)
    commands = start_command_listener()
    trace_path = args.output / TRACE_FILENAME
    with trace_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=trace_fieldnames()).writeheader()
    aborted = False
    published_frames = 0
    last_controller_ready = False
    start_time = 0.0
    try:
        if not wait_for_start(rclpy, node, commands, args.no_gripper):
            aborted = True
            return

        print("Starting LeRobot episode replay. Type `q` + Enter to abort.")
        start_time = time.perf_counter()
        next_publish_time = start_time
        with trace_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trace_fieldnames())
            writer.writeheader()

            for local_idx, frame_index in enumerate(data.frame_indices):
                command = consume_commands(commands)
                if command == "q":
                    aborted = True
                    print("Abort requested by user.")
                    break

                if sleep_with_spin_and_abort(rclpy, node, commands, next_publish_time):
                    aborted = True
                    print("Abort requested by user.")
                    break

                rclpy.spin_once(node, timeout_sec=0.0)
                left_target = np.asarray(targets["left_arm"][local_idx], dtype=float)
                right_target = np.asarray(targets["right_arm"][local_idx], dtype=float)
                left_gripper = None if left_gripper_targets is None else float(left_gripper_targets[local_idx])
                right_gripper = None if right_gripper_targets is None else float(right_gripper_targets[local_idx])
                node.publish_targets(left_target, right_target, left_gripper, right_gripper)
                rclpy.spin_once(node, timeout_sec=0.0)
                last_controller_ready = node.controller_ready(args.no_gripper)

                elapsed_s = time.perf_counter() - start_time
                writer.writerow(
                    build_trace_row(
                        elapsed_s=elapsed_s,
                        frame_index=int(frame_index),
                        dataset_timestamp=float(data.timestamps[local_idx]),
                        mode=args.mode,
                        target_source=args.mode,
                        left_target=left_target,
                        right_target=right_target,
                        left_actual=node.left_actual_q,
                        right_actual=node.right_actual_q,
                        left_gripper_target=left_gripper,
                        right_gripper_target=right_gripper,
                        left_gripper_actual=node.left_gripper_actual,
                        right_gripper_actual=node.right_gripper_actual,
                        abort_requested=aborted,
                        controller_ready=last_controller_ready,
                    )
                )
                published_frames += 1
                next_publish_time += 1.0 / fps
    except KeyboardInterrupt:
        aborted = True
        print("Abort requested by KeyboardInterrupt.")
    finally:
        write_json(
            args.output / SUMMARY_FILENAME,
            {
                "aborted": aborted,
                "completed": bool((not aborted) and published_frames == len(data.frame_indices)),
                "published_frames": published_frames,
                "selected_frames": int(len(data.frame_indices)),
                "controller_ready_at_last_frame": bool(last_controller_ready),
                "duration_s": float(time.perf_counter() - start_time) if start_time else 0.0,
                "trace_path": str(trace_path),
            },
        )
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    args = parse_args()
    data = load_episode_data(args.dataset_root, args.episode)
    data = select_frame_range(data, args.start_frame, args.end_frame, args.max_frames)
    fps = float(args.fps if args.fps is not None else data.fps)

    if args.mode == "policy":
        raise NotImplementedError(
            "policy mode is reserved for closed-loop smoke-test integration and is not implemented in v1."
        )

    if args.dry_run:
        print_dry_run_summary(data, args.mode, fps, args.no_gripper)
        return

    run_replay(args, data, fps)


if __name__ == "__main__":
    main()
