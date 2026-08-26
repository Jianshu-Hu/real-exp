"""Replay a LeRobot episode through the ROS 2 collection controller.

Usage:
    python data_collection/replay_lerobot_episode.py \\
        --dataset-root data/my_dataset --episode 0 --dry-run

The tool first moves both arms to the first selected ``observation.state``, then
publishes the same arm and gripper command topics used during data collection.
Running without ``--dry-run`` commands real hardware.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.trajectory_metadata import (
    EE_ACTION_DIM,
    EE_STATE_DIM,
    require_dataset_trajectory_config,
    validate_action_trajectory_contract,
    validate_setting,
)
from utils.wuji_hand_control import (
    HAND_INITIAL_POSITION_TOLERANCE_RAD,
    make_smoothed_backend_class,
    normalize_hand_positions,
)

ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
TRACE_FILENAME = "trace.csv"
SUMMARY_FILENAME = "summary.json"
RUN_CONFIG_FILENAME = "run_config.json"
JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]
DEFAULT_INITIAL_STATE_POSITION_TOLERANCE_RAD = 0.06
INITIAL_STATE_VELOCITY_TOLERANCE_RAD_PER_S = 0.05
INITIAL_STATE_PUBLISH_PERIOD_S = 0.02
INITIAL_STATE_STABLE_SAMPLES = 5
INITIAL_STATE_PRIME_DURATION_S = 0.5
DEFAULT_INITIAL_STATE_TIMEOUT_S = 120.0
DEFAULT_INITIAL_STATE_MAX_VELOCITY_RAD_PER_S = 0.10
DEFAULT_INITIAL_STATE_MAX_ACCELERATION_RAD_PER_S2 = 0.20
INITIAL_STATE_TRACKING_GAIN_PER_S = 1.5


@dataclass
class EpisodeData:
    states: np.ndarray
    actions: np.ndarray
    frame_indices: np.ndarray
    timestamps: np.ndarray
    fps: float
    action_config: dict[str, Any]
    trajectory_config: dict[str, Any] = field(default_factory=lambda: {
        "end_effector": "gripper", "arm_mode": "duo", "arms": ["left", "right"]
    })
    ee_poses: np.ndarray | None = None
    delta_ee_poses: np.ndarray | None = None
    target_ee_poses: np.ndarray | None = None
    joint_states: np.ndarray | None = None
    target_joints: np.ndarray | None = None
    replay_mode: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a local LeRobot episode through the same ROS 2 collection-controller "
            "topics used during data collection."
        )
    )
    parser.add_argument("--dataset-root", type=Path, help="LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay.")
    parser.add_argument(
        "--replay-mode",
        choices=["joint", "ee"],
        default=None,
        help=(
            "Arm representation to replay. 'joint' publishes stored joint targets; "
            "'ee' verifies stored EE targets against their recorded joint targets "
            "before replaying those geometrically consistent targets. "
            "Defaults to the dataset's state_action_mode."
        ),
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
    parser.add_argument(
        "--initial-state-timeout",
        type=float,
        default=DEFAULT_INITIAL_STATE_TIMEOUT_S,
        help=(
            "Maximum seconds to wait for both arms to reach the first selected observation.state "
            f"before replaying actions. Defaults to {DEFAULT_INITIAL_STATE_TIMEOUT_S:g}."
        ),
    )
    parser.add_argument(
        "--initial-state-max-velocity",
        type=float,
        default=DEFAULT_INITIAL_STATE_MAX_VELOCITY_RAD_PER_S,
        help=(
            "Maximum commanded joint velocity while moving to the initial state in rad/s. "
            f"Defaults to {DEFAULT_INITIAL_STATE_MAX_VELOCITY_RAD_PER_S:g}."
        ),
    )
    parser.add_argument(
        "--initial-state-max-acceleration",
        type=float,
        default=DEFAULT_INITIAL_STATE_MAX_ACCELERATION_RAD_PER_S2,
        help=(
            "Maximum commanded joint acceleration while moving to the initial state in rad/s^2. "
            f"Defaults to {DEFAULT_INITIAL_STATE_MAX_ACCELERATION_RAD_PER_S2:g}."
        ),
    )
    parser.add_argument(
        "--initial-state-position-tolerance",
        type=float,
        default=DEFAULT_INITIAL_STATE_POSITION_TOLERANCE_RAD,
        help=(
            "Maximum per-joint position error accepted for the initial-state gate in rad. "
            f"Defaults to {DEFAULT_INITIAL_STATE_POSITION_TOLERANCE_RAD:g}."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Load and summarize targets without publishing.")
    parser.add_argument("--no-gripper", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--robot-end-effector", choices=["arm", "gripper", "hand"], default=None,
                        help="Connected end-effector setting; normally supplied by scripts/replay.sh.")
    parser.add_argument("--robot-arm-mode", choices=["duo", "left", "right"], default=None,
                        help="Connected arm setting; normally supplied by scripts/replay.sh.")
    parser.add_argument("--left-hand-command-port", type=int, default=5561)
    parser.add_argument("--right-hand-command-port", type=int, default=5562)
    parser.add_argument("--left-hand-status-port", type=int, default=5563, help=argparse.SUPPRESS)
    parser.add_argument("--right-hand-status-port", type=int, default=5564, help=argparse.SUPPRESS)
    parser.add_argument("--internal-wuji-hand", choices=["left", "right"], default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--hand-ip", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--allow-missing-state",
        action="store_true",
        help="Allow replay to start before all required actual arm/gripper state topics have produced a sample.",
    )
    parser.add_argument("--left-target-topic", default="/left/gello/raw_joint_states")
    parser.add_argument("--right-target-topic", default="/right/gello/raw_joint_states")
    parser.add_argument("--left-gripper-topic", default="/left/gripper/gripper_client/target_gripper_width_percent")
    parser.add_argument("--right-gripper-topic", default="/right/gripper/gripper_client/target_gripper_width_percent")
    parser.add_argument("--left-state-topic", default="/left/franka/joint_states")
    parser.add_argument("--right-state-topic", default="/right/franka/joint_states")
    parser.add_argument(
        "--left-robot-state-topic",
        default="/left/franka_robot_state_broadcaster/robot_state",
        help="FrankaRobotState topic used for EE replay and IK.",
    )
    parser.add_argument(
        "--right-robot-state-topic",
        default="/right/franka_robot_state_broadcaster/robot_state",
        help="FrankaRobotState topic used for EE replay and IK.",
    )
    parser.add_argument("--left-gripper-state-topic", default="/left/franka_gripper/joint_states")
    parser.add_argument("--right-gripper-state-topic", default="/right/franka_gripper/joint_states")
    args = parser.parse_args(argv)
    if args.internal_wuji_hand is None and args.dataset_root is None:
        parser.error("the following arguments are required: --dataset-root")
    return args


def run_wuji_hand_process(side: str, command_port: int, status_port: int, hand_ip: str) -> None:
    """Run the Wuji SDK command process in the non-ROS Python environment."""
    example_dir = Path(__file__).resolve().parents[1] / "libs/wuji-retargeting/example"
    os.chdir(example_dir)
    # The replay module imports the repository's ``utils`` package before this
    # worker starts. ``teleop_real`` has a different package with the same name
    # under the submodule's example directory, so clear the cached repository
    # package before loading the submodule.
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]
    example_path = str(example_dir)
    if example_path in sys.path:
        sys.path.remove(example_path)
    sys.path.insert(0, example_path)
    from teleop_real import WujiHand2Backend as OriginalWujiHand2Backend

    backend_class = make_smoothed_backend_class(OriginalWujiHand2Backend)
    backend = backend_class(
        ip=hand_ip,
        kp=3.0,
        kd=0.1,
        current_limit=1.5,
        handedness=side,
    )
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 2)
    socket.bind(f"tcp://127.0.0.1:{command_port}")
    status_socket = context.socket(zmq.REP)
    status_socket.bind(f"tcp://127.0.0.1:{status_port}")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    poller.register(status_socket, zmq.POLLIN)
    initial_target: np.ndarray | None = None
    print(f"{side} Wuji Hand 2 replay is ready on tcp://127.0.0.1:{command_port}", flush=True)
    try:
        while True:
            for ready_socket, _ in poller.poll(50):
                if ready_socket is socket:
                    payload = socket.recv_pyobj()
                    if isinstance(payload, dict):
                        kind = str(payload.get("kind", "target"))
                        target = np.asarray(payload.get("target", []), dtype=np.float64)
                    else:
                        kind = "target"
                        target = np.asarray(payload, dtype=np.float64)
                    if target.shape != (20,) or not np.all(np.isfinite(target)):
                        print(f"Ignoring invalid {side} hand target with shape {target.shape}", flush=True)
                        continue
                    if kind == "initial":
                        initial_target = target.copy()
                    backend.send(target)
                else:
                    request = status_socket.recv_pyobj()
                    if isinstance(request, dict) and str(request.get("kind", "status")) == "initial":
                        requested_target = np.asarray(request.get("target", []), dtype=np.float64)
                        if requested_target.shape == (20,) and np.all(np.isfinite(requested_target)):
                            initial_target = requested_target.copy()
                            backend.send(requested_target)
                    actual = backend.actual_position()
                    target_position = backend.target_position if initial_target is not None else None
                    target_ready = (
                        target_position is not None
                        and initial_target is not None
                        and float(np.max(np.abs(target_position - initial_target)))
                        <= HAND_INITIAL_POSITION_TOLERANCE_RAD
                    )
                    reached = (
                        target_ready
                        and actual is not None
                        and float(np.max(np.abs(actual - target_position)))
                        <= HAND_INITIAL_POSITION_TOLERANCE_RAD
                    )
                    status_socket.send_pyobj(
                        {
                            "ready": True,
                            "initial_reached": reached,
                            "initial_received": initial_target is not None,
                            "actual": None if actual is None else actual.tolist(),
                            "initial_target": backend.target_position.tolist()
                            if initial_target is not None
                            else None,
                            "request": request,
                        }
                    )
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(0)
        status_socket.close(0)
        context.term()
        backend.close()


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
    if arm_action_representation not in {
        "absolute_joint_position",
        "delta_joint_position",
        "delta_end_effector_pose",
        "delta_end_effector_position_rotation_vector",
    }:
        raise ValueError(
            "LeRobot episode replay requires arm_action_representation="
            "a supported joint/EE action representation, got "
            f"{arm_action_representation!r}."
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

    trajectory_config = require_dataset_trajectory_config(dataset_root)
    ee_mode = str(trajectory_config.get("state_action_mode", "joint")).strip().lower() == "end_effector"
    available_columns = set(pq.read_schema(parquet_files[0]).names)
    ee_columns_available = {"observation.ee_pose", "action.delta_ee_pose"}.issubset(available_columns)
    target_ee_column_available = "action.target_ee_pose" in available_columns
    joint_columns_available = {"observation.joint_state", "action.target_joint"}.issubset(available_columns)
    if ee_mode and not ee_columns_available:
        raise ValueError(
            "End-effector replay requires observation.ee_pose and action.delta_ee_pose dataset fields."
        )

    rows: list[tuple[int, float, list[float], list[float], list[float] | None, list[float] | None, list[float] | None, list[float] | None]] = []
    available_episodes: set[int] = set()
    for parquet_file in parquet_files:
        columns = ["episode_index", "frame_index", "timestamp", "observation.state", "action"]
        if ee_columns_available:
            columns.extend(["observation.ee_pose", "action.delta_ee_pose"])
        if target_ee_column_available:
            columns.append("action.target_ee_pose")
        if joint_columns_available:
            columns.extend(["observation.joint_state", "action.target_joint"])
        table = pq.read_table(parquet_file, columns=columns)
        data = table.to_pydict()
        values = [
            data[name]
            for name in ("episode_index", "frame_index", "timestamp", "observation.state", "action")
        ]
        if ee_columns_available:
            values.extend([data["observation.ee_pose"], data["action.delta_ee_pose"]])
        if target_ee_column_available:
            values.append(data["action.target_ee_pose"])
        if joint_columns_available:
            values.extend([data["observation.joint_state"], data["action.target_joint"]])
        for row_values in zip(*values, strict=True):
            row_episode, frame_index, timestamp, state, action = row_values[:5]
            cursor = 5
            ee_pose = row_values[cursor] if ee_columns_available else None
            delta_ee_pose = row_values[cursor + 1] if ee_columns_available else None
            cursor += 2 if ee_columns_available else 0
            target_ee_pose = row_values[cursor] if target_ee_column_available else None
            cursor += 1 if target_ee_column_available else 0
            joint_state = row_values[cursor] if joint_columns_available else None
            target_joint = row_values[cursor + 1] if joint_columns_available else None
            available_episodes.add(int(row_episode))
            if int(row_episode) != episode_index:
                continue
            rows.append(
                (
                    int(frame_index), float(timestamp), state, action, ee_pose,
                    delta_ee_pose, target_ee_pose, joint_state, target_joint,
                )
            )

    if not rows:
        raise ValueError(f"Episode {episode_index} not found. Available episodes: {sorted(available_episodes)}")

    rows.sort(key=lambda item: item[0])
    states = np.asarray([row[2] for row in rows], dtype=float)
    actions = np.asarray([row[3] for row in rows], dtype=float)
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            "Replay requires two-dimensional state/action arrays. "
            f"Got state shape {states.shape}, action shape {actions.shape}."
        )
    validate_action_trajectory_contract(
        action_config, trajectory_config, source=str(dataset_root / "meta")
    )
    expected_dim = int(trajectory_config.get("action_dim", actions.shape[1]))
    if actions.shape[1] != expected_dim:
        raise ValueError(
            f"Trajectory metadata declares action_dim={expected_dim}, but episode data has {actions.shape[1]}."
        )
    if ee_columns_available:
        expected_ee_state_dim = EE_STATE_DIM * len(trajectory_config["arms"])
        expected_ee_action_dim = EE_ACTION_DIM * len(trajectory_config["arms"])
        ee_poses = np.asarray([row[4] for row in rows], dtype=float)
        delta_ee_poses = np.asarray([row[5] for row in rows], dtype=float)
        if (
            ee_poses.shape != (len(rows), expected_ee_state_dim)
            or delta_ee_poses.shape != (len(rows), expected_ee_action_dim)
            or not np.all(np.isfinite(ee_poses))
            or not np.all(np.isfinite(delta_ee_poses))
        ):
            raise ValueError(
                "End-effector replay requires finite EE fields with shape "
                f"({len(rows)}, {expected_ee_state_dim}) state and "
                f"({len(rows)}, {expected_ee_action_dim}) action."
            )
    else:
        ee_poses = None
        delta_ee_poses = None
    if target_ee_column_available:
        expected_ee_dim = EE_STATE_DIM * len(trajectory_config["arms"])
        target_ee_poses = np.asarray([row[6] for row in rows], dtype=float)
        if (
            target_ee_poses.shape != (len(rows), expected_ee_dim)
            or not np.all(np.isfinite(target_ee_poses))
        ):
            raise ValueError(
                "EE replay requires finite action.target_ee_pose values with shape "
                f"({len(rows)}, {expected_ee_dim})."
            )
    else:
        target_ee_poses = None
    if joint_columns_available:
        expected_joint_dim = 7 * len(trajectory_config["arms"])
        if trajectory_config["end_effector"] == "gripper":
            expected_joint_dim += len(trajectory_config["arms"])
        elif trajectory_config["end_effector"] == "hand":
            expected_joint_dim += 20 * len(trajectory_config["arms"])
        joint_states = np.asarray([row[7] for row in rows], dtype=float)
        target_joints = np.asarray([row[8] for row in rows], dtype=float)
        if (
            joint_states.shape != (len(rows), expected_joint_dim)
            or target_joints.shape != (len(rows), expected_joint_dim)
            or not np.all(np.isfinite(joint_states))
            or not np.all(np.isfinite(target_joints))
        ):
            raise ValueError("Replay requires finite joint state/target fields with the expected layout.")
    else:
        joint_states = None
        target_joints = None

    return EpisodeData(
        states=states,
        actions=actions,
        frame_indices=np.asarray([row[0] for row in rows], dtype=int),
        timestamps=np.asarray([row[1] for row in rows], dtype=float),
        fps=float(info["fps"]),
        action_config=action_config,
        ee_poses=ee_poses,
        delta_ee_poses=delta_ee_poses,
        target_ee_poses=target_ee_poses,
        joint_states=joint_states,
        target_joints=target_joints,
        trajectory_config=trajectory_config,
    )


def resolve_replay_mode(data: EpisodeData, requested_mode: str | None) -> str:
    dataset_mode = str(data.trajectory_config.get("state_action_mode", "joint")).strip().lower()
    default_mode = "ee" if dataset_mode == "end_effector" else "joint"
    replay_mode = default_mode if requested_mode is None else requested_mode
    if replay_mode == "joint" and data.joint_states is None:
        raise ValueError(
            "--replay-mode joint requires observation.joint_state and action.target_joint fields."
        )
    if replay_mode == "ee" and (
        data.ee_poses is None
        or data.delta_ee_poses is None
        or data.target_ee_poses is None
        or data.joint_states is None
        or data.target_joints is None
    ):
        raise ValueError(
            "--replay-mode ee requires observation.ee_pose, action.delta_ee_pose, "
            "action.target_ee_pose, observation.joint_state, and action.target_joint fields."
        )
    return replay_mode


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
        ee_poses=None if data.ee_poses is None else data.ee_poses[indices],
        delta_ee_poses=None if data.delta_ee_poses is None else data.delta_ee_poses[indices],
        target_ee_poses=None if data.target_ee_poses is None else data.target_ee_poses[indices],
        joint_states=None if data.joint_states is None else data.joint_states[indices],
        target_joints=None if data.target_joints is None else data.target_joints[indices],
        trajectory_config=data.trajectory_config,
        replay_mode=data.replay_mode,
    )


def split_targets(data: EpisodeData, *, source_kind: str = "action") -> dict[str, np.ndarray]:
    if source_kind not in {"state", "action"}:
        raise ValueError(f"Unsupported target source {source_kind!r}; expected state or action.")
    source = data.states if source_kind == "state" else data.actions
    ee_source = data.ee_poses if source_kind == "state" else data.delta_ee_poses
    joint_source = data.joint_states if source_kind == "state" else data.target_joints
    arm_mode = str(data.trajectory_config["arm_mode"])
    end_effector = str(data.trajectory_config["end_effector"])
    dataset_mode = str(data.trajectory_config.get("state_action_mode", "joint"))
    if dataset_mode == "end_effector":
        stored_arm_size = EE_STATE_DIM if source_kind == "state" else EE_ACTION_DIM
    else:
        stored_arm_size = 7
    replay_mode = data.replay_mode or ("ee" if dataset_mode == "end_effector" else "joint")
    result: dict[str, np.ndarray] = {}
    offset = 0
    joint_offset = 0
    ee_offset = 0
    arms = ["left", "right"] if arm_mode == "duo" else [arm_mode]
    for side in arms:
        stored_arm_values = source[:, offset : offset + stored_arm_size]
        if replay_mode == "joint" and joint_source is not None:
            stored_arm_values = np.asarray(joint_source[:, joint_offset : joint_offset + 7], dtype=float)
        if replay_mode == "ee":
            if ee_source is None:
                raise ValueError(
                    f"EE replay requires the dedicated {source_kind} EE pose field."
                )
            ee_arm_size = EE_STATE_DIM if source_kind == "state" else EE_ACTION_DIM
            arm_values = np.asarray(
                ee_source[:, ee_offset : ee_offset + ee_arm_size], dtype=float
            )
            ee_offset += ee_arm_size
        else:
            arm_values = stored_arm_values
        result[f"{side}_arm"] = arm_values
        if replay_mode == "ee":
            result[f"{side}_delta_ee_pose"] = arm_values
        offset += stored_arm_size
        if end_effector == "gripper":
            result[f"{side}_gripper_raw"] = (
                joint_source[:, joint_offset + 7]
                if joint_source is not None
                else source[:, offset]
            )
            offset += 1
            joint_offset += 8
        elif end_effector == "hand":
            result[f"{side}_hand"] = (
                joint_source[:, joint_offset + 7 : joint_offset + 27]
                if joint_source is not None
                else source[:, offset : offset + 20]
            )
            offset += 20
            joint_offset += 27
        else:
            joint_offset += 7
    if offset != source.shape[1]:
        raise ValueError(
            f"Trajectory layout {end_effector}/{arm_mode} consumes {offset} values, "
            f"but actions have {source.shape[1]}."
        )
    return result


def continuous_gripper_targets(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Gripper targets must contain only finite values.")
    return np.clip(values, 0.0, 1.0)


def print_dry_run_summary(
    data: EpisodeData,
    fps: float,
    no_gripper: bool,
    initial_state_timeout: float,
    initial_state_max_velocity: float,
    initial_state_max_acceleration: float,
    initial_state_position_tolerance: float,
) -> None:
    targets = split_targets(data)
    print("Controller-matched replay dry run")
    print("----------------------------------")
    print(f"frames: {len(data.frame_indices)}")
    print(f"frame range: {int(data.frame_indices[0])}..{int(data.frame_indices[-1])}")
    print(f"dataset fps: {data.fps:g}")
    print(f"replay fps: {fps:g}")
    print(
        "trajectory setting: "
        f"{data.trajectory_config['end_effector']} / {data.trajectory_config['arm_mode']}"
    )
    replay_mode = data.replay_mode or resolve_replay_mode(data, None)
    state_action_mode = "end_effector" if replay_mode == "ee" else "joint"
    initial_source = "observation.ee_pose" if replay_mode == "ee" else "observation.state"
    print(f"initial state source: {initial_source} at frame {int(data.frame_indices[0])}")
    state_data = EpisodeData(
        states=data.states, actions=data.states, frame_indices=data.frame_indices,
        timestamps=data.timestamps, fps=data.fps, action_config=data.action_config,
        trajectory_config=data.trajectory_config,
        ee_poses=data.ee_poses, delta_ee_poses=data.delta_ee_poses,
        target_ee_poses=data.target_ee_poses,
        joint_states=data.joint_states, target_joints=data.target_joints,
    )
    state_data.replay_mode = replay_mode
    initial_states = split_targets(state_data, source_kind="state")
    for side in data.trajectory_config["arms"]:
        label = "initial EE pose" if replay_mode == "ee" else "arm initial q"
        print(f"{side} {label}: {initial_states[f'{side}_arm'][0].round(6).tolist()}")
    print(f"initial move timeout: {initial_state_timeout:g} s")
    print(f"initial move max joint velocity: {initial_state_max_velocity:g} rad/s")
    print(f"initial move max joint acceleration: {initial_state_max_acceleration:g} rad/s^2")
    print(f"initial-state position tolerance: {initial_state_position_tolerance:g} rad")
    print("target source: action")
    print(f"arm action config: {data.action_config.get('arm_action_representation')} / {data.action_config.get('arm_action_definition')}")
    for arm_name in (f"{side}_arm" for side in data.trajectory_config["arms"]):
        values = np.asarray(targets[arm_name], dtype=float)
        label = "delta EE pose" if replay_mode == "ee" else "target"
        print(f"{arm_name} {label} min: {np.min(values, axis=0).round(6).tolist()}")
        print(f"{arm_name} {label} max: {np.max(values, axis=0).round(6).tolist()}")
    if data.trajectory_config["end_effector"] == "gripper" and not no_gripper:
        for side in data.trajectory_config["arms"]:
            values = continuous_gripper_targets(targets[f"{side}_gripper_raw"])
            print(f"{side} gripper target counts: {value_counts(values)}")
    if data.trajectory_config["end_effector"] == "hand":
        for side in data.trajectory_config["arms"]:
            values = targets[f"{side}_hand"]
            print(f"{side} hand target min: {np.min(values, axis=0).round(6).tolist()}")
            print(f"{side} hand target max: {np.max(values, axis=0).round(6).tolist()}")


def value_counts(values: np.ndarray) -> dict[str, int]:
    return {str(float(value)): int(np.sum(values == value)) for value in sorted(set(values.tolist()))}


def flatten_joint(prefix: str, values: np.ndarray | None) -> dict[str, float | str]:
    row: dict[str, float | str] = {}
    for index in range(7):
        row[f"{prefix}_{index + 1}"] = "" if values is None else float(values[index])
    return row


def trace_fieldnames() -> list[str]:
    fields = ["time_s", "frame_index", "dataset_timestamp", "mode", "target_source"]
    for prefix in (
        "left_recorded_state_q",
        "right_recorded_state_q",
        "left_recorded_target_q",
        "right_recorded_target_q",
        "left_target_q",
        "right_target_q",
        "left_actual_q",
        "right_actual_q",
        "left_error_q",
        "right_error_q",
        "left_target_vs_recorded_target_q",
        "right_target_vs_recorded_target_q",
        "left_actual_vs_recorded_state_q",
        "right_actual_vs_recorded_state_q",
    ):
        fields.extend(f"{prefix}_{index}" for index in range(1, 8))
    fields.extend(
        [
            "left_target_vs_recorded_target_max_abs_rad",
            "right_target_vs_recorded_target_max_abs_rad",
            "left_actual_vs_recorded_state_max_abs_rad",
            "right_actual_vs_recorded_state_max_abs_rad",
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


def import_ros_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float32
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ROS 2 Python dependencies are required for LeRobot episode replay. "
            "Run this script in the robot_control Docker/devcontainer environment."
        ) from exc
    return rclpy, ExternalShutdownException, Node, JointState, Float32


def import_franka_robot_state() -> Any:
    try:
        from franka_msgs.msg import FrankaRobotState
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "franka_msgs is required for delta end-effector replay."
        ) from exc
    return FrankaRobotState


def build_replay_node_class(
    Node: Any, JointState: Any, Float32: Any, FrankaRobotState: Any = None
) -> type:
    class LerobotEpisodeReplayNode(Node):  # type: ignore[misc, valid-type]
        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__("lerobot_episode_replay")
            self.left_state_topic = args.left_state_topic
            self.right_state_topic = args.right_state_topic
            self.left_gripper_state_topic = args.left_gripper_state_topic
            self.right_gripper_state_topic = args.right_gripper_state_topic
            self.left_robot_state_topic = getattr(
                args, "left_robot_state_topic", "/left/franka_robot_state_broadcaster/robot_state"
            )
            self.right_robot_state_topic = getattr(
                args, "right_robot_state_topic", "/right/franka_robot_state_broadcaster/robot_state"
            )
            self.left_target_publisher = self.create_publisher(JointState, args.left_target_topic, 10)
            self.right_target_publisher = self.create_publisher(JointState, args.right_target_topic, 10)
            self.left_gripper_publisher = self.create_publisher(Float32, args.left_gripper_topic, 10)
            self.right_gripper_publisher = self.create_publisher(Float32, args.right_gripper_topic, 10)
            self.left_actual_q: np.ndarray | None = None
            self.right_actual_q: np.ndarray | None = None
            self.left_actual_dq: np.ndarray | None = None
            self.right_actual_dq: np.ndarray | None = None
            self.left_gripper_actual: float | None = None
            self.right_gripper_actual: float | None = None
            self.ee_pose_matrices: dict[str, np.ndarray | None] = {"left": None, "right": None}
            self.flange_to_ee: dict[str, np.ndarray | None] = {"left": None, "right": None}
            self.active_arms = list(args.active_arms)
            self.state_action_mode = getattr(args, "state_action_mode", "joint")
            for side in self.active_arms:
                state_topic = args.left_state_topic if side == "left" else args.right_state_topic
                gripper_state_topic = (
                    args.left_gripper_state_topic if side == "left" else args.right_gripper_state_topic
                )
                self.create_subscription(
                    JointState, state_topic,
                    lambda msg, arm=side: self._store_arm_state(arm, msg), 10,
                )
                if args.robot_end_effector == "gripper":
                    self.create_subscription(
                        JointState, gripper_state_topic,
                        lambda msg, arm=side: self._store_gripper_state(arm, msg), 10,
                    )
                if self.state_action_mode == "end_effector":
                    if FrankaRobotState is None:
                        raise RuntimeError("FrankaRobotState is required for delta end-effector replay.")
                    robot_state_topic = (
                        self.left_robot_state_topic if side == "left" else self.right_robot_state_topic
                    )
                    self.create_subscription(
                        FrankaRobotState,
                        robot_state_topic,
                        lambda msg, arm=side: self._store_robot_state(arm, msg),
                        10,
                    )

        def _ordered_arm_values(self, msg: Any, field_name: str) -> np.ndarray | None:
            raw_values = getattr(msg, field_name, [])
            if len(raw_values) < 7:
                return None
            if len(msg.name) >= 7:
                values: list[float | None] = [None] * 7
                for name, value in zip(msg.name, raw_values, strict=False):
                    for joint_index in range(1, 8):
                        if name.endswith(f"joint{joint_index}"):
                            values[joint_index - 1] = float(value)
                if all(value is not None for value in values):
                    return np.asarray([float(value) for value in values], dtype=float)
            return np.asarray(raw_values[:7], dtype=float)

        def _store_arm_state(self, arm_name: str, msg: Any) -> None:
            q = self._ordered_arm_values(msg, "position")
            if q is None:
                return
            dq = self._ordered_arm_values(msg, "velocity")
            if arm_name == "left":
                self.left_actual_q = q
                self.left_actual_dq = dq
            else:
                self.right_actual_q = q
                self.right_actual_dq = dq

        def _store_gripper_state(self, arm_name: str, msg: Any) -> None:
            if not msg.position:
                return
            width = float(sum(msg.position))
            if arm_name == "left":
                self.left_gripper_actual = width
            else:
                self.right_gripper_actual = width

        def _store_robot_state(self, arm_name: str, msg: Any) -> None:
            from data_collection.move_to_target_ee import pose_message_to_matrix

            self.ee_pose_matrices[arm_name] = pose_message_to_matrix(msg.o_t_ee.pose)
            self.flange_to_ee[arm_name] = pose_message_to_matrix(msg.f_t_ee.pose)

        def publish_targets(
            self,
            left_target: np.ndarray | None,
            right_target: np.ndarray | None,
            left_gripper: float | None,
            right_gripper: float | None,
        ) -> None:
            now_msg = self.get_clock().now().to_msg()
            if left_target is not None:
                left_msg = JointState()
                left_msg.header.stamp = now_msg
                left_msg.name = JOINT_NAMES
                left_msg.position = [float(value) for value in left_target]
                self.left_target_publisher.publish(left_msg)
            if right_target is not None:
                right_msg = JointState()
                right_msg.header.stamp = now_msg
                right_msg.name = JOINT_NAMES
                right_msg.position = [float(value) for value in right_target]
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
            return not self.missing_state_topics(no_gripper)

        def missing_state_topics(self, no_gripper: bool) -> list[str]:
            missing: list[str] = []
            if "left" in self.active_arms and self.left_actual_q is None:
                missing.append(self.left_state_topic)
            if "right" in self.active_arms and self.right_actual_q is None:
                missing.append(self.right_state_topic)
            if not no_gripper:
                if "left" in self.active_arms and self.left_gripper_actual is None:
                    missing.append(self.left_gripper_state_topic)
                if "right" in self.active_arms and self.right_gripper_actual is None:
                    missing.append(self.right_gripper_state_topic)
            if self.state_action_mode == "end_effector":
                if "left" in self.active_arms and (
                    self.ee_pose_matrices["left"] is None or self.flange_to_ee["left"] is None
                ):
                    missing.append(self.left_robot_state_topic)
                if "right" in self.active_arms and (
                    self.ee_pose_matrices["right"] is None or self.flange_to_ee["right"] is None
                ):
                    missing.append(self.right_robot_state_topic)
            return missing

    return LerobotEpisodeReplayNode


def solve_delta_ee_pose_target(
    current_q: np.ndarray,
    current_pose: np.ndarray,
    delta_ee_pose: np.ndarray,
    flange_to_ee: np.ndarray,
    model: Any,
    frame_id: int,
) -> np.ndarray:
    """Convert a recorded EE delta into the joint target sent to the robot."""
    from data_collection.move_to_target_ee import solve_fr3_ik
    from utils.fr3_kinematics import apply_ee_delta

    delta = np.asarray(delta_ee_pose, dtype=float)
    if delta.shape != (6,) or not np.all(np.isfinite(delta)):
        raise ValueError(f"EE action delta must be a finite six-value pose vector, got {delta.shape}.")
    current_matrix = np.asarray(current_pose, dtype=float)
    if current_matrix.shape != (4, 4) or not np.all(np.isfinite(current_matrix)):
        raise ValueError("Current EE pose must be a finite 4x4 transform.")
    result = solve_fr3_ik(
        np.asarray(current_q, dtype=float),
        apply_ee_delta(current_matrix, delta),
        np.asarray(flange_to_ee, dtype=float),
        model,
        frame_id,
    )
    target_q = np.asarray(result.q, dtype=float)
    if target_q.shape != (7,) or not np.all(np.isfinite(target_q)):
        raise ValueError(f"FR3 IK returned an invalid joint target with shape {target_q.shape}.")
    return target_q


def build_trace_row(
    *,
    elapsed_s: float,
    frame_index: int,
    dataset_timestamp: float,
    mode: str,
    target_source: str,
    left_recorded_state: np.ndarray | None,
    right_recorded_state: np.ndarray | None,
    left_recorded_target: np.ndarray | None,
    right_recorded_target: np.ndarray | None,
    left_target: np.ndarray | None,
    right_target: np.ndarray | None,
    left_actual: np.ndarray | None,
    right_actual: np.ndarray | None,
    left_gripper_target: float | None,
    right_gripper_target: float | None,
    left_gripper_actual: float | None,
    right_gripper_actual: float | None,
    abort_requested: bool,
    controller_ready: bool,
) -> dict[str, Any]:
    def difference(
        minuend: np.ndarray | None,
        subtrahend: np.ndarray | None,
    ) -> np.ndarray | None:
        if minuend is None or subtrahend is None:
            return None
        return np.asarray(minuend, dtype=float) - np.asarray(subtrahend, dtype=float)

    def max_abs(values: np.ndarray | None) -> float | str:
        return "" if values is None else float(np.max(np.abs(values)))

    left_tracking_error = difference(left_target, left_actual)
    right_tracking_error = difference(right_target, right_actual)
    left_target_recorded_error = difference(left_target, left_recorded_target)
    right_target_recorded_error = difference(right_target, right_recorded_target)
    left_actual_recorded_error = difference(left_actual, left_recorded_state)
    right_actual_recorded_error = difference(right_actual, right_recorded_state)
    row: dict[str, Any] = {
        "time_s": elapsed_s,
        "frame_index": frame_index,
        "dataset_timestamp": dataset_timestamp,
        "mode": mode,
        "target_source": target_source,
        "left_target_vs_recorded_target_max_abs_rad": max_abs(left_target_recorded_error),
        "right_target_vs_recorded_target_max_abs_rad": max_abs(right_target_recorded_error),
        "left_actual_vs_recorded_state_max_abs_rad": max_abs(left_actual_recorded_error),
        "right_actual_vs_recorded_state_max_abs_rad": max_abs(right_actual_recorded_error),
        "left_gripper_target": "" if left_gripper_target is None else float(left_gripper_target),
        "right_gripper_target": "" if right_gripper_target is None else float(right_gripper_target),
        "left_gripper_actual": "" if left_gripper_actual is None else float(left_gripper_actual),
        "right_gripper_actual": "" if right_gripper_actual is None else float(right_gripper_actual),
        "abort_requested": bool(abort_requested),
        "controller_ready": bool(controller_ready),
    }
    row.update(flatten_joint("left_recorded_state_q", left_recorded_state))
    row.update(flatten_joint("right_recorded_state_q", right_recorded_state))
    row.update(flatten_joint("left_recorded_target_q", left_recorded_target))
    row.update(flatten_joint("right_recorded_target_q", right_recorded_target))
    row.update(flatten_joint("left_target_q", left_target))
    row.update(flatten_joint("right_target_q", right_target))
    row.update(flatten_joint("left_actual_q", left_actual))
    row.update(flatten_joint("right_actual_q", right_actual))
    row.update(flatten_joint("left_error_q", left_tracking_error))
    row.update(flatten_joint("right_error_q", right_tracking_error))
    row.update(flatten_joint("left_target_vs_recorded_target_q", left_target_recorded_error))
    row.update(flatten_joint("right_target_vs_recorded_target_q", right_target_recorded_error))
    row.update(flatten_joint("left_actual_vs_recorded_state_q", left_actual_recorded_error))
    row.update(flatten_joint("right_actual_vs_recorded_state_q", right_actual_recorded_error))
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


def wait_for_start(
    rclpy: Any,
    node: Any,
    commands: Queue[str],
    no_gripper: bool,
    allow_missing_state: bool,
) -> bool:
    print(
        "Waiting for ROS state samples. Type `s` + Enter to move to the initial state "
        "and replay, or `q` + Enter to abort."
    )
    last_missing: tuple[str, ...] | None = None
    left_hold_q: np.ndarray | None = None
    right_hold_q: np.ndarray | None = None
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)
        active_arms = getattr(node, "active_arms", ["left", "right"])
        have_active_states = all(getattr(node, f"{side}_actual_q") is not None for side in active_arms)
        if left_hold_q is None and right_hold_q is None and have_active_states:
            if "left" in active_arms:
                left_hold_q = np.asarray(node.left_actual_q, dtype=float).copy()
            if "right" in active_arms:
                right_hold_q = np.asarray(node.right_actual_q, dtype=float).copy()
            print("Priming arm controllers with the measured current pose.")
        if have_active_states:
            node.publish_targets(left_hold_q, right_hold_q, None, None)
        missing = tuple(node.missing_state_topics(no_gripper))
        if missing != last_missing:
            if missing:
                print("Waiting for: " + ", ".join(missing))
            else:
                print("All required actual arm/gripper state samples are available.")
            last_missing = missing
        command = consume_commands(commands)
        if command == "q":
            return False
        if command == "s":
            if node.controller_ready(no_gripper) or allow_missing_state:
                if not node.controller_ready(no_gripper):
                    print("Warning: starting before all actual state samples are available.")
                return True
            print("Still waiting for required state samples. Use --allow-missing-state to override.")
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


def request_hand_status(
    status_socket: Any,
    request: dict[str, Any] | None = None,
    timeout_s: float = 1.0,
) -> dict[str, Any]:
    """Request one hand-worker status response over its REQ/REP channel."""
    status_socket.send_pyobj({"kind": "status"} if request is None else request)
    poller = zmq.Poller()
    poller.register(status_socket, zmq.POLLIN)
    if not poller.poll(max(1, int(timeout_s * 1000))):
        raise TimeoutError("Timed out waiting for Wuji hand replay worker status.")
    response = status_socket.recv_pyobj()
    if not isinstance(response, dict) or not response.get("ready", False):
        raise RuntimeError(f"Invalid Wuji hand replay worker status: {response!r}")
    return response


def move_hands_to_initial_state(
    rclpy: Any,
    node: Any,
    commands: Queue[str],
    data: EpisodeData,
    hand_sockets: dict[str, Any],
    hand_status_sockets: dict[str, Any],
    timeout_s: float,
) -> bool:
    """Move every replay hand to its recorded initial state before actions."""
    state_data = EpisodeData(
        states=data.states,
        actions=data.states,
        frame_indices=data.frame_indices,
        timestamps=data.timestamps,
        fps=data.fps,
        action_config=data.action_config,
        trajectory_config=data.trajectory_config,
        ee_poses=data.ee_poses, delta_ee_poses=data.delta_ee_poses,
        target_ee_poses=data.target_ee_poses,
        joint_states=data.joint_states, target_joints=data.target_joints,
    )
    state_data.replay_mode = data.replay_mode
    initial_states = split_targets(state_data, source_kind="state")
    targets = {
        side: np.asarray(initial_states[f"{side}_hand"][0], dtype=float)
        for side in data.trajectory_config["arms"]
    }
    if not all(np.all(np.isfinite(target)) for target in targets.values()):
        raise ValueError("Episode initial hand state must contain only finite values.")

    print(
        f"Moving {', '.join(targets)} hand(s) to observation.state at frame "
        f"{int(data.frame_indices[0])} before replay. Type `q` + Enter to abort."
    )
    latest_status: dict[str, dict[str, Any]] = {}
    for side, target in targets.items():
        status = request_hand_status(
            hand_status_sockets[side],
            {"kind": "initial", "target": target.tolist()},
        )
        if not bool(status.get("initial_received", False)):
            raise RuntimeError(f"{side} hand worker did not acknowledge its initial target.")
        latest_status[side] = status

    deadline = time.perf_counter() + timeout_s
    last_status_time = 0.0
    next_report_time = 0.0
    while rclpy.ok() and time.perf_counter() < deadline:
        if consume_commands(commands) == "q":
            return False
        now = time.perf_counter()
        if now - last_status_time < 0.05:
            rclpy.spin_once(node, timeout_sec=0.01)
            continue
        last_status_time = now
        reached = True
        for side, status_socket in hand_status_sockets.items():
            status = request_hand_status(status_socket)
            latest_status[side] = status
            if not bool(status.get("initial_reached", False)):
                reached = False
        if reached:
            print("All trajectory hands reached the episode initial state. Starting action replay.")
            return True
        if now >= next_report_time:
            for side, status in latest_status.items():
                actual = status.get("actual")
                commanded = status.get("initial_target")
                if actual is None or commanded is None:
                    error_text = "measured state unavailable"
                else:
                    error = float(
                        np.max(
                            np.abs(
                                np.asarray(actual, dtype=float)
                                - np.asarray(commanded, dtype=float)
                            )
                        )
                    )
                    error_text = f"max joint error={error:.4f} rad"
                print(
                    f"Initial-state hand status: {side} target_received="
                    f"{bool(status.get('initial_received', False))}, {error_text}"
                )
            next_report_time = now + 1.0
        rclpy.spin_once(node, timeout_sec=0.01)
    raise TimeoutError(
        f"Hands did not reach the episode initial state within {timeout_s:g}s."
    )


def arm_reached_initial_state(
    actual_q: np.ndarray | None,
    actual_dq: np.ndarray | None,
    target_q: np.ndarray,
    position_tolerance_rad: float = DEFAULT_INITIAL_STATE_POSITION_TOLERANCE_RAD,
) -> bool:
    if actual_q is None:
        return False
    position_error = float(np.max(np.abs(np.asarray(actual_q, dtype=float) - target_q)))
    if position_error > position_tolerance_rad:
        return False
    if actual_dq is None:
        return True
    joint_speed = float(np.max(np.abs(np.asarray(actual_dq, dtype=float))))
    return joint_speed <= INITIAL_STATE_VELOCITY_TOLERANCE_RAD_PER_S


def ramp_initial_state_command(
    commanded_q: np.ndarray,
    commanded_velocity: np.ndarray,
    target_q: np.ndarray,
    dt: float,
    max_velocity: float,
    max_acceleration: float,
) -> tuple[np.ndarray, np.ndarray]:
    dt = min(max(float(dt), 1e-6), 2.0 * INITIAL_STATE_PUBLISH_PERIOD_S)
    position_error = target_q - commanded_q
    target_velocity = np.clip(
        INITIAL_STATE_TRACKING_GAIN_PER_S * position_error,
        -max_velocity,
        max_velocity,
    )
    velocity_delta = np.clip(
        target_velocity - commanded_velocity,
        -max_acceleration * dt,
        max_acceleration * dt,
    )
    next_velocity = np.clip(commanded_velocity + velocity_delta, -max_velocity, max_velocity)
    position_step = next_velocity * dt
    position_step = np.where(
        np.abs(position_step) > np.abs(position_error),
        position_error,
        position_step,
    )
    next_q = commanded_q + position_step
    next_velocity = np.where(position_step == position_error, 0.0, next_velocity)
    return next_q, next_velocity


def move_arms_to_initial_state(
    rclpy: Any,
    node: Any,
    commands: Queue[str],
    data: EpisodeData,
    timeout_s: float,
    max_velocity: float,
    max_acceleration: float,
    position_tolerance_rad: float = DEFAULT_INITIAL_STATE_POSITION_TOLERANCE_RAD,
    prime_duration_s: float = INITIAL_STATE_PRIME_DURATION_S,
) -> bool:
    """Command and verify the first selected observation before replay starts."""
    state_data = EpisodeData(
        states=data.states, actions=data.states, frame_indices=data.frame_indices,
        timestamps=data.timestamps, fps=data.fps, action_config=data.action_config,
        trajectory_config=data.trajectory_config,
        ee_poses=data.ee_poses, delta_ee_poses=data.delta_ee_poses,
        target_ee_poses=data.target_ee_poses,
        joint_states=data.joint_states, target_joints=data.target_joints,
    )
    replay_mode = data.replay_mode or resolve_replay_mode(data, None)
    state_data.replay_mode = replay_mode
    initial_states = split_targets(state_data, source_kind="state")
    active_arms = list(data.trajectory_config["arms"])
    def initial_joint_target(side: str) -> np.ndarray:
        values = np.asarray(initial_states[f"{side}_arm"][0], dtype=float)
        if replay_mode == "ee":
            if data.joint_states is None:
                raise RuntimeError("EE replay requires recorded observation.joint_state values.")
            side_index = active_arms.index(side)
            end_effector = str(data.trajectory_config["end_effector"])
            stride = 7 + (1 if end_effector == "gripper" else 20 if end_effector == "hand" else 0)
            offset = side_index * stride
            values = np.asarray(data.joint_states[0, offset : offset + 7], dtype=float)
        return values

    left_target = None if "left" not in active_arms else initial_joint_target("left")
    right_target = None if "right" not in active_arms else initial_joint_target("right")
    targets_to_check = [target for target in (left_target, right_target) if target is not None]
    if not all(np.all(np.isfinite(target)) for target in targets_to_check):
        raise ValueError("Episode initial arm state must contain only finite values.")
    if consume_commands(commands) == "q":
        return False
    if any(getattr(node, f"{side}_actual_q") is None for side in active_arms):
        raise RuntimeError("Actual state from every trajectory arm is required for the initial move.")

    left_command = None if left_target is None else np.asarray(node.left_actual_q, dtype=float).copy()
    right_command = None if right_target is None else np.asarray(node.right_actual_q, dtype=float).copy()
    left_commanded_velocity = None if left_target is None else np.zeros(7, dtype=float)
    right_commanded_velocity = None if right_target is None else np.zeros(7, dtype=float)
    frame_index = int(data.frame_indices[0])
    start_time = time.perf_counter()
    deadline_s = start_time + timeout_s
    next_status_time = start_time
    stable_samples = 0

    print(
        f"Moving {', '.join(active_arms)} arm(s) to observation.state at frame {frame_index} before replay "
        f"(max velocity={max_velocity:g} rad/s, max acceleration={max_acceleration:g} rad/s^2). "
        f"Position tolerance={position_tolerance_rad:g} rad. "
        "Type `q` + Enter to abort."
    )

    prime_deadline_s = min(start_time + prime_duration_s, deadline_s)
    next_publish_time = start_time
    while rclpy.ok() and time.perf_counter() < prime_deadline_s:
        node.publish_targets(left_command, right_command, None, None)
        next_publish_time += INITIAL_STATE_PUBLISH_PERIOD_S
        if sleep_with_spin_and_abort(rclpy, node, commands, next_publish_time):
            return False

    last_command_time = time.perf_counter()
    next_publish_time = last_command_time
    while rclpy.ok():
        if consume_commands(commands) == "q":
            return False

        now = time.perf_counter()
        dt = now - last_command_time
        if left_target is not None:
            left_command, left_commanded_velocity = ramp_initial_state_command(
                left_command, left_commanded_velocity, left_target, dt, max_velocity, max_acceleration
            )
        if right_target is not None:
            right_command, right_commanded_velocity = ramp_initial_state_command(
                right_command, right_commanded_velocity, right_target, dt, max_velocity, max_acceleration
            )
        node.publish_targets(left_command, right_command, None, None)
        last_command_time = now
        next_publish_time += INITIAL_STATE_PUBLISH_PERIOD_S
        if sleep_with_spin_and_abort(rclpy, node, commands, next_publish_time):
            return False

        left_reached = left_target is None or arm_reached_initial_state(
            node.left_actual_q, node.left_actual_dq, left_target, position_tolerance_rad
        )
        right_reached = right_target is None or arm_reached_initial_state(
            node.right_actual_q, node.right_actual_dq, right_target, position_tolerance_rad
        )
        if left_reached and right_reached:
            stable_samples += 1
            if stable_samples >= INITIAL_STATE_STABLE_SAMPLES:
                print("All trajectory arms reached the episode initial state. Starting action replay.")
                return True
        else:
            stable_samples = 0

        now = time.perf_counter()
        if now >= deadline_s:
            left_error = (
                None
                if left_target is None or node.left_actual_q is None
                else float(np.max(np.abs(left_target - np.asarray(node.left_actual_q, dtype=float))))
            )
            right_error = (
                None
                if right_target is None or node.right_actual_q is None
                else float(np.max(np.abs(right_target - np.asarray(node.right_actual_q, dtype=float))))
            )
            raise TimeoutError(
                f"Arms did not reach the episode initial state within {timeout_s:g}s "
                f"(left max error={left_error}, right max error={right_error})."
            )
        if now >= next_status_time:
            left_error = (
                "unavailable"
                if left_target is None or node.left_actual_q is None
                else f"{np.max(np.abs(left_target - np.asarray(node.left_actual_q, dtype=float))):.4f} rad"
            )
            right_error = (
                "unavailable"
                if right_target is None or node.right_actual_q is None
                else f"{np.max(np.abs(right_target - np.asarray(node.right_actual_q, dtype=float))):.4f} rad"
            )
            print(f"Initial-state max joint error: left={left_error}, right={right_error}")
            next_status_time = now + 1.0

    return False


def verify_ee_replay_targets(node: Any, data: EpisodeData) -> None:
    """Reject EE replay unless recorded joint and Cartesian targets agree."""
    from utils.fr3_kinematics import (
        Fr3ForwardKinematics,
        apply_ee_delta,
        ee_state_to_matrix,
        pose_error,
    )

    if data.target_ee_poses is None or data.target_joints is None:
        raise RuntimeError("EE replay requires recorded target EE poses and joint targets.")
    arms = list(data.trajectory_config["arms"])
    end_effector = str(data.trajectory_config["end_effector"])
    stride = 7 + (1 if end_effector == "gripper" else 20 if end_effector == "hand" else 0)
    kinematics = Fr3ForwardKinematics()
    for arm_index, side in enumerate(arms):
        flange_to_ee = node.flange_to_ee.get(side)
        if flange_to_ee is None:
            raise RuntimeError(f"{side} F_T_EE is unavailable for EE replay validation.")
        joint_offset = arm_index * stride
        state_pose_offset = arm_index * EE_STATE_DIM
        action_pose_offset = arm_index * EE_ACTION_DIM
        max_position_error = 0.0
        max_orientation_error = 0.0
        worst_frame = int(data.frame_indices[0])
        for local_index, frame_index in enumerate(data.frame_indices):
            actual_target = kinematics.end_effector_pose(
                data.target_joints[local_index, joint_offset : joint_offset + 7],
                flange_to_ee,
            )
            recorded_target = ee_state_to_matrix(
                data.target_ee_poses[
                    local_index, state_pose_offset : state_pose_offset + EE_STATE_DIM
                ]
            )
            reconstructed_target = apply_ee_delta(
                ee_state_to_matrix(
                    data.ee_poses[
                        local_index,
                        state_pose_offset : state_pose_offset + EE_STATE_DIM,
                    ]
                ),
                data.delta_ee_poses[
                    local_index,
                    action_pose_offset : action_pose_offset + EE_ACTION_DIM,
                ],
            )
            delta_position_error, delta_orientation_error = pose_error(
                reconstructed_target, recorded_target
            )
            if delta_position_error > 1e-5 or delta_orientation_error > 1e-5:
                raise ValueError(
                    f"{side} EE delta does not reconstruct its stored target at "
                    f"frame {int(frame_index)}: errors are {delta_position_error:.6g} m "
                    f"and {delta_orientation_error:.6g} rad."
                )
            position_error, orientation_error = pose_error(actual_target, recorded_target)
            if position_error + orientation_error > max_position_error + max_orientation_error:
                worst_frame = int(frame_index)
            max_position_error = max(max_position_error, position_error)
            max_orientation_error = max(max_orientation_error, orientation_error)
        print(
            f"{side} EE/joint target consistency: max position error="
            f"{max_position_error:.6f} m, max orientation error="
            f"{max_orientation_error:.6f} rad"
        )
        if max_position_error > 0.01 or max_orientation_error > 0.03:
            raise ValueError(
                f"{side} recorded EE/joint targets are inconsistent near frame {worst_frame}: "
                f"max errors are {max_position_error:.6f} m and "
                f"{max_orientation_error:.6f} rad. No episode actions were published."
            )


def run_replay(args: argparse.Namespace, data: EpisodeData, fps: float) -> None:
    replay_mode = resolve_replay_mode(data, args.replay_mode)
    data.replay_mode = replay_mode
    args.state_action_mode = "end_effector" if replay_mode == "ee" else "joint"
    rclpy, ExternalShutdownException, Node, JointState, Float32 = import_ros_dependencies()
    FrankaRobotState = (
        import_franka_robot_state() if args.state_action_mode == "end_effector" else None
    )
    ReplayNode = build_replay_node_class(Node, JointState, Float32, FrankaRobotState)

    targets = split_targets(data)
    end_effector = str(data.trajectory_config["end_effector"])
    active_arms = list(data.trajectory_config["arms"])
    recorded_state_q: dict[str, np.ndarray] = {}
    recorded_target_q: dict[str, np.ndarray] = {}
    joint_offset = 0
    joint_stride = 7
    if end_effector == "gripper":
        joint_stride += 1
    elif end_effector == "hand":
        joint_stride += 20
    if data.joint_states is not None and data.target_joints is not None:
        for side in active_arms:
            recorded_state_q[side] = np.asarray(
                data.joint_states[:, joint_offset : joint_offset + 7], dtype=float
            )
            recorded_target_q[side] = np.asarray(
                data.target_joints[:, joint_offset : joint_offset + 7], dtype=float
            )
            joint_offset += joint_stride
    left_gripper_targets = (
        None if end_effector != "gripper" or args.no_gripper or "left" not in active_arms
        else continuous_gripper_targets(targets["left_gripper_raw"])
    )
    right_gripper_targets = (
        None if end_effector != "gripper" or args.no_gripper or "right" not in active_arms
        else continuous_gripper_targets(targets["right_gripper_raw"])
    )
    hand_sockets: dict[str, Any] = {}
    hand_status_sockets: dict[str, Any] = {}
    if end_effector == "hand":
        context = zmq.Context()
        for side, port, status_port in (
            ("left", args.left_hand_command_port, args.left_hand_status_port),
            ("right", args.right_hand_command_port, args.right_hand_status_port),
        ):
            if side in active_arms:
                socket = context.socket(zmq.PUSH)
                socket.setsockopt(zmq.SNDHWM, 2)
                socket.connect(f"tcp://127.0.0.1:{port}")
                hand_sockets[side] = socket
                status_socket = context.socket(zmq.REQ)
                status_socket.setsockopt(zmq.RCVTIMEO, 1000)
                status_socket.setsockopt(zmq.SNDTIMEO, 1000)
                status_socket.connect(f"tcp://127.0.0.1:{status_port}")
                hand_status_sockets[side] = status_socket
    args.output.mkdir(parents=True, exist_ok=True)

    write_json(
        args.output / RUN_CONFIG_FILENAME,
        {
            "dataset_root": str(args.dataset_root),
            "episode": args.episode,
            "mode": "action",
            "replay_mode": data.replay_mode,
            "fps": fps,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "max_frames": args.max_frames,
            "initial_state_frame": int(data.frame_indices[0]),
            "initial_state_timeout_s": args.initial_state_timeout,
            "initial_state_max_velocity_rad_per_s": args.initial_state_max_velocity,
            "initial_state_max_acceleration_rad_per_s2": args.initial_state_max_acceleration,
            "initial_state_position_tolerance_rad": args.initial_state_position_tolerance,
            "no_gripper": args.no_gripper,
            "trajectory_config": data.trajectory_config,
            "allow_missing_state": args.allow_missing_state,
            "trace_file": TRACE_FILENAME,
        },
    )

    rclpy.init()
    args.active_arms = active_arms
    args.robot_end_effector = end_effector
    node = ReplayNode(args)
    commands = start_command_listener()
    trace_path = args.output / TRACE_FILENAME
    with trace_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=trace_fieldnames()).writeheader()
    aborted = False
    failure: str | None = None
    published_frames = 0
    last_controller_ready = False
    initial_state_reached = False
    hand_initial_state_reached = end_effector != "hand"
    initial_state_duration_s = 0.0
    start_time = 0.0
    try:
        if not wait_for_start(
            rclpy,
            node,
            commands,
            args.no_gripper,
            args.allow_missing_state,
        ):
            aborted = True
            return

        if args.state_action_mode == "end_effector":
            verify_ee_replay_targets(node, data)

        initial_state_start_time = time.perf_counter()
        if not move_arms_to_initial_state(
            rclpy,
            node,
            commands,
            data,
            args.initial_state_timeout,
            args.initial_state_max_velocity,
            args.initial_state_max_acceleration,
            args.initial_state_position_tolerance,
        ):
            aborted = True
            print("Abort requested while moving to the episode initial state.")
            return
        initial_state_duration_s = time.perf_counter() - initial_state_start_time
        initial_state_reached = True

        if end_effector == "hand":
            if not move_hands_to_initial_state(
                rclpy,
                node,
                commands,
                data,
                hand_sockets,
                hand_status_sockets,
                args.initial_state_timeout,
            ):
                aborted = True
                print("Abort requested while moving hands to the episode initial state.")
                return
            hand_initial_state_reached = True

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
                if args.state_action_mode == "end_effector":
                    left_target = (
                        None if "left" not in recorded_target_q
                        else recorded_target_q["left"][local_idx]
                    )
                    right_target = (
                        None if "right" not in recorded_target_q
                        else recorded_target_q["right"][local_idx]
                    )
                else:
                    left_target = (
                        None if "left" not in active_arms else np.asarray(targets["left_arm"][local_idx], dtype=float)
                    )
                    right_target = (
                        None if "right" not in active_arms else np.asarray(targets["right_arm"][local_idx], dtype=float)
                    )
                left_gripper = None if left_gripper_targets is None else float(left_gripper_targets[local_idx])
                right_gripper = None if right_gripper_targets is None else float(right_gripper_targets[local_idx])
                node.publish_targets(left_target, right_target, left_gripper, right_gripper)
                if end_effector == "hand":
                    for side, socket in hand_sockets.items():
                        socket.send_pyobj(
                            {
                                "kind": "target",
                                "target": np.asarray(
                                    targets[f"{side}_hand"][local_idx], dtype=float
                                ).tolist(),
                            }
                        )
                rclpy.spin_once(node, timeout_sec=0.0)
                last_controller_ready = node.controller_ready(args.no_gripper)

                elapsed_s = time.perf_counter() - start_time
                writer.writerow(
                    build_trace_row(
                        elapsed_s=elapsed_s,
                        frame_index=int(frame_index),
                        dataset_timestamp=float(data.timestamps[local_idx]),
                        mode="action",
                        target_source="action",
                        left_recorded_state=(
                            None if "left" not in recorded_state_q
                            else recorded_state_q["left"][local_idx]
                        ),
                        right_recorded_state=(
                            None if "right" not in recorded_state_q
                            else recorded_state_q["right"][local_idx]
                        ),
                        left_recorded_target=(
                            None if "left" not in recorded_target_q
                            else recorded_target_q["left"][local_idx]
                        ),
                        right_recorded_target=(
                            None if "right" not in recorded_target_q
                            else recorded_target_q["right"][local_idx]
                        ),
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
                f.flush()
                published_frames += 1
                next_publish_time += 1.0 / fps
    except KeyboardInterrupt:
        aborted = True
        print("Abort requested by KeyboardInterrupt.")
    except ExternalShutdownException:
        aborted = True
        print("ROS shut down while replay was running; stopping replay.")
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(
            args.output / SUMMARY_FILENAME,
            {
                "aborted": aborted,
                "failed": failure is not None,
                "failure": failure,
                "completed": bool((not aborted) and published_frames == len(data.frame_indices)),
                "published_frames": published_frames,
                "selected_frames": int(len(data.frame_indices)),
                "initial_state_reached": initial_state_reached,
                "hand_initial_state_reached": hand_initial_state_reached,
                "initial_state_duration_s": initial_state_duration_s,
                "controller_ready_at_last_frame": bool(last_controller_ready),
                "duration_s": float(time.perf_counter() - start_time) if start_time else 0.0,
                "trace_path": str(trace_path),
            },
        )
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for socket in hand_sockets.values():
            socket.close(0)
        for socket in hand_status_sockets.values():
            socket.close(0)
        if end_effector == "hand":
            context.term()


def main() -> None:
    args = parse_args()
    if args.internal_wuji_hand is not None:
        if args.internal_wuji_hand == "left":
            command_port = args.left_hand_command_port
            status_port = args.left_hand_status_port
        else:
            command_port = args.right_hand_command_port
            status_port = args.right_hand_status_port
        run_wuji_hand_process(args.internal_wuji_hand, command_port, status_port, args.hand_ip)
        return
    data = load_episode_data(args.dataset_root, args.episode)
    data.replay_mode = resolve_replay_mode(data, args.replay_mode)
    recorded_end_effector = str(data.trajectory_config["end_effector"])
    recorded_arm_mode = str(data.trajectory_config["arm_mode"])
    if args.robot_end_effector is None:
        args.robot_end_effector = recorded_end_effector
    if args.robot_arm_mode is None:
        args.robot_arm_mode = recorded_arm_mode
    validate_setting(data.trajectory_config, args.robot_end_effector, args.robot_arm_mode)
    if args.robot_end_effector != "gripper" and args.no_gripper:
        raise ValueError("--no-gripper is only valid when replaying a gripper trajectory.")
    args.no_gripper = args.robot_end_effector != "gripper"
    data = select_frame_range(data, args.start_frame, args.end_frame, args.max_frames)
    fps = float(args.fps if args.fps is not None else data.fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Replay FPS must be a finite positive number, got {fps!r}.")
    if not np.isfinite(args.initial_state_timeout) or args.initial_state_timeout <= 0:
        raise ValueError(
            "Initial-state timeout must be a finite positive number, "
            f"got {args.initial_state_timeout!r}."
        )
    if not np.isfinite(args.initial_state_max_velocity) or args.initial_state_max_velocity <= 0:
        raise ValueError(
            "Initial-state maximum velocity must be a finite positive number, "
            f"got {args.initial_state_max_velocity!r}."
        )
    if not np.isfinite(args.initial_state_max_acceleration) or args.initial_state_max_acceleration <= 0:
        raise ValueError(
            "Initial-state maximum acceleration must be a finite positive number, "
            f"got {args.initial_state_max_acceleration!r}."
        )
    if not np.isfinite(args.initial_state_position_tolerance) or args.initial_state_position_tolerance <= 0:
        raise ValueError(
            "Initial-state position tolerance must be a finite positive number, "
            f"got {args.initial_state_position_tolerance!r}."
        )

    if args.dry_run:
        print_dry_run_summary(
            data,
            fps,
            args.no_gripper,
            args.initial_state_timeout,
            args.initial_state_max_velocity,
            args.initial_state_max_acceleration,
            args.initial_state_position_tolerance,
        )
        return

    run_replay(args, data, fps)


if __name__ == "__main__":
    main()
