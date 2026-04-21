from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Any

import numpy as np
import pylibfranka
import pyarrow.parquet as pq

ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
POSITION_TOLERANCE_RAD = 0.01
VELOCITY_TOLERANCE_RAD_PER_S = 0.05
REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S = np.array([0.35, 0.35, 0.35, 0.35, 0.50, 0.50, 0.50], dtype=float)
REPLAY_MAX_JOINT_ACCELERATIONS_RAD_PER_S2 = np.array([0.80, 0.80, 0.80, 0.80, 1.20, 1.20, 1.20], dtype=float)
DEFAULT_REPLAY_VELOCITY_FILTER_TAU_S = 0.005
DEFAULT_REPLAY_POSITION_TRACKING_GAIN_PER_S = 2.0
INITIAL_POSE_TRACKING_GAIN_PER_S = 1.5
HOLD_POSE_TRACKING_GAIN_PER_S = 1.5
REPLAY_START_BLEND_TIME_S = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a local dual-arm LeRobot dataset using a synchronous joint-position controller."
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Directory where the LeRobot dataset is stored (e.g., ./lerobot_data).",
    )
    parser.add_argument("--ip-left", default="172.16.0.3", help="IP address of the Left Franka robot.")
    parser.add_argument("--ip-right", default="172.16.0.2", help="IP address of the Right Franka robot.")
    parser.add_argument("--episode", type=int, default=0, help="Index of the episode to replay.")
    parser.add_argument("--fps", type=int, default=15, help="FPS at which the data was recorded.")
    parser.add_argument(
        "--velocity-filter-tau",
        type=float,
        default=DEFAULT_REPLAY_VELOCITY_FILTER_TAU_S,
        help=(
            "Low-pass filter time constant for replay velocity commands in seconds. "
            "Set to 0.0 to disable the filter."
        ),
    )
    parser.add_argument(
        "--position-tracking-gain",
        type=float,
        default=DEFAULT_REPLAY_POSITION_TRACKING_GAIN_PER_S,
        help=(
            "Position feedback gain in 1/s added on top of delta-velocity replay. "
            "Set to 0.0 for pure open-loop replay."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the current robot qpos, initial episode state, and stored action sequence without moving the robots.",
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


def load_action_config(dataset_root: Path) -> dict[str, Any] | None:
    action_config_path = dataset_root / ACTION_CONFIG_PATH
    if not action_config_path.exists():
        return None
    return json.loads(action_config_path.read_text())


def get_episode_data(dataset_root: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any] | None]:
    """
    Load local parquet files and extract the observation.state and action sequences
    for the requested episode.
    """
    print(f"Loading local dataset from {dataset_root}...")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    info_path = dataset_root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        state_shape = info.get("features", {}).get("observation.state", {}).get("shape")
        action_shape = info.get("features", {}).get("action", {}).get("shape")
        if state_shape:
            print(f"Dataset state shape from metadata: {state_shape}")
        if action_shape:
            print(f"Dataset action shape from metadata: {action_shape}")

    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet data files found under {dataset_root / 'data'}")

    episode_states: list[list[float]] = []
    episode_actions: list[list[float]] = []
    available_episode_indices: set[int] = set()
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=["episode_index", "observation.state", "action"])
        data = table.to_pydict()
        for row_episode_index, row_state, row_action in zip(
            data["episode_index"], data["observation.state"], data["action"], strict=True
        ):
            available_episode_indices.add(int(row_episode_index))
            if row_episode_index != episode_index:
                continue
            episode_states.append(row_state)
            episode_actions.append(row_action)

    if not episode_actions:
        available_sorted = sorted(available_episode_indices)
        raise ValueError(
            f"Episode {episode_index} not found in frame parquet data. "
            f"Available episode indices: {available_sorted}. "
            "This usually means the dataset metadata and frame data are inconsistent."
        )

    states = np.asarray(episode_states, dtype=float)
    actions = np.asarray(episode_actions, dtype=float)
    print(f"Loaded {len(actions)} frames for episode {episode_index}.")

    action_config = load_action_config(dataset_root)
    if action_config is not None:
        print(
            "Dataset action config: "
            f"arm={action_config.get('arm_action_representation', 'unknown')}, "
            f"gripper={action_config.get('gripper_action_representation', 'unknown')}"
        )

    return states, actions, action_config


def require_supported_arm_actions(action_config: dict[str, Any] | None) -> str:
    if action_config is None:
        raise ValueError(
            "Dataset action metadata is missing. Replay requires a dataset "
            "recorded with meta/real_exp_action_config.json."
        )

    arm_action_representation = str(action_config.get("arm_action_representation", "")).strip().lower()
    arm_action_definition = str(action_config.get("arm_action_definition", "")).strip().lower()
    if arm_action_representation == "absolute_joint_position" and arm_action_definition == "q[t+1]-q[t]":
        print(
            "Dataset metadata says arm=absolute_joint_position, but arm_action_definition=q[t+1]-q[t]. "
            "Treating this dataset as delta_joint_position."
        )
        action_config["arm_action_representation"] = "delta_joint_position"
        return "delta_joint_position"
    if arm_action_representation not in {"absolute_joint_position", "delta_joint_position"}:
        raise ValueError(
            f"Unsupported arm action representation '{arm_action_representation}'. "
            "Replay supports absolute_joint_position and legacy delta_joint_position datasets."
        )
    return arm_action_representation


def read_current_joint_positions(robot_ip: str, arm_name: str) -> np.ndarray | None:
    try:
        robot = pylibfranka.Robot(robot_ip)
        state = robot.read_once()
        return np.asarray(state.q, dtype=float)
    except Exception as exc:
        print(f"[{arm_name}] Failed to read current qpos: {exc}")
        return None


def read_current_gripper_width(robot_ip: str, arm_name: str) -> np.ndarray | None:
    try:
        gripper = pylibfranka.Gripper(robot_ip)
        state = gripper.read_once()
        return np.asarray([float(state.width)], dtype=float)
    except Exception as exc:
        print(f"[{arm_name} Gripper] Failed to read current width: {exc}")
        return None


def print_array(name: str, value: np.ndarray) -> None:
    with np.printoptions(precision=6, suppress=True, threshold=np.inf, linewidth=200):
        print(f"{name}:")
        print(value)


def dry_run_summary(
    ip_left: str,
    ip_right: str,
    states: np.ndarray,
    actions: np.ndarray,
) -> None:
    data = split_dual_arm_data(states, actions)

    print("Dry run summary")
    print("----------------")

    left_current_q = read_current_joint_positions(ip_left, "Left Arm")
    right_current_q = read_current_joint_positions(ip_right, "Right Arm")
    left_current_gripper = read_current_gripper_width(ip_left, "Left")
    right_current_gripper = read_current_gripper_width(ip_right, "Right")
    if left_current_q is not None:
        print_array("Left arm current qpos", left_current_q)
    if right_current_q is not None:
        print_array("Right arm current qpos", right_current_q)
    if left_current_gripper is not None:
        print_array("Left gripper current state", left_current_gripper)
    if right_current_gripper is not None:
        print_array("Right gripper current state", right_current_gripper)

    print_array("Left arm target initial state (observation.state[0])", np.asarray(data["left_arm_state"][0], dtype=float))
    print_array("Right arm target initial state (observation.state[0])", np.asarray(data["right_arm_state"][0], dtype=float))

    if data["left_gripper_state"] is not None:
        print_array(
            "Left gripper target initial state (observation.state[0])",
            np.asarray([data["left_gripper_state"][0]], dtype=float),
        )
    if data["right_gripper_state"] is not None:
        print_array(
            "Right gripper target initial state (observation.state[0])",
            np.asarray([data["right_gripper_state"][0]], dtype=float),
        )


def build_gripper_replay_actions(
    gripper_states: np.ndarray | None,
    gripper_actions: np.ndarray | None,
    gripper_action_representation: str,
) -> np.ndarray | None:
    if gripper_actions is None:
        return None

    representation = gripper_action_representation.strip().lower()
    actions = np.asarray(gripper_actions, dtype=float)
    if representation == "absolute_width":
        return actions

    if representation == "binary_open_close":
        if gripper_states is None:
            raise ValueError("Binary gripper replay requires observation.state gripper widths.")
        state_values = np.asarray(gripper_states, dtype=float)
        open_width = float(np.max(state_values))
        closed_width = float(np.min(state_values))
        return np.where(actions >= 0.5, open_width, closed_width)

    raise ValueError(f"Unsupported gripper action representation '{gripper_action_representation}'.")


def abort_playback(abort_event: threading.Event) -> None:
    abort_event.set()


def duration_to_seconds(time_step: object) -> float:
    if hasattr(time_step, "to_sec"):
        return time_step.to_sec()
    if hasattr(time_step, "toSec"):
        return time_step.toSec()
    return float(time_step)


def smoothstep(alpha: float) -> float:
    alpha = min(max(alpha, 0.0), 1.0)
    return alpha * alpha * alpha * (10.0 + alpha * (-15.0 + 6.0 * alpha))


def robot_at_target(state: object, target_q: np.ndarray) -> bool:
    position_error = float(np.max(np.abs(np.array(state.q, dtype=float) - target_q)))
    joint_speed = float(np.max(np.abs(np.array(state.dq, dtype=float))))
    return (
        position_error <= POSITION_TOLERANCE_RAD
        and joint_speed <= VELOCITY_TOLERANCE_RAD_PER_S
    )


def controller_mode() -> object:
    try:
        return pylibfranka.ControllerMode.kJointImpedance
    except AttributeError:
        return pylibfranka.ControllerMode.JointImpedance


def start_sync_joint_velocity_control(robot: pylibfranka.Robot) -> pylibfranka.ActiveControlBase:
    return robot.start_joint_velocity_control(controller_mode())


def recover_robot_if_needed(robot: pylibfranka.Robot, arm_name: str) -> None:
    try:
        robot.automatic_error_recovery()
        print(f"[{arm_name}] Automatic error recovery completed.")
    except Exception as exc:
        message = str(exc)
        if "no error" in message.lower():
            return
        raise RuntimeError(f"Automatic error recovery failed: {exc}") from exc


def warm_up_velocity_controller(control: pylibfranka.ActiveControlBase, cycles: int = 200) -> object:
    state: object | None = None
    for _ in range(cycles):
        state, _ = control.readOnce()
        control.writeOnce(pylibfranka.JointVelocities([0.0] * 7))
    if state is None:
        raise RuntimeError("Failed to read controller state during warmup.")
    return state


def limit_velocity_command(
    current_velocity: np.ndarray,
    target_velocity: np.ndarray,
    dt: float,
) -> np.ndarray:
    clipped_target_velocity = np.clip(
        np.asarray(target_velocity, dtype=float),
        -REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S,
        REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S,
    )
    max_delta_velocity = REPLAY_MAX_JOINT_ACCELERATIONS_RAD_PER_S2 * max(dt, 1e-6)
    velocity_step = np.clip(
        clipped_target_velocity - current_velocity,
        -max_delta_velocity,
        max_delta_velocity,
    )
    return np.clip(
        current_velocity + velocity_step,
        -REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S,
        REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S,
    )


def move_arm_to_initial_pose(
    control: pylibfranka.ActiveControlBase,
    start_q: np.ndarray,
    arm_name: str,
    abort_event: threading.Event,
) -> np.ndarray:
    state, _ = control.readOnce()
    current_q = np.array(state.q, dtype=float)
    max_delta = float(np.max(np.abs(start_q - current_q)))
    commanded_velocity = np.zeros(7, dtype=float)

    if max_delta < 1e-4:
        print(f"[{arm_name}] Already at the initial pose recorded in the episode.")
        return commanded_velocity

    print(
        f"[{arm_name}] Moving to the initial pose recorded in the episode "
        f"(max joint delta={max_delta:.3f} rad)..."
    )

    warm_up_velocity_controller(control)

    while not abort_event.is_set():
        state, time_step = control.readOnce()
        dt = duration_to_seconds(time_step)
        current_q = np.asarray(state.q, dtype=float)
        if robot_at_target(state, start_q):
            return commanded_velocity

        target_velocity = INITIAL_POSE_TRACKING_GAIN_PER_S * (start_q - current_q)
        commanded_velocity = limit_velocity_command(commanded_velocity, target_velocity, dt)
        control.writeOnce(pylibfranka.JointVelocities(commanded_velocity.tolist()))

    raise RuntimeError("Playback aborted while moving to the initial pose.")


def hold_position_until_start(
    control: pylibfranka.ActiveControlBase,
    hold_q: np.ndarray,
    arm_name: str,
    abort_event: threading.Event,
    start_event: threading.Event,
    commanded_velocity: np.ndarray,
) -> np.ndarray:
    print(f"[{arm_name}] Initial pose reached. Waiting for synchronized playback start...")
    while not start_event.is_set():
        if abort_event.is_set():
            raise RuntimeError("Playback aborted while waiting for synchronized start.")
        state, time_step = control.readOnce()
        dt = duration_to_seconds(time_step)
        current_q = np.asarray(state.q, dtype=float)
        target_velocity = HOLD_POSE_TRACKING_GAIN_PER_S * (hold_q - current_q)
        commanded_velocity = limit_velocity_command(commanded_velocity, target_velocity, dt)
        control.writeOnce(pylibfranka.JointVelocities(commanded_velocity.tolist()))
    return commanded_velocity


def ramp_joint_velocity_to_zero(
    control: pylibfranka.ActiveControlBase,
    current_velocity: np.ndarray,
    abort_event: threading.Event,
    settle_time_s: float = 0.25,
) -> None:
    elapsed = 0.0
    start_velocity = current_velocity.copy()
    while elapsed < settle_time_s and not abort_event.is_set():
        _, time_step = control.readOnce()
        dt = duration_to_seconds(time_step)
        elapsed += dt
        alpha = min(elapsed / settle_time_s, 1.0)
        target_velocity = start_velocity * (1.0 - alpha)
        control.writeOnce(pylibfranka.JointVelocities(target_velocity.tolist()))

    zero_velocity = pylibfranka.JointVelocities([0.0] * 7)
    zero_velocity.motion_finished = True
    control.writeOnce(zero_velocity)


def replay_arm_deltas_as_velocities(
    control: pylibfranka.ActiveControlBase,
    arm_states: np.ndarray,
    arm_deltas: np.ndarray,
    fps: int,
    arm_name: str,
    abort_event: threading.Event,
    velocity_filter_tau_s: float,
    position_tracking_gain_per_s: float,
    initial_commanded_velocity: np.ndarray,
) -> dict[str, float]:
    sample_dt = 1.0 / fps
    total_frames = len(arm_deltas)
    velocity_samples = np.asarray(arm_deltas, dtype=float) / sample_dt
    velocity_samples = np.clip(
        velocity_samples,
        -REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S,
        REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S,
    )
    time_elapsed = 0.0
    commanded_velocity = np.asarray(initial_commanded_velocity, dtype=float).copy()
    summed_abs_error = 0.0
    sample_count = 0
    current_frame_idx = 0
    current_position_error = np.zeros(7, dtype=float)

    while not abort_event.is_set():
        try:
            state, time_step = control.readOnce()
            dt = duration_to_seconds(time_step)
            time_elapsed += dt
            current_frame_idx = int(time_elapsed / sample_dt)

            if current_frame_idx >= total_frames:
                ramp_joint_velocity_to_zero(control, commanded_velocity, abort_event)
                print(f"[{arm_name}] Playback finished successfully.")
                return {
                    "sum_abs_error": summed_abs_error,
                    "num_error_samples": float(sample_count),
                }

            phase = (time_elapsed - current_frame_idx * sample_dt) / sample_dt
            v_start = velocity_samples[current_frame_idx]
            v_end = velocity_samples[min(current_frame_idx + 1, total_frames - 1)]
            reference_velocity = v_start * (1.0 - phase) + v_end * phase
            q_start = np.asarray(arm_states[current_frame_idx], dtype=float)
            q_end = np.asarray(arm_states[min(current_frame_idx + 1, total_frames - 1)], dtype=float)
            reference_state = q_start * (1.0 - phase) + q_end * phase
            executed_state = np.asarray(state.q, dtype=float)
            summed_abs_error += float(np.sum(np.abs(executed_state - reference_state)))
            sample_count += 1
            current_position_error = reference_state - executed_state
            replay_velocity = reference_velocity + (position_tracking_gain_per_s * current_position_error)
            hold_velocity = HOLD_POSE_TRACKING_GAIN_PER_S * current_position_error
            start_blend = min(time_elapsed / REPLAY_START_BLEND_TIME_S, 1.0)
            tracking_velocity = hold_velocity * (1.0 - start_blend) + replay_velocity * start_blend

            filter_alpha = dt / max(velocity_filter_tau_s + dt, 1e-6)
            filtered_velocity = commanded_velocity + filter_alpha * (tracking_velocity - commanded_velocity)
            commanded_velocity = limit_velocity_command(commanded_velocity, filtered_velocity, dt)
            control.writeOnce(pylibfranka.JointVelocities(commanded_velocity.tolist()))
        except Exception as exc:
            mean_abs_error = summed_abs_error / max(sample_count, 1)
            max_abs_joint_error = float(np.max(np.abs(current_position_error)))
            raise RuntimeError(
                f"Replay failed at episode frame {current_frame_idx}/{total_frames - 1} "
                f"(t={time_elapsed:.3f}s, control_sample={sample_count}, "
                f"mean_abs_error_so_far={mean_abs_error:.6f} rad, "
                f"max_abs_joint_error={max_abs_joint_error:.6f} rad): {exc}"
            ) from exc

    raise RuntimeError("Playback aborted while replaying arm actions.")


def replay_arm_targets_as_velocities(
    control: pylibfranka.ActiveControlBase,
    arm_targets: np.ndarray,
    fps: int,
    arm_name: str,
    abort_event: threading.Event,
    velocity_filter_tau_s: float,
    position_tracking_gain_per_s: float,
    initial_commanded_velocity: np.ndarray,
) -> dict[str, float]:
    sample_dt = 1.0 / fps
    targets = np.asarray(arm_targets, dtype=float)
    total_frames = len(targets)
    time_elapsed = 0.0
    commanded_velocity = np.asarray(initial_commanded_velocity, dtype=float).copy()
    summed_abs_error = 0.0
    sample_count = 0
    current_frame_idx = 0
    current_position_error = np.zeros(7, dtype=float)

    while not abort_event.is_set():
        try:
            state, time_step = control.readOnce()
            dt = duration_to_seconds(time_step)
            time_elapsed += dt
            current_frame_idx = int(time_elapsed / sample_dt)

            if current_frame_idx >= total_frames:
                ramp_joint_velocity_to_zero(control, commanded_velocity, abort_event)
                print(f"[{arm_name}] Playback finished successfully.")
                return {
                    "sum_abs_error": summed_abs_error,
                    "num_error_samples": float(sample_count),
                }

            phase = (time_elapsed - current_frame_idx * sample_dt) / sample_dt
            q_start = targets[current_frame_idx]
            q_end = targets[min(current_frame_idx + 1, total_frames - 1)]
            reference_target = q_start * (1.0 - phase) + q_end * phase
            executed_state = np.asarray(state.q, dtype=float)
            current_position_error = reference_target - executed_state
            summed_abs_error += float(np.sum(np.abs(current_position_error)))
            sample_count += 1

            tracking_velocity = position_tracking_gain_per_s * current_position_error
            hold_velocity = HOLD_POSE_TRACKING_GAIN_PER_S * current_position_error
            start_blend = min(time_elapsed / REPLAY_START_BLEND_TIME_S, 1.0)
            blended_velocity = hold_velocity * (1.0 - start_blend) + tracking_velocity * start_blend

            filter_alpha = dt / max(velocity_filter_tau_s + dt, 1e-6)
            filtered_velocity = commanded_velocity + filter_alpha * (blended_velocity - commanded_velocity)
            commanded_velocity = limit_velocity_command(commanded_velocity, filtered_velocity, dt)
            control.writeOnce(pylibfranka.JointVelocities(commanded_velocity.tolist()))
        except Exception as exc:
            mean_abs_error = summed_abs_error / max(sample_count, 1)
            max_abs_joint_error = float(np.max(np.abs(current_position_error)))
            raise RuntimeError(
                f"Replay failed at episode frame {current_frame_idx}/{total_frames - 1} "
                f"(t={time_elapsed:.3f}s, control_sample={sample_count}, "
                f"mean_abs_error_so_far={mean_abs_error:.6f} rad, "
                f"max_abs_joint_error={max_abs_joint_error:.6f} rad): {exc}"
            ) from exc

    raise RuntimeError("Playback aborted while replaying arm targets.")


def register_worker_ready(
    worker_name: str,
    ready_count: Any,
    ready_lock: Any,
    worker_count: int,
    ready_event: Any,
) -> None:
    with ready_lock:
        ready_count.value += 1
        is_last = ready_count.value >= worker_count
    if is_last:
        ready_event.set()
    else:
        print(f"[{worker_name}] Waiting for other workers to reach their initial state...")


def arm_worker(
    robot_ip: str,
    start_q: np.ndarray,
    arm_states: np.ndarray,
    arm_actions: np.ndarray,
    arm_action_representation: str,
    fps: int,
    arm_name: str,
    velocity_filter_tau_s: float,
    position_tracking_gain_per_s: float,
    abort_event: Any,
    ready_event: Any,
    start_event: Any,
    ready_count: Any,
    ready_lock: Any,
    worker_count: int,
    result_queue: Any,
) -> None:
    print(f"[{arm_name}] Connecting to Franka at {robot_ip}...")
    try:
        robot = pylibfranka.Robot(robot_ip)
        recover_robot_if_needed(robot, arm_name)
        robot.set_collision_behavior(
            [20.0] * 7, [20.0] * 7, [20.0] * 6, [20.0] * 6
        )
    except Exception as exc:
        print(f"[{arm_name}] Connection failed: {exc}")
        abort_playback(abort_event)
        start_event.set()
        return

    try:
        velocity_control = start_sync_joint_velocity_control(robot)
        commanded_velocity = move_arm_to_initial_pose(velocity_control, start_q, arm_name, abort_event)
        register_worker_ready(arm_name, ready_count, ready_lock, worker_count, ready_event)
        commanded_velocity = hold_position_until_start(
            velocity_control,
            start_q,
            arm_name,
            abort_event,
            start_event,
            commanded_velocity,
        )
        if arm_action_representation == "absolute_joint_position":
            replay_summary = replay_arm_targets_as_velocities(
                velocity_control,
                arm_actions,
                fps,
                arm_name,
                abort_event,
                velocity_filter_tau_s,
                position_tracking_gain_per_s,
                commanded_velocity,
            )
        else:
            replay_summary = replay_arm_deltas_as_velocities(
                velocity_control,
                arm_states,
                arm_actions,
                fps,
                arm_name,
                abort_event,
                velocity_filter_tau_s,
                position_tracking_gain_per_s,
                commanded_velocity,
            )
        result_queue.put(
            {
                "arm_name": arm_name,
                "sum_abs_error": replay_summary["sum_abs_error"],
                "num_error_samples": replay_summary["num_error_samples"],
            }
        )
    except Exception as exc:
        print(f"[{arm_name}] EXCEPTION during control: {exc}")
        abort_playback(abort_event)
        ready_event.set()
        start_event.set()
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def gripper_worker(
    robot_ip: str,
    start_width: float,
    actions: np.ndarray,
    fps: int,
    arm_name: str,
    abort_event: Any,
    ready_event: Any,
    start_event: Any,
    ready_count: Any,
    ready_lock: Any,
    worker_count: int,
) -> None:
    if actions is None:
        return

    time_per_frame = 1.0 / fps
    total_frames = len(actions)
    total_duration = total_frames * time_per_frame

    try:
        gripper = pylibfranka.Gripper(robot_ip)
        current_w = float(start_width)
        print(f"[{arm_name} Gripper] Moving to the initial width recorded in the episode...")
        gripper.move(current_w, 0.1)

        register_worker_ready(f"{arm_name} Gripper", ready_count, ready_lock, worker_count, ready_event)
        print(f"[{arm_name} Gripper] Initial pose reached. Waiting for synchronized playback start...")
        while not start_event.is_set():
            if abort_event.is_set():
                return
            time.sleep(0.01)

        start_time = time.time()
        last_frame_idx = 0
        while not abort_event.is_set():
            elapsed = time.time() - start_time
            if elapsed > total_duration:
                break

            current_frame_idx = int(elapsed / time_per_frame)
            if current_frame_idx > last_frame_idx and current_frame_idx < total_frames:
                target_w = float(actions[current_frame_idx])
                if abs(target_w - current_w) > 0.002:
                    gripper.move(target_w, 0.1)
                    current_w = target_w
                last_frame_idx = current_frame_idx

            time.sleep(0.01)
    except Exception as exc:
        print(f"[{arm_name} Gripper] Error: {exc}")
        abort_playback(abort_event)
        ready_event.set()
        start_event.set()


def wait_for_manual_start(
    commands: Queue[str],
    abort_event: Any,
    ready_event: Any,
    start_event: Any,
) -> None:
    print("Waiting for all arms and grippers to reach the initial state...")
    while not ready_event.is_set():
        if abort_event.is_set():
            return
        time.sleep(0.05)

    if abort_event.is_set():
        return

    print("All workers are holding the initial state.")
    print("Type `s` + Enter to start replay, or `q` + Enter to abort.")

    while not abort_event.is_set():
        try:
            command = commands.get(timeout=0.1)
        except Empty:
            continue

        if command == "s":
            start_event.set()
            return
        if command == "q":
            abort_playback(abort_event)
            start_event.set()
            return
        print("Unknown command. Use: s or q")


def split_dual_arm_data(states: np.ndarray, actions: np.ndarray) -> dict[str, Any]:
    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    if state_dim != action_dim:
        raise ValueError(
            f"State/action dimension mismatch: state_dim={state_dim}, action_dim={action_dim}."
        )

    if action_dim == 16:
        print("Detected 16-dim action/state space: [Left Arm(7), Left Grip(1), Right Arm(7), Right Grip(1)]")
        return {
            "left_arm_state": states[:, 0:7],
            "left_gripper_state": states[:, 7],
            "right_arm_state": states[:, 8:15],
            "right_gripper_state": states[:, 15],
            "left_arm_action": actions[:, 0:7],
            "left_gripper_action": actions[:, 7],
            "right_arm_action": actions[:, 8:15],
            "right_gripper_action": actions[:, 15],
        }

    if action_dim == 14:
        print("Detected 14-dim action/state space: [Left Arm(7), Right Arm(7)] (No Grippers)")
        return {
            "left_arm_state": states[:, 0:7],
            "left_gripper_state": None,
            "right_arm_state": states[:, 7:14],
            "right_gripper_state": None,
            "left_arm_action": actions[:, 0:7],
            "left_gripper_action": None,
            "right_arm_action": actions[:, 7:14],
            "right_gripper_action": None,
        }

    raise ValueError(f"Unsupported state/action dimension: {action_dim}. Expected 14 or 16.")


def replay_dual_trajectory(
    ip_left: str,
    ip_right: str,
    states: np.ndarray,
    actions: np.ndarray,
    fps: int,
    velocity_filter_tau_s: float,
    position_tracking_gain_per_s: float,
    action_config: dict[str, Any] | None,
) -> None:
    data = split_dual_arm_data(states, actions)
    left_start_q = np.asarray(data["left_arm_state"][0], dtype=float)
    right_start_q = np.asarray(data["right_arm_state"][0], dtype=float)
    left_arm_states = np.asarray(data["left_arm_state"], dtype=float)
    right_arm_states = np.asarray(data["right_arm_state"], dtype=float)
    left_arm_actions = np.asarray(data["left_arm_action"], dtype=float)
    right_arm_actions = np.asarray(data["right_arm_action"], dtype=float)
    arm_action_representation = str(
        (action_config or {}).get("arm_action_representation", "absolute_joint_position")
    ).strip().lower()
    gripper_action_representation = str(
        (action_config or {}).get("gripper_action_representation", "absolute_width")
    )
    print(f"Arm action mode: {arm_action_representation}")
    if data["left_gripper_action"] is not None or data["right_gripper_action"] is not None:
        print(f"Gripper action mode: {gripper_action_representation}")

    left_gripper_actions = build_gripper_replay_actions(
        np.asarray(data["left_gripper_state"], dtype=float) if data["left_gripper_state"] is not None else None,
        np.asarray(data["left_gripper_action"], dtype=float) if data["left_gripper_action"] is not None else None,
        gripper_action_representation,
    )
    right_gripper_actions = build_gripper_replay_actions(
        np.asarray(data["right_gripper_state"], dtype=float) if data["right_gripper_state"] is not None else None,
        np.asarray(data["right_gripper_action"], dtype=float) if data["right_gripper_action"] is not None else None,
        gripper_action_representation,
    )
    left_start_width = float(data["left_gripper_state"][0]) if data["left_gripper_state"] is not None else 0.0
    right_start_width = float(data["right_gripper_state"][0]) if data["right_gripper_state"] is not None else 0.0

    abort_event = mp.Event()
    ready_event = mp.Event()
    start_event = mp.Event()
    ready_count = mp.Value("i", 0)
    ready_lock = mp.Lock()
    result_queue = mp.Queue()
    commands = start_command_listener()
    worker_count = 2
    if left_gripper_actions is not None:
        worker_count += 1
    if right_gripper_actions is not None:
        worker_count += 1

    arm_processes = [
        mp.Process(
            target=arm_worker,
            args=(
                ip_left,
                left_start_q,
                left_arm_states,
                left_arm_actions,
                arm_action_representation,
                fps,
                "Left Arm",
                velocity_filter_tau_s,
                position_tracking_gain_per_s,
                abort_event,
                ready_event,
                start_event,
                ready_count,
                ready_lock,
                worker_count,
                result_queue,
            ),
            name="left_arm_replay",
        ),
        mp.Process(
            target=arm_worker,
            args=(
                ip_right,
                right_start_q,
                right_arm_states,
                right_arm_actions,
                arm_action_representation,
                fps,
                "Right Arm",
                velocity_filter_tau_s,
                position_tracking_gain_per_s,
                abort_event,
                ready_event,
                start_event,
                ready_count,
                ready_lock,
                worker_count,
                result_queue,
            ),
            name="right_arm_replay",
        ),
    ]

    gripper_threads: list[threading.Thread] = []
    if left_gripper_actions is not None:
        gripper_threads.append(
            threading.Thread(
                target=gripper_worker,
                args=(
                    ip_left,
                    left_start_width,
                    np.asarray(left_gripper_actions, dtype=float),
                    fps,
                    "Left",
                    abort_event,
                    ready_event,
                    start_event,
                    ready_count,
                    ready_lock,
                    worker_count,
                ),
            )
        )
    if right_gripper_actions is not None:
        gripper_threads.append(
            threading.Thread(
                target=gripper_worker,
                args=(
                    ip_right,
                    right_start_width,
                    np.asarray(right_gripper_actions, dtype=float),
                    fps,
                    "Right",
                    abort_event,
                    ready_event,
                    start_event,
                    ready_count,
                    ready_lock,
                    worker_count,
                ),
            )
        )

    for process in arm_processes:
        process.start()
    for thread in gripper_threads:
        thread.start()

    wait_for_manual_start(commands, abort_event, ready_event, start_event)

    for process in arm_processes:
        process.join()
        if process.exitcode not in (0, None):
            abort_playback(abort_event)
            ready_event.set()
            start_event.set()
    for thread in gripper_threads:
        thread.join()

    replay_error_summaries: list[dict[str, float | str]] = []
    while not result_queue.empty():
        replay_error_summaries.append(result_queue.get())

    for replay_error_summary in sorted(replay_error_summaries, key=lambda item: str(item["arm_name"])):
        sample_count = int(float(replay_error_summary["num_error_samples"]))
        sum_abs_error = float(replay_error_summary["sum_abs_error"])
        mean_abs_error_per_timestep = sum_abs_error / max(sample_count, 1)
        print(
            f"[{replay_error_summary['arm_name']}] Mean absolute state error per timestep: "
            f"{mean_abs_error_per_timestep:.6f} rad "
            f"(sum={sum_abs_error:.6f} rad over {sample_count} control samples)."
        )

    if abort_event.is_set():
        print("Dual-arm replay aborted due to an error.")
    else:
        print("Dual-arm replay completed.")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_dir).expanduser()
    states, actions, action_config = get_episode_data(dataset_root, args.episode)
    require_supported_arm_actions(action_config)
    if args.dry_run:
        dry_run_summary(args.ip_left, args.ip_right, states, actions)
        return
    replay_dual_trajectory(
        args.ip_left,
        args.ip_right,
        states,
        actions,
        args.fps,
        args.velocity_filter_tau,
        args.position_tracking_gain,
        action_config,
    )


if __name__ == "__main__":
    main()
