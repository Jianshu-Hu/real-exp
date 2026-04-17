from __future__ import annotations

import argparse
import multiprocessing as mp
import threading
import time

import numpy as np
import pylibfranka

POSITION_TOLERANCE_RAD = 0.01
VELOCITY_TOLERANCE_RAD_PER_S = 0.05
RESET_MAX_JOINT_VELOCITIES_RAD_PER_S = np.array([0.35, 0.35, 0.35, 0.35, 0.50, 0.50, 0.50], dtype=float)
RESET_MAX_JOINT_ACCELERATIONS_RAD_PER_S2 = np.array([0.80, 0.80, 0.80, 0.80, 1.20, 1.20, 1.20], dtype=float)
INITIAL_POSE_TRACKING_GAIN_PER_S = 1.5

# Episode 0 initial state from data/pick_and_place_test:
# [Left Arm(7), Left Grip(1), Right Arm(7), Right Grip(1)]
INITIAL_STATE = np.array(
    [
        0.04882633313536644,
        0.5285813212394714,
        -0.5865817666053772,
        -1.853965163230896,
        0.9884126782417297,
        1.7107422351837158,
        -1.065798044204712,
        0.08010675758123398,
        -0.07229013741016388,
        0.5121952891349792,
        0.6051291227340698,
        -1.866397500038147,
        -0.9738773703575134,
        1.726406455039978,
        1.002806544303894,
        0.08030491322278976,
    ],
    dtype=float,
)

LEFT_ARM_START_Q = INITIAL_STATE[0:7]
LEFT_GRIPPER_START_WIDTH = float(INITIAL_STATE[7])
RIGHT_ARM_START_Q = INITIAL_STATE[8:15]
RIGHT_GRIPPER_START_WIDTH = float(INITIAL_STATE[15])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset both Franka arms to the hardcoded initial state from pick_and_place_test episode 0."
    )
    parser.add_argument("--ip-left", default="172.16.0.3", help="IP address of the Left Franka robot.")
    parser.add_argument("--ip-right", default="172.16.0.2", help="IP address of the Right Franka robot.")
    parser.add_argument(
        "--skip-grippers",
        action="store_true",
        help="Move only the arms and leave grippers unchanged.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the hardcoded initial targets without moving the robots.",
    )
    return parser.parse_args()


def print_array(name: str, value: np.ndarray) -> None:
    with np.printoptions(precision=6, suppress=True, threshold=np.inf, linewidth=200):
        print(f"{name}:")
        print(value)


def robot_at_target(state: object, target_q: np.ndarray) -> bool:
    position_error = float(np.max(np.abs(np.array(state.q, dtype=float) - target_q)))
    joint_speed = float(np.max(np.abs(np.array(state.dq, dtype=float))))
    return position_error <= POSITION_TOLERANCE_RAD and joint_speed <= VELOCITY_TOLERANCE_RAD_PER_S


def duration_to_seconds(time_step: object) -> float:
    if hasattr(time_step, "to_sec"):
        return time_step.to_sec()
    if hasattr(time_step, "toSec"):
        return time_step.toSec()
    return float(time_step)


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
        if "no error" not in str(exc).lower():
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
        -RESET_MAX_JOINT_VELOCITIES_RAD_PER_S,
        RESET_MAX_JOINT_VELOCITIES_RAD_PER_S,
    )
    max_delta_velocity = RESET_MAX_JOINT_ACCELERATIONS_RAD_PER_S2 * max(dt, 1e-6)
    velocity_step = np.clip(
        clipped_target_velocity - current_velocity,
        -max_delta_velocity,
        max_delta_velocity,
    )
    return np.clip(
        current_velocity + velocity_step,
        -RESET_MAX_JOINT_VELOCITIES_RAD_PER_S,
        RESET_MAX_JOINT_VELOCITIES_RAD_PER_S,
    )


def move_arm_to_initial_pose(
    control: pylibfranka.ActiveControlBase,
    start_q: np.ndarray,
    arm_name: str,
    abort_event: threading.Event | mp.synchronize.Event,
) -> None:
    state, _ = control.readOnce()
    current_q = np.array(state.q, dtype=float)
    max_delta = float(np.max(np.abs(start_q - current_q)))
    commanded_velocity = np.zeros(7, dtype=float)

    if max_delta < 1e-4:
        print(f"[{arm_name}] Already at the requested initial pose.")
        return

    print(f"[{arm_name}] Moving to the requested initial pose (max joint delta={max_delta:.3f} rad)...")
    warm_up_velocity_controller(control)

    while not abort_event.is_set():
        state, time_step = control.readOnce()
        dt = duration_to_seconds(time_step)
        current_q = np.asarray(state.q, dtype=float)
        if robot_at_target(state, start_q):
            zero_velocity = pylibfranka.JointVelocities([0.0] * 7)
            zero_velocity.motion_finished = True
            control.writeOnce(zero_velocity)
            print(f"[{arm_name}] Initial pose reached.")
            return

        target_velocity = INITIAL_POSE_TRACKING_GAIN_PER_S * (start_q - current_q)
        commanded_velocity = limit_velocity_command(commanded_velocity, target_velocity, dt)
        control.writeOnce(pylibfranka.JointVelocities(commanded_velocity.tolist()))

    raise RuntimeError("Reset aborted while moving to the initial pose.")


def arm_worker(
    robot_ip: str,
    start_q: np.ndarray,
    arm_name: str,
    abort_event: mp.synchronize.Event,
) -> None:
    print(f"[{arm_name}] Connecting to Franka at {robot_ip}...")
    robot: pylibfranka.Robot | None = None
    try:
        robot = pylibfranka.Robot(robot_ip)
        recover_robot_if_needed(robot, arm_name)
        robot.set_collision_behavior([20.0] * 7, [20.0] * 7, [20.0] * 6, [20.0] * 6)
        velocity_control = start_sync_joint_velocity_control(robot)
        move_arm_to_initial_pose(velocity_control, start_q, arm_name, abort_event)
    except Exception as exc:
        print(f"[{arm_name}] Reset failed: {exc}")
        abort_event.set()
    finally:
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass


def move_gripper(robot_ip: str, width: float, arm_name: str, abort_event: mp.synchronize.Event) -> None:
    try:
        gripper = pylibfranka.Gripper(robot_ip)
        print(f"[{arm_name} Gripper] Moving to width {width:.6f} m...")
        gripper.move(width, 0.1)
        print(f"[{arm_name} Gripper] Target width reached.")
    except Exception as exc:
        print(f"[{arm_name} Gripper] Reset failed: {exc}")
        abort_event.set()


def dry_run_summary(ip_left: str, ip_right: str) -> None:
    print("Dry run summary")
    print("----------------")
    print(f"Left arm IP: {ip_left}")
    print(f"Right arm IP: {ip_right}")
    print_array("Left arm target qpos", LEFT_ARM_START_Q)
    print_array("Right arm target qpos", RIGHT_ARM_START_Q)
    print_array("Left gripper target width", np.asarray([LEFT_GRIPPER_START_WIDTH], dtype=float))
    print_array("Right gripper target width", np.asarray([RIGHT_GRIPPER_START_WIDTH], dtype=float))


def main() -> None:
    args = parse_args()

    if args.dry_run:
        dry_run_summary(args.ip_left, args.ip_right)
        return

    abort_event = mp.Event()
    arm_processes = [
        mp.Process(
            target=arm_worker,
            args=(args.ip_left, np.asarray(LEFT_ARM_START_Q, dtype=float), "Left Arm", abort_event),
            name="left_arm_reset",
        ),
        mp.Process(
            target=arm_worker,
            args=(args.ip_right, np.asarray(RIGHT_ARM_START_Q, dtype=float), "Right Arm", abort_event),
            name="right_arm_reset",
        ),
    ]

    for process in arm_processes:
        process.start()

    for process in arm_processes:
        process.join()

    if abort_event.is_set():
        raise SystemExit("Arm reset aborted due to an error.")

    if not args.skip_grippers:
        move_gripper(args.ip_left, LEFT_GRIPPER_START_WIDTH, "Left", abort_event)
        move_gripper(args.ip_right, RIGHT_GRIPPER_START_WIDTH, "Right", abort_event)

    if abort_event.is_set():
        raise SystemExit("Reset completed with errors.")

    print("Dual-arm reset completed.")


if __name__ == "__main__":
    main()
