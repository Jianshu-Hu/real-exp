from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

import reset_pylibfranka as reset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_FILE = REPO_ROOT / "outputs" / "homing" / "gello_initial_state_result.json"
HOME_CONFIRMATION = "HOME_GELLOS_TO_INITIAL_STATE_WITH_EXTERNAL_POWER"


@dataclass(frozen=True)
class SideSpec:
    name: str
    node_name: str
    topic_name: str


SIDES = (
    SideSpec("left", "/left/gello_publisher", "/left/gello/joint_states"),
    SideSpec("right", "/right/gello_publisher", "/right/gello/joint_states"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Home both GELLOs to reset_pylibfranka.INITIAL_STATE through the running "
            "ROS GELLO publishers. The arm controllers can therefore follow the measured "
            "GELLO motion continuously during data-collection setup."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("target", "preview", "home"),
        default="target",
        help=(
            "target: offline INITIAL_STATE only; preview: read ROS state and mapping; "
            "home: actively command the running GELLO publishers."
        ),
    )
    parser.add_argument("--result-file", type=Path, default=DEFAULT_RESULT_FILE)
    parser.add_argument(
        "--reset-frankas-first",
        action="store_true",
        help=(
            "Call reset_pylibfranka before GELLO homing. Only use this when the ROS arm "
            "controllers are stopped. Normally omit it and let them follow GELLO live."
        ),
    )
    parser.add_argument("--ip-left", default="172.16.0.3")
    parser.add_argument("--ip-right", default="172.16.0.2")
    parser.add_argument(
        "--power-mode",
        choices=("usb", "external"),
        default="usb",
        help=(
            "Power budget for GELLO torque. USB is the conservative default and only "
            "permits low-current fine alignment after manual pre-positioning."
        ),
    )
    parser.add_argument(
        "--usb-goal-current-ma",
        type=int,
        default=25,
        help="Per-joint current cap in USB mode (default: 25 mA; maximum: 30 mA).",
    )
    parser.add_argument(
        "--usb-max-joint-move-deg",
        type=float,
        default=2.0,
        help=(
            "USB mode refuses active homing until every joint is manually within this "
            "distance of INITIAL_STATE (default: 2 degrees)."
        ),
    )
    parser.add_argument(
        "--external-goal-current-ma",
        type=int,
        default=500,
        help="Per-joint current cap with verified terminal power (default: 500 mA).",
    )
    parser.add_argument(
        "--max-speed-rad-s",
        type=float,
        default=0.50,
        help="Peak commanded GELLO joint speed (default: 0.50 rad/s).",
    )
    parser.add_argument(
        "--command-hz",
        type=float,
        default=15.0,
        help="ROS parameter command rate during the quintic trajectory (default: 15 Hz).",
    )
    parser.add_argument("--coarse-kp-p", type=int, default=700)
    parser.add_argument("--coarse-kp-d", type=int, default=120)
    parser.add_argument(
        "--fine-entry-deg",
        type=float,
        default=4.0,
        help="Switch to fine gains inside this maximum error (default: 4 degrees).",
    )
    parser.add_argument("--fine-kp-p", type=int, default=350)
    parser.add_argument("--fine-kp-d", type=int, default=100)
    parser.add_argument(
        "--position-tolerance-deg",
        type=float,
        default=0.75,
        help="Required final maximum error (default: 0.75 degrees).",
    )
    parser.add_argument(
        "--velocity-tolerance-deg-s",
        type=float,
        default=1.5,
        help="Required final maximum speed (default: 1.5 degrees/s).",
    )
    parser.add_argument("--stable-time-sec", type=float, default=0.30)
    parser.add_argument("--settle-timeout-sec", type=float, default=8.0)
    parser.add_argument(
        "--max-target-crossings",
        type=int,
        default=2,
        help="Abort instead of hunting if a joint repeatedly crosses its target.",
    )
    parser.add_argument("--max-joint-move-rad", type=float, default=1.30)
    parser.add_argument(
        "--confirm",
        default="",
        help=f"Required for --stage home: {HOME_CONFIRMATION}",
    )
    return parser.parse_args()


def initial_targets() -> dict[str, np.ndarray]:
    return {
        "left": np.asarray(reset.LEFT_ARM_START_Q, dtype=float).copy(),
        "right": np.asarray(reset.RIGHT_ARM_START_Q, dtype=float).copy(),
    }


def print_vector(label: str, values: np.ndarray) -> None:
    with np.printoptions(precision=6, suppress=True, linewidth=180):
        print(f"{label}: {np.asarray(values, dtype=float)}")


def print_targets(targets: dict[str, np.ndarray]) -> None:
    print("Exact data-collection initial-state target")
    print("------------------------------------------")
    print("Source: reset_pylibfranka.INITIAL_STATE")
    print_vector("Left Franka-space target q", targets["left"])
    print_vector("Right Franka-space target q", targets["right"])


def validate_args(args: argparse.Namespace) -> None:
    if args.reset_frankas_first and args.stage != "home":
        raise SystemExit("--reset-frankas-first is only valid with --stage home.")
    if not 10 <= args.usb_goal_current_ma <= 30:
        raise SystemExit("--usb-goal-current-ma must be between 10 and 30.")
    if not 1.0 <= args.usb_max_joint_move_deg <= 5.0:
        raise SystemExit("--usb-max-joint-move-deg must be between 1 and 5.")
    if not 50 <= args.external_goal_current_ma <= 600:
        raise SystemExit("--external-goal-current-ma must be between 50 and 600.")
    if not 0.10 <= args.max_speed_rad_s <= 0.60:
        raise SystemExit("--max-speed-rad-s must be between 0.10 and 0.60.")
    if not 5.0 <= args.command_hz <= 25.0:
        raise SystemExit("--command-hz must be between 5 and 25.")
    if not 50 <= args.fine_kp_p <= args.coarse_kp_p <= 1200:
        raise SystemExit("P gains must satisfy 50 <= fine <= coarse <= 1200.")
    if not 0 <= args.coarse_kp_d <= 1000 or not 0 <= args.fine_kp_d <= 1000:
        raise SystemExit("D gains must be between 0 and 1000.")
    if not 1.0 <= args.fine_entry_deg <= 10.0:
        raise SystemExit("--fine-entry-deg must be between 1 and 10.")
    if not 0.25 <= args.position_tolerance_deg <= 2.0:
        raise SystemExit("--position-tolerance-deg must be between 0.25 and 2.0.")
    if args.position_tolerance_deg >= args.fine_entry_deg:
        raise SystemExit("--position-tolerance-deg must be smaller than --fine-entry-deg.")
    if not 0.5 <= args.velocity_tolerance_deg_s <= 5.0:
        raise SystemExit("--velocity-tolerance-deg-s must be between 0.5 and 5.0.")
    if not 0.15 <= args.stable_time_sec <= 1.0:
        raise SystemExit("--stable-time-sec must be between 0.15 and 1.0.")
    if not 2.0 <= args.settle_timeout_sec <= 15.0:
        raise SystemExit("--settle-timeout-sec must be between 2 and 15.")
    if not 0 <= args.max_target_crossings <= 4:
        raise SystemExit("--max-target-crossings must be between 0 and 4.")
    if not 0.2 <= args.max_joint_move_rad <= 1.5:
        raise SystemExit("--max-joint-move-rad must be between 0.2 and 1.5.")


def load_ros_types() -> dict[str, Any]:
    try:
        import rclpy
        from rcl_interfaces.srv import GetParameters, SetParameters
        from rclpy.node import Node
        from rclpy.parameter import Parameter, parameter_value_to_python
        from sensor_msgs.msg import JointState
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "ROS 2 Python packages are unavailable. Source "
            "~/real-exp/gello_software/ros2/install/setup.bash first."
        ) from exc
    return {
        "rclpy": rclpy,
        "Node": Node,
        "Parameter": Parameter,
        "parameter_value_to_python": parameter_value_to_python,
        "GetParameters": GetParameters,
        "SetParameters": SetParameters,
        "JointState": JointState,
    }


class RosGelloSession:
    PARAMETER_NAMES = (
        "joint_signs",
        "best_offsets",
        "dynamixel_kp_p",
        "dynamixel_goal_current",
        "dynamixel_kp_i",
        "dynamixel_kp_d",
        "dynamixel_torque_enable",
        "dynamixel_goal_position",
    )

    def __init__(self, ros: dict[str, Any]) -> None:
        self.ros = ros
        self.rclpy = ros["rclpy"]
        self.rclpy.init()
        self.node = ros["Node"]("gello_initial_state_homing")
        self.get_clients = {
            side.name: self.node.create_client(
                ros["GetParameters"], f"{side.node_name}/get_parameters"
            )
            for side in SIDES
        }
        self.set_clients = {
            side.name: self.node.create_client(
                ros["SetParameters"], f"{side.node_name}/set_parameters"
            )
            for side in SIDES
        }
        self.latest_q: dict[str, np.ndarray] = {}
        self.latest_time: dict[str, float] = {}
        self.parameters: dict[str, dict[str, Any]] = {}
        self.subscriptions = [
            self.node.create_subscription(
                ros["JointState"],
                side.topic_name,
                self._callback(side.name),
                10,
            )
            for side in SIDES
        ]

    def _callback(self, side_name: str) -> Callable[[Any], None]:
        def receive(message: Any) -> None:
            if len(message.position) < 7:
                return
            self.latest_q[side_name] = np.asarray(message.position[:7], dtype=float)
            self.latest_time[side_name] = time.monotonic()

        return receive

    def close(self) -> None:
        try:
            self.node.destroy_node()
        finally:
            self.rclpy.try_shutdown()

    def spin_until(self, predicate: Callable[[], bool], timeout_sec: float, failure: str) -> None:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if predicate():
                return
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        raise RuntimeError(failure)

    def wait_ready(self) -> None:
        for side in SIDES:
            if not self.get_clients[side.name].wait_for_service(timeout_sec=5.0):
                raise RuntimeError(
                    f"ROS get-parameters service unavailable for {side.node_name}. "
                    "Launch franka_gello_state_publisher with gello_duo.yaml first."
                )
            if not self.set_clients[side.name].wait_for_service(timeout_sec=5.0):
                raise RuntimeError(
                    f"ROS set-parameters service unavailable for {side.node_name}."
                )
        self.spin_until(
            lambda: all(side.name in self.latest_q for side in SIDES),
            5.0,
            "Timed out waiting for both GELLO joint-state topics.",
        )
        for side in SIDES:
            request = self.ros["GetParameters"].Request()
            request.names = list(self.PARAMETER_NAMES)
            future = self.get_clients[side.name].call_async(request)
            self.spin_until(
                future.done,
                5.0,
                f"Timed out reading parameters from {side.node_name}.",
            )
            response = future.result()
            if response is None:
                raise RuntimeError(f"Failed to read parameters from {side.node_name}.")
            values = [
                self.ros["parameter_value_to_python"](value)
                for value in response.values
            ]
            self.parameters[side.name] = dict(zip(self.PARAMETER_NAMES, values, strict=True))

    def snapshot_q(self) -> dict[str, np.ndarray]:
        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        now = time.monotonic()
        for side in SIDES:
            if side.name not in self.latest_q or now - self.latest_time[side.name] > 0.5:
                raise RuntimeError(f"{side.name} GELLO joint state is stale.")
        return {name: values.copy() for name, values in self.latest_q.items()}

    def _wait_futures(self, futures: dict[str, Any], action: str) -> None:
        self.spin_until(
            lambda: all(future.done() for future in futures.values()),
            5.0,
            f"Timed out while {action}.",
        )
        for side_name, future in futures.items():
            response = future.result()
            if response is None:
                raise RuntimeError(f"{action} failed for {side_name}.")
            failed = [result.reason for result in response.results if not result.successful]
            if failed:
                raise RuntimeError(f"{action} failed for {side_name}: {failed}")

    def set_parameters(self, updates: dict[str, dict[str, Any]], action: str) -> None:
        parameter_type = self.ros["Parameter"]
        futures = {}
        for side_name, values in updates.items():
            parameters = [
                parameter_type(name=name, value=value)
                for name, value in values.items()
            ]
            request = self.ros["SetParameters"].Request()
            request.parameters = [
                parameter.to_parameter_msg() for parameter in parameters
            ]
            futures[side_name] = self.set_clients[side_name].call_async(request)
        self._wait_futures(futures, action)

    def set_goal_positions(self, goals: dict[str, np.ndarray]) -> None:
        updates: dict[str, dict[str, Any]] = {}
        for side in SIDES:
            old_goal = list(self.parameters[side.name]["dynamixel_goal_position"])
            if len(old_goal) < 8:
                raise RuntimeError(
                    f"{side.name} dynamixel_goal_position must contain arm(7)+gripper(1)."
                )
            goal = [float(value) for value in goals[side.name]]
            updates[side.name] = {
                "dynamixel_goal_position": goal + [float(old_goal[7])]
            }
            self.parameters[side.name]["dynamixel_goal_position"] = goal + [float(old_goal[7])]
        self.set_parameters(updates, "setting GELLO goal positions")

    def set_gains(self, kp_p: int, kp_d: int) -> None:
        updates: dict[str, dict[str, Any]] = {}
        for side in SIDES:
            old_p = list(self.parameters[side.name]["dynamixel_kp_p"])
            old_i = list(self.parameters[side.name]["dynamixel_kp_i"])
            old_d = list(self.parameters[side.name]["dynamixel_kp_d"])
            if min(len(old_p), len(old_i), len(old_d)) < 8:
                raise RuntimeError(f"{side.name} gain arrays must contain eight entries.")
            updates[side.name] = {
                "dynamixel_kp_p": [int(kp_p)] * 7 + [int(old_p[7])],
                "dynamixel_kp_i": [0] * 7 + [int(old_i[7])],
                "dynamixel_kp_d": [int(kp_d)] * 7 + [int(old_d[7])],
            }
        self.set_parameters(updates, "setting GELLO gains")

    def set_goal_current(self, goal_current_ma: int) -> None:
        updates: dict[str, dict[str, Any]] = {}
        for side in SIDES:
            old_current = list(
                self.parameters[side.name]["dynamixel_goal_current"]
            )
            if len(old_current) < 8:
                raise RuntimeError(
                    f"{side.name} goal-current array must contain eight entries."
                )
            updates[side.name] = {
                "dynamixel_goal_current": [int(goal_current_ma)] * 7
                + [int(old_current[7])]
            }
        self.set_parameters(updates, "setting GELLO current limits")

    def set_arm_torque(self, enabled: bool) -> None:
        value = 1 if enabled else 0
        updates = {
            side.name: {"dynamixel_torque_enable": [value] * 7 + [0]}
            for side in SIDES
        }
        self.set_parameters(
            updates,
            "enabling GELLO torque" if enabled else "disabling GELLO torque",
        )

    def restore_gains_torque_off(self) -> None:
        try:
            self.set_arm_torque(False)
        finally:
            updates: dict[str, dict[str, Any]] = {}
            for side in SIDES:
                saved = self.parameters[side.name]
                updates[side.name] = {
                    "dynamixel_kp_p": [int(value) for value in saved["_original_kp_p"]],
                    "dynamixel_kp_i": [int(value) for value in saved["_original_kp_i"]],
                    "dynamixel_kp_d": [int(value) for value in saved["_original_kp_d"]],
                    "dynamixel_goal_current": [
                        int(value) for value in saved["_original_goal_current"]
                    ],
                }
            self.set_parameters(updates, "restoring GELLO gains")

    def preserve_original_gains(self) -> None:
        for side in SIDES:
            saved = self.parameters[side.name]
            saved["_original_kp_p"] = list(saved["dynamixel_kp_p"])
            saved["_original_kp_i"] = list(saved["dynamixel_kp_i"])
            saved["_original_kp_d"] = list(saved["dynamixel_kp_d"])
            saved["_original_goal_current"] = list(
                saved["dynamixel_goal_current"]
            )


def inverse_map_targets(
    session: RosGelloSession,
    targets: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    raw_targets: dict[str, np.ndarray] = {}
    print()
    print("Live GELLO mapping preview")
    print("--------------------------")
    current = session.snapshot_q()
    for side in SIDES:
        params = session.parameters[side.name]
        signs = np.asarray(params["joint_signs"], dtype=float)
        offsets = np.asarray(params["best_offsets"], dtype=float)
        if signs.shape != (7,) or offsets.shape != (7,) or not np.all(np.isin(signs, [-1, 1])):
            raise RuntimeError(
                f"{side.name} publisher has invalid joint_signs/best_offsets."
            )
        # Publisher read path:
        #   q_franka = (q_gello_raw - offsets) * signs
        # Since each sign is +/-1, the exact inverse is:
        #   q_gello_raw = q_franka * signs + offsets
        raw_targets[side.name] = targets[side.name] * signs + offsets
        raw_current = current[side.name] * signs + offsets
        delta = targets[side.name] - current[side.name]
        print(f"{side.name.capitalize()} via {side.node_name}")
        print_vector("  current published Franka-space q", current[side.name])
        print_vector("  Franka INITIAL_STATE target q", targets[side.name])
        print_vector("  joint_signs", signs)
        print_vector("  best_offsets", offsets)
        print_vector("  inverse-mapped GELLO raw target", raw_targets[side.name])
        print_vector("  GELLO raw target delta", raw_targets[side.name] - raw_current)
        print(f"  max required move: {np.max(np.abs(delta)):.3f} rad")
    return raw_targets


def run_franka_reset_first(
    targets: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    print()
    print("Resetting both Frankas before GELLO homing.")
    print("The ROS arm controllers must be stopped to avoid competing command sources.")
    abort_event = mp.Event()
    processes = [
        mp.Process(
            target=reset.arm_worker,
            args=(args.ip_left, targets["left"], "Left Arm", abort_event),
            name="left_franka_initial_reset",
        ),
        mp.Process(
            target=reset.arm_worker,
            args=(args.ip_right, targets["right"], "Right Arm", abort_event),
            name="right_franka_initial_reset",
        ),
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [process.name for process in processes if process.exitcode not in (0, None)]
    if failed or abort_event.is_set():
        raise RuntimeError(f"Franka reset failed or aborted: {failed}")
    print("Both Frankas reached INITIAL_STATE; grippers were not moved.")


def quintic_smoothstep(progress: float) -> float:
    progress = min(1.0, max(0.0, progress))
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


def execute_home(
    session: RosGelloSession,
    targets: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    start_q = session.snapshot_q()
    largest_move = max(
        float(np.max(np.abs(targets[side.name] - start_q[side.name])))
        for side in SIDES
    )
    if largest_move > args.max_joint_move_rad:
        raise RuntimeError(
            f"Required movement {largest_move:.3f} rad exceeds "
            f"--max-joint-move-rad {args.max_joint_move_rad:.3f}."
        )
    if args.power_mode == "usb":
        usb_move_limit = math.radians(args.usb_max_joint_move_deg)
        if largest_move > usb_move_limit:
            raise RuntimeError(
                f"USB mode refuses {math.degrees(largest_move):.2f} deg of active "
                f"movement. Manually guide every GELLO joint to within "
                f"{args.usb_max_joint_move_deg:.2f} deg of INITIAL_STATE, rerun "
                "--stage preview, and only then use motor-assisted fine alignment."
            )
        goal_current_ma = args.usb_goal_current_ma
        active_max_speed = min(args.max_speed_rad_s, 0.15)
    else:
        goal_current_ma = args.external_goal_current_ma
        active_max_speed = args.max_speed_rad_s

    duration = max(0.8, 1.875 * largest_move / active_max_speed)
    command_samples = max(1, int(math.ceil(duration * args.command_hz)))
    period = 1.0 / args.command_hz
    fine_entry_rad = math.radians(args.fine_entry_deg)
    position_tolerance_rad = math.radians(args.position_tolerance_deg)
    velocity_tolerance_rad_s = math.radians(args.velocity_tolerance_deg_s)
    stable_required = max(1, int(math.ceil(args.stable_time_sec * args.command_hz)))

    print()
    print("LIVE GELLO/FRANKA INITIAL-STATE HOMING")
    print("--------------------------------------")
    print(
        f"power_mode={args.power_mode}, per_joint_current_cap={goal_current_ma}mA"
    )
    print(
        f"duration={duration:.2f}s, peak_goal_speed<={active_max_speed:.2f}rad/s"
    )
    print(
        f"coarse kp/kd={args.coarse_kp_p}/{args.coarse_kp_d}; "
        f"fine kp/kd={args.fine_kp_p}/{args.fine_kp_d}"
    )
    print(
        "The GELLO publishers stay online, so running arm controllers receive the "
        "measured smooth motion continuously."
    )
    print("Press Ctrl-C to request immediate GELLO torque-off.")

    session.preserve_original_gains()
    fine_mode = False
    final_q: dict[str, np.ndarray] = {}
    try:
        session.set_goal_positions(start_q)
        session.set_goal_current(goal_current_ma)
        session.set_gains(args.coarse_kp_p, args.coarse_kp_d)
        session.set_arm_torque(True)

        for step in range(1, command_samples + 1):
            scale = quintic_smoothstep(step / command_samples)
            goals = {
                side.name: start_q[side.name]
                + scale * (targets[side.name] - start_q[side.name])
                for side in SIDES
            }
            session.set_goal_positions(goals)
            loop_start = time.monotonic()
            while time.monotonic() - loop_start < period:
                session.rclpy.spin_once(session.node, timeout_sec=0.01)
            live = session.snapshot_q()
            max_error = max(
                float(np.max(np.abs(targets[side.name] - live[side.name])))
                for side in SIDES
            )
            if not fine_mode and max_error <= fine_entry_rad:
                session.set_gains(args.fine_kp_p, args.fine_kp_d)
                fine_mode = True
                print(
                    f"Entered fine approach at max error "
                    f"{math.degrees(max_error):.2f} deg."
                )

        if not fine_mode:
            session.set_gains(args.fine_kp_p, args.fine_kp_d)
        session.set_goal_positions(targets)

        previous_q = session.snapshot_q()
        previous_time = time.monotonic()
        previous_error = {
            side.name: targets[side.name] - previous_q[side.name]
            for side in SIDES
        }
        crossings = {side.name: np.zeros(7, dtype=int) for side in SIDES}
        stable_samples = 0
        deadline = time.monotonic() + args.settle_timeout_sec

        while time.monotonic() < deadline:
            loop_start = time.monotonic()
            while time.monotonic() - loop_start < period:
                session.rclpy.spin_once(session.node, timeout_sec=0.01)
            now = time.monotonic()
            dt = max(now - previous_time, 1e-6)
            current_q = session.snapshot_q()
            max_error = 0.0
            max_speed = 0.0

            for side in SIDES:
                error = targets[side.name] - current_q[side.name]
                speed = (current_q[side.name] - previous_q[side.name]) / dt
                max_error = max(max_error, float(np.max(np.abs(error))))
                max_speed = max(max_speed, float(np.max(np.abs(speed))))
                current_sign = np.sign(error)
                previous_sign = np.sign(previous_error[side.name])
                crossed = (
                    (current_sign != 0)
                    & (previous_sign != 0)
                    & (current_sign != previous_sign)
                    & (np.abs(error) > position_tolerance_rad)
                    & (np.abs(previous_error[side.name]) > position_tolerance_rad)
                )
                crossings[side.name][crossed] += 1
                if np.max(crossings[side.name]) > args.max_target_crossings:
                    joint = int(np.argmax(crossings[side.name])) + 1
                    raise RuntimeError(
                        f"{side.name} joint {joint} repeatedly crossed INITIAL_STATE; "
                        "torque disabled instead of continuing to oscillate."
                    )
                previous_error[side.name] = error
                previous_q[side.name] = current_q[side.name]

            previous_time = now
            if (
                max_error <= position_tolerance_rad
                and max_speed <= velocity_tolerance_rad_s
            ):
                stable_samples += 1
                if stable_samples >= stable_required:
                    final_q = current_q
                    break
            else:
                stable_samples = 0
        else:
            raise RuntimeError(
                "Fine approach timed out before reaching stable position/velocity tolerance; "
                "torque disabled instead of hunting around INITIAL_STATE."
            )
    finally:
        session.restore_gains_torque_off()

    time.sleep(0.25)
    for _ in range(5):
        session.rclpy.spin_once(session.node, timeout_sec=0.05)
    passive_q = session.snapshot_q()
    passive_limit = 2.0 * position_tolerance_rad
    success = all(
        float(np.max(np.abs(targets[side.name] - passive_q[side.name])))
        <= passive_limit
        for side in SIDES
    )

    print()
    print("Homing result (GELLO torque off)")
    print("--------------------------------")
    for side in SIDES:
        print_vector(f"{side.name.capitalize()} passive q", passive_q[side.name])
        print_vector(
            f"{side.name.capitalize()} error q",
            targets[side.name] - passive_q[side.name],
        )
    print(f"Passive verification success: {success}")
    if not success:
        raise RuntimeError(
            "A GELLO moved outside twice the position tolerance after torque-off."
        )

    return {
        "schema_version": 3,
        "created_wall_time": datetime.now().isoformat(timespec="seconds"),
        "success": True,
        "source": "reset_pylibfranka.INITIAL_STATE",
        "position_tolerance_deg": args.position_tolerance_deg,
        "velocity_tolerance_deg_s": args.velocity_tolerance_deg_s,
        "reset_frankas_first": bool(args.reset_frankas_first),
        "left": {
            "start_q": start_q["left"].tolist(),
            "target_q": targets["left"].tolist(),
            "stable_final_q": final_q["left"].tolist(),
            "passive_final_q": passive_q["left"].tolist(),
        },
        "right": {
            "start_q": start_q["right"].tolist(),
            "target_q": targets["right"].tolist(),
            "stable_final_q": final_q["right"].tolist(),
            "passive_final_q": passive_q["right"].tolist(),
        },
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Result written to: {path}")


def run_ros_stage(
    args: argparse.Namespace,
    targets: dict[str, np.ndarray],
    execute: bool,
) -> None:
    ros = load_ros_types()
    session = RosGelloSession(ros)
    try:
        session.wait_ready()
        inverse_map_targets(session, targets)
        if not execute:
            print()
            print("Preview complete: no parameter was changed and no torque was enabled.")
            print(
                "For live data-collection homing, keep both GELLO publishers and arm "
                "controllers running, verify external 5 V Dynamixel power, then use:"
            )
            print(f"  --stage home --confirm {HOME_CONFIRMATION}")
            return
        if args.confirm != HOME_CONFIRMATION:
            raise SystemExit(
                "Refusing active motion: --stage home requires "
                f"--confirm {HOME_CONFIRMATION}."
            )
        if args.reset_frankas_first:
            run_franka_reset_first(targets, args)
        else:
            print()
            print(
                "Direct Franka reset skipped: the running arm controllers will follow "
                "the GELLO publishers throughout this smooth homing trajectory."
            )
        result = execute_home(session, targets, args)
        write_result(args.result_file, result)
    except KeyboardInterrupt:
        try:
            session.set_arm_torque(False)
        except Exception as exc:
            print(f"WARNING: ROS torque-off request failed: {exc}", file=sys.stderr)
        raise SystemExit("\nInterrupted; GELLO torque-off requested.") from None
    finally:
        session.close()


def main() -> None:
    args = parse_args()
    validate_args(args)
    targets = initial_targets()
    print_targets(targets)
    if args.stage == "target":
        print()
        print("Offline target only: no ROS or robot connection was opened.")
        return
    run_ros_stage(args, targets, execute=args.stage == "home")


if __name__ == "__main__":
    main()
