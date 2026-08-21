#!/usr/bin/env python3
"""Move selected FR3 end effectors and optional grippers/Wuji hands to targets.

Pose values are ``x,y,z,roll,pitch,yaw`` in the Franka base frame, using meters
and radians. The same six-value target is sent to every selected side in duo
mode. End-effector targets are one width or 20 hand joints and are likewise
broadcast to every selected side.

Running this program without ``--dry-run`` commands real hardware.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

DEFAULT_LEFT_ROBOT_IP = "172.16.0.3"
DEFAULT_RIGHT_ROBOT_IP = "172.16.0.2"
MAX_TRANSLATION_SPEED_M_PER_S = 0.08
MAX_ROTATION_SPEED_RAD_PER_S = 0.5
MAX_HAND_JOINT_SPEED_RAD_PER_S = 1.0
MIN_TRAJECTORY_DURATION_S = 1.0
POSE_SETTLE_DURATION_S = 0.30
POSE_SETTLE_TIMEOUT_S = 5.0
POSE_POSITION_TOLERANCE_M = 2e-4
POSE_ORIENTATION_TOLERANCE_RAD = 2e-3
POSE_VELOCITY_TOLERANCE_M_PER_S = 2e-3
POSE_ANGULAR_VELOCITY_TOLERANCE_RAD_PER_S = 5e-3
POSE_VELOCITY_GAIN_PER_S = 1.5
POSE_VELOCITY_RAMP_S = 0.25
GRIPPER_SPEED_M_PER_S = 0.05
HAND_COMMAND_RATE_HZ = 50.0
HAND_ENABLE_TIMEOUT_S = 5.0

# A Cartesian command is checked before opening a hardware connection. These
# are a conservative FR3 end-effector envelope in the robot base frame (the
# exact reachable set is configuration-dependent, so the envelope intentionally
# rejects points near the physical boundary rather than clipping them).
EE_POSITION_LOWER_M = np.asarray([-0.40, -1.00, -0.60], dtype=float)
EE_POSITION_UPPER_M = np.asarray([1.00, 1.00, 1.20], dtype=float)
EE_POSITION_MAX_RADIUS_M = 1.25
MAX_FRANKA_GRIPPER_WIDTH_M = 0.08


@dataclass(frozen=True)
class SideTarget:
    side: str
    pose: np.ndarray
    end_effector_joint: np.ndarray | None


class TargetArgumentParser(argparse.ArgumentParser):
    """Treat numeric vector fragments such as ``-0.1,`` as argument values."""

    def _parse_optional(self, arg_string: str) -> Any:
        # argparse normally interprets a token starting with '-' as an option.
        # That breaks NumPy-style values such as ``-0.154496,`` and
        # ``-0.019371]``. A token that becomes a float after trimming vector
        # punctuation is unambiguously a target-value fragment instead.
        numeric_fragment = arg_string.strip("[] ,")
        try:
            float(numeric_fragment)
        except ValueError:
            return super()._parse_optional(arg_string)
        return None


def parse_target_values(raw_values: Sequence[str], argument_name: str, parser: argparse.ArgumentParser) -> list[float]:
    """Accept vectors separated by commas, spaces, or a mixture of both."""
    raw_text = " ".join(raw_values).strip()
    fields = [field for field in re.split(r"[\s,\[\]]+", raw_text) if field]
    if not fields:
        parser.error(f"{argument_name} requires at least one numeric value")
    try:
        values = [float(field) for field in fields]
    except ValueError as exc:
        parser.error(f"{argument_name} contains an invalid numeric value: {exc}")
    if not all(math.isfinite(item) for item in values):
        parser.error(f"{argument_name} values must all be finite")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = TargetArgumentParser(
        description=(
            "Move selected FR3 end effectors to Cartesian poses and optionally "
            "move their Franka grippers or Wuji Hand 2 joints."
        )
    )
    side_group = parser.add_mutually_exclusive_group(required=True)
    side_group.add_argument("--duo", dest="arm_mode", action="store_const", const="duo")
    side_group.add_argument("--left", dest="arm_mode", action="store_const", const="left")
    side_group.add_argument("--right", dest="arm_mode", action="store_const", const="right")

    end_effector_group = parser.add_mutually_exclusive_group(required=True)
    end_effector_group.add_argument("--arm", dest="end_effector", action="store_const", const="arm")
    end_effector_group.add_argument(
        "--gripper", dest="end_effector", action="store_const", const="gripper"
    )
    end_effector_group.add_argument("--hand", dest="end_effector", action="store_const", const="hand")

    parser.add_argument(
        "--target-ee-pose",
        "--target_ee_pose",
        required=True,
        nargs="+",
        metavar="X,Y,Z,ROLL,PITCH,YAW",
        help=(
            "Target pose in meters/radians. Pass exactly 6 comma- or space-separated "
            "values; in duo mode the target is sent to both sides."
        ),
    )
    parser.add_argument(
        "--target-ee-joint",
        "--target_ee_joint",
        nargs="+",
        default=None,
        metavar="VALUES",
        help=(
            "Target physical gripper width (1 value) or Wuji joint angles "
            "(20 values). Omit with --arm. In duo mode the target is sent to both sides."
        ),
    )
    parser.add_argument("--ip-left", default=DEFAULT_LEFT_ROBOT_IP)
    parser.add_argument("--ip-right", default=DEFAULT_RIGHT_ROBOT_IP)
    parser.add_argument(
        "--left-hand-ip",
        default=os.environ.get("WUJI_LEFT_HAND_IP", ""),
        help="Left Wuji Hand 2 SDK address (IP:port); defaults to WUJI_LEFT_HAND_IP.",
    )
    parser.add_argument(
        "--right-hand-ip",
        default=os.environ.get("WUJI_RIGHT_HAND_IP", ""),
        help="Right Wuji Hand 2 SDK address (IP:port); defaults to WUJI_RIGHT_HAND_IP.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate targets and print current/target hardware state without commanding motion.",
    )
    return parser


def selected_sides(arm_mode: str) -> list[str]:
    return ["left", "right"] if arm_mode == "duo" else [arm_mode]


def resolve_targets(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[SideTarget]:
    pose_values = parse_target_values(args.target_ee_pose, "--target-ee-pose", parser)
    joint_values = (
        None
        if args.target_ee_joint is None
        else parse_target_values(args.target_ee_joint, "--target-ee-joint", parser)
    )
    sides = selected_sides(args.arm_mode)
    expected_pose_values = 6
    if len(pose_values) != expected_pose_values:
        parser.error(
            f"--target-ee-pose requires {expected_pose_values} values with --{args.arm_mode}; "
            f"got {len(pose_values)}"
        )

    joints_per_side = {"arm": 0, "gripper": 1, "hand": 20}[args.end_effector]
    expected_joint_values = joints_per_side
    if joints_per_side == 0:
        if joint_values is not None:
            parser.error("--target-ee-joint must be omitted with --arm")
        resolved_joint_values: list[float] = []
    else:
        if joint_values is None:
            parser.error(f"--target-ee-joint is required with --{args.end_effector}")
        if len(joint_values) != expected_joint_values:
            parser.error(
                f"--target-ee-joint requires {expected_joint_values} values with "
                f"--{args.end_effector} --{args.arm_mode}; got {len(joint_values)}"
            )
        resolved_joint_values = joint_values

    if args.end_effector == "gripper":
        if any(width < 0.0 for width in resolved_joint_values):
            parser.error("gripper target widths must be non-negative")
        if any(width > MAX_FRANKA_GRIPPER_WIDTH_M for width in resolved_joint_values):
            parser.error(
                f"gripper target widths must be at most {MAX_FRANKA_GRIPPER_WIDTH_M:g} m"
            )
    if args.end_effector == "hand" and args.arm_mode == "duo":
        if not args.left_hand_ip or not args.right_hand_ip:
            parser.error(
                "--hand --duo requires both --left-hand-ip and --right-hand-ip "
                "(or WUJI_LEFT_HAND_IP and WUJI_RIGHT_HAND_IP)"
            )
        if args.left_hand_ip == args.right_hand_ip:
            parser.error("left and right Wuji hand addresses must differ")

    targets: list[SideTarget] = []
    pose = np.asarray(pose_values, dtype=float)
    for side in sides:
        validate_pose(pose, side, parser)
        joint = None
        if joints_per_side:
            joint = np.asarray(resolved_joint_values, dtype=float)
        targets.append(SideTarget(side=side, pose=pose.copy(), end_effector_joint=joint))
    return targets


def validate_pose(pose: np.ndarray, side: str, parser: argparse.ArgumentParser) -> None:
    """Reject Cartesian targets outside the pre-connection FR3 safety envelope."""
    position = np.asarray(pose[:3], dtype=float)
    if np.any(position < EE_POSITION_LOWER_M) or np.any(position > EE_POSITION_UPPER_M):
        parser.error(
            f"{side} target position {position.tolist()} is outside the conservative FR3 "
            f"robot-coordinate envelope [{EE_POSITION_LOWER_M.tolist()}, {EE_POSITION_UPPER_M.tolist()}]"
        )
    radius = float(np.linalg.norm(position))
    if radius > EE_POSITION_MAX_RADIUS_M:
        parser.error(
            f"{side} target position has radius {radius:.3f} m, exceeding the conservative "
            f"FR3 hardware limit of {EE_POSITION_MAX_RADIUS_M:.3f} m"
        )
    # The parser already guarantees finite values, but keep this check close to
    # the hardware-envelope check for callers that construct a Namespace.
    if not np.all(np.isfinite(pose)):
        parser.error(f"{side} target pose must contain only finite values")


def rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def pose_vector_to_matrix(pose: Sequence[float]) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rpy_to_rotation(float(pose[3]), float(pose[4]), float(pose[5]))
    transform[:3, 3] = np.asarray(pose[:3], dtype=float)
    return transform


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to the same ZYX roll/pitch/yaw convention."""
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=float)


def matrix_to_pose_vector(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float).reshape((4, 4), order="F")
    return np.concatenate((matrix[:3, 3], rotation_to_rpy(matrix[:3, :3])))


def rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [scale / 4.0, (rotation[2, 1] - rotation[1, 2]) / scale,
             (rotation[0, 2] - rotation[2, 0]) / scale,
             (rotation[1, 0] - rotation[0, 1]) / scale],
            dtype=float,
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [(rotation[2, 1] - rotation[1, 2]) / scale, scale / 4.0,
                 (rotation[0, 1] + rotation[1, 0]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale], dtype=float
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [(rotation[0, 2] - rotation[2, 0]) / scale,
                 (rotation[0, 1] + rotation[1, 0]) / scale, scale / 4.0,
                 (rotation[1, 2] + rotation[2, 1]) / scale], dtype=float
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                [(rotation[1, 0] - rotation[0, 1]) / scale,
                 (rotation[0, 2] + rotation[2, 0]) / scale,
                 (rotation[1, 2] + rotation[2, 1]) / scale, scale / 4.0], dtype=float
            )
    return quaternion / np.linalg.norm(quaternion)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def slerp(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(np.dot(start, target))
    if dot < 0.0:
        target = -target
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = start + fraction * (target - start)
        return result / np.linalg.norm(result)
    angle = math.acos(dot)
    return (
        math.sin((1.0 - fraction) * angle) / math.sin(angle) * start
        + math.sin(fraction * angle) / math.sin(angle) * target
    )


def smooth_step(fraction: float) -> float:
    value = float(np.clip(fraction, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def rotation_distance(start: np.ndarray, target: np.ndarray) -> float:
    cosine = (float(np.trace(start.T @ target)) - 1.0) / 2.0
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle vector for a rotation matrix."""
    angle = rotation_distance(np.eye(3), rotation)
    if angle < 1e-9:
        return np.zeros(3, dtype=float)
    sine = math.sin(angle)
    if abs(sine) > 1e-6:
        axis = np.asarray(
            [rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]],
            dtype=float,
        ) / (2.0 * sine)
    else:
        # Near pi, extract the axis from the largest diagonal element.
        axis_index = int(np.argmax(np.diag(rotation)))
        axis = np.zeros(3, dtype=float)
        axis[axis_index] = math.sqrt(max(0.0, (float(rotation[axis_index, axis_index]) + 1.0) / 2.0))
        other = (axis_index + 1) % 3
        third = (axis_index + 2) % 3
        denominator = max(4.0 * axis[axis_index], 1e-9)
        axis[other] = (rotation[other, axis_index] + rotation[axis_index, other]) / denominator
        axis[third] = (rotation[third, axis_index] + rotation[axis_index, third]) / denominator
    return axis * angle


def cartesian_velocity_toward_pose(current: np.ndarray, target: np.ndarray, ramp: float) -> np.ndarray:
    """Compute a bounded base-frame Cartesian velocity toward ``target``."""
    position_error = target[:3, 3] - current[:3, 3]
    orientation_error = rotation_vector(target[:3, :3] @ current[:3, :3].T)
    velocity = np.concatenate(
        (
            POSE_VELOCITY_GAIN_PER_S * position_error,
            POSE_VELOCITY_GAIN_PER_S * orientation_error,
        )
    )
    velocity[:3] = np.clip(velocity[:3], -MAX_TRANSLATION_SPEED_M_PER_S, MAX_TRANSLATION_SPEED_M_PER_S)
    velocity[3:] = np.clip(velocity[3:], -MAX_ROTATION_SPEED_RAD_PER_S, MAX_ROTATION_SPEED_RAD_PER_S)
    return float(np.clip(ramp, 0.0, 1.0)) * velocity


def trajectory_duration(start: np.ndarray, target: np.ndarray) -> float:
    translation_time = (
        float(np.linalg.norm(target[:3, 3] - start[:3, 3])) / MAX_TRANSLATION_SPEED_M_PER_S
    )
    rotation_time = rotation_distance(start[:3, :3], target[:3, :3]) / MAX_ROTATION_SPEED_RAD_PER_S
    return max(MIN_TRAJECTORY_DURATION_S, translation_time, rotation_time)


def pose_error(start: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return translational and rotational error between homogeneous poses."""
    translation_error = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
    orientation_error = rotation_distance(start[:3, :3], target[:3, :3])
    return translation_error, orientation_error


def state_pose_matrix(state: Any) -> np.ndarray:
    """Return the FR3 commanded Cartesian pose used to seed pose control."""
    # O_T_EE_c is the pose currently commanded by libfranka's generator. The
    # measured O_T_EE can lag it, and starting from that lagging pose creates a
    # velocity/acceleration discontinuity on the next 1 kHz command cycle.
    values = getattr(state, "O_T_EE_c", None)
    if values is None:
        values = state.O_T_EE
    matrix = np.asarray(values, dtype=float).reshape((4, 4), order="F")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("FR3 returned a non-finite Cartesian command pose")
    if not np.allclose(matrix[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise RuntimeError(f"FR3 returned an invalid homogeneous command pose: {matrix.tolist()}")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise RuntimeError(f"FR3 returned an invalid rotation in command pose: {rotation.tolist()}")
    return matrix


def cartesian_state_is_settled(state: Any, target: np.ndarray) -> bool:
    """Check measured pose and Cartesian velocity before setting motion_finished."""
    current = np.asarray(state.O_T_EE, dtype=float).reshape((4, 4), order="F")
    translation_error, orientation_error = pose_error(current, target)
    velocity_values = getattr(state, "O_dP_EE", None)
    if velocity_values is None:
        # pylibfranka exposes commanded/desired Cartesian velocity as O_dP_EE_c
        # and O_dP_EE_d; some test doubles expose the shorter O_dP_EE name.
        velocity_values = getattr(state, "O_dP_EE_c", getattr(state, "O_dP_EE_d", None))
    velocity = np.asarray(np.zeros(6) if velocity_values is None else velocity_values, dtype=float)
    if velocity.shape != (6,) or not np.all(np.isfinite(velocity)):
        return (
            translation_error <= POSE_POSITION_TOLERANCE_M
            and orientation_error <= POSE_ORIENTATION_TOLERANCE_RAD
        )
    return (
        translation_error <= POSE_POSITION_TOLERANCE_M
        and orientation_error <= POSE_ORIENTATION_TOLERANCE_RAD
        and float(np.linalg.norm(velocity[:3])) <= POSE_VELOCITY_TOLERANCE_M_PER_S
        and float(np.linalg.norm(velocity[3:])) <= POSE_ANGULAR_VELOCITY_TOLERANCE_RAD_PER_S
    )


def interpolate_pose(start: np.ndarray, target: np.ndarray, fraction: float) -> np.ndarray:
    blend = smooth_step(fraction)
    result = np.eye(4, dtype=float)
    result[:3, 3] = start[:3, 3] + blend * (target[:3, 3] - start[:3, 3])
    result[:3, :3] = quaternion_to_rotation(
        slerp(rotation_to_quaternion(start[:3, :3]), rotation_to_quaternion(target[:3, :3]), blend)
    )
    return result


def duration_to_seconds(duration: Any) -> float:
    if hasattr(duration, "to_sec"):
        return float(duration.to_sec())
    if hasattr(duration, "toSec"):
        return float(duration.toSec())
    return float(duration)


def cartesian_impedance_mode(pylibfranka: Any) -> Any:
    try:
        return pylibfranka.ControllerMode.kCartesianImpedance
    except AttributeError:
        return pylibfranka.ControllerMode.CartesianImpedance


def joint_impedance_mode(pylibfranka: Any) -> Any:
    try:
        return pylibfranka.ControllerMode.kJointImpedance
    except AttributeError:
        return pylibfranka.ControllerMode.JointImpedance


def recover_robot_if_needed(robot: Any, side: str) -> None:
    try:
        robot.automatic_error_recovery()
        print(f"[{side}] Automatic error recovery completed.", flush=True)
    except Exception as exc:
        if "no error" not in str(exc).lower():
            raise RuntimeError(f"automatic error recovery failed: {exc}") from exc


def move_arm(robot_ip: str, side: str, pose: np.ndarray) -> None:
    import pylibfranka

    print(f"[{side}] Connecting to FR3 at {robot_ip}...", flush=True)
    robot = pylibfranka.Robot(robot_ip)
    try:
        recover_robot_if_needed(robot, side)
        control = robot.start_cartesian_velocity_control(joint_impedance_mode(pylibfranka))
        control.readOnce()
        target = pose_vector_to_matrix(pose)
        print(f"[{side}] Moving end effector with Cartesian velocity control...", flush=True)
        # Active control requires the first read to be paired with a write
        # before requesting the next state sample.
        control.writeOnce(pylibfranka.CartesianVelocities([0.0] * 6))
        elapsed = 0.0
        settle_elapsed = 0.0
        settled_for = 0.0
        while settle_elapsed < POSE_SETTLE_TIMEOUT_S:
            current_state, period = control.readOnce()
            dt = max(duration_to_seconds(period), 1e-4)
            elapsed += dt
            settle_elapsed += dt
            current = np.asarray(current_state.O_T_EE, dtype=float).reshape((4, 4), order="F")
            translation_error, orientation_error = pose_error(current, target)
            if translation_error <= POSE_POSITION_TOLERANCE_M and orientation_error <= POSE_ORIENTATION_TOLERANCE_RAD:
                command_velocity = np.zeros(6, dtype=float)
                settled_for += dt
                if settled_for >= POSE_SETTLE_DURATION_S:
                    break
            else:
                settled_for = 0.0
                ramp = min(1.0, elapsed / POSE_VELOCITY_RAMP_S)
                command_velocity = cartesian_velocity_toward_pose(current, target, ramp)
            control.writeOnce(pylibfranka.CartesianVelocities(command_velocity.tolist()))
        else:
            raise RuntimeError(
                f"[{side}] FR3 did not reach the target pose within "
                f"{POSE_SETTLE_TIMEOUT_S:.1f} s; refusing motion-finished command"
            )

        final_command = pylibfranka.CartesianVelocities([0.0] * 6)
        final_command.motion_finished = True
        control.writeOnce(final_command)
        print(f"[{side}] End-effector target reached.", flush=True)
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def read_current_arm_pose(robot_ip: str, side: str) -> np.ndarray:
    import pylibfranka

    print(f"[{side}] Reading current FR3 end-effector pose from {robot_ip}...", flush=True)
    robot = pylibfranka.Robot(robot_ip)
    try:
        return matrix_to_pose_vector(np.asarray(robot.read_once().O_T_EE, dtype=float))
    finally:
        try:
            robot.stop()
        except Exception:
            pass


def move_gripper(robot_ip: str, side: str, target_width: float) -> None:
    import pylibfranka

    gripper = pylibfranka.Gripper(robot_ip)
    state = gripper.read_once()
    max_width = float(state.max_width)
    if target_width > max_width + 1e-6:
        raise ValueError(
            f"[{side}] target gripper width {target_width:.6f} m exceeds "
            f"the measured maximum {max_width:.6f} m"
        )
    print(f"[{side}] Moving gripper to {target_width:.6f} m...", flush=True)
    if not gripper.move(target_width, GRIPPER_SPEED_M_PER_S):
        raise RuntimeError(f"[{side}] gripper rejected or failed to reach the target")
    print(f"[{side}] Gripper target reached.", flush=True)


def read_current_gripper_width(robot_ip: str, side: str) -> np.ndarray:
    import pylibfranka

    print(f"[{side}] Reading current Franka gripper width from {robot_ip}...", flush=True)
    gripper = pylibfranka.Gripper(robot_ip)
    return np.asarray([float(gripper.read_once().width)], dtype=float)


def joint_state_positions(state: Any) -> np.ndarray | None:
    positions = getattr(state, "position", None)
    if positions is not None:
        values = np.asarray(positions, dtype=float)
    else:
        joints = getattr(state, "joints", None)
        if joints is None:
            return None
        values = np.asarray(
            [joint.position for joint in sorted(joints, key=lambda joint: int(joint.nid))],
            dtype=float,
        )
    return values if values.shape == (20,) and np.all(np.isfinite(values)) else None


def connect_wuji_hand(manager: Any, side: str, address: str) -> Any:
    if address:
        return manager.connect(address=address, device_name="wuji_hand_2")
    devices = [device for device in manager.scan() if str(device.sn).upper().startswith("WH")]
    if not devices:
        raise RuntimeError(f"[{side}] no Wuji Hand 2 found; pass --{side}-hand-ip")
    if len(devices) == 1:
        return manager.connect(address=devices[0].address, device_name="wuji_hand_2")
    for device in devices:
        hand = manager.connect(address=device.address, device_name="wuji_hand_2")
        try:
            if str(hand.handedness().get()).lower() == side:
                return hand
        except Exception:
            pass
        hand.disconnect()
        time.sleep(0.2)
    raise RuntimeError(f"[{side}] no matching Wuji Hand 2 found; pass --{side}-hand-ip")


def wait_for_wuji_enabled(hand: Any, side: str) -> None:
    deadline = time.monotonic() + HAND_ENABLE_TIMEOUT_S
    subscription = hand.joint_diagnostics().subscribe()
    try:
        while time.monotonic() < deadline:
            time.sleep(0.1)
            frame = subscription.recv()
            if frame is None or not frame.joints:
                continue
            live = [joint for joint in frame.joints if joint.vbus_v_fb > 0.5]
            if live and all(joint.status_word.ext_state == 2 for joint in live):
                return
    finally:
        subscription.close()
    raise RuntimeError(f"[{side}] Wuji Hand 2 did not enable within {HAND_ENABLE_TIMEOUT_S:g} s")


def read_wuji_positions(hand: Any, side: str) -> np.ndarray:
    deadline = time.monotonic() + HAND_ENABLE_TIMEOUT_S
    subscription = hand.joint_states().subscribe()
    try:
        while time.monotonic() < deadline:
            time.sleep(0.02)
            state = subscription.recv()
            if state is None:
                continue
            positions = joint_state_positions(state)
            if positions is not None:
                return positions
    finally:
        subscription.close()
    raise RuntimeError(f"[{side}] no valid 20-joint Wuji state received")


def read_current_hand_positions(side: str, address: str) -> np.ndarray:
    import wuji_sdk

    manager = wuji_sdk.SdkManager.instance()
    hand = None
    try:
        print(f"[{side}] Reading current Wuji Hand 2 joint positions...", flush=True)
        hand = connect_wuji_hand(manager, side, address)
        return read_wuji_positions(hand, side)
    finally:
        if hand is not None:
            try:
                hand.disconnect()
            except Exception:
                pass
        try:
            manager.disconnect_all()
        except Exception:
            pass


def move_hand(side: str, address: str, target: np.ndarray) -> None:
    import wuji_sdk

    manager = wuji_sdk.SdkManager.instance()
    hand = None
    publisher = None
    try:
        print(f"[{side}] Connecting to Wuji Hand 2...", flush=True)
        hand = connect_wuji_hand(manager, side, address)
        online_count = int(hand.online_joints_count().get())
        if online_count != 20:
            raise RuntimeError(f"[{side}] expected 20 online hand joints, found {online_count}")
        hand.effort_limit().set(1.5)
        hand.mit_params().set((3.0, 0.1))
        hand.enable()
        wait_for_wuji_enabled(hand, side)
        start = read_wuji_positions(hand, side)
        duration = max(
            MIN_TRAJECTORY_DURATION_S,
            float(np.max(np.abs(target - start))) / MAX_HAND_JOINT_SPEED_RAD_PER_S,
        )
        publisher = hand.joint_command().publish()
        period = 1.0 / HAND_COMMAND_RATE_HZ
        start_time = time.monotonic()
        print(f"[{side}] Moving 20 hand joints over {duration:.2f} s...", flush=True)
        while True:
            elapsed = time.monotonic() - start_time
            blend = smooth_step(elapsed / duration)
            positions = start + blend * (target - start)
            publisher.send(
                [wuji_sdk.JointCommand(float(position), 0.0, 0.0) for position in positions]
            )
            if elapsed >= duration:
                break
            time.sleep(period)
        time.sleep(0.2)
        print(f"[{side}] Hand-joint target reached.", flush=True)
    finally:
        if publisher is not None:
            try:
                publisher.close()
            except Exception:
                pass
        if hand is not None:
            try:
                hand.disable()
            except Exception:
                pass
        try:
            manager.disconnect_all()
        except Exception:
            pass


def run_parallel(process_specs: list[tuple[str, Any, tuple[Any, ...]]]) -> None:
    context = mp.get_context("spawn")
    processes: list[tuple[str, mp.Process]] = []
    for name, function, arguments in process_specs:
        process = context.Process(target=function, args=arguments, name=name)
        process.start()
        processes.append((name, process))
    failed: list[str] = []
    for name, process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(f"{name} (exit {process.exitcode})")
    if failed:
        raise RuntimeError("hardware command failed: " + ", ".join(failed))


def read_current_targets(args: argparse.Namespace, targets: list[SideTarget]) -> dict[str, dict[str, np.ndarray]]:
    """Read all values displayed to the operator before an actual move."""
    robot_ips = {"left": args.ip_left, "right": args.ip_right}
    hand_addresses = {"left": args.left_hand_ip, "right": args.right_hand_ip}
    current: dict[str, dict[str, np.ndarray]] = {}
    for target in targets:
        side_values = {"pose": read_current_arm_pose(robot_ips[target.side], target.side)}
        if args.end_effector == "gripper":
            side_values["joint"] = read_current_gripper_width(robot_ips[target.side], target.side)
        elif args.end_effector == "hand":
            side_values["joint"] = read_current_hand_positions(
                target.side, hand_addresses[target.side]
            )
        current[target.side] = side_values
    return current


def format_values(values: np.ndarray) -> str:
    return np.array2string(
        np.asarray(values, dtype=float),
        precision=6,
        separator=", ",
        max_line_width=10_000,
    )


def format_command_values(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.6f}" for value in np.asarray(values, dtype=float))


def print_move_summary(
    args: argparse.Namespace,
    targets: list[SideTarget],
    current: dict[str, dict[str, np.ndarray]],
) -> None:
    print("\nRequested robot motion")
    print("======================")
    for target in targets:
        print(f"{target.side} current ee pose [x, y, z, roll, pitch, yaw]: {format_values(current[target.side]['pose'])}")
        print(f"{target.side} target  ee pose [x, y, z, roll, pitch, yaw]: {format_values(target.pose)}")
        if target.end_effector_joint is not None:
            label = "gripper width [m]" if args.end_effector == "gripper" else "hand joint angles [rad]"
            print(f"{target.side} current {label}: {format_values(current[target.side]['joint'])}")
            print(f"{target.side} target  {label}: {format_values(target.end_effector_joint)}")
        command = f"--target-ee-pose {format_command_values(target.pose)}"
        if target.end_effector_joint is not None:
            command += f" --target-ee-joint {format_command_values(target.end_effector_joint)}"
        print(f"{target.side} copy target arguments: {command}")


def require_approval() -> None:
    try:
        response = input("Move the real robot to these targets? [y/N]: ").strip().lower()
    except EOFError as exc:
        raise SystemExit("No approval received; real-robot motion cancelled.") from exc
    if response not in {"y", "yes"}:
        raise SystemExit("Real-robot motion cancelled.")


def print_dry_run(args: argparse.Namespace, targets: list[SideTarget]) -> None:
    current = read_current_targets(args, targets)
    print_move_summary(args, targets, current)
    print("Dry run: no hardware motion was commanded.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    targets = resolve_targets(args, parser)
    robot_ips = {"left": args.ip_left, "right": args.ip_right}
    if args.dry_run:
        print_dry_run(args, targets)
        return

    current = read_current_targets(args, targets)
    print_move_summary(args, targets, current)
    require_approval()
    run_parallel(
        [
            (f"{target.side}_arm_move", move_arm, (robot_ips[target.side], target.side, target.pose))
            for target in targets
        ]
    )

    if args.end_effector == "gripper":
        run_parallel(
            [
                (
                    f"{target.side}_gripper_move",
                    move_gripper,
                    (robot_ips[target.side], target.side, float(target.end_effector_joint[0])),
                )
                for target in targets
            ]
        )
    elif args.end_effector == "hand":
        hand_addresses = {"left": args.left_hand_ip, "right": args.right_hand_ip}
        run_parallel(
            [
                (
                    f"{target.side}_hand_move",
                    move_hand,
                    (target.side, hand_addresses[target.side], target.end_effector_joint),
                )
                for target in targets
            ]
        )

    print("All requested targets reached.")


if __name__ == "__main__":
    main()
