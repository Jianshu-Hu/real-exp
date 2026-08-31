#!/usr/bin/env python3
"""Plan and execute a collision-aware FR3 move to an end-effector pose.

Pose values are ``x,y,z,roll,pitch,yaw`` in the Franka base frame, using meters
and radians. MoveIt/OMPL plans a collision-checked configuration-space path to
the Cartesian goal and a FollowJointTrajectory controller executes it.

Running this program without ``--dry-run`` commands real hardware.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

DEFAULT_LEFT_ROBOT_IP = "172.16.0.3"
DEFAULT_RIGHT_ROBOT_IP = "172.16.0.2"
JOINT_NAMES = [f"fr3_joint{index}" for index in range(1, 8)]
MOVEIT_PLANNING_TIME_S = 10.0
MOVEIT_PLANNING_ATTEMPTS = 5
MOVEIT_VELOCITY_SCALING = 0.20
MOVEIT_ACCELERATION_SCALING = 0.15
MOVEIT_POSITION_TOLERANCE_M = 0.002
MOVEIT_ORIENTATION_TOLERANCE_RAD = 0.01
MOVEIT_SERVER_TIMEOUT_S = 30.0
TRAJECTORY_EXECUTION_MARGIN_S = 10.0
EE_FINAL_POSITION_TOLERANCE_M = 0.01
EE_FINAL_ORIENTATION_TOLERANCE_RAD = 0.03
IK_POSITION_TOLERANCE_M = 5e-4
IK_ORIENTATION_TOLERANCE_RAD = 5e-3
IK_MAX_FUNCTION_EVALUATIONS = 1200
END_EFFECTOR_MOVE_TIMEOUT_S = 30.0

# Match JointReferenceGenerator's operational envelope in the ROS controller.
# It is deliberately narrower than the mechanical URDF limits, so an IK target
# accepted here cannot be silently clipped by the controller later.
ARM_POSITION_LOWER_RAD = np.asarray(
    [-2.6937, -1.7337, -2.8507, -2.9921, -2.7565, 0.5945, -2.9659], dtype=float
)
ARM_POSITION_UPPER_RAD = np.asarray(
    [2.6937, 1.7337, 2.8507, -0.2018, 2.7565, 4.4669, 2.9659], dtype=float
)

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


@dataclass(frozen=True)
class IkResult:
    q: np.ndarray
    achieved_pose: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    function_evaluations: int


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
            "values. The pose is expressed in the selected FR3 base frame."
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
            "(20 values). Omit with --arm."
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


def resolve_targets(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[SideTarget]:
    pose_values = parse_target_values(args.target_ee_pose, "--target-ee-pose", parser)
    joint_values = (
        None
        if args.target_ee_joint is None
        else parse_target_values(args.target_ee_joint, "--target-ee-joint", parser)
    )
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
    pose = np.asarray(pose_values, dtype=float)
    validate_pose(pose, args.arm_mode, parser)
    joint = None
    if joints_per_side:
        joint = np.asarray(resolved_joint_values, dtype=float)
    return [SideTarget(side=args.arm_mode, pose=pose, end_effector_joint=joint)]


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
    """Convert a 4x4 matrix or libfranka's flat column-major transform."""
    values = np.asarray(transform, dtype=float)
    matrix = values.reshape((4, 4), order="F") if values.ndim == 1 else values.reshape(4, 4)
    return np.concatenate((matrix[:3, 3], rotation_to_rpy(matrix[:3, :3])))


def rotation_distance(start: np.ndarray, target: np.ndarray) -> float:
    cosine = (float(np.trace(start.T @ target)) - 1.0) / 2.0
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def pose_error(start: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return translational and rotational error between homogeneous poses."""
    translation_error = float(np.linalg.norm(target[:3, 3] - start[:3, 3]))
    orientation_error = rotation_distance(start[:3, :3], target[:3, :3])
    return translation_error, orientation_error


def pose_message_to_matrix(pose: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    transform = np.eye(4, dtype=float)
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    transform[:3, :3] = Rotation.from_quat(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    ).as_matrix()
    return transform


def matrix_to_pose_message(transform: np.ndarray, Pose: Any) -> Any:
    """Convert a homogeneous transform to geometry_msgs/Pose."""
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    message = Pose()
    message.position.x, message.position.y, message.position.z = matrix[:3, 3].tolist()
    (
        message.orientation.x,
        message.orientation.y,
        message.orientation.z,
        message.orientation.w,
    ) = quaternion.tolist()
    return message


def build_move_group_goal(
    side: str,
    target_ee_pose: np.ndarray,
    flange_to_ee: np.ndarray,
    message_types: dict[str, Any],
    current_q: np.ndarray | None = None,
) -> Any:
    """Create a plan-only OMPL request for the flange pose behind an EE goal."""
    MoveGroup = message_types["MoveGroup"]
    Constraints = message_types["Constraints"]
    OrientationConstraint = message_types["OrientationConstraint"]
    PositionConstraint = message_types["PositionConstraint"]
    Pose = message_types["Pose"]
    SolidPrimitive = message_types["SolidPrimitive"]

    target_flange = np.asarray(target_ee_pose, dtype=float) @ np.linalg.inv(
        np.asarray(flange_to_ee, dtype=float)
    )
    frame_id = f"{side}_fr3_link0"
    link_name = f"{side}_fr3_link8"
    target_pose = matrix_to_pose_message(target_flange, Pose)

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.SPHERE
    primitive.dimensions = [MOVEIT_POSITION_TOLERANCE_M]

    position = PositionConstraint()
    position.header.frame_id = frame_id
    position.link_name = link_name
    position.constraint_region.primitives = [primitive]
    position.constraint_region.primitive_poses = [target_pose]
    position.weight = 1.0

    orientation = OrientationConstraint()
    orientation.header.frame_id = frame_id
    orientation.link_name = link_name
    orientation.orientation = target_pose.orientation
    orientation.absolute_x_axis_tolerance = MOVEIT_ORIENTATION_TOLERANCE_RAD
    orientation.absolute_y_axis_tolerance = MOVEIT_ORIENTATION_TOLERANCE_RAD
    orientation.absolute_z_axis_tolerance = MOVEIT_ORIENTATION_TOLERANCE_RAD
    orientation.weight = 1.0

    constraints = Constraints()
    constraints.name = "target_end_effector_pose"
    constraints.position_constraints = [position]
    constraints.orientation_constraints = [orientation]

    goal = MoveGroup.Goal()
    goal.request.group_name = f"{side}_fr3_arm"
    # Leave this empty to select the launch-configured OMPL pipeline.  MoveIt
    # names the default pipeline from its parameter namespace on ROS 2 Humble.
    goal.request.pipeline_id = ""
    goal.request.planner_id = "RRTConnectkConfigDefault"
    goal.request.num_planning_attempts = MOVEIT_PLANNING_ATTEMPTS
    goal.request.allowed_planning_time = MOVEIT_PLANNING_TIME_S
    goal.request.max_velocity_scaling_factor = MOVEIT_VELOCITY_SCALING
    goal.request.max_acceleration_scaling_factor = MOVEIT_ACCELERATION_SCALING
    goal.request.goal_constraints = [constraints]
    if current_q is not None:
        measured_q = np.asarray(current_q, dtype=float)
        if measured_q.shape != (7,) or not np.all(np.isfinite(measured_q)):
            raise ValueError("MoveIt start state must be a finite seven-joint configuration")
        goal.request.start_state.joint_state.name = [
            f"{side}_{name}" for name in JOINT_NAMES
        ]
        goal.request.start_state.joint_state.position = measured_q.tolist()
        goal.request.start_state.is_diff = False
    goal.planning_options.plan_only = True
    goal.planning_options.look_around = False
    goal.planning_options.replan = False
    return goal


def spin_until_future(rclpy: Any, node: Any, future: Any, timeout_s: float, description: str) -> Any:
    """Wait for an rclpy future with a bounded, descriptive timeout."""
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_s)
    if not future.done():
        raise TimeoutError(f"Timed out after {timeout_s:g} s waiting for {description}")
    exception = future.exception()
    if exception is not None:
        raise RuntimeError(f"{description} failed: {exception}") from exception
    return future.result()


def cancel_action_goal(rclpy: Any, node: Any, goal_handle: Any, description: str) -> None:
    """Best-effort cancellation used when planning or execution times out."""
    if not rclpy.ok():
        return
    cancel_future = goal_handle.cancel_goal_async()
    rclpy.spin_until_future_complete(node, cancel_future, timeout_sec=2.0)
    if not cancel_future.done():
        node.get_logger().error(f"Timed out cancelling {description}")


def plan_moveit_trajectory(
    rclpy: Any,
    node: Any,
    side: str,
    target_ee_pose: np.ndarray,
    flange_to_ee: np.ndarray,
    current_q: np.ndarray,
    message_types: dict[str, Any],
) -> tuple[Any, float]:
    """Request a collision-checked, time-parameterized joint trajectory."""
    from rclpy.action import ActionClient

    MoveGroup = message_types["MoveGroup"]
    action_name = f"/{side}/move_action"
    client = ActionClient(node, MoveGroup, action_name)
    if not client.wait_for_server(timeout_sec=MOVEIT_SERVER_TIMEOUT_S):
        raise TimeoutError(f"MoveIt action server is unavailable: {action_name}")
    goal = build_move_group_goal(
        side, target_ee_pose, flange_to_ee, message_types, current_q=current_q
    )
    goal_handle = spin_until_future(
        rclpy,
        node,
        client.send_goal_async(goal),
        MOVEIT_SERVER_TIMEOUT_S,
        f"{side} MoveIt plan acceptance",
    )
    if not goal_handle.accepted:
        raise RuntimeError(f"[{side}] MoveIt rejected the pose-goal planning request")
    try:
        result_wrapper = spin_until_future(
            rclpy,
            node,
            goal_handle.get_result_async(),
            MOVEIT_PLANNING_TIME_S * MOVEIT_PLANNING_ATTEMPTS
            + MOVEIT_SERVER_TIMEOUT_S,
            f"{side} MoveIt plan",
        )
    except TimeoutError:
        cancel_action_goal(rclpy, node, goal_handle, f"{side} MoveIt plan")
        raise
    result = result_wrapper.result
    if result.error_code.val != result.error_code.SUCCESS:
        raise RuntimeError(
            f"[{side}] MoveIt/OMPL planning failed with error code {result.error_code.val}; "
            "no trajectory was executed"
        )
    trajectory = result.planned_trajectory.joint_trajectory
    if not trajectory.points:
        raise RuntimeError(f"[{side}] MoveIt returned an empty trajectory")
    expected_names = {f"{side}_{name}" for name in JOINT_NAMES}
    if set(trajectory.joint_names) != expected_names:
        raise RuntimeError(
            f"[{side}] MoveIt trajectory joints {trajectory.joint_names} do not match "
            f"the controlled arm joints {sorted(expected_names)}"
        )
    return trajectory, float(result.planning_time)


def trajectory_duration_s(trajectory: Any) -> float:
    duration = trajectory.points[-1].time_from_start
    return float(duration.sec) + 1e-9 * float(duration.nanosec)


def trajectory_joint_positions(side: str, trajectory: Any, point_index: int) -> np.ndarray:
    """Return one trajectory point in canonical FR3 joint order."""
    positions_by_name = dict(
        zip(trajectory.joint_names, trajectory.points[point_index].positions, strict=True)
    )
    return np.asarray(
        [positions_by_name[f"{side}_{name}"] for name in JOINT_NAMES], dtype=float
    )


def verify_planned_endpoint(
    side: str,
    trajectory: Any,
    target_pose: np.ndarray,
    flange_to_ee: np.ndarray,
) -> np.ndarray:
    """Verify the selected IK goal before allowing trajectory execution."""
    final_q = trajectory_joint_positions(side, trajectory, -1)
    model, frame_id = build_fr3_model()
    planned_pose = forward_end_effector_pose(model, frame_id, final_q, flange_to_ee)
    position_error, orientation_error = pose_error(planned_pose, target_pose)
    print(
        f"{side} planned final ee pose [x, y, z, roll, pitch, yaw]: "
        f"{format_values(matrix_to_pose_vector(planned_pose))}",
        flush=True,
    )
    print(
        f"{side} planned Cartesian residual: position={position_error:.6f} m, "
        f"orientation={orientation_error:.6f} rad",
        flush=True,
    )
    if (
        position_error > EE_FINAL_POSITION_TOLERANCE_M
        or orientation_error > EE_FINAL_ORIENTATION_TOLERANCE_RAD
    ):
        raise RuntimeError(
            f"[{side}] planned endpoint is outside the final Cartesian tolerance: "
            f"position={position_error:.6f} m, orientation={orientation_error:.6f} rad; "
            "no trajectory was executed"
        )
    return final_q


def print_trajectory_summary(side: str, trajectory: Any, planning_time_s: float) -> None:
    final = trajectory.points[-1]
    print(
        f"{side} collision-checked OMPL plan: {len(trajectory.points)} points, "
        f"duration={trajectory_duration_s(trajectory):.3f} s, "
        f"planning_time={planning_time_s:.3f} s",
        flush=True,
    )
    print(f"{side} planned final arm joint angles [rad]: {format_values(final.positions)}")


def execute_joint_trajectory(
    rclpy: Any, node: Any, side: str, trajectory: Any, action_type: Any
) -> bool:
    """Execute a trajectory; return whether its joint goal tolerance passed.

    A pose-goal command still measures and enforces its Cartesian residual when
    the controller narrowly misses the particular redundant IK joint endpoint.
    All other controller errors remain hard execution failures.
    """
    from rclpy.action import ActionClient

    action_name = f"/{side}/fr3_arm_controller/follow_joint_trajectory"
    current_q = node.arm_q[side]
    if current_q is None:
        raise RuntimeError(f"[{side}] measured joint state is unavailable before execution")
    planned_start = trajectory_joint_positions(side, trajectory, 0)
    start_error = float(np.max(np.abs(np.asarray(current_q, dtype=float) - planned_start)))
    if start_error > 0.03:
        raise RuntimeError(
            f"[{side}] robot moved {start_error:.4f} rad away from the approved plan start; "
            "replan before execution"
        )
    client = ActionClient(node, action_type, action_name)
    if not client.wait_for_server(timeout_sec=MOVEIT_SERVER_TIMEOUT_S):
        raise TimeoutError(f"Trajectory controller action server is unavailable: {action_name}")
    goal = action_type.Goal()
    goal.trajectory = trajectory
    goal_handle = spin_until_future(
        rclpy,
        node,
        client.send_goal_async(goal),
        MOVEIT_SERVER_TIMEOUT_S,
        f"{side} trajectory acceptance",
    )
    if not goal_handle.accepted:
        raise RuntimeError(f"[{side}] trajectory controller rejected the planned trajectory")
    try:
        result_wrapper = spin_until_future(
            rclpy,
            node,
            goal_handle.get_result_async(),
            trajectory_duration_s(trajectory) + TRAJECTORY_EXECUTION_MARGIN_S,
            f"{side} trajectory execution",
        )
    except TimeoutError:
        cancel_action_goal(rclpy, node, goal_handle, f"{side} trajectory execution")
        raise
    result = result_wrapper.result
    if result.error_code == result.SUCCESSFUL:
        return True
    if result.error_code == result.GOAL_TOLERANCE_VIOLATED:
        print(
            f"[{side}] trajectory controller missed its joint endpoint tolerance: "
            f"{result.error_string}. Measuring the authoritative Cartesian pose goal.",
            flush=True,
        )
        return False
    else:
        raise RuntimeError(
            f"[{side}] trajectory execution failed with code {result.error_code}: "
            f"{result.error_string}"
        )


def verify_final_ee_pose(
    rclpy: Any,
    node: Any,
    side: str,
    target_pose: np.ndarray,
    planned_final_q: np.ndarray,
) -> tuple[float, float]:
    """Refresh robot state and enforce the final measured Cartesian tolerance."""
    deadline = time.monotonic() + 2.0
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    measured = node.ee_pose[side]
    if measured is None:
        raise RuntimeError(f"[{side}] final measured EE pose is unavailable")
    position_error, orientation_error = pose_error(measured, target_pose)
    measured_q = node.arm_q[side]
    if measured_q is None:
        raise RuntimeError(f"[{side}] final measured joint state is unavailable")
    joint_error = float(
        np.max(np.abs(np.asarray(measured_q, dtype=float) - planned_final_q))
    )
    print(
        f"{side} final measured Cartesian residual: position={position_error:.6f} m, "
        f"orientation={orientation_error:.6f} rad",
        flush=True,
    )
    print(f"{side} final maximum joint residual: {joint_error:.6f} rad", flush=True)
    if (
        position_error > EE_FINAL_POSITION_TOLERANCE_M
        or orientation_error > EE_FINAL_ORIENTATION_TOLERANCE_RAD
    ):
        raise RuntimeError(
            f"[{side}] final EE pose is outside tolerance: position={position_error:.6f} m "
            f"(limit {EE_FINAL_POSITION_TOLERANCE_M:g}), orientation={orientation_error:.6f} "
            f"rad (limit {EE_FINAL_ORIENTATION_TOLERANCE_RAD:g}); maximum joint residual "
            f"to the planned endpoint={joint_error:.6f} rad"
        )
    return position_error, orientation_error


def build_fr3_model() -> tuple[Any, int]:
    """Build the same no-gripper FR3 model used by the ROS controller stack."""
    import pinocchio as pin
    import xacro
    from ament_index_python.packages import get_package_share_directory
    from pathlib import Path

    xacro_path = (
        Path(get_package_share_directory("franka_description"))
        / "robots"
        / "fr3"
        / "fr3.urdf.xacro"
    )
    xml = xacro.process_file(
        str(xacro_path),
        mappings={
            "ros2_control": "false",
            "arm_id": "fr3",
            "arm_prefix": "",
            "robot_ip": "",
            "hand": "false",
            "use_fake_hardware": "false",
            "fake_sensor_commands": "false",
        },
    ).toxml()
    model = pin.buildModelFromXML(xml)
    frame_id = model.getFrameId("fr3_link8")
    if frame_id >= len(model.frames):
        raise RuntimeError("FR3 model does not contain the fr3_link8 flange frame")
    return model, frame_id


def forward_end_effector_pose(
    model: Any, frame_id: int, q: np.ndarray, flange_to_ee: np.ndarray
) -> np.ndarray:
    import pinocchio as pin

    data = model.createData()
    pin.forwardKinematics(model, data, np.asarray(q, dtype=float))
    pin.updateFramePlacements(model, data)
    return np.asarray(data.oMf[frame_id].homogeneous, dtype=float) @ flange_to_ee


def solve_fr3_ik(
    current_q: np.ndarray,
    target_ee_pose: np.ndarray,
    flange_to_ee: np.ndarray,
    model: Any | None = None,
    frame_id: int | None = None,
    *,
    kinematics: Any | None = None,
    try_alternative_seeds: bool = True,
    max_function_evaluations: int = IK_MAX_FUNCTION_EVALUATIONS,
) -> IkResult:
    """Solve bounded final-pose IK and independently verify its FK residual."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    seed = np.asarray(current_q, dtype=float)
    target = np.asarray(target_ee_pose, dtype=float)
    flange_to_ee = np.asarray(flange_to_ee, dtype=float)
    if seed.shape != (7,) or not np.all(np.isfinite(seed)):
        raise ValueError("IK seed must be a finite seven-joint FR3 configuration")
    if target.shape != (4, 4) or flange_to_ee.shape != (4, 4):
        raise ValueError("IK target and F_T_EE must be 4x4 transforms")
    if max_function_evaluations <= 0:
        raise ValueError("IK maximum function evaluations must be positive")
    if kinematics is not None and (model is not None or frame_id is not None):
        raise ValueError("Pass either an FK kinematics backend or a Pinocchio model/frame pair")
    if kinematics is None and (model is None or frame_id is None):
        model, frame_id = build_fr3_model()

    target_flange = target @ np.linalg.inv(flange_to_ee)
    data = model.createData() if kinematics is None else None

    def flange_pose(q: np.ndarray) -> np.ndarray:
        if kinematics is not None:
            return np.asarray(kinematics.flange_pose(q), dtype=float)
        import pinocchio as pin

        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        return np.asarray(data.oMf[frame_id].homogeneous, dtype=float)

    def end_effector_pose(q: np.ndarray) -> np.ndarray:
        if kinematics is not None:
            return np.asarray(kinematics.end_effector_pose(q, flange_to_ee), dtype=float)
        return forward_end_effector_pose(model, frame_id, q, flange_to_ee)

    def residual(q: np.ndarray) -> np.ndarray:
        current = flange_pose(q)
        translation = current[:3, 3] - target_flange[:3, 3]
        orientation = Rotation.from_matrix(
            target_flange[:3, :3].T @ current[:3, :3]
        ).as_rotvec()
        # A very weak redundancy preference selects a solution near the live
        # configuration without materially relaxing the six pose constraints.
        return np.concatenate((translation, orientation, 1e-5 * (q - seed)))

    candidate_seeds = [seed]
    if try_alternative_seeds:
        for joint, delta in ((6, 0.45), (6, -0.45), (2, 0.30), (2, -0.30)):
            candidate = seed.copy()
            candidate[joint] = np.clip(
                candidate[joint] + delta,
                ARM_POSITION_LOWER_RAD[joint] + 1e-6,
                ARM_POSITION_UPPER_RAD[joint] - 1e-6,
            )
            candidate_seeds.append(candidate)

    results: list[IkResult] = []
    total_evaluations = 0
    for candidate_seed in candidate_seeds:
        solution = least_squares(
            residual,
            np.clip(candidate_seed, ARM_POSITION_LOWER_RAD, ARM_POSITION_UPPER_RAD),
            bounds=(ARM_POSITION_LOWER_RAD, ARM_POSITION_UPPER_RAD),
            max_nfev=max_function_evaluations,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        total_evaluations += int(solution.nfev)
        achieved = end_effector_pose(solution.x)
        position_error, orientation_error = pose_error(achieved, target)
        results.append(
            IkResult(
                q=np.asarray(solution.x, dtype=float),
                achieved_pose=achieved,
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
                function_evaluations=int(solution.nfev),
            )
        )

    valid = [
        result
        for result in results
        if result.position_error_m <= IK_POSITION_TOLERANCE_M
        and result.orientation_error_rad <= IK_ORIENTATION_TOLERANCE_RAD
    ]
    if not valid:
        best = min(results, key=lambda item: item.position_error_m + item.orientation_error_rad)
        raise RuntimeError(
            "No FR3 IK solution satisfies the verified pose tolerance: "
            f"best position error={best.position_error_m:.6f} m, "
            f"orientation error={best.orientation_error_rad:.6f} rad after "
            f"{total_evaluations} function evaluations. No arm command was published."
        )
    return min(valid, key=lambda item: float(np.linalg.norm(item.q - seed)))


def move_gripper(robot_ip: str, side: str, target_width: float) -> None:
    """Move a Franka Hand after the ROS-controlled arm has settled."""
    import pylibfranka

    gripper = pylibfranka.Gripper(robot_ip)
    state = gripper.read_once()
    if target_width > float(state.max_width) + 1e-6:
        raise ValueError(
            f"[{side}] target width {target_width:.6f} m exceeds the measured "
            f"maximum {float(state.max_width):.6f} m"
        )
    if not gripper.move(target_width, 0.05):
        raise RuntimeError(f"[{side}] Franka gripper failed to reach its target")
    print(f"[{side}] Franka gripper target reached.", flush=True)


def ordered_joint_values(message: Any, field_name: str) -> np.ndarray | None:
    raw = getattr(message, field_name, [])
    if len(raw) < 7:
        return None
    ordered: list[float | None] = [None] * 7
    for name, value in zip(message.name, raw, strict=False):
        for index in range(1, 8):
            if name.endswith(f"joint{index}"):
                ordered[index - 1] = float(value)
    if all(value is not None for value in ordered):
        return np.asarray(ordered, dtype=float)
    return np.asarray(raw[:7], dtype=float)


def build_move_node_class(Node: Any, JointState: Any, FrankaRobotState: Any) -> type:
    class MoveToTargetNode(Node):  # type: ignore[misc, valid-type]
        def __init__(self, args: argparse.Namespace) -> None:
            super().__init__("move_to_target_ee")
            self.active_sides = [args.arm_mode]
            self.arm_q: dict[str, np.ndarray | None] = {side: None for side in self.active_sides}
            self.ee_pose: dict[str, np.ndarray | None] = {side: None for side in self.active_sides}
            self.flange_to_ee: dict[str, np.ndarray | None] = {
                side: None for side in self.active_sides
            }
            for side in self.active_sides:
                self.create_subscription(
                    JointState,
                    f"/{side}/franka/joint_states",
                    lambda message, selected=side: self._store_joint_state(selected, message),
                    10,
                )
                self.create_subscription(
                    FrankaRobotState,
                    f"/{side}/franka_robot_state_broadcaster/robot_state",
                    lambda message, selected=side: self._store_robot_state(selected, message),
                    10,
                )

        def _store_joint_state(self, side: str, message: Any) -> None:
            q = ordered_joint_values(message, "position")
            if q is not None and np.all(np.isfinite(q)):
                self.arm_q[side] = q

        def _store_robot_state(self, side: str, message: Any) -> None:
            self.ee_pose[side] = pose_message_to_matrix(message.o_t_ee.pose)
            self.flange_to_ee[side] = pose_message_to_matrix(message.f_t_ee.pose)

    return MoveToTargetNode


def wait_until(rclpy: Any, node: Any, predicate: Any, timeout_s: float, description: str) -> None:
    deadline = time.monotonic() + timeout_s
    while rclpy.ok() and time.monotonic() < deadline:
        if predicate():
            return
        rclpy.spin_once(node, timeout_sec=0.05)
    raise TimeoutError(f"Timed out after {timeout_s:g} s waiting for {description}")


def wait_for_robot_state(rclpy: Any, node: Any) -> None:
    def ready() -> bool:
        for side in node.active_sides:
            if (
                node.arm_q[side] is None
                or node.ee_pose[side] is None
                or node.flange_to_ee[side] is None
            ):
                return False
        return True

    wait_until(rclpy, node, ready, 30.0, "FR3 state topics")


def request_hand_status(socket: Any, request: dict[str, Any] | None = None) -> dict[str, Any]:
    import zmq

    socket.send_pyobj({"kind": "status"} if request is None else request)
    if not socket.poll(1000, zmq.POLLIN):
        raise TimeoutError("Timed out waiting for Wuji hand worker status")
    response = socket.recv_pyobj()
    if not isinstance(response, dict) or not response.get("ready", False):
        raise RuntimeError(f"Invalid Wuji hand worker status: {response!r}")
    return response


def open_hand_status_sockets(sides: list[str]) -> tuple[Any, dict[str, Any]]:
    import zmq

    context = zmq.Context()
    ports = {"left": 5563, "right": 5564}
    sockets: dict[str, Any] = {}
    for side in sides:
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.connect(f"tcp://127.0.0.1:{ports[side]}")
        sockets[side] = socket
    return context, sockets


def read_hand_states(sockets: dict[str, Any]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for side, socket in sockets.items():
        status = request_hand_status(socket)
        actual = status.get("actual")
        if actual is None or np.asarray(actual).shape != (20,):
            raise RuntimeError(f"[{side}] Wuji worker has no valid measured joint state")
        result[side] = np.asarray(actual, dtype=float)
    return result


def move_hands(sockets: dict[str, Any], targets: list[SideTarget]) -> None:
    for target in targets:
        status = request_hand_status(
            sockets[target.side],
            {"kind": "initial", "target": target.end_effector_joint.tolist()},
        )
        if not status.get("initial_received", False):
            raise RuntimeError(f"[{target.side}] Wuji worker rejected the hand target")
    deadline = time.monotonic() + END_EFFECTOR_MOVE_TIMEOUT_S
    while time.monotonic() < deadline:
        statuses = {side: request_hand_status(socket) for side, socket in sockets.items()}
        if all(status.get("initial_reached", False) for status in statuses.values()):
            print("All selected Wuji hands reached their targets.", flush=True)
            return
        time.sleep(0.05)
    errors = {}
    for target in targets:
        actual = request_hand_status(sockets[target.side]).get("actual")
        errors[target.side] = None if actual is None else float(
            np.max(np.abs(np.asarray(actual, dtype=float) - target.end_effector_joint))
        )
    raise TimeoutError(f"Hands did not settle within {END_EFFECTOR_MOVE_TIMEOUT_S:g} s; errors={errors}")


def read_current_targets(
    args: argparse.Namespace,
    targets: list[SideTarget],
    node: Any | None = None,
    hand_states: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Read all values displayed to the operator before an actual move.

    Arm pose and joint state are intentionally read from the ROS controller
    state streams. Hardware execution always supplies ``node``.
    """
    current: dict[str, dict[str, np.ndarray]] = {}
    for target in targets:
        if node is None:
            raise RuntimeError("ROS node is required to read current arm state")
        pose = node.ee_pose[target.side]
        q = node.arm_q[target.side]
        if pose is None or q is None:
            raise RuntimeError(f"[{target.side}] ROS state is not available")
        side_values = {"pose": matrix_to_pose_vector(pose), "arm_q": np.asarray(q, dtype=float)}
        if args.end_effector == "gripper":
            import pylibfranka

            gripper = pylibfranka.Gripper(
                args.ip_left if target.side == "left" else args.ip_right
            )
            width = float(gripper.read_once().width)
            side_values["joint"] = np.asarray([width], dtype=float)
        elif args.end_effector == "hand":
            if hand_states is None or target.side not in hand_states:
                raise RuntimeError(f"[{target.side}] hand state is not available")
            side_values["joint"] = np.asarray(hand_states[target.side], dtype=float)
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
        if "arm_q" in current[target.side]:
            print(f"{target.side} current arm joint angles [rad]: {format_values(current[target.side]['arm_q'])}")
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    targets = resolve_targets(args, parser)
    try:
        from control_msgs.action import FollowJointTrajectory
        import rclpy
        from franka_msgs.msg import FrankaRobotState
        from geometry_msgs.msg import Pose
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from shape_msgs.msg import SolidPrimitive
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ROS 2 Python dependencies are required. Run scripts/move_to_target_ee.sh "
            "so the ROS Humble environment is sourced."
        ) from exc

    rclpy.init()
    node_class = build_move_node_class(Node, JointState, FrankaRobotState)
    node = node_class(args)
    hand_context = None
    hand_sockets: dict[str, Any] = {}
    try:
        wait_for_robot_state(rclpy, node)
        hand_states = None
        if args.end_effector == "hand":
            hand_context, hand_sockets = open_hand_status_sockets(node.active_sides)
            hand_states = read_hand_states(hand_sockets)
        current = read_current_targets(args, targets, node, hand_states)

        # Preserve these diagnostics even when the planner cannot find a path.
        print_move_summary(args, targets, current)
        target = targets[0]
        target_matrix = pose_vector_to_matrix(target.pose)
        message_types = {
            "MoveGroup": MoveGroup,
            "Constraints": Constraints,
            "OrientationConstraint": OrientationConstraint,
            "PositionConstraint": PositionConstraint,
            "Pose": Pose,
            "SolidPrimitive": SolidPrimitive,
        }
        trajectory, planning_time = plan_moveit_trajectory(
            rclpy,
            node,
            target.side,
            target_matrix,
            np.asarray(node.flange_to_ee[target.side], dtype=float),
            current[target.side]["arm_q"],
            message_types,
        )
        print_trajectory_summary(target.side, trajectory, planning_time)
        planned_final_q = verify_planned_endpoint(
            target.side,
            trajectory,
            target_matrix,
            np.asarray(node.flange_to_ee[target.side], dtype=float),
        )
        print(
            "The plan was checked against the MoveIt self-collision model and the current "
            "planning scene. Confirm that the scene contains every real obstacle before moving."
        )
        if args.dry_run:
            print("Dry run: the motion was planned but no trajectory was executed.")
            return

        require_approval()
        joint_goal_satisfied = execute_joint_trajectory(
            rclpy, node, target.side, trajectory, FollowJointTrajectory
        )
        verify_final_ee_pose(
            rclpy, node, target.side, target_matrix, planned_final_q
        )
        if not joint_goal_satisfied:
            print(
                f"[{target.side}] Cartesian pose goal reached despite the redundant IK "
                "joint endpoint miss.",
                flush=True,
            )
        if args.end_effector == "gripper":
            robot_ips = {"left": args.ip_left, "right": args.ip_right}
            for target in targets:
                move_gripper(
                    robot_ips[target.side],
                    target.side,
                    float(target.end_effector_joint[0]),
                )
        elif args.end_effector == "hand":
            move_hands(hand_sockets, targets)
        print("All requested targets reached.")
    finally:
        node.destroy_node()
        if hand_context is not None:
            for socket in hand_sockets.values():
                socket.close(0)
            hand_context.term()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
