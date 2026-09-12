#!/usr/bin/env python3
"""Execute the local arm and hand motion sequence for one grasp command."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from data_collection import move_to_target_ee as move
from grasp.common import xyz_rpy_to_matrix


HAND_GRASP_CONTRACT_FRACTION = 0.2
HAND_FINGER_LATERAL_CONTRACT_INDICES = (5, 9, 13, 17)
HAND_INITIAL_COMMAND_WAIT_S = 2.0
HAND_GRASP_CONTRACT_WAIT_S = 2.0
GRASP_MOVEIT_VELOCITY_SCALING = 0.40
GRASP_MOVEIT_ACCELERATION_SCALING = 0.30
GRASP_CARTESIAN_JOINT_SPEED_RAD_S = np.asarray(
    [0.30, 0.30, 0.30, 0.30, 0.40, 0.40, 0.40], dtype=float
)
GRASP_CARTESIAN_JOINT_ACCELERATION_RAD_S2 = 1.00


def build_parser() -> argparse.ArgumentParser:
    parser = move.build_parser()
    parser.description = "Execute a collision-checked FR3 grasp motion."
    parser.add_argument(
        "--grasp-contract",
        action="store_true",
        help=(
            "Contract the non-lateral Wuji joints after applying the inferred "
            "grasp target."
        ),
    )
    parser.add_argument(
        "--post-grasp-pose",
        nargs="+",
        default=None,
        metavar="X,Y,Z,ROLL,PITCH,YAW",
        help="Second Cartesian pose executed after the hand grasp phase.",
    )
    return parser


def hand_position_limits(side: str) -> tuple[np.ndarray, np.ndarray]:
    path = (
        Path(__file__).resolve().parents[1]
        / "libs"
        / "wuji-retargeting"
        / "wuji_retargeting"
        / "wuji-description"
        / "hand2"
        / "body"
        / "urdf"
        / f"{side}.urdf"
    )
    root = ET.parse(path).getroot()
    limits = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            limits.append((float(limit.attrib["lower"]), float(limit.attrib["upper"])))
    if len(limits) != 20:
        raise RuntimeError(f"Could not load 20 Wuji Hand 2 joint limits from {path}")
    lower, upper = np.asarray(limits, dtype=float).T
    return lower, upper


def move_grasp_hands(
    sockets: dict[str, Any],
    targets: list[move.SideTarget],
    *,
    contract: bool,
) -> None:
    for target in targets:
        status = move.request_hand_status(
            sockets[target.side],
            {"kind": "initial", "target": target.end_effector_joint.tolist()},
        )
        if not status.get("initial_received", False):
            raise RuntimeError(f"[{target.side}] Wuji worker rejected the hand target")
    print(
        "Initial grasp targets sent; waiting "
        f"{HAND_INITIAL_COMMAND_WAIT_S:g} s before the next grasp phase.",
        flush=True,
    )
    time.sleep(HAND_INITIAL_COMMAND_WAIT_S)
    if not contract:
        return

    for target in targets:
        lower, upper = hand_position_limits(target.side)
        q = np.asarray(target.end_effector_joint, dtype=float)
        contracted = np.where(
            q >= 0.0,
            q + HAND_GRASP_CONTRACT_FRACTION * (upper - q),
            q - HAND_GRASP_CONTRACT_FRACTION * (q - lower),
        )
        contracted[list(HAND_FINGER_LATERAL_CONTRACT_INDICES)] = q[
            list(HAND_FINGER_LATERAL_CONTRACT_INDICES)
        ]
        move.request_hand_status(
            sockets[target.side],
            {"kind": "initial", "target": contracted.tolist()},
        )
    print(
        f"Applying grasp contract for {HAND_GRASP_CONTRACT_WAIT_S:g} s before lifting.",
        flush=True,
    )
    time.sleep(HAND_GRASP_CONTRACT_WAIT_S)


def main() -> None:
    move.MOVEIT_VELOCITY_SCALING = GRASP_MOVEIT_VELOCITY_SCALING
    move.MOVEIT_ACCELERATION_SCALING = GRASP_MOVEIT_ACCELERATION_SCALING
    move.CARTESIAN_JOINT_SPEED_RAD_S = GRASP_CARTESIAN_JOINT_SPEED_RAD_S
    move.CARTESIAN_JOINT_ACCELERATION_RAD_S2 = (
        GRASP_CARTESIAN_JOINT_ACCELERATION_RAD_S2
    )
    parser = build_parser()
    args = parser.parse_args()
    targets = move.resolve_targets(args, parser)
    if args.post_grasp_pose is not None and args.end_effector != "hand":
        parser.error("--post-grasp-pose requires --hand")

    try:
        from control_msgs.action import FollowJointTrajectory
        import rclpy
        from franka_msgs.msg import FrankaRobotState
        from moveit_msgs.msg import Constraints, JointConstraint
        from moveit_msgs.srv import GetStateValidity
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ROS 2 Python dependencies are required; start this command through "
            "grasp/start_grasp_execution_client.sh."
        ) from exc

    rclpy.init()
    node_class = move.build_move_node_class(Node, JointState, FrankaRobotState)
    node = node_class(args)
    hand_context = None
    hand_sockets: dict[str, Any] = {}
    try:
        move.wait_for_robot_state(rclpy, node)
        hand_states = None
        if args.end_effector == "hand":
            hand_context, hand_sockets = move.open_hand_status_sockets(node.active_sides)
            hand_states = move.read_hand_states(hand_sockets)
        current = move.read_current_targets(args, targets, node, hand_states)
        move.print_move_summary(args, targets, current)

        target = targets[0]
        target_matrix = xyz_rpy_to_matrix(target.pose)
        post_grasp_matrix = None
        if args.post_grasp_pose is not None:
            post_values = move.parse_target_values(
                args.post_grasp_pose, "--post-grasp-pose", parser
            )
            if len(post_values) != 6:
                parser.error("--post-grasp-pose requires 6 values")
            post_pose = np.asarray(post_values, dtype=float)
            move.validate_pose(post_pose, args.arm_mode, parser)
            post_grasp_matrix = xyz_rpy_to_matrix(post_pose)

        def plan_from_live_state() -> tuple[Any, np.ndarray]:
            start_pose = node.ee_pose[target.side]
            start_q = node.arm_q[target.side]
            flange_to_ee = node.flange_to_ee[target.side]
            if start_pose is None or start_q is None or flange_to_ee is None:
                raise RuntimeError(f"[{target.side}] live state is unavailable for planning")
            start_pose = np.asarray(start_pose, dtype=float).copy()
            start_q = np.asarray(start_q, dtype=float).copy()
            flange_to_ee = np.asarray(flange_to_ee, dtype=float).copy()
            ik_model, ik_frame_id = move.build_fr3_model()
            reference_lower = np.maximum(
                move.ARM_POSITION_LOWER_RAD,
                start_q - move.CARTESIAN_REFERENCE_MAX_JOINT_CHANGE_RAD,
            )
            reference_upper = np.minimum(
                move.ARM_POSITION_UPPER_RAD,
                start_q + move.CARTESIAN_REFERENCE_MAX_JOINT_CHANGE_RAD,
            )
            reference_ik = move.solve_fr3_ik(
                start_q,
                target_matrix,
                flange_to_ee,
                ik_model,
                ik_frame_id,
                try_alternative_seeds=False,
                position_lower_rad=reference_lower,
                position_upper_rad=reference_upper,
            )
            trajectory, planning_time, _ = move.plan_deterministic_cartesian_trajectory(
                rclpy,
                node,
                target.side,
                start_pose,
                target_matrix,
                flange_to_ee,
                start_q,
                reference_ik.q,
                GetStateValidity,
                JointConstraint,
                Constraints,
            )
            audit = move.audit_trajectory(
                target.side,
                trajectory,
                start_pose,
                target_matrix,
                flange_to_ee,
                reference_ik.q,
            )
            move.print_trajectory_summary(target.side, trajectory, planning_time, audit)
            planned_final_q = move.verify_planned_endpoint(
                target.side,
                trajectory,
                target_matrix,
                flange_to_ee,
            )
            return trajectory, planned_final_q

        trajectory, planned_final_q = plan_from_live_state()
        print(
            "The Cartesian plan was checked against the MoveIt self-collision model, the "
            "current planning scene, joint-jump limits, and the requested EE interpolation. "
            "Confirm that the scene contains every real obstacle before moving."
        )
        if args.dry_run:
            print("Dry run: the grasp motion was planned but no trajectory was executed.")
            return

        reached = False
        for attempt in range(1, move.MAX_EXECUTION_ATTEMPTS + 1):
            prompt = (
                "Execute this Cartesian path on the real robot? [y/N]: "
                if attempt == 1
                else "Execute the replanned correction from the current pose? [y/N]: "
            )
            move.require_approval(prompt)
            joint_goal_satisfied = move.execute_joint_trajectory(
                rclpy, node, target.side, trajectory, FollowJointTrajectory
            )
            reached, position_error, orientation_error = move.measure_final_ee_pose(
                rclpy, node, target.side, target_matrix, planned_final_q
            )
            if reached:
                if not joint_goal_satisfied:
                    print(
                        f"[{target.side}] Cartesian pose goal reached despite the redundant "
                        "joint endpoint miss.",
                        flush=True,
                    )
                break
            if attempt == move.MAX_EXECUTION_ATTEMPTS:
                raise RuntimeError(
                    f"[{target.side}] target still not reached after the correction attempt: "
                    f"position={position_error:.6f} m, "
                    f"orientation={orientation_error:.6f} rad"
                )
            print(
                f"[{target.side}] target is outside the relaxed Cartesian tolerance. "
                "Replanning one correction from the measured current pose...",
                flush=True,
            )
            trajectory, planned_final_q = plan_from_live_state()

        if not reached:
            raise RuntimeError(f"[{target.side}] arm target was not reached")
        if args.end_effector == "gripper":
            robot_ip = args.ip_left if target.side == "left" else args.ip_right
            move.move_gripper(robot_ip, target.side, float(target.end_effector_joint[0]))
        elif args.end_effector == "hand":
            move_grasp_hands(hand_sockets, targets, contract=args.grasp_contract)
            if post_grasp_matrix is not None:
                print("Hand grasp phase complete; planning the configured post-grasp lift.")
                target_matrix = post_grasp_matrix
                trajectory, planned_final_q = plan_from_live_state()
                print("Executing the configured post-grasp lift.", flush=True)
                move.execute_joint_trajectory(
                    rclpy, node, target.side, trajectory, FollowJointTrajectory
                )
                reached, position_error, orientation_error = move.measure_final_ee_pose(
                    rclpy, node, target.side, post_grasp_matrix, planned_final_q
                )
                if not reached:
                    raise RuntimeError(
                        f"[{target.side}] post-grasp pose was not reached: "
                        f"position={position_error:.6f} m, "
                        f"orientation={orientation_error:.6f} rad"
                    )
        print("All requested grasp targets reached.")
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
