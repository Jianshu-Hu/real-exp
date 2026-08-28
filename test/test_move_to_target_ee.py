from __future__ import annotations

import argparse

import numpy as np
import pytest

from data_collection import move_to_target_ee as move


def parse(*arguments: str) -> tuple[argparse.Namespace, list[move.SideTarget]]:
    parser = move.build_parser()
    args = parser.parse_args(list(arguments))
    return args, move.resolve_targets(args, parser)


def test_single_arm_target_requires_six_pose_values_and_no_joint_target() -> None:
    _, targets = parse("--left", "--arm", "--target_ee_pose", "0.4,0.2,0.3,0.1,0.2,0.3")
    assert len(targets) == 1
    assert targets[0].side == "left"
    np.testing.assert_allclose(targets[0].pose, [0.4, 0.2, 0.3, 0.1, 0.2, 0.3])
    assert targets[0].end_effector_joint is None


def test_duo_mode_is_not_available_for_independent_single_arm_planning() -> None:
    parser = move.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--duo", "--arm", "--target-ee-pose", "0.4,0.2,0.3,0,0,0"]
        )


def test_hand_target_requires_twenty_values_per_side() -> None:
    joint_values = ",".join(str(index / 10) for index in range(20))
    _, targets = parse(
        "--right",
        "--hand",
        "--target-ee-pose",
        "0.4,-0.2,0.3,3.14,0,0",
        "--target-ee-joint",
        joint_values,
        "--right-hand-ip",
        "192.168.1.5:50001",
    )
    assert targets[0].end_effector_joint.shape == (20,)


def test_space_separated_hand_target_accepts_numpy_style_punctuation() -> None:
    expected_joint_values = np.asarray(
        [
            -0.154496,
            0.433012,
            0.009958,
            -0.063993,
            -0.040934,
            -0.201617,
            0.182062,
            0.691199,
            0.177983,
            -0.236337,
            0.002687,
            0.335948,
            0.178005,
            -0.218885,
            -0.025013,
            0.267149,
            0.137674,
            -0.202433,
            -0.096723,
            -0.019371,
        ]
    )
    joint_values = [
        "-0.154496,",
        "0.433012,",
        "0.009958,",
        "-0.063993,",
        "-0.040934,",
        "-0.201617,",
        "0.182062,",
        "0.691199,",
        "0.177983,",
        "-0.236337,",
        "0.002687,",
        "0.335948,",
        "0.178005,",
        "-0.218885,",
        "-0.025013,",
        "0.267149,",
        "0.137674,",
        "-0.202433,",
        "-0.096723,",
        "-0.019371]",
    ]
    _, targets = parse(
        "--right",
        "--hand",
        "--target-ee-pose",
        "[0.4308",
        ",",
        "0.265915,",
        "0.162497,",
        "3.069743,",
        "0.918623,",
        "1.486756]",
        "--target-ee-joint",
        *joint_values,
        "--right-hand-ip",
        "192.168.1.5:50001",
    )
    np.testing.assert_allclose(
        targets[0].pose,
        [0.4308, 0.265915, 0.162497, 3.069743, 0.918623, 1.486756],
    )
    np.testing.assert_allclose(targets[0].end_effector_joint, expected_joint_values)


def test_copyable_target_values_are_space_separated_on_one_line() -> None:
    assert (
        move.format_command_values(np.asarray([0.4, -0.2, 0.3]))
        == "0.400000 -0.200000 0.300000"
    )


def test_displayed_values_are_comma_separated_on_one_line() -> None:
    formatted = move.format_values(np.asarray([0.447619, -0.315507, 0.215857]))
    assert formatted == "[ 0.447619, -0.315507,  0.215857]"
    assert "\n" not in formatted


def test_wrong_target_dimension_exits_during_validation() -> None:
    with pytest.raises(SystemExit):
        parse("--left", "--gripper", "--target-ee-pose", "1,2,3", "--target-ee-joint", "0.04")


def test_out_of_workspace_target_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse("--left", "--arm", "--target-ee-pose", "1.1,0,0,0,0,0")


def test_pose_matrix_uses_xyz_and_roll_pitch_yaw() -> None:
    matrix = move.pose_vector_to_matrix([1, 2, 3, 0, 0, np.pi / 2])
    np.testing.assert_allclose(matrix[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(
        matrix[:3, :3],
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
        atol=1e-12,
    )


def test_move_group_goal_converts_ee_target_to_flange_pose() -> None:
    from geometry_msgs.msg import Pose
    from moveit_msgs.action import MoveGroup
    from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
    from shape_msgs.msg import SolidPrimitive

    target_ee = move.pose_vector_to_matrix([0.4, -0.2, 0.3, 0.1, -0.2, 0.3])
    flange_to_ee = move.pose_vector_to_matrix([0.0, 0.0, 0.1034, 0.0, 0.0, -np.pi / 4])
    goal = move.build_move_group_goal(
        "right",
        target_ee,
        flange_to_ee,
        {
            "MoveGroup": MoveGroup,
            "Constraints": Constraints,
            "OrientationConstraint": OrientationConstraint,
            "PositionConstraint": PositionConstraint,
            "Pose": Pose,
            "SolidPrimitive": SolidPrimitive,
        },
        current_q=np.asarray([0.1, -0.7, 0.2, -2.0, 0.3, 1.5, -0.4]),
    )

    constraints = goal.request.goal_constraints[0]
    planned_flange = move.pose_message_to_matrix(
        constraints.position_constraints[0].constraint_region.primitive_poses[0]
    )
    expected_flange = target_ee @ np.linalg.inv(flange_to_ee)
    np.testing.assert_allclose(planned_flange, expected_flange, atol=1e-12)
    assert goal.request.group_name == "right_fr3_arm"
    assert goal.request.planner_id == "RRTConnectkConfigDefault"
    assert goal.request.pipeline_id == ""
    assert goal.planning_options.plan_only
    assert constraints.position_constraints[0].header.frame_id == "right_fr3_link0"
    assert constraints.position_constraints[0].link_name == "right_fr3_link8"
    assert goal.request.start_state.joint_state.name == [
        f"right_fr3_joint{index}" for index in range(1, 8)
    ]
    np.testing.assert_allclose(
        goal.request.start_state.joint_state.position,
        [0.1, -0.7, 0.2, -2.0, 0.3, 1.5, -0.4],
    )
    np.testing.assert_allclose(
        constraints.position_constraints[0].constraint_region.primitives[0].dimensions,
        [move.MOVEIT_POSITION_TOLERANCE_M],
    )


def test_trajectory_duration_uses_ros_duration() -> None:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    trajectory = JointTrajectory()
    point = JointTrajectoryPoint()
    point.time_from_start.sec = 3
    point.time_from_start.nanosec = 250_000_000
    trajectory.points = [point]
    assert move.trajectory_duration_s(trajectory) == pytest.approx(3.25)


def test_trajectory_joint_positions_reorders_moveit_joint_names() -> None:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    trajectory = JointTrajectory()
    trajectory.joint_names = [
        "right_fr3_joint3",
        "right_fr3_joint1",
        "right_fr3_joint7",
        "right_fr3_joint2",
        "right_fr3_joint6",
        "right_fr3_joint4",
        "right_fr3_joint5",
    ]
    point = JointTrajectoryPoint()
    point.positions = [3.0, 1.0, 7.0, 2.0, 6.0, 4.0, 5.0]
    trajectory.points = [point]

    np.testing.assert_allclose(
        move.trajectory_joint_positions("right", trajectory, 0),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )


def test_planned_endpoint_verification_uses_final_trajectory_pose() -> None:
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    model, frame_id = move.build_fr3_model()
    final_q = np.asarray([0.2, 0.6, 0.5, -1.8, -0.5, 1.5, 0.3])
    flange_to_ee = move.pose_vector_to_matrix([0, 0, 0.1034, 0, 0, -np.pi / 4])
    target = move.forward_end_effector_pose(model, frame_id, final_q, flange_to_ee)
    trajectory = JointTrajectory()
    trajectory.joint_names = [f"right_fr3_joint{index}" for index in range(1, 8)]
    point = JointTrajectoryPoint()
    point.positions = final_q.tolist()
    trajectory.points = [point]

    np.testing.assert_allclose(
        move.verify_planned_endpoint("right", trajectory, target, flange_to_ee),
        final_q,
    )


def test_goal_tolerance_result_can_continue_to_cartesian_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Result:
        SUCCESSFUL = 0
        GOAL_TOLERANCE_VIOLATED = -5
        error_code = GOAL_TOLERANCE_VIOLATED
        error_string = "joint endpoint tolerance missed"

    class ResultWrapper:
        result = Result()

    class GoalHandle:
        accepted = True

        def get_result_async(self) -> object:
            return object()

    class ActionClient:
        def __init__(self, *_args: object) -> None:
            pass

        def wait_for_server(self, timeout_sec: float) -> bool:
            return timeout_sec > 0.0

        def send_goal_async(self, _goal: object) -> object:
            return object()

    class Goal:
        trajectory = None

    class ActionType:
        pass

    ActionType.Goal = Goal

    class Node:
        arm_q = {"right": np.zeros(7)}

    class Duration:
        sec = 1
        nanosec = 0

    class Point:
        positions = np.zeros(7)
        time_from_start = Duration()

    class Trajectory:
        joint_names = [f"right_fr3_joint{index}" for index in range(1, 8)]
        points = [Point()]

    responses = iter([GoalHandle(), ResultWrapper()])
    monkeypatch.setattr("rclpy.action.ActionClient", ActionClient)
    monkeypatch.setattr(move, "spin_until_future", lambda *_args, **_kwargs: next(responses))

    assert not move.execute_joint_trajectory(
        object(), Node(), "right", Trajectory(), ActionType
    )


def test_forward_kinematics_applies_live_flange_to_ee_transform() -> None:
    model, frame_id = move.build_fr3_model()
    q = np.asarray([0.1, 0.5, 0.2, -1.5, -0.3, 1.2, 0.4])
    flange_to_ee = move.pose_vector_to_matrix([0, 0, 0.1034, 0, 0, -np.pi / 4])
    achieved = move.forward_end_effector_pose(model, frame_id, q, flange_to_ee)
    flange = move.forward_end_effector_pose(model, frame_id, q, np.eye(4))
    np.testing.assert_allclose(achieved, flange @ flange_to_ee)


def test_ik_recovers_a_known_reachable_pose_with_tool_transform() -> None:
    model, frame_id = move.build_fr3_model()
    known_q = np.asarray([0.2, 0.6, 0.5, -1.8, -0.5, 1.5, 0.3])
    flange_to_ee = move.pose_vector_to_matrix([0, 0, 0.1034, 0, 0, -np.pi / 4])
    target = move.forward_end_effector_pose(model, frame_id, known_q, flange_to_ee)
    seed = known_q + np.asarray([0.03, -0.02, 0.02, -0.03, 0.02, -0.02, 0.03])

    result = move.solve_fr3_ik(seed, target, flange_to_ee, model, frame_id)

    assert result.position_error_m <= move.IK_POSITION_TOLERANCE_M
    assert result.orientation_error_rad <= move.IK_ORIENTATION_TOLERANCE_RAD
    assert np.all(result.q >= move.ARM_POSITION_LOWER_RAD)
    assert np.all(result.q <= move.ARM_POSITION_UPPER_RAD)


def test_ik_recovers_pose_with_dependency_free_fr3_backend() -> None:
    from utils.fr3_kinematics import Fr3ForwardKinematics

    kinematics = Fr3ForwardKinematics(backend="numpy")
    known_q = np.asarray([0.2, 0.6, 0.5, -1.8, -0.5, 1.5, 0.3])
    flange_to_ee = move.pose_vector_to_matrix([0, 0, 0.1034, 0, 0, -np.pi / 4])
    target = kinematics.end_effector_pose(known_q, flange_to_ee)
    seed = known_q + np.asarray([0.03, -0.02, 0.02, -0.03, 0.02, -0.02, 0.03])

    result = move.solve_fr3_ik(
        seed,
        target,
        flange_to_ee,
        kinematics=kinematics,
        try_alternative_seeds=False,
        max_function_evaluations=100,
    )

    assert result.position_error_m <= move.IK_POSITION_TOLERANCE_M
    assert result.orientation_error_rad <= move.IK_ORIENTATION_TOLERANCE_RAD
    assert np.all(result.q >= move.ARM_POSITION_LOWER_RAD)
    assert np.all(result.q <= move.ARM_POSITION_UPPER_RAD)


def test_ik_rejects_unreachable_pose_before_transport() -> None:
    model, frame_id = move.build_fr3_model()
    seed = np.asarray([0.2, 0.6, 0.5, -1.8, -0.5, 1.5, 0.3])
    unreachable = move.pose_vector_to_matrix([0.99, 0.99, 1.1, 0, 0, 0])

    with pytest.raises(RuntimeError, match="No FR3 IK solution"):
        move.solve_fr3_ik(seed, unreachable, np.eye(4), model, frame_id)


def test_robot_pose_matrix_round_trip_preserves_robot_coordinates() -> None:
    pose = np.asarray([0.4, -0.2, 0.3, 0.2, -0.3, 0.4])
    matrix = move.pose_vector_to_matrix(pose)
    np.testing.assert_allclose(move.matrix_to_pose_vector(matrix), pose)
    np.testing.assert_allclose(move.matrix_to_pose_vector(matrix.reshape(16, order="F")), pose)


def test_approval_requires_explicit_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "yes")
    move.require_approval()

    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    with pytest.raises(SystemExit, match="cancelled"):
        move.require_approval()


def test_move_summary_prints_current_and_target_state(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args, targets = parse(
        "--left",
        "--gripper",
        "--target-ee-pose",
        "0.4,0.2,0.3,0,0,0",
        "--target-ee-joint",
        "0.04",
    )
    current = {
        "left": {
            "pose": np.asarray([0.3, 0.1, 0.2, 0.1, 0.2, 0.3]),
            "joint": np.asarray([0.02]),
        }
    }
    move.print_move_summary(args, targets, current)

    output = capsys.readouterr().out
    assert "left current ee pose" in output
    assert "left target  ee pose" in output
    assert "left current gripper width [m]" in output
    assert "left target  gripper width [m]" in output
