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


def test_duo_gripper_target_is_broadcast_to_left_and_right() -> None:
    _, targets = parse(
        "--duo",
        "--gripper",
        "--target-ee-pose",
        "0.4,0.2,0.3,0,0,0",
        "--target-ee-joint",
        "0.04",
    )
    assert [target.side for target in targets] == ["left", "right"]
    np.testing.assert_allclose(targets[0].end_effector_joint, [0.04])
    np.testing.assert_allclose(targets[1].end_effector_joint, [0.04])


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


def test_arm_ramp_bounds_velocity_and_acceleration_without_overshoot() -> None:
    q = np.zeros(7)
    velocity = np.zeros(7)
    target = np.asarray([1.0, -1.0, 0.5, -0.4, 0.3, 0.8, -0.7])
    dt = move.ARM_PUBLISH_PERIOD_S
    previous_velocity = velocity.copy()
    for _ in range(1000):
        q, velocity = move.ramp_arm_command(q, velocity, target, dt)
        assert np.max(np.abs(velocity)) <= move.ARM_MAX_VELOCITY_RAD_PER_S + 1e-12
        assert np.max(np.abs(velocity - previous_velocity)) <= (
            move.ARM_MAX_ACCELERATION_RAD_PER_S2 * dt + 1e-12
        )
        assert np.all(np.abs(q) <= np.abs(target) + 1e-12)
        previous_velocity = velocity.copy()
    np.testing.assert_allclose(q, target, atol=1e-9)


def test_arm_ramp_clamps_delayed_cycle_duration() -> None:
    q, velocity = move.ramp_arm_command(
        np.zeros(7), np.zeros(7), np.ones(7), dt=1.0
    )
    maximum_dt = 2.0 * move.ARM_PUBLISH_PERIOD_S
    np.testing.assert_allclose(
        velocity, np.full(7, move.ARM_MAX_ACCELERATION_RAD_PER_S2 * maximum_dt)
    )
    np.testing.assert_allclose(q, velocity * maximum_dt)


def test_arm_reached_uses_replay_position_tolerance() -> None:
    class Node:
        arm_q = {"right": np.full(7, 0.044)}
        arm_dq = {"right": np.zeros(7)}

    assert move.ARM_POSITION_TOLERANCE_RAD == pytest.approx(0.06)
    assert move.arm_reached(Node(), "right", np.zeros(7))

    Node.arm_q["right"] = np.full(7, 0.061)
    assert not move.arm_reached(Node(), "right", np.zeros(7))


def test_arm_reached_requires_low_velocity_inside_position_tolerance() -> None:
    class Node:
        arm_q = {"right": np.full(7, 0.044)}
        arm_dq = {"right": np.full(7, move.ARM_VELOCITY_TOLERANCE_RAD_PER_S + 0.001)}

    assert not move.arm_reached(Node(), "right", np.zeros(7))


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


def test_dry_run_reads_and_prints_current_state(
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
    move.print_dry_run(args, targets, current)

    output = capsys.readouterr().out
    assert "left current ee pose" in output
    assert "left target  ee pose" in output
    assert "left current gripper width [m]" in output
    assert "left target  gripper width [m]" in output
    assert "Dry run: no hardware motion was commanded." in output
