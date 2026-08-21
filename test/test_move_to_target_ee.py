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


def test_pose_interpolation_has_exact_endpoints() -> None:
    start = move.pose_vector_to_matrix([0, 0, 0, 0, 0, 0])
    target = move.pose_vector_to_matrix([1, 2, 3, 0.2, -0.3, 0.4])
    np.testing.assert_allclose(move.interpolate_pose(start, target, 0.0), start, atol=1e-12)
    np.testing.assert_allclose(move.interpolate_pose(start, target, 1.0), target, atol=1e-12)


def test_pose_settle_duration_is_positive() -> None:
    assert move.POSE_SETTLE_DURATION_S > 0.0
    assert move.POSE_SETTLE_TIMEOUT_S > move.POSE_SETTLE_DURATION_S


def test_cartesian_state_settled_checks_pose_and_velocity() -> None:
    target = move.pose_vector_to_matrix([0.4, 0.2, 0.3, 0.1, 0.2, 0.3])

    class State:
        O_T_EE = target.reshape(16, order="F")
        O_dP_EE = np.zeros(6)

    assert move.cartesian_state_is_settled(State(), target)
    State.O_dP_EE = np.asarray([0.01, 0, 0, 0, 0, 0])
    assert not move.cartesian_state_is_settled(State(), target)


def test_state_pose_matrix_prefers_commanded_pose() -> None:
    measured = move.pose_vector_to_matrix([0.1, 0.2, 0.3, 0, 0, 0])
    commanded = move.pose_vector_to_matrix([0.4, 0.2, 0.3, 0, 0, 0])

    class State:
        O_T_EE = measured.reshape(16, order="F")
        O_T_EE_c = commanded.reshape(16, order="F")

    np.testing.assert_allclose(move.state_pose_matrix(State()), commanded)


def test_cartesian_velocity_toward_pose_is_bounded() -> None:
    current = move.pose_vector_to_matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    target = move.pose_vector_to_matrix([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    velocity = move.cartesian_velocity_toward_pose(current, target, 1.0)
    assert velocity.shape == (6,)
    assert np.all(np.abs(velocity[:3]) <= move.MAX_TRANSLATION_SPEED_M_PER_S)
    assert np.all(np.abs(velocity[3:]) <= move.MAX_ROTATION_SPEED_RAD_PER_S)


def test_robot_pose_matrix_round_trip_preserves_robot_coordinates() -> None:
    pose = np.asarray([0.4, -0.2, 0.3, 0.2, -0.3, 0.4])
    matrix = move.pose_vector_to_matrix(pose)
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
    monkeypatch.setattr(move, "read_current_targets", lambda _args, _targets: current)

    move.print_dry_run(args, targets)

    output = capsys.readouterr().out
    assert "left current ee pose" in output
    assert "left target  ee pose" in output
    assert "left current gripper width [m]" in output
    assert "left target  gripper width [m]" in output
    assert "Dry run: no hardware motion was commanded." in output
