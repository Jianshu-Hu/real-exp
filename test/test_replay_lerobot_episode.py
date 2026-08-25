from __future__ import annotations

import pytest

from queue import Queue
from types import SimpleNamespace
from typing import Any

import numpy as np

from data_collection.replay_lerobot_episode import (
    INITIAL_STATE_STABLE_SAMPLES,
    EpisodeData,
    arm_reached_initial_state,
    build_replay_node_class,
    build_trace_row,
    move_arms_to_initial_state,
    parse_args,
    request_hand_status,
    ramp_initial_state_command,
    wait_for_start,
    trace_fieldnames,
)


class FakeRosNode:
    def __init__(self, name: str) -> None:
        self.name = name

    def create_publisher(self, *args: Any) -> object:
        del args
        return SimpleNamespace(publish=lambda message: None)

    def create_subscription(self, *args: Any) -> object:
        del args
        return object()


def make_replay_node_args(state_action_mode: str) -> SimpleNamespace:
    return SimpleNamespace(
        left_state_topic="/left/joint_states",
        right_state_topic="/right/joint_states",
        left_gripper_state_topic="/left/gripper/joint_states",
        right_gripper_state_topic="/right/gripper/joint_states",
        left_robot_state_topic="/left/robot_state",
        right_robot_state_topic="/right/robot_state",
        left_target_topic="/left/target_joint_states",
        right_target_topic="/right/target_joint_states",
        left_gripper_topic="/left/gripper/target",
        right_gripper_topic="/right/gripper/target",
        active_arms=["left", "right"],
        robot_end_effector="gripper",
        state_action_mode=state_action_mode,
    )


@pytest.mark.parametrize(
    ("state_action_mode", "expected_missing"),
    [
        ("joint", []),
        ("end_effector", ["/left/robot_state", "/right/robot_state"]),
    ],
)
def test_missing_state_topics_uses_node_replay_mode_after_initialization(
    state_action_mode: str,
    expected_missing: list[str],
) -> None:
    fake_message_type = object()
    ReplayNode = build_replay_node_class(
        FakeRosNode,
        fake_message_type,
        fake_message_type,
        fake_message_type,
    )
    node = ReplayNode(make_replay_node_args(state_action_mode))
    node.left_actual_q = np.zeros(7)
    node.right_actual_q = np.zeros(7)
    node.left_gripper_actual = 0.08
    node.right_gripper_actual = 0.08

    assert node.missing_state_topics(no_gripper=False) == expected_missing

    node.ee_pose_matrices = {"left": np.eye(4), "right": np.eye(4)}
    node.flange_to_ee = {"left": np.eye(4), "right": np.eye(4)}
    assert node.missing_state_topics(no_gripper=False) == []


def test_trace_compares_ee_ik_commands_with_recorded_and_actual_joints() -> None:
    recorded_state = np.arange(7, dtype=float)
    recorded_target = recorded_state + 0.1
    replay_target = recorded_target + 0.2
    actual = recorded_state + 0.05
    row = build_trace_row(
        elapsed_s=1.0,
        frame_index=3,
        dataset_timestamp=0.2,
        mode="action",
        target_source="action",
        left_recorded_state=recorded_state,
        right_recorded_state=recorded_state,
        left_recorded_target=recorded_target,
        right_recorded_target=recorded_target,
        left_target=replay_target,
        right_target=replay_target,
        left_actual=actual,
        right_actual=actual,
        left_gripper_target=None,
        right_gripper_target=None,
        left_gripper_actual=None,
        right_gripper_actual=None,
        abort_requested=False,
        controller_ready=True,
    )

    assert set(row) == set(trace_fieldnames())
    assert row["left_target_vs_recorded_target_max_abs_rad"] == pytest.approx(0.2)
    assert row["right_actual_vs_recorded_state_max_abs_rad"] == pytest.approx(0.05)


def test_request_hand_status_forwards_initial_target() -> None:
    class FakeSocket:
        def __init__(self) -> None:
            self.sent = None

        def send_pyobj(self, payload: object) -> None:
            self.sent = payload

        def recv_pyobj(self) -> dict[str, object]:
            return {"ready": True, "initial_received": True}

    class FakePoller:
        def register(self, socket: object, event: object) -> None:
            del socket, event

        def poll(self, timeout_ms: int) -> list[tuple[object, int]]:
            del timeout_ms
            return [(object(), 1)]

    import data_collection.replay_lerobot_episode as replay_module

    original_poller = replay_module.zmq.Poller
    replay_module.zmq.Poller = FakePoller
    socket = FakeSocket()
    try:
        response = request_hand_status(socket, {"kind": "initial", "target": [0.0] * 20})
    finally:
        replay_module.zmq.Poller = original_poller
    assert response["initial_received"] is True
    assert socket.sent == {"kind": "initial", "target": [0.0] * 20}


def make_episode_data() -> EpisodeData:
    state = np.asarray(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.08, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, 0.08]],
        dtype=float,
    )
    return EpisodeData(
        states=state,
        actions=np.zeros((1, 16), dtype=float),
        frame_indices=np.asarray([12], dtype=int),
        timestamps=np.asarray([0.0], dtype=float),
        fps=15.0,
        action_config={},
        joint_states=state.copy(),
        target_joints=state.copy(),
    )


def test_internal_wuji_worker_does_not_require_dataset_root() -> None:
    args = parse_args(["--internal-wuji-hand", "right", "--right-hand-command-port", "5562"])

    assert args.internal_wuji_hand == "right"
    assert args.dataset_root is None
    assert args.right_hand_command_port == 5562


def test_episode_replay_still_requires_dataset_root() -> None:
    with pytest.raises(SystemExit) as error:
        parse_args([])

    assert error.value.code == 2


def test_arm_reached_initial_state_checks_position_and_velocity() -> None:
    target = np.zeros(7, dtype=float)

    assert arm_reached_initial_state(np.full(7, 0.05), np.full(7, 0.01), target)
    assert not arm_reached_initial_state(np.full(7, 0.08), np.zeros(7), target)
    assert not arm_reached_initial_state(np.full(7, 0.05), np.zeros(7), target, position_tolerance_rad=0.04)
    assert not arm_reached_initial_state(np.zeros(7), np.full(7, 0.1), target)
    assert not arm_reached_initial_state(None, None, target)


def test_initial_state_command_respects_velocity_and_acceleration_limits() -> None:
    commanded_q = np.zeros(7, dtype=float)
    commanded_velocity = np.zeros(7, dtype=float)
    target_q = np.full(7, 0.3, dtype=float)
    positions = [commanded_q.copy()]

    for _ in range(250):
        commanded_q, commanded_velocity = ramp_initial_state_command(
            commanded_q,
            commanded_velocity,
            target_q,
            dt=0.02,
            max_velocity=0.1,
            max_acceleration=0.2,
        )
        positions.append(commanded_q.copy())

    finite_difference_velocity = np.diff(np.asarray(positions), axis=0) / 0.02
    finite_difference_acceleration = np.diff(
        np.vstack([np.zeros((1, 7)), finite_difference_velocity]),
        axis=0,
    ) / 0.02
    assert float(np.max(np.abs(finite_difference_velocity))) <= 0.1 + 1e-12
    assert float(np.max(np.abs(finite_difference_acceleration))) <= 0.2 + 1e-12
    assert float(np.max(np.asarray(positions))) <= 0.3


def test_move_arms_holds_first_observation_until_both_arms_are_stable() -> None:
    data = make_episode_data()

    class FakeNode:
        def __init__(self) -> None:
            self.left_actual_q = np.zeros(7, dtype=float)
            self.right_actual_q = np.zeros(7, dtype=float)
            self.left_actual_dq = np.zeros(7, dtype=float)
            self.right_actual_dq = np.zeros(7, dtype=float)
            self.published: list[tuple[np.ndarray, np.ndarray, None, None]] = []

        def publish_targets(
            self,
            left: np.ndarray,
            right: np.ndarray,
            left_gripper: None,
            right_gripper: None,
        ) -> None:
            self.published.append((left.copy(), right.copy(), left_gripper, right_gripper))

    node = FakeNode()

    class FakeRclpy:
        def ok(self) -> bool:
            return True

        def spin_once(self, unused_node: Any, timeout_sec: float) -> None:
            del unused_node, timeout_sec
            node.left_actual_q = data.states[0, 0:7].copy()
            node.right_actual_q = data.states[0, 8:15].copy()

    assert move_arms_to_initial_state(
        FakeRclpy(),
        node,
        Queue(),
        data,
        timeout_s=1.0,
        max_velocity=0.1,
        max_acceleration=0.2,
        prime_duration_s=0.0,
    )
    assert len(node.published) == INITIAL_STATE_STABLE_SAMPLES
    assert np.max(np.abs(node.published[0][0] - data.states[0, 0:7])) > 0.05
    assert np.max(np.abs(node.published[0][1] - data.states[0, 8:15])) > 0.05
    for left, right, left_gripper, right_gripper in node.published:
        assert np.all(np.abs(left) < np.abs(data.states[0, 0:7]))
        assert np.all(np.abs(right) < np.abs(data.states[0, 8:15]))
        assert left_gripper is None
        assert right_gripper is None


def test_move_arms_can_be_aborted_before_publishing() -> None:
    data = make_episode_data()
    commands: Queue[str] = Queue()
    commands.put("q")

    class FakeNode:
        left_actual_q = None
        right_actual_q = None
        left_actual_dq = None
        right_actual_dq = None

        def publish_targets(self, *args: Any) -> None:
            raise AssertionError("No target should be published after an abort request.")

    class FakeRclpy:
        def ok(self) -> bool:
            return True

    assert not move_arms_to_initial_state(
        FakeRclpy(),
        FakeNode(),
        commands,
        data,
        timeout_s=1.0,
        max_velocity=0.1,
        max_acceleration=0.2,
        prime_duration_s=0.0,
    )


def test_wait_for_start_primes_controllers_at_current_pose() -> None:
    commands: Queue[str] = Queue()
    commands.put("s")

    class FakeNode:
        left_actual_q = np.arange(7, dtype=float)
        right_actual_q = -np.arange(7, dtype=float)

        def __init__(self) -> None:
            self.published: list[tuple[np.ndarray, np.ndarray, None, None]] = []

        def publish_targets(
            self,
            left: np.ndarray,
            right: np.ndarray,
            left_gripper: None,
            right_gripper: None,
        ) -> None:
            self.published.append((left.copy(), right.copy(), left_gripper, right_gripper))

        def missing_state_topics(self, no_gripper: bool) -> list[str]:
            del no_gripper
            return []

        def controller_ready(self, no_gripper: bool) -> bool:
            del no_gripper
            return True

    class FakeRclpy:
        def ok(self) -> bool:
            return True

        def spin_once(self, node: Any, timeout_sec: float) -> None:
            del node, timeout_sec

    node = FakeNode()
    assert wait_for_start(FakeRclpy(), node, commands, no_gripper=True, allow_missing_state=False)
    assert len(node.published) == 1
    np.testing.assert_array_equal(node.published[0][0], node.left_actual_q)
    np.testing.assert_array_equal(node.published[0][1], node.right_actual_q)
