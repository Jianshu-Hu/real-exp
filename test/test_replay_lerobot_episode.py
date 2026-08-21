from __future__ import annotations

from queue import Queue
from typing import Any

import numpy as np

from data_collection.replay_lerobot_episode import (
    INITIAL_STATE_STABLE_SAMPLES,
    EpisodeData,
    arm_reached_initial_state,
    move_arms_to_initial_state,
    ramp_initial_state_command,
    wait_for_start,
)


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
    )


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
