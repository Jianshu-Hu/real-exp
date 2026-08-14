from __future__ import annotations

import numpy as np

from utils.limit import (
    FR3_SAFE_MAX_ACCELERATION_RAD_S2,
    FR3_SAFE_MAX_VELOCITY_RAD_S,
    FR3_SAFE_POSITION_LOWER_RAD,
    FR3_SAFE_POSITION_UPPER_RAD,
)
from utils.trajectory import QuinticJointTrajectory


def midpoint() -> np.ndarray:
    return 0.5 * (FR3_SAFE_POSITION_LOWER_RAD + FR3_SAFE_POSITION_UPPER_RAD)


def test_quintic_reaches_target_with_zero_terminal_derivatives() -> None:
    initial = midpoint()
    target = initial.copy()
    target[0] += 0.2
    trajectory = QuinticJointTrajectory()
    trajectory.reset(initial, 0.0)
    trajectory.update_target(target, 0.0)

    position = trajectory.sample(trajectory.duration_s)

    np.testing.assert_allclose(position, target, atol=1e-12)
    np.testing.assert_allclose(trajectory._velocity, 0.0, atol=1e-12)
    np.testing.assert_allclose(trajectory._acceleration, 0.0, atol=1e-12)


def test_quintic_respects_position_velocity_and_acceleration_limits() -> None:
    initial = midpoint()
    target = initial.copy()
    target[0] += 0.2
    trajectory = QuinticJointTrajectory()
    trajectory.reset(initial, 0.0)
    trajectory.update_target(target, 0.0)

    samples = [
        trajectory._sample_without_mutating(float(time_s))
        for time_s in np.linspace(0.0, trajectory.duration_s, 501)
    ]
    positions = np.asarray([sample[0] for sample in samples])
    velocities = np.asarray([sample[1] for sample in samples])
    accelerations = np.asarray([sample[2] for sample in samples])

    assert np.all(positions >= FR3_SAFE_POSITION_LOWER_RAD - 1e-10)
    assert np.all(positions <= FR3_SAFE_POSITION_UPPER_RAD + 1e-10)
    assert np.all(np.abs(velocities) <= FR3_SAFE_MAX_VELOCITY_RAD_S + 1e-9)
    assert np.all(np.abs(accelerations) <= FR3_SAFE_MAX_ACCELERATION_RAD_S2 + 1e-8)


def test_quintic_replanning_preserves_position_velocity_and_acceleration() -> None:
    initial = midpoint()
    first_target = initial.copy()
    first_target[0] += 0.4
    second_target = initial.copy()
    second_target[0] -= 0.2
    trajectory = QuinticJointTrajectory()
    trajectory.reset(initial, 0.0)
    trajectory.update_target(first_target, 0.0)

    replan_time = 0.2
    before = trajectory._sample_without_mutating(replan_time)
    trajectory.update_target(second_target, replan_time)
    after = trajectory._sample_without_mutating(replan_time)

    for expected, actual in zip(before, after, strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
