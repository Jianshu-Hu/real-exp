from __future__ import annotations

import numpy as np

from utils.limit import (
    FR3_SAFE_MAX_ACCELERATION_RAD_S2,
    FR3_SAFE_MAX_VELOCITY_RAD_S,
    FR3_SAFE_POSITION_LOWER_RAD,
    FR3_SAFE_POSITION_UPPER_RAD,
    JointPositionLimiter,
    validate_joint_trajectory,
)


def safe_midpoint() -> np.ndarray:
    return 0.5 * (FR3_SAFE_POSITION_LOWER_RAD + FR3_SAFE_POSITION_UPPER_RAD)


def test_trajectory_checks_each_eligible_timestep() -> None:
    positions = np.tile(safe_midpoint(), (5, 1))
    timestamps = np.arange(5, dtype=float) * 0.1

    positions[1:, 0] += np.array([0.05, 0.10, 0.30, 0.35])
    positions[4, 1] = FR3_SAFE_POSITION_UPPER_RAD[1] + 0.01

    counts = validate_joint_trajectory(positions, timestamps)

    assert counts.position_indices == (4,)
    assert counts.velocity_indices == (3, 4)
    assert counts.acceleration_indices == (3, 4)
    assert counts.any_indices == (3, 4)


def test_trajectory_accepts_safe_constant_velocity_at_every_timestep() -> None:
    positions = np.tile(safe_midpoint(), (5, 1))
    timestamps = np.arange(5, dtype=float) * 0.1
    positions[:, 0] += timestamps * (0.5 * FR3_SAFE_MAX_VELOCITY_RAD_S[0])

    counts = validate_joint_trajectory(positions, timestamps)

    assert counts.position_indices == ()
    assert counts.velocity_indices == ()
    assert counts.acceleration_indices == ()
    assert counts.any_indices == ()


def test_trajectory_acceleration_uses_velocity_midpoint_times() -> None:
    positions = np.tile(safe_midpoint(), (3, 1))
    timestamps = np.array([0.0, 0.1, 0.3])
    first_velocity = 0.2
    acceleration = FR3_SAFE_MAX_ACCELERATION_RAD_S2[0] + 0.1
    midpoint_dt = 0.5 * (0.1 + 0.2)
    second_velocity = first_velocity + acceleration * midpoint_dt
    positions[1, 0] += first_velocity * 0.1
    positions[2, 0] = positions[1, 0] + second_velocity * 0.2

    counts = validate_joint_trajectory(positions, timestamps)

    assert second_velocity < FR3_SAFE_MAX_VELOCITY_RAD_S[0]
    assert counts.velocity_indices == ()
    assert counts.acceleration_indices == (2,)


def test_trajectory_reports_bad_timing_separately_from_joint_limits() -> None:
    positions = np.tile(safe_midpoint(), (3, 1))
    timestamps = np.array([0.0, 0.0, 0.1])

    counts = validate_joint_trajectory(positions, timestamps)

    assert counts.timing_indices == (1,)
    assert counts.non_finite_indices == ()
    assert counts.velocity_indices == ()
    assert counts.acceleration_indices == ()


def test_position_limiter_brakes_at_a_constant_target_without_overshoot() -> None:
    dt = 1.0 / 15.0
    initial = safe_midpoint()
    target = initial.copy()
    target[0] += 0.02
    limiter = JointPositionLimiter(max_dt=2.0 * dt)
    limiter.reset(initial, 0.0)

    positions = [initial[0]]
    velocities = [0.0]
    for step in range(1, 31):
        result = limiter.filter(target, step * dt)
        positions.append(result.position[0])
        velocities.append(result.velocity[0])

    position_steps = np.diff(positions)
    accelerations = np.diff(velocities) / dt
    assert np.all(position_steps >= -1e-12)
    assert np.all(np.asarray(positions) <= target[0] + 1e-12)
    assert positions[-1] == target[0]
    assert velocities[-1] == 0.0
    assert np.max(np.abs(np.asarray(velocities))) <= (
        FR3_SAFE_MAX_VELOCITY_RAD_S[0] + 1e-12
    )
    assert np.max(np.abs(accelerations)) <= (
        FR3_SAFE_MAX_ACCELERATION_RAD_S2[0] + 1e-10
    )


def test_position_limiter_large_timestamp_gap_does_not_pass_small_step() -> None:
    dt = 1.0 / 15.0
    initial = safe_midpoint()
    target = initial.copy()
    target[0] += 0.02
    limiter = JointPositionLimiter(max_dt=2.0 * dt)
    limiter.reset(initial, 0.0)

    result = limiter.filter(target, 10.0)

    assert initial[0] < result.position[0] < target[0]
    assert result.velocity[0] <= FR3_SAFE_MAX_VELOCITY_RAD_S[0] + 1e-12
    assert result.velocity[0] <= (
        FR3_SAFE_MAX_ACCELERATION_RAD_S2[0] * limiter.max_dt + 1e-12
    )


def test_position_limiter_bounds_motion_when_target_reverses() -> None:
    dt = 1.0 / 15.0
    initial = safe_midpoint()
    forward_target = initial.copy()
    forward_target[0] += 0.5
    reverse_target = initial.copy()
    reverse_target[0] -= 0.5
    limiter = JointPositionLimiter(max_dt=2.0 * dt)
    limiter.reset(initial, 0.0)

    results = []
    for step in range(1, 6):
        results.append(limiter.filter(forward_target, step * dt))
    for step in range(6, 21):
        results.append(limiter.filter(reverse_target, step * dt))

    velocities = np.asarray([0.0, *(result.velocity[0] for result in results)])
    accelerations = np.diff(velocities) / dt
    assert np.max(np.abs(velocities)) <= FR3_SAFE_MAX_VELOCITY_RAD_S[0] + 1e-12
    assert np.max(np.abs(accelerations)) <= (
        FR3_SAFE_MAX_ACCELERATION_RAD_S2[0] + 1e-10
    )
    assert velocities[5] > 0.0
    assert np.any(velocities[6:] < 0.0)
