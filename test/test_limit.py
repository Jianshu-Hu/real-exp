from __future__ import annotations

import numpy as np

from utils.limit import (
    FR3_SAFE_MAX_ACCELERATION_RAD_S2,
    FR3_SAFE_MAX_VELOCITY_RAD_S,
    FR3_SAFE_POSITION_LOWER_RAD,
    FR3_SAFE_POSITION_UPPER_RAD,
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
