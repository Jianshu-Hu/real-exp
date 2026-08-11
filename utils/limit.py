"""Shared Franka FR3 joint safety limits and trajectory filtering.

The hardware constants mirror the installed Franka description and libfranka.
The command limits are intentionally more conservative: position bounds use
the narrower legacy FR3 range with an additional 0.05 rad margin so the same
envelope remains valid across robot system-image versions, while velocity and
acceleration match the controller's existing 0.2-speed startup trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

FR3_JOINT_COUNT = 7

# Nominal hardware ceilings from franka_description/robots/fr3/joint_limits.yaml
# and libfranka's franka/rate_limiting.h in the installed Franka workspace.
FR3_HARD_POSITION_LOWER_RAD = np.array(
    [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508],
    dtype=np.float64,
)
FR3_HARD_POSITION_UPPER_RAD = np.array(
    [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508],
    dtype=np.float64,
)
FR3_HARD_MAX_VELOCITY_RAD_S = np.array(
    [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26],
    dtype=np.float64,
)
FR3_HARD_MAX_ACCELERATION_RAD_S2 = np.full(
    FR3_JOINT_COUNT,
    10.0,
    dtype=np.float64,
)

# Older FR3 system images expose this narrower position range. Keeping the
# operational envelope inside it makes recorded data portable across versions.
FR3_LEGACY_POSITION_LOWER_RAD = np.array(
    [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159],
    dtype=np.float64,
)
FR3_LEGACY_POSITION_UPPER_RAD = np.array(
    [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159],
    dtype=np.float64,
)
FR3_POSITION_MARGIN_RAD = 0.05
FR3_SAFE_POSITION_LOWER_RAD = FR3_LEGACY_POSITION_LOWER_RAD + FR3_POSITION_MARGIN_RAD
FR3_SAFE_POSITION_UPPER_RAD = FR3_LEGACY_POSITION_UPPER_RAD - FR3_POSITION_MARGIN_RAD

# Use a 0.5 operational speed factor on the local MotionGenerator's base
# velocity limits. The independently selected acceleration limit remains 2.0 rad/s^2.
FR3_CONTROLLER_BASE_MAX_VELOCITY_RAD_S = np.array(
    [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5],
    dtype=np.float64,
)
FR3_SAFE_SPEED_FACTOR = 0.7
FR3_SAFE_MAX_VELOCITY_RAD_S = (
    FR3_CONTROLLER_BASE_MAX_VELOCITY_RAD_S * FR3_SAFE_SPEED_FACTOR
)
FR3_SAFE_MAX_ACCELERATION_RAD_S2 = 5.0 * np.ones(
    FR3_JOINT_COUNT,
    dtype=np.float64,
)

SAFETY_TOLERANCE = 1e-9


@dataclass(frozen=True)
class JointLimitViolation:
    """Per-joint violations found in one target update."""

    non_finite: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    @property
    def any(self) -> bool:
        return bool(
            np.any(self.non_finite)
            or np.any(self.position)
            or np.any(self.velocity)
            or np.any(self.acceleration)
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in ("non_finite", "position", "velocity", "acceleration")
            if np.any(getattr(self, name))
        )


@dataclass(frozen=True)
class JointLimitResult:
    """A safe target and the constraints violated by the unfiltered target."""

    position: np.ndarray
    velocity: np.ndarray
    violation: JointLimitViolation

    @property
    def clipped(self) -> bool:
        return self.violation.any


@dataclass(frozen=True)
class TrajectoryViolationCounts:
    """Number of trajectory steps violating each safety constraint."""

    non_finite_steps: int
    position_steps: int
    velocity_steps: int
    acceleration_steps: int
    any_steps: int

    @property
    def any(self) -> bool:
        return self.any_steps > 0


class SustainedViolationMonitor:
    """Latch after a constraint remains violated for a continuous duration."""

    def __init__(self, stop_after_s: float = 1.0) -> None:
        if stop_after_s <= 0.0:
            raise ValueError("stop_after_s must be positive.")
        self.stop_after_s = float(stop_after_s)
        self._violation_start: float | None = None

    @property
    def violation_start(self) -> float | None:
        return self._violation_start

    def update(self, violated: bool, timestamp: float) -> bool:
        timestamp = float(timestamp)
        if not violated:
            self._violation_start = None
            return False
        if self._violation_start is None or timestamp < self._violation_start:
            self._violation_start = timestamp
        return timestamp - self._violation_start >= self.stop_after_s

    def duration(self, timestamp: float) -> float:
        if self._violation_start is None:
            return 0.0
        return max(0.0, float(timestamp) - self._violation_start)


def _joint_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (FR3_JOINT_COUNT,):
        raise ValueError(
            f"{name} must have shape ({FR3_JOINT_COUNT},), got {result.shape}."
        )
    return result


def arm_joint_slices(vector_size: int) -> tuple[tuple[str, slice], ...]:
    """Return arm slices for supported single- and dual-arm LeRobot layouts."""
    layouts = {
        7: (("left", slice(0, 7)),),
        8: (("left", slice(0, 7)),),
        14: (("left", slice(0, 7)), ("right", slice(7, 14))),
        16: (("left", slice(0, 7)), ("right", slice(8, 15))),
    }
    try:
        return layouts[int(vector_size)]
    except KeyError as exc:
        raise ValueError(
            "Expected a 7/8-D single-arm or 14/16-D dual-arm vector, "
            f"got {vector_size} dimensions."
        ) from exc


class JointPositionLimiter:
    """Stateful position-target limiter with velocity and acceleration bounds."""

    def __init__(self, *, max_dt: float = 0.1) -> None:
        if max_dt <= 0.0:
            raise ValueError("max_dt must be positive.")
        self.max_dt = float(max_dt)
        self._position: np.ndarray | None = None
        self._velocity = np.zeros(FR3_JOINT_COUNT, dtype=np.float64)
        self._last_input_position: np.ndarray | None = None
        self._last_input_velocity = np.zeros(FR3_JOINT_COUNT, dtype=np.float64)
        self._timestamp: float | None = None

    @property
    def initialized(self) -> bool:
        return self._position is not None

    def reset(self, position: Sequence[float] | np.ndarray, timestamp: float) -> None:
        position_array = _joint_vector(position, "position")
        if not np.all(np.isfinite(position_array)):
            raise ValueError(
                "Cannot initialize joint limiter from non-finite positions."
            )
        self._position = np.clip(
            position_array,
            FR3_SAFE_POSITION_LOWER_RAD,
            FR3_SAFE_POSITION_UPPER_RAD,
        )
        self._velocity = np.zeros(FR3_JOINT_COUNT, dtype=np.float64)
        self._last_input_position = self._position.copy()
        self._last_input_velocity = np.zeros(FR3_JOINT_COUNT, dtype=np.float64)
        self._timestamp = float(timestamp)

    def filter(
        self,
        target: Sequence[float] | np.ndarray,
        timestamp: float,
        *,
        initial_position: Sequence[float] | np.ndarray | None = None,
    ) -> JointLimitResult:
        raw_target = _joint_vector(target, "target")
        timestamp = float(timestamp)

        if not self.initialized:
            reference = (
                raw_target
                if initial_position is None
                else _joint_vector(initial_position, "initial_position")
            )
            finite_reference = np.where(
                np.isfinite(reference),
                reference,
                0.5 * (FR3_SAFE_POSITION_LOWER_RAD + FR3_SAFE_POSITION_UPPER_RAD),
            )
            self.reset(finite_reference, timestamp)

        assert self._position is not None
        assert self._last_input_position is not None
        assert self._timestamp is not None

        non_finite = ~np.isfinite(raw_target)
        finite_input = np.where(non_finite, self._last_input_position, raw_target)
        filter_target = np.where(non_finite, self._position, raw_target)
        position_violation = np.logical_or(
            finite_input < FR3_SAFE_POSITION_LOWER_RAD - SAFETY_TOLERANCE,
            finite_input > FR3_SAFE_POSITION_UPPER_RAD + SAFETY_TOLERANCE,
        )
        position_violation = np.logical_and(position_violation, ~non_finite)
        bounded_target = np.clip(
            filter_target,
            FR3_SAFE_POSITION_LOWER_RAD,
            FR3_SAFE_POSITION_UPPER_RAD,
        )

        elapsed = timestamp - self._timestamp
        dt = min(max(elapsed, 1e-6), self.max_dt)
        input_velocity = (finite_input - self._last_input_position) / dt
        input_velocity = np.where(
            non_finite,
            self._last_input_velocity,
            input_velocity,
        )
        velocity_violation = (
            np.abs(input_velocity) > FR3_SAFE_MAX_VELOCITY_RAD_S + SAFETY_TOLERANCE
        )
        velocity_violation = np.logical_and(velocity_violation, ~non_finite)
        input_acceleration = (input_velocity - self._last_input_velocity) / dt
        acceleration_violation = (
            np.abs(input_acceleration)
            > FR3_SAFE_MAX_ACCELERATION_RAD_S2 + SAFETY_TOLERANCE
        )
        acceleration_violation = np.logical_and(acceleration_violation, ~non_finite)

        requested_velocity = (bounded_target - self._position) / dt
        desired_velocity = np.clip(
            requested_velocity,
            -FR3_SAFE_MAX_VELOCITY_RAD_S,
            FR3_SAFE_MAX_VELOCITY_RAD_S,
        )

        # Reserve enough distance to decelerate before reaching either position bound.
        distance_to_upper = np.maximum(
            FR3_SAFE_POSITION_UPPER_RAD - self._position, 0.0
        )
        distance_to_lower = np.maximum(
            self._position - FR3_SAFE_POSITION_LOWER_RAD, 0.0
        )
        acceleration = FR3_SAFE_MAX_ACCELERATION_RAD_S2
        upper_braking_velocity = -acceleration * dt + np.sqrt(
            np.square(acceleration * dt) + 2.0 * acceleration * distance_to_upper
        )
        lower_braking_velocity = acceleration * dt - np.sqrt(
            np.square(acceleration * dt) + 2.0 * acceleration * distance_to_lower
        )
        desired_velocity = np.clip(
            desired_velocity,
            np.maximum(-FR3_SAFE_MAX_VELOCITY_RAD_S, lower_braking_velocity),
            np.minimum(FR3_SAFE_MAX_VELOCITY_RAD_S, upper_braking_velocity),
        )

        velocity_step = np.clip(
            desired_velocity - self._velocity,
            -FR3_SAFE_MAX_ACCELERATION_RAD_S2 * dt,
            FR3_SAFE_MAX_ACCELERATION_RAD_S2 * dt,
        )
        safe_velocity = np.clip(
            self._velocity + velocity_step,
            -FR3_SAFE_MAX_VELOCITY_RAD_S,
            FR3_SAFE_MAX_VELOCITY_RAD_S,
        )
        safe_position = np.clip(
            self._position + safe_velocity * dt,
            FR3_SAFE_POSITION_LOWER_RAD,
            FR3_SAFE_POSITION_UPPER_RAD,
        )
        safe_velocity = (safe_position - self._position) / dt

        self._position = safe_position
        self._velocity = safe_velocity
        self._last_input_position = finite_input
        self._last_input_velocity = input_velocity
        self._timestamp = timestamp

        return JointLimitResult(
            position=safe_position.copy(),
            velocity=safe_velocity.copy(),
            violation=JointLimitViolation(
                non_finite=non_finite,
                position=position_violation,
                velocity=velocity_violation,
                acceleration=acceleration_violation,
            ),
        )


def validate_joint_trajectory(
    positions: Sequence[Sequence[float]] | np.ndarray,
    timestamps: Sequence[float] | np.ndarray,
) -> TrajectoryViolationCounts:
    """Count unsafe steps in a sampled 7-DoF absolute-position trajectory."""
    position_array = np.asarray(positions, dtype=np.float64)
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    if position_array.ndim != 2 or position_array.shape[1] != FR3_JOINT_COUNT:
        raise ValueError(
            f"positions must have shape (steps, {FR3_JOINT_COUNT}), got {position_array.shape}."
        )
    if timestamp_array.shape != (position_array.shape[0],):
        raise ValueError(
            f"timestamps must have shape ({position_array.shape[0]},), got {timestamp_array.shape}."
        )
    if position_array.shape[0] == 0:
        return TrajectoryViolationCounts(0, 0, 0, 0, 0)

    finite_position = np.all(np.isfinite(position_array), axis=1)
    finite_timestamp = np.isfinite(timestamp_array)
    non_finite_steps = np.logical_not(np.logical_and(finite_position, finite_timestamp))
    sanitized = np.where(np.isfinite(position_array), position_array, 0.0)
    position_steps = np.any(
        np.logical_or(
            sanitized < FR3_SAFE_POSITION_LOWER_RAD - SAFETY_TOLERANCE,
            sanitized > FR3_SAFE_POSITION_UPPER_RAD + SAFETY_TOLERANCE,
        ),
        axis=1,
    )
    position_steps = np.logical_and(position_steps, finite_position)

    velocity_steps = np.zeros(position_array.shape[0], dtype=bool)
    acceleration_steps = np.zeros(position_array.shape[0], dtype=bool)
    if position_array.shape[0] >= 2:
        dt = np.diff(timestamp_array)
        valid_velocity = np.logical_and.reduce(
            (dt > 0.0, np.isfinite(dt), finite_position[:-1], finite_position[1:])
        )
        velocities = np.zeros(
            (position_array.shape[0] - 1, FR3_JOINT_COUNT), dtype=np.float64
        )
        velocities[valid_velocity] = (
            np.diff(sanitized, axis=0)[valid_velocity] / dt[valid_velocity, None]
        )
        velocity_steps[1:] = np.logical_or(
            ~valid_velocity,
            np.any(
                np.abs(velocities) > FR3_SAFE_MAX_VELOCITY_RAD_S + SAFETY_TOLERANCE,
                axis=1,
            ),
        )

        if position_array.shape[0] >= 3:
            acceleration_dt = dt[1:]
            valid_acceleration = np.logical_and.reduce(
                (valid_velocity[:-1], valid_velocity[1:], acceleration_dt > 0.0)
            )
            accelerations = np.zeros(
                (position_array.shape[0] - 2, FR3_JOINT_COUNT),
                dtype=np.float64,
            )
            accelerations[valid_acceleration] = (
                np.diff(velocities, axis=0)[valid_acceleration]
                / acceleration_dt[valid_acceleration, None]
            )
            acceleration_steps[2:] = np.logical_or(
                ~valid_acceleration,
                np.any(
                    np.abs(accelerations)
                    > FR3_SAFE_MAX_ACCELERATION_RAD_S2 + SAFETY_TOLERANCE,
                    axis=1,
                ),
            )

    any_steps = np.logical_or.reduce(
        (non_finite_steps, position_steps, velocity_steps, acceleration_steps)
    )
    return TrajectoryViolationCounts(
        non_finite_steps=int(np.count_nonzero(non_finite_steps)),
        position_steps=int(np.count_nonzero(position_steps)),
        velocity_steps=int(np.count_nonzero(velocity_steps)),
        acceleration_steps=int(np.count_nonzero(acceleration_steps)),
        any_steps=int(np.count_nonzero(any_steps)),
    )
