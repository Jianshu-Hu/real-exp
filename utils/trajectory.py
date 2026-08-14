"""High-rate, state-continuous joint reference generation.

The policy and bridge command socket may run at a low rate (normally 15 Hz),
while the ROS controller consumes position references at 1 kHz.  This module
turns each absolute joint-position waypoint into a quintic segment whose
position, velocity, and acceleration are continuous when a waypoint changes.
The trajectory duration is increased until the configured velocity and
acceleration limits are satisfied.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from utils.limit import (
    FR3_JOINT_COUNT,
    FR3_SAFE_MAX_ACCELERATION_RAD_S2,
    FR3_SAFE_MAX_VELOCITY_RAD_S,
    FR3_SAFE_POSITION_LOWER_RAD,
    FR3_SAFE_POSITION_UPPER_RAD,
)


def _joint_vector(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (FR3_JOINT_COUNT,):
        raise ValueError(
            f"{name} must have shape ({FR3_JOINT_COUNT},), got {result.shape}."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _quintic_coefficients(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    target: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Return coefficients c[ joint, degree ] for q(t) = sum(c_i t**i)."""
    t = float(duration)
    coefficients = np.zeros((FR3_JOINT_COUNT, 6), dtype=np.float64)
    coefficients[:, 0] = position
    coefficients[:, 1] = velocity
    coefficients[:, 2] = 0.5 * acceleration

    # Solve the three terminal constraints for c3, c4, c5.
    matrix = np.array(
        [
            [t**3, t**4, t**5],
            [3.0 * t**2, 4.0 * t**3, 5.0 * t**4],
            [6.0 * t, 12.0 * t**2, 20.0 * t**3],
        ],
        dtype=np.float64,
    )
    rhs = np.stack(
        [
            target - (coefficients[:, 0] + coefficients[:, 1] * t + coefficients[:, 2] * t**2),
            -(coefficients[:, 1] + 2.0 * coefficients[:, 2] * t),
            -2.0 * coefficients[:, 2],
        ],
        axis=0,
    )
    coefficients[:, 3:] = np.linalg.solve(matrix, rhs).T
    return coefficients


def _polynomial_extrema(coefficients: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """Return exact sampled maxima of |velocity| and |acceleration|."""
    max_velocity = np.zeros(FR3_JOINT_COUNT, dtype=np.float64)
    max_acceleration = np.zeros(FR3_JOINT_COUNT, dtype=np.float64)
    for joint in range(FR3_JOINT_COUNT):
        c = coefficients[joint]
        velocity_coefficients = np.array([c[1], 2.0 * c[2], 3.0 * c[3], 4.0 * c[4], 5.0 * c[5]])
        acceleration_coefficients = np.array(
            [2.0 * c[2], 6.0 * c[3], 12.0 * c[4], 20.0 * c[5]]
        )
        velocity_roots = np.roots(velocity_coefficients[::-1])
        acceleration_roots = np.roots(acceleration_coefficients[::-1])
        velocity_times = [0.0, duration] + [
            float(root.real)
            for root in velocity_roots
            if abs(root.imag) < 1e-8 and 0.0 < root.real < duration
        ]
        acceleration_times = [0.0, duration] + [
            float(root.real)
            for root in acceleration_roots
            if abs(root.imag) < 1e-8 and 0.0 < root.real < duration
        ]
        velocity_values = np.polyval(velocity_coefficients[::-1], velocity_times)
        acceleration_values = np.polyval(acceleration_coefficients[::-1], acceleration_times)
        max_velocity[joint] = np.max(np.abs(velocity_values))
        max_acceleration[joint] = np.max(np.abs(acceleration_values))
    return max_velocity, max_acceleration


class QuinticJointTrajectory:
    """State-continuous, per-joint quintic trajectory generator."""

    def __init__(
        self,
        *,
        max_velocity: Sequence[float] | np.ndarray = FR3_SAFE_MAX_VELOCITY_RAD_S,
        max_acceleration: Sequence[float] | np.ndarray = FR3_SAFE_MAX_ACCELERATION_RAD_S2,
        minimum_duration_s: float = 0.1,
    ) -> None:
        self.max_velocity = _joint_vector(max_velocity, "max_velocity")
        self.max_acceleration = _joint_vector(max_acceleration, "max_acceleration")
        if np.any(self.max_velocity <= 0.0) or np.any(self.max_acceleration <= 0.0):
            raise ValueError("Trajectory limits must be positive.")
        if minimum_duration_s <= 0.0:
            raise ValueError("minimum_duration_s must be positive.")
        self.minimum_duration_s = float(minimum_duration_s)
        self._position: np.ndarray | None = None
        self._velocity = np.zeros(FR3_JOINT_COUNT, dtype=np.float64)
        self._acceleration = np.zeros(FR3_JOINT_COUNT, dtype=np.float64)
        self._target: np.ndarray | None = None
        self._coefficients: np.ndarray | None = None
        self._start_time_s: float | None = None
        self._duration_s = 0.0

    @property
    def initialized(self) -> bool:
        return self._position is not None

    @property
    def target(self) -> np.ndarray | None:
        return None if self._target is None else self._target.copy()

    @property
    def duration_s(self) -> float:
        return self._duration_s

    def reset(self, position: Sequence[float] | np.ndarray, timestamp_s: float) -> None:
        position_array = np.clip(
            _joint_vector(position, "position"),
            FR3_SAFE_POSITION_LOWER_RAD,
            FR3_SAFE_POSITION_UPPER_RAD,
        )
        self._position = position_array.copy()
        self._velocity.fill(0.0)
        self._acceleration.fill(0.0)
        self._target = position_array.copy()
        self._coefficients = None
        self._start_time_s = float(timestamp_s)
        self._duration_s = 0.0

    def _sample_without_mutating(self, timestamp_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self._position is not None
        if self._coefficients is None or self._start_time_s is None:
            return self._position.copy(), self._velocity.copy(), self._acceleration.copy()
        t = min(max(float(timestamp_s) - self._start_time_s, 0.0), self._duration_s)
        c = self._coefficients
        powers = np.array([1.0, t, t**2, t**3, t**4, t**5])
        velocity_powers = np.array([0.0, 1.0, 2.0 * t, 3.0 * t**2, 4.0 * t**3, 5.0 * t**4])
        acceleration_powers = np.array([0.0, 0.0, 2.0, 6.0 * t, 12.0 * t**2, 20.0 * t**3])
        return c @ powers, c @ velocity_powers, c @ acceleration_powers

    def sample(self, timestamp_s: float) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Trajectory must be reset before sampling.")
        position, velocity, acceleration = self._sample_without_mutating(timestamp_s)
        self._position = position
        self._velocity = velocity
        self._acceleration = acceleration
        if self._start_time_s is not None and float(timestamp_s) >= self._start_time_s + self._duration_s:
            assert self._target is not None
            self._position = self._target.copy()
            self._velocity.fill(0.0)
            self._acceleration.fill(0.0)
            self._coefficients = None
            self._duration_s = 0.0
        return self._position.copy()

    def update_target(self, target: Sequence[float] | np.ndarray, timestamp_s: float) -> None:
        if not self.initialized:
            self.reset(target, timestamp_s)
            return
        target_array = np.clip(
            _joint_vector(target, "target"),
            FR3_SAFE_POSITION_LOWER_RAD,
            FR3_SAFE_POSITION_UPPER_RAD,
        )
        current_position, current_velocity, current_acceleration = self._sample_without_mutating(timestamp_s)
        self._position = current_position
        self._velocity = current_velocity
        self._acceleration = current_acceleration
        if np.allclose(target_array, self._target, rtol=0.0, atol=1e-7):
            return
        self._target = target_array.copy()
        distance = np.abs(target_array - current_position)
        duration = max(
            self.minimum_duration_s,
            float(np.max(1.875 * distance / self.max_velocity)),
            float(np.max(np.sqrt(5.773503 * distance / self.max_acceleration))),
        )
        for _ in range(40):
            coefficients = _quintic_coefficients(
                current_position,
                current_velocity,
                current_acceleration,
                target_array,
                duration,
            )
            peak_velocity, peak_acceleration = _polynomial_extrema(coefficients, duration)
            if np.all(peak_velocity <= self.max_velocity * 0.98) and np.all(
                peak_acceleration <= self.max_acceleration * 0.98
            ):
                self._coefficients = coefficients
                self._start_time_s = float(timestamp_s)
                self._duration_s = duration
                return
            duration *= 1.25
        raise RuntimeError("Could not construct a quintic trajectory within configured limits.")
