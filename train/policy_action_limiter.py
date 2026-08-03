from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time

import numpy as np


FR3_JOINT_LOWER_RAD = np.array(
    [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508],
    dtype=float,
)
FR3_JOINT_UPPER_RAD = np.array(
    [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508],
    dtype=float,
)
FR3_MAX_VELOCITY_RAD_S = np.array(
    [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26],
    dtype=float,
)
FR3_MAX_ACCELERATION_RAD_S2 = np.full(7, 10.0, dtype=float)

DEFAULT_LIMIT_SAFETY_FACTOR = 0.8
DEFAULT_POSITION_MARGIN_RAD = 0.02
MAX_INTERPOLATION_STEPS = 10_000
POLICY_START_MAX_VELOCITY_RAD_S = np.array(
    [0.35, 0.35, 0.35, 0.35, 0.50, 0.50, 0.50],
    dtype=float,
)
POLICY_START_MAX_ACCELERATION_RAD_S2 = np.array(
    [0.80, 0.80, 0.80, 0.80, 1.20, 1.20, 1.20],
    dtype=float,
)
POLICY_START_POSITION_TOLERANCE_RAD = np.array(
    [0.03, 0.03, 0.03, 0.03, 0.04, 0.05, 0.07],
    dtype=float,
)


def add_policy_action_limit_arguments(parser: argparse.ArgumentParser) -> None:
    limit_group = parser.add_mutually_exclusive_group()
    limit_group.add_argument(
        "--limit",
        dest="limit",
        action="store_true",
        help=(
            "Enable conservative FR3 joint position, velocity, and acceleration "
            "limits. This is the default and the flag is retained for compatibility."
        ),
    )
    limit_group.add_argument(
        "--no-limit",
        dest="limit",
        action="store_false",
        help="Disable policy action limiting and send valid targets without time-stretching.",
    )
    parser.set_defaults(limit=True)


@dataclass(frozen=True)
class LimitedArmStep:
    left_joint_target: list[float]
    right_joint_target: list[float]
    step_index: int
    step_count: int

    @property
    def is_final(self) -> bool:
        return self.step_index == self.step_count


@dataclass(frozen=True)
class LimitedActionPlan:
    steps: tuple[LimitedArmStep, ...]
    source_duration_s: float
    limited_duration_s: float
    max_velocity_ratio: float
    max_acceleration_ratio: float

    @property
    def was_stretched(self) -> bool:
        return len(self.steps) > 1


@dataclass(frozen=True)
class PolicyStartAlignmentUpdate:
    step: LimitedArmStep
    send_grippers: bool
    max_position_error_rad: float | None
    alignment_finished: bool


def arm_joints_from_robot_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"Expected a one-dimensional robot vector, got shape {values.shape}.")
    if values.shape[0] == 16:
        arms = np.stack((values[0:7], values[8:15]))
    elif values.shape[0] == 14:
        arms = np.stack((values[0:7], values[7:14]))
    else:
        raise ValueError(f"Expected a 14D or 16D robot vector, got {values.shape[0]}D.")
    if not np.all(np.isfinite(arms)):
        raise ValueError("Robot vector contains NaN or infinity in an arm joint.")
    return arms


class PolicyActionLimiter:
    def __init__(
        self,
        fps: int,
        *,
        safety_factor: float = DEFAULT_LIMIT_SAFETY_FACTOR,
        position_margin_rad: float = DEFAULT_POSITION_MARGIN_RAD,
        max_velocity_rad_s: np.ndarray | None = None,
        max_acceleration_rad_s2: np.ndarray | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")
        if not 0.0 < safety_factor <= 1.0:
            raise ValueError(
                f"safety_factor must be in (0, 1], got {safety_factor}."
            )
        if position_margin_rad < 0.0:
            raise ValueError(
                f"position_margin_rad must be non-negative, got {position_margin_rad}."
            )

        self.fps = fps
        self.dt = 1.0 / fps
        self.safety_factor = safety_factor
        self.position_margin_rad = position_margin_rad
        self.max_velocity_rad_s = self._resolve_limit(
            "max_velocity_rad_s",
            max_velocity_rad_s,
            FR3_MAX_VELOCITY_RAD_S * safety_factor,
        )
        self.max_acceleration_rad_s2 = self._resolve_limit(
            "max_acceleration_rad_s2",
            max_acceleration_rad_s2,
            FR3_MAX_ACCELERATION_RAD_S2 * safety_factor,
        )

    def config_dict(self) -> dict[str, object]:
        return {
            "enabled": True,
            "method": "quintic_time_stretch",
            "safety_factor": self.safety_factor,
            "position_margin_rad": self.position_margin_rad,
            "max_velocity_rad_s": self.max_velocity_rad_s.tolist(),
            "max_acceleration_rad_s2": self.max_acceleration_rad_s2.tolist(),
        }

    @staticmethod
    def _resolve_limit(
        label: str,
        override: np.ndarray | None,
        default: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(default if override is None else override, dtype=float)
        if values.shape != (7,):
            raise ValueError(f"{label} must contain seven joint limits, got shape {values.shape}.")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"{label} must contain finite positive values.")
        return values.copy()

    def plan(
        self,
        current_robot_state: np.ndarray,
        policy_action: np.ndarray,
    ) -> LimitedActionPlan:
        start = arm_joints_from_robot_vector(current_robot_state)
        target = arm_joints_from_robot_vector(policy_action)
        self._validate_position("current robot state", start, use_margin=False)
        self._validate_position("policy target", target, use_margin=True)

        delta = target - start
        velocity_lower_bound = np.max(
            np.abs(delta) / (self.max_velocity_rad_s[np.newaxis, :] * self.dt)
        )
        first_step_count = max(1, math.ceil(float(velocity_lower_bound)))

        for step_count in range(first_step_count, MAX_INTERPOLATION_STEPS + 1):
            positions = self._quintic_positions(start, delta, step_count)
            max_velocity_ratio, max_acceleration_ratio = self._limit_ratios(
                start, positions
            )
            if max_velocity_ratio <= 1.0 + 1e-12 and max_acceleration_ratio <= 1.0 + 1e-12:
                steps = tuple(
                    LimitedArmStep(
                        left_joint_target=position[0].tolist(),
                        right_joint_target=position[1].tolist(),
                        step_index=index,
                        step_count=step_count,
                    )
                    for index, position in enumerate(positions, start=1)
                )
                return LimitedActionPlan(
                    steps=steps,
                    source_duration_s=self.dt,
                    limited_duration_s=step_count * self.dt,
                    max_velocity_ratio=max_velocity_ratio,
                    max_acceleration_ratio=max_acceleration_ratio,
                )

        raise RuntimeError(
            "Could not construct a joint trajectory inside the configured limits "
            f"within {MAX_INTERPOLATION_STEPS} control steps."
        )

    @staticmethod
    def _quintic_positions(
        start: np.ndarray,
        delta: np.ndarray,
        step_count: int,
    ) -> np.ndarray:
        phase = np.arange(1, step_count + 1, dtype=float) / step_count
        blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
        positions = start[np.newaxis, :, :] + blend[:, np.newaxis, np.newaxis] * delta
        positions[-1] = start + delta
        return positions

    def _limit_ratios(
        self,
        start: np.ndarray,
        positions: np.ndarray,
    ) -> tuple[float, float]:
        points = np.concatenate((start[np.newaxis, :, :], positions), axis=0)
        velocities = np.diff(points, axis=0) / self.dt
        zero_velocity = np.zeros((1, 2, 7), dtype=float)
        padded_velocities = np.concatenate(
            (zero_velocity, velocities, zero_velocity), axis=0
        )
        accelerations = np.diff(padded_velocities, axis=0) / self.dt

        velocity_ratio = np.max(
            np.abs(velocities) / self.max_velocity_rad_s[np.newaxis, np.newaxis, :]
        )
        acceleration_ratio = np.max(
            np.abs(accelerations)
            / self.max_acceleration_rad_s2[np.newaxis, np.newaxis, :]
        )
        return float(velocity_ratio), float(acceleration_ratio)

    def _validate_position(
        self,
        label: str,
        positions: np.ndarray,
        *,
        use_margin: bool,
    ) -> None:
        margin = self.position_margin_rad if use_margin else 0.0
        lower = FR3_JOINT_LOWER_RAD + margin
        upper = FR3_JOINT_UPPER_RAD - margin
        invalid = (positions < lower[np.newaxis, :]) | (
            positions > upper[np.newaxis, :]
        )
        if not np.any(invalid):
            return

        arm_index, joint_index = np.argwhere(invalid)[0]
        arm_name = "left" if arm_index == 0 else "right"
        value = positions[arm_index, joint_index]
        raise ValueError(
            f"{label} is outside the allowed FR3 range: {arm_name} joint "
            f"{joint_index + 1}={value:.6f} rad, allowed "
            f"[{lower[joint_index]:.6f}, {upper[joint_index]:.6f}] rad."
        )


class PolicyStartAligner:
    WAITING_FIRST_ACTION = "waiting_first_action"
    ALIGNING = "aligning"
    SETTLING = "settling"
    WAITING_REPLAN = "waiting_replan"
    COMPLETE = "complete"

    def __init__(
        self,
        fps: int,
        *,
        position_tolerance_rad: float | np.ndarray = POLICY_START_POSITION_TOLERANCE_RAD,
        settle_frames: int = 5,
        settle_timeout_s: float = 10.0,
    ) -> None:
        position_tolerance = np.asarray(position_tolerance_rad, dtype=float)
        if position_tolerance.ndim == 0:
            position_tolerance = np.full(7, float(position_tolerance), dtype=float)
        if (
            position_tolerance.shape != (7,)
            or not np.all(np.isfinite(position_tolerance))
            or np.any(position_tolerance <= 0.0)
        ):
            raise ValueError(
                "position_tolerance_rad must be one positive value or seven "
                "finite positive joint tolerances."
            )
        if settle_frames <= 0:
            raise ValueError("settle_frames must be positive.")
        if settle_timeout_s <= 0.0:
            raise ValueError("settle_timeout_s must be positive.")

        self.limiter = PolicyActionLimiter(
            fps,
            safety_factor=1.0,
            position_margin_rad=0.05,
            max_velocity_rad_s=POLICY_START_MAX_VELOCITY_RAD_S,
            max_acceleration_rad_s2=POLICY_START_MAX_ACCELERATION_RAD_S2,
        )
        self.position_tolerance_rad = position_tolerance.copy()
        self.settle_frames = settle_frames
        self.settle_timeout_s = settle_timeout_s
        self.phase = self.WAITING_FIRST_ACTION
        self.plan: LimitedActionPlan | None = None
        self._steps: list[LimitedArmStep] = []
        self._next_step_index = 0
        self._target_arms: np.ndarray | None = None
        self._settled_frames = 0
        self._settle_started_monotonic: float | None = None

    @property
    def allows_observations(self) -> bool:
        return self.phase in {
            self.WAITING_FIRST_ACTION,
            self.WAITING_REPLAN,
            self.COMPLETE,
        }

    @property
    def blocks_normal_execution(self) -> bool:
        return self.phase != self.COMPLETE

    def config_dict(self) -> dict[str, object]:
        return {
            "enabled": True,
            "position_tolerance_rad": self.position_tolerance_rad.tolist(),
            "settle_frames": self.settle_frames,
            "settle_timeout_s": self.settle_timeout_s,
            "trajectory": self.limiter.config_dict(),
        }

    def start(
        self,
        current_robot_state: np.ndarray,
        first_policy_action: np.ndarray,
    ) -> LimitedActionPlan:
        if self.phase != self.WAITING_FIRST_ACTION:
            raise RuntimeError(f"Cannot start policy alignment from phase {self.phase!r}.")
        self.plan = self.limiter.plan(current_robot_state, first_policy_action)
        self._steps = list(self.plan.steps)
        self._next_step_index = 0
        self._target_arms = arm_joints_from_robot_vector(first_policy_action)
        self._settled_frames = 0
        self._settle_started_monotonic = None
        self.phase = self.ALIGNING
        return self.plan

    def update(
        self,
        current_robot_state: np.ndarray,
        *,
        now_monotonic: float | None = None,
    ) -> PolicyStartAlignmentUpdate:
        if self.phase not in {self.ALIGNING, self.SETTLING, self.WAITING_REPLAN}:
            raise RuntimeError(f"Policy alignment has no command in phase {self.phase!r}.")
        if self.plan is None or self._target_arms is None:
            raise RuntimeError("Policy alignment target has not been initialized.")

        if self.phase == self.ALIGNING:
            step = self._steps[self._next_step_index]
            self._next_step_index += 1
            final_trajectory_step = self._next_step_index == len(self._steps)
            if final_trajectory_step:
                self.phase = self.SETTLING
                self._settle_started_monotonic = (
                    time.monotonic() if now_monotonic is None else now_monotonic
                )
            return PolicyStartAlignmentUpdate(
                step=step,
                send_grippers=final_trajectory_step,
                max_position_error_rad=None,
                alignment_finished=False,
            )

        current_arms = arm_joints_from_robot_vector(current_robot_state)
        position_error = np.abs(current_arms - self._target_arms)
        max_error = float(np.max(position_error))
        alignment_finished = False
        if self.phase == self.SETTLING:
            if np.all(
                position_error
                <= self.position_tolerance_rad[np.newaxis, :]
            ):
                self._settled_frames += 1
            else:
                self._settled_frames = 0

            now = time.monotonic() if now_monotonic is None else now_monotonic
            if self._settle_started_monotonic is None:
                self._settle_started_monotonic = now
            if now - self._settle_started_monotonic > self.settle_timeout_s:
                raise TimeoutError(
                    "Policy-start alignment did not settle within "
                    f"{self.settle_timeout_s:.1f}s; max joint error is "
                    f"{max_error:.6f} rad and per-joint tolerances are "
                    f"{self.position_tolerance_rad.tolist()} rad."
                )
            if self._settled_frames >= self.settle_frames:
                self.phase = self.WAITING_REPLAN
                alignment_finished = True

        final_step = self.plan.steps[-1]
        return PolicyStartAlignmentUpdate(
            step=final_step,
            send_grippers=False,
            max_position_error_rad=max_error,
            alignment_finished=alignment_finished,
        )

    def mark_replan_received(self) -> None:
        if self.phase != self.WAITING_REPLAN:
            raise RuntimeError(f"Cannot complete policy alignment from phase {self.phase!r}.")
        self.phase = self.COMPLETE
