"""Convert normalized SimToolReal actions to absolute FR3/Wuji targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policy_contract import NUM_ARM_JOINTS, NUM_JOINTS


@dataclass(frozen=True)
class TargetResult:
    full: np.ndarray
    arm: np.ndarray
    hand: np.ndarray


class ActionPipeline:
    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        *,
        dt: float = 1.0 / 60.0,
        dof_speed_scale: float = 1.5,
        arm_moving_average: float = 0.1,
        hand_moving_average: float = 0.1,
        command_lower_limits: np.ndarray | None = None,
        command_upper_limits: np.ndarray | None = None,
    ) -> None:
        self.lower = np.asarray(lower_limits, dtype=np.float64)
        self.upper = np.asarray(upper_limits, dtype=np.float64)
        self.dt = float(dt)
        self.dof_speed_scale = float(dof_speed_scale)
        self.arm_average = float(arm_moving_average)
        self.hand_average = float(hand_moving_average)
        self.command_lower = np.asarray(
            self.lower if command_lower_limits is None else command_lower_limits,
            dtype=np.float64,
        )
        self.command_upper = np.asarray(
            self.upper if command_upper_limits is None else command_upper_limits,
            dtype=np.float64,
        )
        if self.lower.shape != (NUM_JOINTS,) or self.upper.shape != (NUM_JOINTS,):
            raise ValueError("action limits must contain 27 joints")
        if self.command_lower.shape != (NUM_JOINTS,) or self.command_upper.shape != (NUM_JOINTS,):
            raise ValueError("command safety limits must contain 27 joints")
        if np.any(self.command_lower < self.lower) or np.any(self.command_upper > self.upper):
            raise ValueError("command safety limits must be contained within training limits")
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("action dt must be positive")
        for name, value in (("arm", self.arm_average), ("hand", self.hand_average)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} moving average must be within [0, 1]")
        self.previous: np.ndarray | None = None

    def reset(self, measured_position: np.ndarray) -> None:
        measured = np.asarray(measured_position, dtype=np.float64)
        if measured.shape != (NUM_JOINTS,) or not np.all(np.isfinite(measured)):
            raise ValueError("initial action targets require 27 finite joint positions")
        self.previous = np.clip(measured, self.command_lower, self.command_upper)

    def targets(self, normalized_action: np.ndarray) -> TargetResult:
        action = np.asarray(normalized_action, dtype=np.float64)
        if action.shape != (NUM_JOINTS,) or not np.all(np.isfinite(action)):
            raise ValueError("policy action must contain 27 finite values")
        if self.previous is None:
            raise RuntimeError("action pipeline must be reset from measured state before inference")
        action = np.clip(action, -1.0, 1.0)
        arm_raw = self.previous[:NUM_ARM_JOINTS] + self.dof_speed_scale * self.dt * action[:NUM_ARM_JOINTS]
        arm_raw = np.clip(arm_raw, self.lower[:NUM_ARM_JOINTS], self.upper[:NUM_ARM_JOINTS])
        arm = self.arm_average * arm_raw + (1.0 - self.arm_average) * self.previous[:NUM_ARM_JOINTS]
        hand_raw = self.lower[NUM_ARM_JOINTS:] + 0.5 * (action[NUM_ARM_JOINTS:] + 1.0) * (
            self.upper[NUM_ARM_JOINTS:] - self.lower[NUM_ARM_JOINTS:]
        )
        hand = self.hand_average * hand_raw + (1.0 - self.hand_average) * self.previous[NUM_ARM_JOINTS:]
        full = np.concatenate((arm, hand))
        full = np.clip(full, self.command_lower, self.command_upper)
        self.previous = full.copy()
        return TargetResult(full=full, arm=full[:NUM_ARM_JOINTS], hand=full[NUM_ARM_JOINTS:])
