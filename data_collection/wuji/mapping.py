"""Mapping and safety filtering for Wuji glove samples.

The glove SDK has changed Python bindings across releases.  This module keeps
the robot-facing contract stable: seven finite joint targets in radians and a
normalized gripper command in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GloveSample:
    """One glove sample in the glove's native joint order."""

    joints: tuple[float, ...]
    gripper: float


class WujiJointMapper:
    """Convert glove angles to FR3 targets with calibration and rate limits."""

    def __init__(
        self,
        *,
        joint_indices: Sequence[int] = tuple(range(7)),
        joint_signs: Sequence[float] = (1, -1, 1, 1, 1, -1, 1),
        joint_offsets: Sequence[float] = (0.0,) * 7,
        joint_min: Sequence[float] | None = None,
        joint_max: Sequence[float] | None = None,
        max_step_rad: float = 0.15,
    ) -> None:
        if len(joint_indices) != 7 or len(joint_signs) != 7 or len(joint_offsets) != 7:
            raise ValueError("joint_indices, joint_signs and joint_offsets must contain seven values")
        if max_step_rad <= 0:
            raise ValueError("max_step_rad must be positive")
        self.indices = tuple(int(i) for i in joint_indices)
        self.signs = np.asarray(joint_signs, dtype=np.float64)
        self.offsets = np.asarray(joint_offsets, dtype=np.float64)
        self.minimum = None if joint_min is None else np.asarray(joint_min, dtype=np.float64)
        self.maximum = None if joint_max is None else np.asarray(joint_max, dtype=np.float64)
        if (self.minimum is None) != (self.maximum is None):
            raise ValueError("joint_min and joint_max must be supplied together")
        if self.minimum is not None and (self.minimum.shape != (7,) or self.maximum.shape != (7,)):
            raise ValueError("joint_min and joint_max must contain seven values")
        self.max_step_rad = float(max_step_rad)
        self._last: np.ndarray | None = None

    def reset(self, target: Iterable[float] | None = None) -> None:
        self._last = None if target is None else np.asarray(tuple(target), dtype=np.float64)

    def map(self, native_joints: Sequence[float]) -> np.ndarray:
        values = np.asarray(native_joints, dtype=np.float64)
        if values.ndim != 1 or values.size <= max(self.indices):
            raise ValueError(f"expected at least {max(self.indices) + 1} glove joints, got {values.shape}")
        target = values[list(self.indices)] * self.signs + self.offsets
        if not np.all(np.isfinite(target)):
            raise ValueError("glove joint sample contains non-finite values")
        if self.minimum is not None:
            target = np.clip(target, self.minimum, self.maximum)
        if self._last is not None:
            target = self._last + np.clip(target - self._last, -self.max_step_rad, self.max_step_rad)
        self._last = target.copy()
        return target

    @staticmethod
    def gripper(value: float) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("gripper sample is not finite")
        return float(np.clip(value, 0.0, 1.0))
