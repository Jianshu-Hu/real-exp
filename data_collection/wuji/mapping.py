"""Command safety filtering for Python SDK control of Wuji Hand 2."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class WujiHandCommandLimiter:
    """Bound each 20-joint command step to a configured velocity limit."""

    def __init__(
        self,
        initial_position: Sequence[float],
        *,
        max_velocity_rad_s: float,
        rate_hz: float,
    ) -> None:
        initial = np.asarray(initial_position, dtype=np.float64)
        if initial.shape != (20,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_position must contain 20 finite values")
        if max_velocity_rad_s <= 0 or rate_hz <= 0:
            raise ValueError("max_velocity_rad_s and rate_hz must be positive")
        self._last = initial.copy()
        self.max_step_rad = float(max_velocity_rad_s) / float(rate_hz)

    @property
    def last(self) -> np.ndarray:
        return self._last.copy()

    def limit(self, target: Sequence[float]) -> np.ndarray:
        values = np.asarray(target, dtype=np.float64)
        if values.shape != (20,) or not np.all(np.isfinite(values)):
            raise ValueError("target must contain 20 finite values")
        delta = np.clip(values - self._last, -self.max_step_rad, self.max_step_rad)
        self._last = self._last + delta
        return self._last.copy()
