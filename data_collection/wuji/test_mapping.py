from __future__ import annotations

import numpy as np
import pytest

from .mapping import WujiHandCommandLimiter


def test_hand_command_limiter_starts_at_actual_position_and_limits_each_step() -> None:
    limiter = WujiHandCommandLimiter(
        [0.0] * 20, max_velocity_rad_s=1.0, rate_hz=50.0
    )
    np.testing.assert_allclose(limiter.limit([1.0] * 20), [0.02] * 20)
    np.testing.assert_allclose(limiter.limit([-1.0] * 20), [0.0] * 20)


@pytest.mark.parametrize(
    "initial, velocity, rate",
    [([0.0] * 19, 1.0, 50.0), ([0.0] * 20, 0.0, 50.0), ([0.0] * 20, 1.0, 0.0)],
)
def test_hand_command_limiter_rejects_invalid_configuration(initial, velocity, rate) -> None:
    with pytest.raises(ValueError):
        WujiHandCommandLimiter(initial, max_velocity_rad_s=velocity, rate_hz=rate)


def test_hand_command_limiter_rejects_invalid_target() -> None:
    limiter = WujiHandCommandLimiter(
        [0.0] * 20, max_velocity_rad_s=1.0, rate_hz=50.0
    )
    with pytest.raises(ValueError):
        limiter.limit([0.0] * 19)
    with pytest.raises(ValueError):
        limiter.limit([0.0] * 19 + [float("nan")])
