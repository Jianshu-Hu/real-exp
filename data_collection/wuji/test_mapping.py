from __future__ import annotations

import numpy as np
import pytest

from .mapping import WujiJointMapper


def test_mapping_applies_signs_offsets_and_rate_limit() -> None:
    mapper = WujiJointMapper(joint_offsets=[1] * 7, max_step_rad=0.25)
    np.testing.assert_allclose(mapper.map([0] * 7), [1] * 7)
    np.testing.assert_allclose(mapper.map([1] * 7), [1.25, 0.75, 1.25, 1.25, 1.25, 0.75, 1.25])


def test_mapping_rejects_short_or_nonfinite_samples() -> None:
    mapper = WujiJointMapper()
    with pytest.raises(ValueError):
        mapper.map([0] * 6)
    with pytest.raises(ValueError):
        mapper.map([0, 0, 0, 0, 0, 0, float("nan")])


def test_gripper_is_normalized() -> None:
    assert WujiJointMapper.gripper(-1) == 0.0
    assert WujiJointMapper.gripper(2) == 1.0
