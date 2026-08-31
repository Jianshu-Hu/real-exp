from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from foundation_pose_runner import _normalize_estimator_geometry


class _Mesh:
    def __init__(self) -> None:
        self._data = {"vertices": np.ones((4, 3), dtype=np.float64)}
        self._cache = {"stale": object()}

    @property
    def vertices(self) -> np.ndarray:
        return self._data["vertices"]


def test_estimator_geometry_is_normalized_to_float32() -> None:
    estimator = SimpleNamespace(diameter=np.float64(0.25), mesh=_Mesh())

    _normalize_estimator_geometry(estimator)

    assert type(estimator.diameter) is float
    # A Python float follows torch's float32 default; a NumPy float32 is
    # promoted back to float64 when FoundationPose multiplies by crop_ratio.
    assert type(estimator.diameter * 1.2 / 2) is float
    assert estimator.mesh.vertices.dtype == np.float32
    assert estimator.mesh._cache == {}
