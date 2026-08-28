from __future__ import annotations

from typing import Any

import numpy as np


MANO_LANDMARK_NAMES = (
    "palm",
    "thumb_middle",
    "thumb_distal",
    "thumb_tip",
    "index_middle",
    "index_distal",
    "index_tip",
    "middle_middle",
    "middle_distal",
    "middle_tip",
    "ring_middle",
    "ring_distal",
    "ring_tip",
    "pinky_middle",
    "pinky_distal",
    "pinky_tip",
)

# MANO joint order from the local smplx MANO layer:
# wrist, index 1-3, middle 1-3, pinky 1-3, ring 1-3, thumb 1-3.
MANO_LANDMARK_SOURCES = (
    ("joint", 0),
    ("joint", 14),
    ("joint", 15),
    ("vertex", 745),
    ("joint", 2),
    ("joint", 3),
    ("vertex", 317),
    ("joint", 5),
    ("joint", 6),
    ("vertex", 444),
    ("joint", 11),
    ("joint", 12),
    ("vertex", 556),
    ("joint", 8),
    ("joint", 9),
    ("vertex", 673),
)


def target_landmark_array(landmarks: dict[str, np.ndarray]) -> tuple[list[str], np.ndarray]:
    names = list(MANO_LANDMARK_NAMES)
    points = np.stack(
        [np.asarray(landmarks[name], dtype=np.float32).reshape(3) for name in names],
        axis=0,
    )
    return names, points.astype(np.float32)


def select_mano_landmarks(joints: Any, vertices: Any) -> Any:
    return _stack_like(
        [
            joints[index] if source == "joint" else vertices[index]
            for source, index in MANO_LANDMARK_SOURCES
        ]
    )


def _stack_like(points: list[Any]) -> Any:
    first = points[0]
    if hasattr(first, "new_stack"):
        return first.new_stack(points)
    try:
        import torch

        if isinstance(first, torch.Tensor):
            return torch.stack(points, dim=0)
    except ModuleNotFoundError:
        pass
    return np.stack(points, axis=0)
