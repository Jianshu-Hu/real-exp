from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from grasp.runtime.retargeting.mano_landmarks import MANO_LANDMARK_NAMES
from grasp.runtime.retargeting.robot_hand_model import RobotHandSpec


_FINGER_NUMBER = {
    "thumb": 1,
    "index": 2,
    "middle": 3,
    "ring": 4,
    "pinky": 5,
}

# MANO local joint order after the wrist/global orientation:
# index, middle, pinky, ring, thumb; three axis-angle joints per finger.
_MANO_FINGER_JOINTS = {
    "thumb": (12, 13, 14),
    "index": (0, 1, 2),
    "middle": (3, 4, 5),
    "ring": (9, 10, 11),
    "pinky": (6, 7, 8),
}

WUJI_HAND_RIGHT_MANO_ORIENT_CORRECTION = Rotation.from_rotvec(
    np.asarray((1.39457832, -1.29370800, -1.01752498), dtype=np.float64)
)


def create_wuji_hand_right_spec(
    *,
    robodex_root: Path = Path("data/githubRepo/RoboDex"),
    urdf_name: str = "panda_wuji_hand_right_handonly.urdf",
) -> RobotHandSpec:
    """Build the canonical right-Wuji specification used by both stages.

    The hand-only URDF is generated and maintained by RoboDex and uses the same
    joint/link names as ``PandaWujiHandRight``.  Keeping this as the single
    geometry source prevents static evaluation and free-hand replay from
    silently evaluating different hands.
    """

    return RobotHandSpec(
        name="right_wuji_hand",
        urdf_path=Path(robodex_root) / "task" / "assets" / "urdf" / urdf_name,
        landmark_links=_wuji_landmark_links(),
        finger_contact_links={
            finger: (
                f"right_finger{number}_link2",
                f"right_finger{number}_link3",
                f"right_finger{number}_link4",
                f"right_finger{number}_tip_link",
            )
            for finger, number in _FINGER_NUMBER.items()
        },
        mano_to_joint_seed=wuji_hand_right_joints_from_mano,
        mano_to_global_orient_seed=wuji_hand_right_global_orient_from_mano,
    )


def _wuji_landmark_links() -> dict[str, str]:
    mapping = {"palm": "right_palm_link"}
    for finger, number in _FINGER_NUMBER.items():
        mapping.update(
            {
                f"{finger}_middle": f"right_finger{number}_link3",
                f"{finger}_distal": f"right_finger{number}_link4",
                f"{finger}_tip": f"right_finger{number}_tip_link",
            }
        )
    return {name: mapping[name] for name in MANO_LANDMARK_NAMES}


def wuji_hand_right_joints_from_mano(
    mano_hand_pose: np.ndarray,
    joint_names: list[str],
) -> np.ndarray:
    """Initialize Wuji flexion while leaving its lateral joint neutral.

    Wuji joint2 is the signed lateral/base degree of freedom.  A MANO local
    rotvec component is not portable into that joint's rotated local frame, so
    it starts at the valid neutral position and landmark fitting resolves it.
    Joints 1, 3, and 4 receive the MANO MCP, PIP, and DIP bend magnitudes.
    """

    hand_pose = np.asarray(mano_hand_pose, dtype=np.float32)
    if hand_pose.ndim != 2 or hand_pose.shape[1] != 45:
        raise ValueError(f"mano_hand_pose must have shape (B, 45), got {hand_pose.shape}")
    mano = hand_pose.reshape(hand_pose.shape[0], 15, 3)
    seed = np.zeros((hand_pose.shape[0], len(joint_names)), dtype=np.float32)
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    for finger, number in _FINGER_NUMBER.items():
        mano_indices = _MANO_FINGER_JOINTS[finger]
        bends = np.linalg.norm(mano[:, mano_indices, :], axis=2)
        values = (
            bends[:, 0],
            np.zeros(hand_pose.shape[0], dtype=np.float32),
            bends[:, 1],
            bends[:, 2],
        )
        for joint_number, value in enumerate(values, start=1):
            name = f"right_finger{number}_joint{joint_number}"
            index = name_to_index.get(name)
            if index is not None:
                seed[:, index] = value
    return seed.astype(np.float32)


def wuji_hand_right_global_orient_from_mano(
    mano_global_orient: np.ndarray,
) -> np.ndarray:
    """Apply the calibrated local-frame offset from MANO to right Wuji."""
    orient = np.asarray(mano_global_orient, dtype=np.float32)
    squeeze = orient.ndim == 1
    if squeeze:
        orient = orient.reshape(1, 3)
    if orient.ndim != 2 or orient.shape[1] != 3:
        raise ValueError(
            "mano_global_orient must have shape (3,) or (B, 3), got "
            f"{orient.shape}"
        )
    corrected = np.stack(
        [
            (
                Rotation.from_rotvec(value)
                * WUJI_HAND_RIGHT_MANO_ORIENT_CORRECTION
            ).as_rotvec()
            for value in orient
        ],
        axis=0,
    ).astype(np.float32)
    return corrected[0] if squeeze else corrected
