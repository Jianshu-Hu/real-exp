from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from grasp.runtime.retargeting.mano_landmarks import MANO_LANDMARK_NAMES
from grasp.runtime.retargeting.robot_hand_model import RobotHandSpec


WUJI_HAND2_BETA1_RIGHT_MANO_ORIENT_CORRECTION = Rotation.from_rotvec(
    np.asarray((2.05418207, 0.11426044, 2.37324510), dtype=np.float64)
)

_MANO_FINGER_JOINTS = {
    "thumb": (12, 13, 14),
    "index": (0, 1, 2),
    "middle": (3, 4, 5),
    "ring": (9, 10, 11),
    "pinky": (6, 7, 8),
}


def create_wuji_hand2_beta1_right_spec(
    *,
    hand_root: Path = Path("grasp/assets/Wuji_hand2"),
) -> RobotHandSpec:
    """Build the canonical right Wuji Hand 2 Beta 1 specification."""
    return RobotHandSpec(
        name="right_wuji_hand2_beta1",
        urdf_path=Path(hand_root) / "hand2_beta1" / "body" / "urdf" / "right.urdf",
        landmark_links=_wuji_hand2_beta1_landmark_links(),
        finger_contact_links={
            finger: (
                f"r_{stem}_proximal_abd",
                f"r_{stem}_middle",
                f"r_{stem}_distal",
            )
            for finger, stem in _finger_stems().items()
        },
        mano_to_joint_seed=wuji_hand2_beta1_right_joints_from_mano,
        mano_to_global_orient_seed=wuji_hand2_beta1_right_global_orient_from_mano,
    )


def _finger_stems() -> dict[str, str]:
    return {
        "thumb": "thumb",
        "index": "index_finger",
        "middle": "middle_finger",
        "ring": "ring_finger",
        "pinky": "pinky",
    }


def _wuji_hand2_beta1_landmark_links() -> dict[str, str]:
    mapping = {"palm": "r_wrist"}
    for finger, stem in _finger_stems().items():
        mapping.update(
            {
                f"{finger}_middle": f"r_{stem}_middle",
                f"{finger}_distal": f"r_{stem}_distal",
                f"{finger}_tip": f"r_{stem}_tip",
            }
        )
    return {name: mapping[name] for name in MANO_LANDMARK_NAMES}


def wuji_hand2_beta1_right_joints_from_mano(
    mano_hand_pose: np.ndarray,
    joint_names: list[str],
) -> np.ndarray:
    """Seed Hand 2 flexion from MANO and leave abduction neutral."""
    hand_pose = np.asarray(mano_hand_pose, dtype=np.float32)
    if hand_pose.ndim != 2 or hand_pose.shape[1] != 45:
        raise ValueError(f"mano_hand_pose must have shape (B, 45), got {hand_pose.shape}")
    mano = hand_pose.reshape(hand_pose.shape[0], 15, 3)
    seed = np.zeros((hand_pose.shape[0], len(joint_names)), dtype=np.float32)
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    for finger, stem in _finger_stems().items():
        bends = np.linalg.norm(mano[:, _MANO_FINGER_JOINTS[finger], :], axis=2)
        joint_suffixes = (
            ("cmc_flex", "cmc_abd", "mcp", "ip")
            if finger == "thumb"
            else ("mcp_flex", "mcp_abd", "pip", "dip")
        )
        values = (
            bends[:, 0],
            np.zeros(hand_pose.shape[0], dtype=np.float32),
            bends[:, 1],
            bends[:, 2],
        )
        for suffix, value in zip(joint_suffixes, values, strict=True):
            index = name_to_index.get(f"r_{stem}_{suffix}")
            if index is not None:
                seed[:, index] = value
    return seed


def wuji_hand2_beta1_right_global_orient_from_mano(
    mano_global_orient: np.ndarray,
) -> np.ndarray:
    """Apply the calibrated MANO-to-Hand-2-Beta-1 frame offset."""
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
                * WUJI_HAND2_BETA1_RIGHT_MANO_ORIENT_CORRECTION
            ).as_rotvec()
            for value in orient
        ],
        axis=0,
    ).astype(np.float32)
    return corrected[0] if squeeze else corrected
