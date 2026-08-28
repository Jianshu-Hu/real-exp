#!/usr/bin/env python3
"""Convert a world-frame EE pose to the right FR3 base frame and optionally move it.

The input and output pose convention is ``x y z roll pitch yaw`` in metres and
radians.  RPY uses the controller's ZYX composition:
``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOVE_SCRIPT = REPOSITORY_ROOT / "scripts" / "move_to_target_ee.sh"

# Copied from calibration/matrix.md.  A_T_B maps coordinates in B into A.
WORLD_T_CAMERA = np.asarray(
    [
        [0.016116505, -0.947169025, 0.320329670, -0.394891761],
        [-0.998707711, 0.000194370, 0.050821951, -0.041552817],
        [-0.048199240, -0.320734783, -0.945941876, 1.159142768],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
CAMERA_T_RIGHT_BASE = np.asarray(
    [
        [0.061077178, -0.724658647, 0.686395967, 0.162840030],
        [-0.927423222, -0.295425134, -0.229369041, 0.562456279],
        [0.368992880, -0.622570346, -0.690108991, 0.901632663],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def rpy_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll), matching the arm controller."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotation_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Extract controller-compatible roll, pitch, yaw from a rotation matrix."""
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        # At gimbal lock there are infinitely many equivalent RPY triples.
        # Use the same deterministic representative as move_to_target_ee.py.
        roll = 0.0
        yaw = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("pose must contain six finite XYZ/RPY values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rpy_to_rotation(*values[3:])
    result[:3, 3] = values[:3]
    return result


def matrix_to_pose(transform: np.ndarray) -> np.ndarray:
    return np.concatenate((transform[:3, 3], rotation_to_rpy(transform[:3, :3])))


def world_pose_to_right_base_pose(world_pose: Sequence[float]) -> np.ndarray:
    """Convert W_T_E XYZ/RPY to B_R_T_E XYZ/RPY."""
    world_t_right_base = WORLD_T_CAMERA @ CAMERA_T_RIGHT_BASE
    right_base_t_world = invert_transform(world_t_right_base)
    right_base_t_ee = right_base_t_world @ pose_to_matrix(world_pose)
    return matrix_to_pose(right_base_t_ee)


def format_pose(pose: Sequence[float]) -> str:
    return " ".join(f"{float(value):.9f}" for value in pose)


def build_move_command(move_script: Path, right_base_pose: Sequence[float], dry_run: bool) -> list[str]:
    command = [
        str(move_script),
        "--right",
        "--arm",
        "--target-ee-pose",
        *(f"{float(value):.12g}" for value in right_base_pose),
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world-ee-pose",
        nargs=6,
        type=float,
        required=True,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help="world-frame target in metres/radians",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help=(
            "invoke the right-arm controller; its existing interactive yes/no "
            "confirmation is still required before motion"
        ),
    )
    mode.add_argument(
        "--controller-dry-run",
        action="store_true",
        help="invoke the controller in --dry-run mode to read state and validate IK without motion",
    )
    parser.add_argument(
        "--move-script",
        type=Path,
        default=DEFAULT_MOVE_SCRIPT,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in args.world_ee_pose):
        parser.error("--world-ee-pose values must all be finite")

    right_base_pose = world_pose_to_right_base_pose(args.world_ee_pose)
    command = build_move_command(args.move_script, right_base_pose, args.controller_dry_run)
    print(f"world EE pose       [x y z r p y]: {format_pose(args.world_ee_pose)}")
    print(f"right-base EE pose  [x y z r p y]: {format_pose(right_base_pose)}")
    print(f"controller command: {shlex.join(command)}")

    if not (args.execute or args.controller_dry_run):
        print("Conversion only: controller was not started.")
        return 0
    if not args.move_script.is_file():
        parser.error(f"move script does not exist: {args.move_script}")
    if not args.move_script.stat().st_mode & 0o111:
        parser.error(f"move script is not executable: {args.move_script}")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
