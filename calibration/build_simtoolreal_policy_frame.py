#!/usr/bin/env python3
"""Build the SimToolReal policy-frame transforms for scheme 2.

Convention: ``A_T_B`` maps coordinates in frame B into frame A.  The input
calibration JSON contains ``Wreal_T_C`` and ``C_T_B_R``.  The fixed mount
transform ``U_T_B_R`` is read from the selected URDF, where U is the
``trapezoid_base`` root used by the trained policy.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_POLICY_T_U = np.asarray(
    [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.45],
     [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
)
DEFAULT_URDF = Path(__file__).resolve().parents[1] / "simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf"


def checked_transform(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation must be right-handed and orthonormal")
    return matrix


def invert(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                      [-sp, cp * sr, cp * cr]], dtype=np.float64)


def read_urdf_mount(path: Path) -> np.ndarray:
    root = ET.parse(path).getroot()
    joint = root.find("./joint[@name='trapezoid_base_to_right_fr3']")
    if joint is None:
        raise ValueError(f"{path} does not contain trapezoid_base_to_right_fr3")
    # Inspect the link names explicitly so a malformed or different joint
    # cannot silently be used.
    parent_node, child_node = joint.find("parent"), joint.find("child")
    if parent_node is None or child_node is None or parent_node.get("link") != "trapezoid_base" or child_node.get("link") != "right_fr3_base":
        raise ValueError("right FR3 mount joint must connect trapezoid_base to right_fr3_base")
    origin = joint.find("origin")
    if origin is None:
        raise ValueError("right FR3 mount joint has no origin")
    xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
    rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
    if len(xyz) != 3 or len(rpy) != 3:
        raise ValueError("URDF mount xyz/rpy must have three values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rpy_matrix(*rpy)
    result[:3, 3] = xyz
    return checked_transform(result, "U_T_B_R")


def read_matrix_arg(value: str, name: str) -> np.ndarray:
    path = Path(value).expanduser()
    if path.is_file():
        raw = json.loads(path.read_text())
    else:
        # A value that looks like a path is almost certainly a missing input
        # file, not a malformed comma-separated matrix.  Report that directly
        # so operators do not mistake a missing goal for bad matrix numbers.
        if path.suffix.lower() in {".json", ".npy"} or "/" in value:
            raise FileNotFoundError(f"{name} input file does not exist: {path}")
        try:
            raw = [float(item) for item in value.split(",")]
        except ValueError as exc:
            raise ValueError(
                f"{name} must be an existing JSON file or 16 comma-separated numbers"
            ) from exc
    return checked_transform(raw, name)


def load_inputs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    try:
        world = data["Wreal_T_C"] if "Wreal_T_C" in data else data["world_from_real_camera"]
        camera_base = data["C_T_B_R"] if "C_T_B_R" in data else data["camera_from_right_base"]
    except (AttributeError, KeyError) as exc:
        raise ValueError("input JSON must contain Wreal_T_C and C_T_B_R") from exc
    return checked_transform(world, "Wreal_T_C"), checked_transform(camera_base, "C_T_B_R")


def matrix_list(value: np.ndarray) -> list[list[float]]:
    return np.asarray(value, dtype=np.float64).tolist()


def build(world_real_camera: np.ndarray, camera_base: np.ndarray, mount: np.ndarray, policy_root: np.ndarray) -> dict[str, np.ndarray]:
    real_base = world_real_camera @ camera_base
    policy_base = policy_root @ mount
    policy_real = policy_base @ invert(real_base)
    return {
        "Wreal_T_B_R": real_base,
        "Wreal_T_U": real_base @ invert(mount),
        "Wp_T_B_R": policy_base,
        "Wp_T_Wreal": policy_real,
        "Wp_T_C": policy_real @ world_real_camera,
        "Wp_T_U": policy_root,
        "U_T_B_R": mount,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON with Wreal_T_C and C_T_B_R")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--policy-world-from-robot", default=None, help="Wp_T_U JSON path or 16 comma-separated values")
    parser.add_argument("--real-world-goal", default=None, help="Wreal_T_G JSON path or 16 comma-separated values")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "generated/simtoolreal_policy_frame")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    world_real_camera, camera_base = load_inputs(args.input)
    mount = read_urdf_mount(args.urdf)
    policy_root = DEFAULT_POLICY_T_U if args.policy_world_from_robot is None else read_matrix_arg(args.policy_world_from_robot, "Wp_T_U")
    transforms = build(world_real_camera, camera_base, mount, policy_root)
    residual_camera = np.max(np.abs(transforms["Wp_T_C"] - transforms["Wp_T_Wreal"] @ world_real_camera))
    residual_base = np.max(np.abs(transforms["Wp_T_Wreal"] @ transforms["Wreal_T_B_R"] - transforms["Wp_T_B_R"]))
    if max(residual_camera, residual_base) > 1e-8:
        raise ValueError("internal transform composition check failed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = {"Wp_T_C": "world_from_camera_policy.json", "Wp_T_U": "world_from_robot_policy.json",
             "Wp_T_Wreal": "policy_from_real_world.json", "Wreal_T_U": "real_world_from_robot_root.json",
             "Wreal_T_B_R": "real_world_from_right_base.json", "Wp_T_B_R": "policy_from_right_base.json",
             "U_T_B_R": "urdf_root_from_right_base.json"}
    for key, filename in names.items():
        (args.output_dir / filename).write_text(json.dumps(matrix_list(transforms[key]), indent=2) + "\n")
    if args.real_world_goal is not None:
        try:
            real_world_goal = read_matrix_arg(args.real_world_goal, "Wreal_T_G")
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"error: {exc}") from exc
        policy_goal = transforms["Wp_T_Wreal"] @ real_world_goal
        (args.output_dir / "goal_policy.json").write_text(json.dumps(matrix_list(policy_goal), indent=2) + "\n")
    manifest = {"convention": "A_T_B maps B coordinates into A", "source_input": str(args.input),
                "urdf": str(args.urdf), "formula": "Wp_T_Wreal = Wp_T_B_R @ inv(Wreal_T_B_R)",
                "residuals": {"camera": float(residual_camera), "right_base": float(residual_base)},
                "transforms": {key: matrix_list(value) for key, value in transforms.items()}}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated policy-frame calibration in {args.output_dir}")
    print("Dry-run executor arguments:")
    print(f"  --pose-frame camera --world-from-camera {args.output_dir / names['Wp_T_C']}")
    print(f"  --world-from-robot {args.output_dir / names['Wp_T_U']}")
    if args.real_world_goal is None:
        print("Pass --real-world-goal Wreal_T_G to generate goal_policy.json (Wp_T_G).")
    else:
        print(f"  --goal-pose {args.output_dir / 'goal_policy.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
