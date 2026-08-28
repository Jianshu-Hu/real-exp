#!/usr/bin/env python3
"""Run one RGB-D grasp inference using only the connected D435 camera.

The script captures a frame, transforms the valid depth points into the
calibrated world frame, runs generator/retargeting/refinement, and writes all
point clouds and hand meshes in the world coordinate frame.
No control host or ZeroMQ connection is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from grasp.inference_client import (
    DEFAULT_CAMERA_SERIAL,
    DEFAULT_GENERATOR_CHECKPOINT,
    DEFAULT_MANO_ROOT,
    DEFAULT_ROBODEX_ROOT,
    _posed_wuji_mesh,
    _write_ply_mesh,
    _write_ply_points,
    backproject_depth,
    filter_and_sample_points,
    load_calibration_transforms,
    run_model,
)
from grasp.common import (
    WUJI_RIGHT_JOINT_NAMES,
    hand_pose_to_ee_pose,
    matrix_to_xyz_rpy,
    reorder_wuji_joints,
    transform_points,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator-checkpoint", type=Path, default=DEFAULT_GENERATOR_CHECKPOINT)
    parser.add_argument("--mount-calibration", type=Path, default=Path(__file__).resolve().parent / "ee_to_wuji_nominal.json")
    parser.add_argument("--camera-serial", default=DEFAULT_CAMERA_SERIAL)
    parser.add_argument(
        "--camera-python",
        default="/usr/bin/python3",
        help="Python interpreter containing pyrealsense2 (default: /usr/bin/python3).",
    )
    parser.add_argument(
        "--capture-script",
        type=Path,
        default=Path(__file__).resolve().parent / "realsense_capture.py",
        help="System-Python RGB-D capture helper.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.50)
    parser.add_argument("--world-min", type=float, nargs=3, default=(-0.50, -0.50, 0.005))
    parser.add_argument("--world-max", type=float, nargs=3, default=(0.50, 0.50, 0.50))
    parser.add_argument("--min-filtered-points", type=int, default=300)
    parser.add_argument("--num-points", type=int, default=8192)
    parser.add_argument("--world-z-segmentation-min-m", type=float, default=0.002)
    parser.add_argument("--generator-weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--posterior-conditioning",
        choices=("auto", "target_film", "full_feature_only"),
        default="auto",
    )
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--retarget-landmark-fit-steps", type=int, default=75)
    parser.add_argument("--semantic-refine-steps", type=int, default=40)
    parser.add_argument("--semantic-learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-penetration-m", type=float, default=0.02)
    parser.add_argument("--contact-sigma-m", type=float, default=0.01)
    parser.add_argument("--contact-d-max-m", type=float, default=0.03)
    parser.add_argument("--contact-binary-threshold-m", type=float, default=0.010)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.min_depth_m < 0 or args.max_depth_m <= args.min_depth_m:
        parser.error("depth bounds must satisfy 0 <= min < max")
    if np.any(np.asarray(args.world_max) <= np.asarray(args.world_min)):
        parser.error("every --world-max value must exceed --world-min")
    if min(args.num_points, args.min_filtered_points) <= 0:
        parser.error("point counts must be positive")
    if args.semantic_refine_steps < 0 or args.max_penetration_m < 0:
        parser.error("refinement steps and penetration limit must be non-negative")
    for path, name in (
        (args.generator_checkpoint, "generator checkpoint"),
        (DEFAULT_MANO_ROOT / "models" / "MANO_RIGHT.pkl", "MANO right-hand model"),
        (
            DEFAULT_ROBODEX_ROOT / "task/assets/urdf/panda_wuji_hand_right_handonly.urdf",
            "Wuji hand-only URDF",
        ),
    ):
        if not path.is_file():
            parser.error(f"bundled {name} is missing: {path}")


def _world_hand_pose(refined: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_rotvec(refined.robot_global_orient).as_matrix()
    pose[:3, 3] = np.asarray(refined.robot_trans, dtype=np.float64)
    return pose


def _capture_with_system_python(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Capture RGB-D in a separate interpreter that has pyrealsense2 installed."""
    camera_python = shutil.which(str(args.camera_python)) or str(args.camera_python)
    if not Path(camera_python).is_file():
        raise FileNotFoundError(f"camera Python interpreter does not exist: {args.camera_python}")
    if not args.capture_script.is_file():
        raise FileNotFoundError(f"camera capture helper does not exist: {args.capture_script}")
    with tempfile.TemporaryDirectory(prefix="real_exp_camera_") as temp_dir:
        capture_dir = Path(temp_dir)
        command = [
            camera_python,
            str(args.capture_script),
            "--output-dir",
            str(capture_dir),
            "--camera-serial",
            str(args.camera_serial),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--fps",
            str(args.fps),
            "--warmup-frames",
            str(args.warmup_frames),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"system Python camera capture failed with exit code {completed.returncode}"
                + (f": {detail}" if detail else "")
            )
        rgb = np.load(capture_dir / "rgb.npy")
        depth = np.load(capture_dir / "depth.npy")
        metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="utf-8"))
    return rgb, depth, metadata


def _write_coordinate_outputs(
    directory: Path,
    *,
    unfiltered_points: np.ndarray,
    filtered_points: np.ndarray,
    sampled_points: np.ndarray,
    grasp_points: np.ndarray,
    mano_vertices: np.ndarray,
    mano_faces: np.ndarray,
    wuji_retargeted: tuple[np.ndarray, np.ndarray],
    wuji_refined: tuple[np.ndarray, np.ndarray],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_ply_points(directory / "scene_points_unfiltered.ply", unfiltered_points)
    _write_ply_points(directory / "scene_points_filtered.ply", filtered_points)
    _write_ply_points(directory / "generator_input.ply", sampled_points)
    _write_ply_points(directory / "grasp_object_points.ply", grasp_points)
    _write_ply_mesh(directory / "mano.ply", mano_vertices, mano_faces)
    _write_ply_mesh(directory / "wuji_retargeted.ply", *wuji_retargeted)
    _write_ply_mesh(directory / "wuji_refined.ply", *wuji_refined)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    transforms = load_calibration_transforms(args.mount_calibration)

    rgb, depth, camera = _capture_with_system_python(args)
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and aligned depth image dimensions differ")
    camera_points, depth_m = backproject_depth(depth, camera["intrinsics"], camera["depth_scale_m"])
    valid_camera, filtered_world, sampled_world = filter_and_sample_points(
        camera_points, depth_m, transforms["world_T_camera"], args
    )
    unfiltered_world = transform_points(transforms["world_T_camera"], valid_camera)

    # run_model expects the complete set of refinement/contact parameters; the
    # parser above deliberately keeps these local to this camera-only tool.
    generated, refined = run_model(sampled_world, args)
    mano_faces = np.asarray(generated.mano_faces, dtype=np.int64)
    wuji_retargeted_world = _posed_wuji_mesh(
        generated.robot_trans, generated.robot_global_orient, generated.robot_joints
    )
    wuji_refined_world = _posed_wuji_mesh(
        refined.robot_trans, refined.robot_global_orient, refined.robot_joints
    )
    world_outputs = {
        "unfiltered_points": unfiltered_world,
        "filtered_points": filtered_world,
        "sampled_points": sampled_world,
        "grasp_points": np.asarray(generated.object_points, dtype=np.float32),
        "mano_vertices": np.asarray(generated.mano_vertices, dtype=np.float32),
        "mano_faces": mano_faces,
        "wuji_retargeted": wuji_retargeted_world,
        "wuji_refined": wuji_refined_world,
    }
    _write_coordinate_outputs(args.output_dir / "world", **world_outputs)

    world_t_hand = _world_hand_pose(refined)
    base_t_world = transforms["base_T_world"]
    base_t_hand = base_t_world @ world_t_hand
    final_hand_joints = reorder_wuji_joints(
        refined.robot_joints, generated.robot_joint_names
    )
    record = {
        "camera": camera,
        "calibration": {
            "world_T_camera": transforms["world_T_camera"].tolist(),
            "base_T_camera": transforms["base_T_camera"].tolist(),
            "base_T_world": base_t_world.tolist(),
            "ee_T_hand": transforms["ee_T_hand"].tolist(),
        },
        "filter": {
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "world_min": args.world_min,
            "world_max": args.world_max,
            "valid_depth_points": int(valid_camera.shape[0]),
            "filtered_world_points": int(filtered_world.shape[0]),
            "generator_input_points": int(sampled_world.shape[0]),
        },
        "poses": {
            "world_T_hand": world_t_hand.tolist(),
            "base_T_hand": base_t_hand.tolist(),
            "base_T_ee": hand_pose_to_ee_pose(
                world_t_hand, base_t_world, transforms["ee_T_hand"]
            ).tolist(),
            "base_T_ee_xyz_rpy": matrix_to_xyz_rpy(
                hand_pose_to_ee_pose(world_t_hand, base_t_world, transforms["ee_T_hand"])
            ).tolist(),
            "hand_joint_names": list(WUJI_RIGHT_JOINT_NAMES),
            "hand_joints_rad": final_hand_joints.tolist(),
        },
        "inference": {
            "generator_checkpoint": str(args.generator_checkpoint.resolve()),
            "seed": args.seed,
            "retarget_fit_error": generated.retarget_fit_error,
            "refinement": refined.metadata,
        },
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(record, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (args.output_dir / "poses.json").write_text(
        json.dumps(record["poses"], indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    np.save(args.output_dir / "rgb_bgr.npy", rgb)
    np.save(args.output_dir / "depth_raw.npy", depth)
    print(f"Inference complete; outputs saved to {args.output_dir}")
    print(f"World points: {filtered_world.shape[0]} filtered / {unfiltered_world.shape[0]} valid")
    print(f"World-frame outputs: {args.output_dir / 'world'}")
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
