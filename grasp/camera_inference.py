#!/usr/bin/env python3
"""Run RGB-D grasp inference using only the connected D435 camera.

The script fuses consecutive aligned depth frames, transforms the valid depth
points into the calibrated world frame, runs generator/retargeting/refinement,
and writes all point clouds and hand meshes in the world coordinate frame.
No control host or ZeroMQ connection is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
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


def add_camera_inference_arguments(
    parser: argparse.ArgumentParser, *, include_output_dir: bool = True
) -> argparse.ArgumentParser:
    """Add camera/model options shared by one-shot and server inference."""
    if include_output_dir:
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
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--observation-frames",
        type=int,
        default=15,
        help=(
            "Number of consecutive aligned depth frames to fuse with a per-pixel "
            "median (default: 15). Keep the camera and scene still during capture."
        ),
    )
    parser.add_argument(
        "--min-valid-depth-ratio",
        type=float,
        default=0.5,
        help=(
            "Minimum fraction of observation frames in which a pixel must have "
            "nonzero depth to be retained (default: 0.5)."
        ),
    )
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
    parser.add_argument("--max-penetration-m", type=float, default=0.05)
    parser.add_argument("--contact-sigma-m", type=float, default=0.01)
    parser.add_argument("--contact-d-max-m", type=float, default=0.03)
    parser.add_argument("--contact-binary-threshold-m", type=float, default=0.010)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def build_parser() -> argparse.ArgumentParser:
    return add_camera_inference_arguments(argparse.ArgumentParser(description=__doc__))


def validate_camera_inference_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("camera width, height, and fps must be positive")
    if args.warmup_frames < 0:
        parser.error("warmup-frames must be non-negative")
    if args.observation_frames <= 0:
        parser.error("observation-frames must be positive")
    if not 0.0 < args.min_valid_depth_ratio <= 1.0:
        parser.error("min-valid-depth-ratio must be in (0, 1]")
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
            "--observation-frames",
            str(args.observation_frames),
            "--min-valid-depth-ratio",
            str(args.min_valid_depth_ratio),
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


def _write_json(path: Path, value: Any) -> None:
    """Atomically write JSON so an interrupted run does not leave truncated metadata."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_run_status(
    output_dir: Path,
    *,
    status: str,
    stage: str,
    error: BaseException | None = None,
) -> None:
    value: dict[str, Any] = {
        "status": status,
        "stage": stage,
        "updated_unix_s": time.time(),
    }
    if error is not None:
        value["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    _write_json(output_dir / "run_status.json", value)


def _write_failure(
    output_dir: Path,
    *,
    status: str,
    stage: str,
    error: BaseException,
) -> None:
    available_outputs = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "failure.json"
    )
    _write_json(
        output_dir / "failure.json",
        {
            "status": status,
            "stage": stage,
            "failed_unix_s": time.time(),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "available_outputs": available_outputs,
        },
    )


def run_camera_inference(args: argparse.Namespace) -> dict[str, Any]:
    """Capture, infer, persist one trial, and return its JSON-compatible record."""
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stage = "initialization"
    _write_run_status(args.output_dir, status="running", stage=stage)
    try:
        stage = "calibration"
        _write_run_status(args.output_dir, status="running", stage=stage)
        transforms = load_calibration_transforms(args.mount_calibration)

        stage = "camera_capture"
        _write_run_status(args.output_dir, status="running", stage=stage)
        rgb, depth, camera = _capture_with_system_python(args)
        if rgb.shape[:2] != depth.shape:
            raise ValueError("RGB and aligned depth image dimensions differ")
        np.save(args.output_dir / "rgb_bgr.npy", rgb)
        np.save(args.output_dir / "depth_raw.npy", depth)
        _write_json(args.output_dir / "camera.json", camera)

        stage = "point_cloud_filtering"
        _write_run_status(args.output_dir, status="running", stage=stage)
        camera_points, depth_m = backproject_depth(
            depth, camera["intrinsics"], camera["depth_scale_m"]
        )
        valid_camera, filtered_world, sampled_world = filter_and_sample_points(
            camera_points, depth_m, transforms["world_T_camera"], args
        )
        unfiltered_world = transform_points(transforms["world_T_camera"], valid_camera)
        filter_record = {
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "world_min": args.world_min,
            "world_max": args.world_max,
            "valid_depth_points": int(valid_camera.shape[0]),
            "filtered_world_points": int(filtered_world.shape[0]),
            "generator_input_points": int(sampled_world.shape[0]),
        }
        _write_json(args.output_dir / "filter.json", filter_record)
        world_dir = args.output_dir / "world"
        world_dir.mkdir(parents=True, exist_ok=True)
        _write_ply_points(world_dir / "scene_points_unfiltered.ply", unfiltered_world)
        _write_ply_points(world_dir / "scene_points_filtered.ply", filtered_world)
        _write_ply_points(world_dir / "generator_input.ply", sampled_world)

        stage = "model_inference"
        _write_run_status(args.output_dir, status="running", stage=stage)
        # run_model expects the complete set of refinement/contact parameters;
        # the parser deliberately keeps these local to this camera-only tool.
        generated, refined = run_model(sampled_world, args)
        mano_faces = np.asarray(generated.mano_faces, dtype=np.int64)
        wuji_retargeted_world = _posed_wuji_mesh(
            generated.robot_trans,
            generated.robot_global_orient,
            generated.robot_joints,
        )
        wuji_refined_world = _posed_wuji_mesh(
            refined.robot_trans, refined.robot_global_orient, refined.robot_joints
        )
        _write_coordinate_outputs(
            world_dir,
            unfiltered_points=unfiltered_world,
            filtered_points=filtered_world,
            sampled_points=sampled_world,
            grasp_points=np.asarray(generated.object_points, dtype=np.float32),
            mano_vertices=np.asarray(generated.mano_vertices, dtype=np.float32),
            mano_faces=mano_faces,
            wuji_retargeted=wuji_retargeted_world,
            wuji_refined=wuji_refined_world,
        )

        stage = "result_serialization"
        _write_run_status(args.output_dir, status="running", stage=stage)
        world_t_hand = _world_hand_pose(refined)
        base_t_world = transforms["base_T_world"]
        base_t_hand = base_t_world @ world_t_hand
        base_t_ee = hand_pose_to_ee_pose(
            world_t_hand, base_t_world, transforms["ee_T_hand"]
        )
        final_hand_joints = reorder_wuji_joints(
            refined.robot_joints, generated.robot_joint_names
        )
        final_penetration = float(
            refined.metadata["final_penetration"]["max_penetration_depth_m"]
        )
        penetration_accepted = bool(
            np.isfinite(final_penetration)
            and final_penetration <= args.max_penetration_m
        )
        rejection_reason = None
        if not penetration_accepted:
            rejection_reason = (
                f"refined grasp penetration {final_penetration:.6f} m exceeds "
                f"limit {args.max_penetration_m:.6f} m"
            )
        record = {
            "status": "completed" if penetration_accepted else "rejected",
            "rejection_reason": rejection_reason,
            "camera": camera,
            "calibration": {
                "world_T_camera": transforms["world_T_camera"].tolist(),
                "base_T_camera": transforms["base_T_camera"].tolist(),
                "base_T_world": base_t_world.tolist(),
                "ee_T_hand": transforms["ee_T_hand"].tolist(),
            },
            "filter": filter_record,
            "poses": {
                "world_T_hand": world_t_hand.tolist(),
                "base_T_hand": base_t_hand.tolist(),
                "base_T_ee": base_t_ee.tolist(),
                "base_T_ee_xyz_rpy": matrix_to_xyz_rpy(base_t_ee).tolist(),
                "hand_joint_names": list(WUJI_RIGHT_JOINT_NAMES),
                "hand_joints_rad": final_hand_joints.tolist(),
            },
            "safety": {
                "penetration_accepted": penetration_accepted,
                "final_penetration_m": final_penetration,
                "max_penetration_m": args.max_penetration_m,
            },
            "inference": {
                "generator_checkpoint": str(args.generator_checkpoint.resolve()),
                "seed": args.seed,
                "retarget_fit_error": generated.retarget_fit_error,
                "refinement": refined.metadata,
            },
        }
        record_json = json.dumps(record, indent=2, default=_json_default)
        _write_json(args.output_dir / "result.json", record)
        _write_json(args.output_dir / "poses.json", record["poses"])

        if not penetration_accepted:
            stage = "safety_validation"
            raise ValueError(rejection_reason)

        _write_run_status(args.output_dir, status="completed", stage="completed")
        print(f"Inference complete; outputs saved to {args.output_dir}")
        print(
            f"World points: {filtered_world.shape[0]} filtered / "
            f"{unfiltered_world.shape[0]} valid"
        )
        print(f"World-frame outputs: {world_dir}")
        # Normalize any NumPy scalar nested in model metadata before returning
        # this record to the JSON-based daemon.
        return json.loads(record_json)
    except Exception as exc:
        failure_status = "rejected" if stage == "safety_validation" else "failed"
        try:
            _write_failure(
                args.output_dir,
                status=failure_status,
                stage=stage,
                error=exc,
            )
            _write_run_status(
                args.output_dir,
                status=failure_status,
                stage=stage,
                error=exc,
            )
        except Exception as persistence_error:
            print(
                f"Warning: could not persist failure metadata: {persistence_error}",
                flush=True,
            )
        print(
            f"Inference {failure_status}; available outputs saved to {args.output_dir}",
            flush=True,
        )
        raise


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_camera_inference_args(args, parser)
    run_camera_inference(args)
    return 0


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    raise SystemExit(main())
