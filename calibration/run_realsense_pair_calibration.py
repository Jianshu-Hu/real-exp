#!/usr/bin/env python3
"""Run the complete D435i/L515 shared-AprilTag calibration workflow.

The script captures both cameras, runs AprilTag pose estimation for each
capture, composes D435I_T_L515, saves a versioned result JSON, and prints a
Python snippet for manual insertion into the inference client.  It never
modifies runtime calibration constants itself.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess  # nosec B404 - commands are fixed argument lists
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = REPO_ROOT / "calibration" / "runs"
DEFAULT_L515_BINDING = REPO_ROOT / ".vendor" / "l515_realsense"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d435i-serial", default="401622071701")
    parser.add_argument("--l515-serial", default="f1480539")
    parser.add_argument("--tag-size-m", type=float, default=0.095)
    parser.add_argument(
        "--tag-family",
        choices=("tag36h11", "tag25h9", "tag16h5"),
        default="tag36h11",
    )
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-detections", type=int, default=10)
    parser.add_argument("--max-median-reprojection-rmse-px", type=float, default=0.5)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Output suffix (default: current local time, YYYYmmdd_HHMMSS).",
    )
    parser.add_argument("--capture-conda-env", default="pose")
    parser.add_argument("--solver-conda-env", default="wjh_grasp")
    parser.add_argument("--l515-binding", type=Path, default=DEFAULT_L515_BINDING)
    args = parser.parse_args()
    if args.tag_size_m <= 0:
        parser.error("tag-size-m must be positive")
    if min(args.frames, args.fps, args.min_detections) <= 0 or args.warmup_frames < 0:
        parser.error("frames, fps, and min-detections must be positive")
    if args.max_median_reprojection_rmse_px <= 0:
        parser.error("max-median-reprojection-rmse-px must be positive")
    allowed_run_id_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    )
    if args.run_id is not None and (
        not args.run_id
        or any(char not in allowed_run_id_characters for char in args.run_id)
    ):
        parser.error("run-id may contain only letters, digits, underscores, and hyphens")
    return args


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n+ " + " ".join(command), flush=True)
    try:
        subprocess.run(  # nosec B603 - no shell and arguments are explicit
            command,
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"Required command is unavailable: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"Calibration stopped because this command failed with exit code {exc.returncode}:\n"
            + " ".join(command)
        ) from exc


def conda_python(env_name: str, script: Path, arguments: list[str]) -> list[str]:
    return [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        str(script),
        *arguments,
    ]


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read calibration result {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"Calibration result {path} must contain a JSON object")
    return value


def read_transform(value: Any, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise SystemExit(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise SystemExit(f"{name} has an invalid homogeneous bottom row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise SystemExit(f"{name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise SystemExit(f"{name} rotation determinant is not +1")
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def timestamp_gaps_ms(capture_dir: Path) -> tuple[float, float]:
    metadata = read_json_object(capture_dir / "metadata.json")
    gaps = np.asarray(
        [
            abs(float(frame["color_timestamp_ms"]) - float(frame["depth_timestamp_ms"]))
            for frame in metadata.get("frames", [])
        ],
        dtype=np.float64,
    )
    if gaps.size == 0:
        raise SystemExit(f"No frame timestamps in {capture_dir / 'metadata.json'}")
    return float(np.median(gaps)), float(np.max(gaps))


def calibration_quality(result: dict[str, Any], label: str, threshold: float) -> dict[str, Any]:
    frames = result.get("frames", [])
    errors = np.asarray(
        [float(frame["reprojection_rmse_px"]) for frame in frames], dtype=np.float64
    )
    if errors.size == 0 or not np.all(np.isfinite(errors)):
        raise SystemExit(f"{label} solver result has no finite reprojection errors")
    median = float(np.median(errors))
    if median > threshold:
        raise SystemExit(
            f"{label} median reprojection RMSE {median:.3f}px exceeds "
            f"the configured {threshold:.3f}px limit"
        )
    cluster = result.get("dominant_pose_cluster", {})
    return {
        "valid_detections": int(result["valid_detections"]),
        "dominant_pose_cluster": int(cluster["frame_count"]),
        "excluded_frame_indices": [
            int(index) for index in cluster.get("excluded_frame_indices", [])
        ],
        "reprojection_rmse_px_median": median,
        "reprojection_rmse_px_mean": float(np.mean(errors)),
        "reprojection_rmse_px_max": float(np.max(errors)),
    }


def ensure_matching_target(reference: dict[str, Any], secondary: dict[str, Any]) -> None:
    if reference.get("tag") != secondary.get("tag"):
        raise SystemExit("D435i and L515 solver results do not use identical Tag parameters")
    for field in ("world_T_tag", "physical_tag_T_pnp_tag"):
        left = np.asarray(reference.get(field), dtype=np.float64)
        right = np.asarray(secondary.get(field), dtype=np.float64)
        if left.shape != (4, 4) or right.shape != (4, 4) or not np.allclose(left, right):
            raise SystemExit(f"D435i and L515 solver results have different {field}")


def print_copyable_result(transform: np.ndarray, l515_serial: str) -> None:
    print("\nCalibration succeeded. Copy the following values into the required runtime file:\n")
    print(f'CALIBRATED_L515_SERIAL = "{l515_serial}"')
    print("CALIBRATED_D435I_T_L515 = np.asarray(")
    print("    [")
    for row in transform:
        values = ", ".join(f"{float(value):.9f}" for value in row)
        print(f"        [{values}],")
    print("    ],")
    print("    dtype=np.float64,")
    print(")")


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = args.runs_dir.resolve()
    d435i_dir = runs_dir / f"d435i_recal_{run_id}"
    l515_dir = runs_dir / f"l515_recal_{run_id}"
    pair_path = runs_dir / f"d435i_T_l515_{run_id}.json"
    collisions = [path for path in (d435i_dir, l515_dir, pair_path) if path.exists()]
    if collisions:
        raise SystemExit(
            "Refusing to overwrite existing calibration evidence: "
            + ", ".join(map(str, collisions))
        )
    if not args.l515_binding.is_dir():
        raise SystemExit(
            f"L515-compatible pyrealsense2 binding not found: {args.l515_binding}\n"
            "Install it as documented in calibration/README.md."
        )
    runs_dir.mkdir(parents=True, exist_ok=True)

    collector = REPO_ROOT / "calibration" / "collect_camera_world_data.py"
    solver = REPO_ROOT / "calibration" / "calibrate_camera_to_world.py"
    capture_env = os.environ.copy()
    existing_pythonpath = capture_env.get("PYTHONPATH")
    capture_env["PYTHONPATH"] = str(args.l515_binding.resolve()) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )

    print(
        "Keep both cameras and the shared AprilTag completely fixed until both captures finish.\n"
        f"Run ID: {run_id}\n"
        f"Tag: {args.tag_family}, ID {args.tag_id}, black-square size {args.tag_size_m:.6f} m",
        flush=True,
    )

    common_capture = [
        "--frames", str(args.frames),
        "--warmup-frames", str(args.warmup_frames),
        "--fps", str(args.fps),
    ]
    run_checked(
        conda_python(
            args.capture_conda_env,
            collector,
            [
                "--camera-serial", args.d435i_serial,
                "--output", str(d435i_dir),
                "--width", "640", "--height", "480",
                *common_capture,
            ],
        ),
        env=capture_env,
    )
    common_solver = [
        "--tag-size-m", str(args.tag_size_m),
        "--tag-family", args.tag_family,
        "--tag-id", str(args.tag_id),
        "--min-detections", str(args.min_detections),
    ]
    run_checked(
        conda_python(
            args.solver_conda_env,
            solver,
            ["--input", str(d435i_dir), *common_solver],
        )
    )
    d435i_result_path = d435i_dir / "camera_to_world.json"
    d435i_result = read_json_object(d435i_result_path)
    d435i_quality = calibration_quality(
        d435i_result, "D435i", args.max_median_reprojection_rmse_px
    )
    print(
        f"D435i quality gate passed: {d435i_quality['valid_detections']} detections, "
        f"median RMSE {d435i_quality['reprojection_rmse_px_median']:.3f}px",
        flush=True,
    )

    run_checked(
        conda_python(
            args.capture_conda_env,
            collector,
            [
                "--camera-serial", args.l515_serial,
                "--output", str(l515_dir),
                "--width", "1280", "--height", "720",
                "--depth-width", "640", "--depth-height", "480",
                *common_capture,
            ],
        ),
        env=capture_env,
    )
    run_checked(
        conda_python(
            args.solver_conda_env,
            solver,
            ["--input", str(l515_dir), *common_solver],
        )
    )

    l515_result_path = l515_dir / "camera_to_world.json"
    l515_result = read_json_object(l515_result_path)
    ensure_matching_target(d435i_result, l515_result)
    l515_quality = calibration_quality(
        l515_result, "L515", args.max_median_reprojection_rmse_px
    )

    world_t_d435i = read_transform(d435i_result.get("world_T_camera"), "WORLD_T_D435I")
    world_t_l515 = read_transform(l515_result.get("world_T_camera"), "WORLD_T_L515")
    d435i_t_l515 = read_transform(
        invert_transform(world_t_d435i) @ world_t_l515, "D435I_T_L515"
    )
    d435i_gap_median, d435i_gap_max = timestamp_gaps_ms(d435i_dir)
    l515_gap_median, l515_gap_max = timestamp_gaps_ms(l515_dir)

    result = {
        "format": "real_exp_realsense_pair_calibration_v1",
        "reference_camera": {
            "serial": args.d435i_serial,
            "source": str(d435i_result_path),
        },
        "secondary_camera": {
            "serial": args.l515_serial,
            "source": str(l515_result_path),
        },
        "tag": d435i_result["tag"],
        "reference_T_secondary_camera": d435i_t_l515.tolist(),
        "quality": {
            "d435i": d435i_quality,
            "l515": l515_quality,
            "optical_origin_baseline_m": float(np.linalg.norm(d435i_t_l515[:3, 3])),
            "d435i_color_depth_timestamp_gap_median_ms": d435i_gap_median,
            "d435i_color_depth_timestamp_gap_max_ms": d435i_gap_max,
            "l515_color_depth_timestamp_gap_median_ms": l515_gap_median,
            "l515_color_depth_timestamp_gap_max_ms": l515_gap_max,
        },
        "method": "shared_static_apriltag_target",
    }
    try:
        with pair_path.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
            output.write("\n")
    except OSError as exc:
        raise SystemExit(f"Cannot save pair calibration {pair_path}: {exc}") from exc

    print(
        f"\nD435i: {d435i_quality['valid_detections']} detections, "
        f"{d435i_quality['dominant_pose_cluster']} in dominant cluster, "
        f"median RMSE {d435i_quality['reprojection_rmse_px_median']:.3f}px"
    )
    print(
        f"L515:  {l515_quality['valid_detections']} detections, "
        f"{l515_quality['dominant_pose_cluster']} in dominant cluster, "
        f"median RMSE {l515_quality['reprojection_rmse_px_median']:.3f}px"
    )
    print(f"Pair result saved to: {pair_path}")
    print_copyable_result(d435i_t_l515, args.l515_serial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
