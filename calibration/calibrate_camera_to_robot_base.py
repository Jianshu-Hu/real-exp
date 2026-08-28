#!/usr/bin/env python3
"""Calibrate camera-to-Franka-base from captured eye-to-hand samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def invert(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def pose_error(observed: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    relative = invert(observed) @ predicted
    return np.concatenate((relative[:3, 3], Rotation.from_matrix(relative[:3, :3]).as_rotvec()))


def detect_tag(cv2: Any, image: np.ndarray, family: str, tag_id: int) -> np.ndarray | None:
    aruco = cv2.aruco
    dictionary_names = {
        "tag36h11": "DICT_APRILTAG_36h11",
        "tag25h9": "DICT_APRILTAG_25h9",
        "tag16h5": "DICT_APRILTAG_16h5",
    }
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_names[family]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(aruco, "ArucoDetector"):
        corners, ids, _ = aruco.ArucoDetector(dictionary).detectMarkers(gray)
    else:
        corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=aruco.DetectorParameters_create())
    if ids is None:
        return None
    for corner, detected_id in zip(corners, ids.reshape(-1), strict=False):
        if int(detected_id) == tag_id:
            return np.asarray(corner, dtype=np.float64).reshape(4, 2)
    return None


def read_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="camera host eye-to-hand sample directory")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tag-size-m", type=float, required=True)
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-family", choices=("tag36h11", "tag25h9", "tag16h5"), default="tag36h11")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument(
        "--exclude-sample",
        action="append",
        default=[],
        metavar="ID",
        help="Exclude a sample ID such as 000014; repeat for multiple samples.",
    )
    args = parser.parse_args()
    if args.tag_size_m <= 0 or args.min_samples < 3:
        parser.error("tag-size-m must be positive and min-samples must be at least 3")

    try:
        import cv2
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise SystemExit(f"calibration processing requires opencv-contrib-python, scipy, and numpy: missing {exc.name}") from exc

    sample_dirs = sorted(path for path in args.input.glob("sample_*") if (path / "sample.json").is_file())
    if not sample_dirs:
        raise SystemExit(f"no sample_*/sample.json entries found under {args.input}")
    first = json.loads((sample_dirs[0] / "sample.json").read_text(encoding="utf-8"))
    intrinsics = first["color_intrinsics"]
    camera_serial = first.get("camera_serial")
    camera_matrix = np.asarray([[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.asarray(intrinsics.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
    half = args.tag_size_m / 2.0
    tag_points = np.asarray([[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float64)

    observations: list[dict[str, Any]] = []
    excluded_samples = set(args.exclude_sample)
    seen_sample_ids: set[str] = set()
    for sample_dir in sample_dirs:
        sample = json.loads((sample_dir / "sample.json").read_text(encoding="utf-8"))
        sample_id = str(sample["sample_id"])
        seen_sample_ids.add(sample_id)
        if sample.get("color_intrinsics") != intrinsics:
            raise ValueError(f"{sample_dir} color_intrinsics differ from the first sample")
        if sample.get("camera_serial") != camera_serial:
            raise ValueError(f"{sample_dir} camera_serial differs from the first sample")
        if sample_id in excluded_samples:
            continue
        image = np.load(sample_dir / "rgb.npy")
        corners = detect_tag(cv2, image, args.tag_family, args.tag_id)
        if corners is None:
            continue
        ok, rvec, tvec = cv2.solvePnP(tag_points, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        camera_t_tag = np.eye(4, dtype=np.float64)
        camera_t_tag[:3, :3] = rotation
        camera_t_tag[:3, 3] = tvec.reshape(3)
        projected, _ = cv2.projectPoints(tag_points, rvec, tvec, camera_matrix, distortion)
        reprojection = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1))))
        observations.append({"sample_id": sample_id, "camera_T_tag": camera_t_tag, "base_T_ee": read_matrix(sample["B_T_E"], "B_T_E"), "reprojection_rmse_px": reprojection})

    unknown_exclusions = excluded_samples - seen_sample_ids
    if unknown_exclusions:
        raise ValueError(f"excluded sample IDs were not found: {sorted(unknown_exclusions)}")

    if len(observations) < args.min_samples:
        raise SystemExit(f"only {len(observations)} valid Tag observations; need at least {args.min_samples}")

    def decode(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera_t_base = np.eye(4, dtype=np.float64)
        ee_t_tag = np.eye(4, dtype=np.float64)
        camera_t_base[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
        camera_t_base[:3, 3] = parameters[3:6]
        ee_t_tag[:3, :3] = Rotation.from_rotvec(parameters[6:9]).as_matrix()
        ee_t_tag[:3, 3] = parameters[9:12]
        return camera_t_base, ee_t_tag

    def residual(parameters: np.ndarray) -> np.ndarray:
        camera_t_base, ee_t_tag = decode(parameters)
        return np.concatenate([pose_error(item["camera_T_tag"], camera_t_base @ item["base_T_ee"] @ ee_t_tag) for item in observations])

    initial_guesses: list[np.ndarray] = []
    for ee_t_tag in (np.eye(4, dtype=np.float64), np.block([[Rotation.from_euler("z", 135, degrees=True).as_matrix(), np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])):
        camera_t_base = observations[0]["camera_T_tag"] @ invert(observations[0]["base_T_ee"] @ ee_t_tag)
        initial_guesses.append(np.concatenate((Rotation.from_matrix(camera_t_base[:3, :3]).as_rotvec(), camera_t_base[:3, 3], Rotation.from_matrix(ee_t_tag[:3, :3]).as_rotvec(), ee_t_tag[:3, 3])))

    solutions = []
    for initial in initial_guesses:
        solution = least_squares(residual, initial, loss="soft_l1", f_scale=0.01, max_nfev=3000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        solutions.append(solution)
    solution = min(solutions, key=lambda item: float(np.mean(residual(item.x) ** 2)))
    camera_t_base, ee_t_tag = decode(solution.x)
    per_sample = []
    for item in observations:
        error = pose_error(item["camera_T_tag"], camera_t_base @ item["base_T_ee"] @ ee_t_tag)
        per_sample.append({"sample_id": item["sample_id"], "reprojection_rmse_px": item["reprojection_rmse_px"], "translation_error_m": float(np.linalg.norm(error[:3])), "rotation_error_rad": float(np.linalg.norm(error[3:])), "camera_T_tag": item["camera_T_tag"].tolist(), "base_T_ee": item["base_T_ee"].tolist()})

    output = args.output or (args.input / "camera_to_robot_base.json")
    result = {"format": "real_exp_camera_to_robot_base_v1", "input": str(args.input), "camera_serial": camera_serial, "color_intrinsics": intrinsics, "tag": {"family": args.tag_family, "id": args.tag_id, "size_m": args.tag_size_m}, "excluded_samples": sorted(excluded_samples), "camera_T_base": camera_t_base.tolist(), "base_T_camera": invert(camera_t_base).tolist(), "ee_T_tag": ee_t_tag.tolist(), "valid_samples": len(observations), "median_reprojection_rmse_px": float(np.median([item["reprojection_rmse_px"] for item in observations])), "median_translation_residual_m": float(np.median([item["translation_error_m"] for item in per_sample])), "median_rotation_residual_rad": float(np.median([item["rotation_error_rad"] for item in per_sample])), "samples": per_sample}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output} from {len(observations)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
