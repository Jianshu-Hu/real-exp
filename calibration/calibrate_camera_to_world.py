#!/usr/bin/env python3
"""Estimate camera-to-world from a captured tabletop AprilTag sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_WORLD_T_TAG = np.asarray(
    [
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# For the installed marker, the ArUco corner order makes IPPE_SQUARE +x point
# toward the physical marker's left edge, +y toward its bottom edge, and +z out
# of the paper.  The physical frame uses +x right, +y bottom, and +z into the
# paper, so the two frames differ by 180 degrees about physical +y.
PHYSICAL_TAG_T_PNP_TAG = np.asarray(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def world_t_tag_from_metadata(metadata: dict[str, Any]) -> np.ndarray:
    """Return the fixed tabletop Tag pose, including legacy-data fallback."""
    transform = np.asarray(
        metadata.get("world_T_tag", DEFAULT_WORLD_T_TAG), dtype=np.float64
    )
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("metadata world_T_tag must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("metadata world_T_tag must be a homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("metadata world_T_tag rotation must be right-handed and orthonormal")
    return transform


def invert(transform: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def matrix_to_list(transform: np.ndarray) -> list[list[float]]:
    return np.asarray(transform, dtype=np.float64).tolist()


def detect_tag(cv2: Any, image: np.ndarray, family: str, tag_id: int) -> np.ndarray | None:
    aruco = cv2.aruco
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    names = {
        "tag36h11": "DICT_APRILTAG_36h11",
        "tag25h9": "DICT_APRILTAG_25h9",
        "tag16h5": "DICT_APRILTAG_16h5",
    }
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, names[family]))
    if hasattr(aruco, "ArucoDetector"):
        corners, ids, _ = aruco.ArucoDetector(dictionary).detectMarkers(image)
    else:
        corners, ids, _ = aruco.detectMarkers(
            image, dictionary, parameters=aruco.DetectorParameters_create()
        )
    if ids is None:
        return None
    for corner, detected_id in zip(corners, ids.reshape(-1), strict=False):
        if int(detected_id) == tag_id:
            return np.asarray(corner, dtype=np.float64).reshape(4, 2)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Collector output directory")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tag-size-m", type=float, required=True)
    parser.add_argument("--tag-id", type=int, default=0)
    parser.add_argument("--tag-family", choices=("tag36h11", "tag25h9", "tag16h5"), default="tag36h11")
    parser.add_argument("--min-detections", type=int, default=10)
    args = parser.parse_args()
    if args.tag_size_m <= 0 or args.min_detections <= 0:
        parser.error("tag-size-m and min-detections must be positive")

    try:
        import cv2
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise SystemExit(f"Calibration processing requires opencv-contrib-python and scipy: missing {exc.name}") from exc

    metadata_path = args.input / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    intrinsics = metadata["color_intrinsics"]
    camera_matrix = np.array(
        [[intrinsics["fx"], 0.0, intrinsics["cx"]], [0.0, intrinsics["fy"], intrinsics["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.asarray(intrinsics.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
    half = args.tag_size_m / 2.0
    object_points = np.asarray(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )
    # Metadata describes the physical printed Tag frame, while solvePnP below
    # returns the IPPE Tag frame dictated by object_points.
    world_t_tag = world_t_tag_from_metadata(metadata)
    world_t_pnp_tag = world_t_tag @ PHYSICAL_TAG_T_PNP_TAG
    records: list[dict[str, Any]] = []
    for frame in metadata.get("frames", []):
        image = np.load(args.input / frame["rgb_file"])
        corners = detect_tag(cv2, image, args.tag_family, args.tag_id)
        if corners is None:
            continue
        ok, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        rotation, _ = cv2.Rodrigues(rvec)
        camera_t_tag = np.eye(4, dtype=np.float64)
        camera_t_tag[:3, :3] = rotation
        camera_t_tag[:3, 3] = tvec.reshape(3)
        world_t_camera = world_t_pnp_tag @ invert(camera_t_tag)
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        reprojection_px = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - corners) ** 2, axis=1))))
        records.append({"index": frame["index"], "world_T_camera": matrix_to_list(world_t_camera), "camera_T_tag": matrix_to_list(camera_t_tag), "reprojection_rmse_px": reprojection_px})

    if len(records) < args.min_detections:
        raise SystemExit(f"Only {len(records)} valid tag detections; need at least {args.min_detections}")
    translations = np.asarray([record["world_T_camera"] for record in records], dtype=np.float64)[:, :3, 3]
    rotations = Rotation.from_matrix(np.asarray([record["world_T_camera"] for record in records])[:, :3, :3])
    median_translation = np.median(translations, axis=0)
    mean_rotation = Rotation.from_quat(rotations.as_quat()).mean()
    estimate = np.eye(4, dtype=np.float64)
    estimate[:3, :3] = mean_rotation.as_matrix()
    estimate[:3, 3] = median_translation
    result = {
        "format": "real_exp_camera_to_world_v1",
        "input": str(args.input),
        "tag": {"family": args.tag_family, "id": args.tag_id, "size_m": args.tag_size_m},
        "world_T_tag": matrix_to_list(world_t_tag),
        "physical_tag_T_pnp_tag": matrix_to_list(PHYSICAL_TAG_T_PNP_TAG),
        "world_T_camera": matrix_to_list(estimate),
        "valid_detections": len(records),
        "reprojection_rmse_px_median": float(
            np.median([record["reprojection_rmse_px"] for record in records])
        ),
        "frames": records,
    }
    output = args.output or (args.input / "camera_to_world.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved {output} from {len(records)} detections; median reprojection RMSE={result['reprojection_rmse_px_median']:.3f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
