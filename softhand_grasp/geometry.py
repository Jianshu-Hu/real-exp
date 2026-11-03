"""Aligned depth to a visible-surface grasp point and calibrated EE poses."""

import json
from pathlib import Path

import numpy as np

from data_collection import move_to_target_ee as move

DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def rigid_transform(value, name):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f"{name} must be a rigid 4x4 transform")
    return matrix


def validate_pose(pose):
    pose = vector(pose, 6, "EE pose")
    if (np.any(pose[:3] < move.EE_POSITION_LOWER_M)
            or np.any(pose[:3] > move.EE_POSITION_UPPER_M)
            or np.linalg.norm(pose[:3]) > move.EE_POSITION_MAX_RADIUS_M):
        raise ValueError(f"EE position outside the right-arm workspace: {pose[:3]}")
    return pose


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    vector(config["grasp_rpy"], 3, "grasp_rpy")
    vector(config["ee_to_grasp_point_m"], 3, "ee_to_grasp_point_m")
    for name in ("camera_T_right_base", "world_T_camera"):
        rigid_transform(config[name], name)
    for name in ("lift_distance_m", "depth_inlier_band_m"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name, lower in (("mask_erode_pixels", 0), ("min_depth_pixels", 1)):
        if not isinstance(config[name], int) or config[name] < lower:
            raise ValueError(f"{name} must be an integer >= {lower}")
    low, high = vector(config["depth_range_m"], 2, "depth_range_m")
    if not 0 < low < high:
        raise ValueError("depth_range_m must be positive and increasing")
    if not 0 <= config["score_threshold"] <= 1:
        raise ValueError("score_threshold must be in [0, 1]")
    if config["lift_frame"] not in ("world", "base", "ee"):
        raise ValueError("lift_frame must be world, base, or ee")
    if not config["camera_serial"]:
        raise ValueError("camera_serial is required")
    return config


def surface_point(mask, depth_m, intrinsics, config):
    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=float)
    if mask.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("Mask and aligned depth must have identical HxW shapes")
    for _ in range(config["mask_erode_pixels"]):
        padded = np.pad(mask, 1, constant_values=False)
        mask = np.logical_and.reduce([
            padded[y:y + mask.shape[0], x:x + mask.shape[1]]
            for y in range(3) for x in range(3)
        ])
    low, high = config["depth_range_m"]
    valid = mask & np.isfinite(depth) & (depth >= low) & (depth <= high)
    if valid.sum() < config["min_depth_pixels"]:
        raise ValueError("Selected mask has too few valid interior depth pixels")
    median_z = np.median(depth[valid])
    valid &= np.abs(depth - median_z) <= config["depth_inlier_band_m"]
    if valid.sum() < config["min_depth_pixels"]:
        raise ValueError("Selected mask has too few depth inliers")
    fx, fy, cx, cy = vector([intrinsics[k] for k in ("fx", "fy", "cx", "cy")], 4, "intrinsics")
    if fx <= 0 or fy <= 0:
        raise ValueError("Camera focal lengths must be positive")
    if not np.allclose(intrinsics.get("coeffs", [0] * 5), 0):
        raise ValueError("Nonzero distortion requires undistortion before deprojection")
    v, u = np.nonzero(valid)
    z = depth[v, u]
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return np.median(points, axis=0), int(len(z))


def target_poses(point_camera, config):
    point_camera = vector(point_camera, 3, "point_camera")
    base_T_camera = np.linalg.inv(rigid_transform(config["camera_T_right_base"], "camera_T_right_base"))
    point_base = (base_T_camera @ np.r_[point_camera, 1])[:3]
    rpy = vector(config["grasp_rpy"], 3, "grasp_rpy")
    rotation = move.rpy_to_rotation(*rpy)
    # The configured point is expressed in the controller's EE frame.
    ee_position = point_base - rotation @ np.asarray(config["ee_to_grasp_point_m"])
    grasp = validate_pose(np.r_[ee_position, rpy])
    delta = np.array([0., 0., config["lift_distance_m"]])
    if config["lift_frame"] == "world":
        world_T_base = np.asarray(config["world_T_camera"]) @ np.asarray(config["camera_T_right_base"])
        delta = np.linalg.inv(world_T_base)[:3, :3] @ delta
    elif config["lift_frame"] == "ee":
        delta = rotation @ delta
    lift = validate_pose(np.r_[ee_position + delta, rpy])
    return {"object_point_camera_m": point_camera.tolist(),
            "object_point_base_m": point_base.tolist(),
            "grasp_pose_xyzrpy": grasp.tolist(), "lift_pose_xyzrpy": lift.tolist()}
