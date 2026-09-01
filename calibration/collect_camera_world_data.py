#!/usr/bin/env python3
"""Collect RealSense data for later camera-to-world AprilTag calibration.

This program deliberately does not estimate an extrinsic transform. It records
the raw RGB-D observations, the intrinsics/extrinsics reported by RealSense,
and timestamps. AprilTag detection and all PnP/SE(3) computation belong to a
separate offline script.

Example:
    python calibration/collect_camera_world_data.py \
        --output calibration/runs/table_tag_20260824 \
        --frames 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


# The tag center is the world origin. Columns are the AprilTag +x, +y, +z
# axes expressed in the world frame: right, backward, down respectively.
WORLD_T_TAG = np.asarray(
    [
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _intrinsics(profile: Any) -> dict[str, Any]:
    intr = profile.get_intrinsics()
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(item) for item in intr.coeffs],
    }


def _extrinsics(extr: Any) -> dict[str, Any]:
    return {
        "rotation_row_major": [float(item) for item in extr.rotation],
        "translation_m": [float(item) for item in extr.translation],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--camera-serial",
        default=None,
        help=(
            "RealSense serial to open. Required when more than one camera is "
            "connected; use `rs-enumerate-devices -s` to list serials."
        ),
    )
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--depth-width",
        type=int,
        default=None,
        help="Native depth width (default: --width).",
    )
    parser.add_argument(
        "--depth-height",
        type=int,
        default=None,
        help="Native depth height (default: --height).",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frame-gap-ms", type=float, default=100.0)
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup_frames < 0:
        parser.error("frames must be positive and warmup-frames non-negative")
    depth_width = args.depth_width or args.width
    depth_height = args.depth_height or args.height
    if min(args.width, args.height, depth_width, depth_height, args.fps) <= 0:
        parser.error("color/depth dimensions and fps must be positive")

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "This collector requires pyrealsense2 and numpy on the real-exp runtime. "
            f"Missing dependency: {exc.name}"
        ) from exc

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "rgb").mkdir()
    (args.output / "depth").mkdir()
    pipeline = rs.pipeline()
    config = rs.config()
    if args.camera_serial:
        config.enable_device(args.camera_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    sensor = profile.get_device().first_depth_sensor()
    metadata: dict[str, Any] = {
        "format": "real_exp_camera_world_calibration_v1",
        "created_unix_s": time.time(),
        "device": {
            "name": str(profile.get_device().get_info(rs.camera_info.name)),
            "serial": str(profile.get_device().get_info(rs.camera_info.serial_number)),
            "firmware": str(profile.get_device().get_info(rs.camera_info.firmware_version)),
        },
        "stream": {
            "color": [args.width, args.height, args.fps],
            "depth": [depth_width, depth_height, args.fps],
        },
        "coordinate_frames": {
            "world": "+x forward, +y left, +z up",
            "apriltag": "+x right, +y backward, +z down",
        },
        "tag_pose_definition": "Tag center coincides with the world origin; world_T_tag maps the AprilTag axes into the world axes.",
        "world_T_tag": WORLD_T_TAG,
        "color_intrinsics": _intrinsics(color_profile),
        "depth_intrinsics": _intrinsics(depth_profile),
        "saved_depth_frame": "depth_aligned_to_color",
        "saved_depth_intrinsics": "color_intrinsics",
        "depth_scale_m": float(sensor.get_depth_scale()),
        "depth_to_color": _extrinsics(depth_profile.get_extrinsics_to(color_profile)),
        "frames": [],
    }

    saved = 0
    try:
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames()
        while saved < args.frames:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            color_stamp_ms = float(color_frame.get_timestamp())
            depth_stamp_ms = float(depth_frame.get_timestamp())
            if abs(color_stamp_ms - depth_stamp_ms) > args.max_frame_gap_ms:
                continue

            stem = f"{saved:06d}"
            np.save(args.output / "rgb" / f"{stem}.npy", color)
            np.save(args.output / "depth" / f"{stem}.npy", depth)
            metadata["frames"].append({
                "index": saved,
                "rgb_file": f"rgb/{stem}.npy",
                "depth_file": f"depth/{stem}.npy",
                "color_timestamp_ms": color_stamp_ms,
                "depth_timestamp_ms": depth_stamp_ms,
            })
            saved += 1
            print(f"saved {saved}/{args.frames}", flush=True)
    finally:
        pipeline.stop()
        metadata["saved_frame_count"] = saved
        (args.output / "metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2) + "\n", encoding="utf-8")

    if saved < args.frames:
        raise SystemExit(f"Only saved {saved} frames before capture stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
