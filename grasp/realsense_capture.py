#!/usr/bin/env python3
"""Capture one aligned D435 RGB-D frame for the lerobot inference process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _intrinsics_dict(profile: Any) -> dict[str, Any]:
    intrinsics = profile.get_intrinsics()
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-serial", default="401622071701")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            f"system camera capture requires pyrealsense2: missing {exc.name}"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.camera_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames()
        aligned = align.process(pipeline.wait_for_frames())
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("D435 returned an incomplete aligned RGB-D frame")
        rgb = np.asanyarray(color_frame.get_data()).copy()
        depth = np.asanyarray(depth_frame.get_data()).copy()
        metadata = {
            "intrinsics": _intrinsics_dict(color_frame.profile.as_video_stream_profile()),
            "depth_scale_m": float(profile.get_device().first_depth_sensor().get_depth_scale()),
            "camera_serial": str(profile.get_device().get_info(rs.camera_info.serial_number)),
            "color_timestamp_ms": float(color_frame.get_timestamp()),
            "depth_timestamp_ms": float(depth_frame.get_timestamp()),
            "source": "system_python_realsense",
        }
        np.save(args.output_dir / "rgb.npy", rgb)
        np.save(args.output_dir / "depth.npy", depth)
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
