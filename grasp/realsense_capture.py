#!/usr/bin/env python3
"""Capture and temporally fuse aligned D435 RGB-D frames for inference."""

from __future__ import annotations

import argparse
import json
import math
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


def _fuse_depth_frames(
    depth_frames: list[np.ndarray], min_valid_depth_ratio: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a per-pixel valid-only median and its temporal support count."""
    if not depth_frames:
        raise ValueError("at least one depth frame is required")
    stack = np.stack(depth_frames, axis=0)
    if stack.ndim != 3:
        raise ValueError(f"depth frames must be 2-D and equally shaped, got {stack.shape}")

    valid = stack != 0
    valid_count = valid.sum(axis=0)
    required_count = math.ceil(len(depth_frames) * min_valid_depth_ratio)
    masked = np.ma.array(stack, mask=~valid)
    fused = np.ma.median(masked, axis=0).filled(0)
    fused = np.rint(fused).astype(stack.dtype, copy=False)
    fused[valid_count < required_count] = 0
    return fused, valid_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-serial", default="401622071701")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--observation-frames", type=int, default=15)
    parser.add_argument("--min-valid-depth-ratio", type=float, default=0.5)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("camera width, height, and fps must be positive")
    if args.warmup_frames < 0:
        parser.error("warmup-frames must be non-negative")
    if args.observation_frames <= 0:
        parser.error("observation-frames must be positive")
    if not 0.0 < args.min_valid_depth_ratio <= 1.0:
        parser.error("min-valid-depth-ratio must be in (0, 1]")

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
        depth_frames: list[np.ndarray] = []
        color_timestamps_ms: list[float] = []
        depth_timestamps_ms: list[float] = []
        rgb: np.ndarray | None = None
        color_frame = None
        while len(depth_frames) < args.observation_frames:
            aligned = align.process(pipeline.wait_for_frames())
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            rgb = np.asanyarray(color_frame.get_data()).copy()
            depth_frames.append(np.asanyarray(depth_frame.get_data()).copy())
            color_timestamps_ms.append(float(color_frame.get_timestamp()))
            depth_timestamps_ms.append(float(depth_frame.get_timestamp()))

        if rgb is None or color_frame is None:
            raise RuntimeError("D435 returned no complete aligned RGB-D frames")
        depth, valid_count = _fuse_depth_frames(
            depth_frames, args.min_valid_depth_ratio
        )
        required_valid_frames = math.ceil(
            args.observation_frames * args.min_valid_depth_ratio
        )
        metadata = {
            "intrinsics": _intrinsics_dict(color_frame.profile.as_video_stream_profile()),
            "depth_scale_m": float(profile.get_device().first_depth_sensor().get_depth_scale()),
            "camera_serial": str(profile.get_device().get_info(rs.camera_info.serial_number)),
            "color_timestamp_ms": color_timestamps_ms[-1],
            "depth_timestamp_ms": depth_timestamps_ms[-1],
            "observation": {
                "frame_count": args.observation_frames,
                "aggregation": "per_pixel_nonzero_median",
                "min_valid_depth_ratio": args.min_valid_depth_ratio,
                "required_valid_frames": required_valid_frames,
                "retained_depth_pixels": int(np.count_nonzero(depth)),
                "pixels_valid_in_all_frames": int(
                    np.count_nonzero(valid_count == args.observation_frames)
                ),
                "first_depth_timestamp_ms": depth_timestamps_ms[0],
                "last_depth_timestamp_ms": depth_timestamps_ms[-1],
                "duration_ms": depth_timestamps_ms[-1] - depth_timestamps_ms[0],
                "rgb_frame": "last_observation_frame",
            },
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
