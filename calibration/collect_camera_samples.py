#!/usr/bin/env python3
"""Capture one local D435 RGB-D sample and the latest robot state on Enter."""

from __future__ import annotations

import argparse
import json
import select
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

TOP_CAMERA_WIDTH = 640
TOP_CAMERA_HEIGHT = 480
TOP_CAMERA_FPS = 30
TOP_CAMERA_SERIAL = "401622071701"


def pose_to_matrix(pose: Any) -> list[list[float]]:
    x, y, z, w = (float(pose.orientation.x), float(pose.orientation.y), float(pose.orientation.z), float(pose.orientation.w))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [float(pose.position.x), float(pose.position.y), float(pose.position.z)]
    transform[:3, :3] = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return transform.tolist()


def intrinsics(profile: Any) -> dict[str, Any]:
    value = profile.get_intrinsics()
    return {
        "width": int(value.width), "height": int(value.height),
        "fx": float(value.fx), "fy": float(value.fy),
        "cx": float(value.ppx), "cy": float(value.ppy),
        "model": str(value.model), "coeffs": [float(item) for item in value.coeffs],
    }


class CameraCollector:
    def __init__(self, args: argparse.Namespace, rs: Any, node: Any, JointState: Any, FrankaRobotState: Any) -> None:
        import pyrealsense2 as _rs

        self.args = args
        self.rs = rs
        self.node = node
        self.pipeline = _rs.pipeline()
        config = _rs.config()
        config.enable_device(TOP_CAMERA_SERIAL)
        config.enable_stream(_rs.stream.color, TOP_CAMERA_WIDTH, TOP_CAMERA_HEIGHT, _rs.format.bgr8, TOP_CAMERA_FPS)
        config.enable_stream(_rs.stream.depth, TOP_CAMERA_WIDTH, TOP_CAMERA_HEIGHT, _rs.format.z16, TOP_CAMERA_FPS)
        self.profile = self.pipeline.start(config)
        self.align = _rs.align(_rs.stream.color)
        self.color_profile = self.profile.get_stream(_rs.stream.color).as_video_stream_profile()
        self.depth_profile = self.profile.get_stream(_rs.stream.depth).as_video_stream_profile()
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.node.create_subscription(FrankaRobotState, f"/{args.side}/franka_robot_state_broadcaster/robot_state", self.on_robot_state, 10)
        self.node.create_subscription(JointState, f"/{args.side}/franka/joint_states", self.on_joint_state, 10)
        self.robot_state: Any | None = None
        self.joint_state: Any | None = None
        self.robot_state_received_unix_s: float | None = None
        self.next_sample = 0
        self.get_logger().info("Camera collector ready; press Enter to capture a sample")

    def on_robot_state(self, message: Any) -> None:
        self.robot_state = message
        self.robot_state_received_unix_s = time.time()

    def on_joint_state(self, message: Any) -> None:
        self.joint_state = message

    def capture(self) -> None:
        if self.robot_state is None or self.joint_state is None:
            self.get_logger().warning("robot state is not ready; sample not captured")
            return
        sample_id = f"{self.next_sample:06d}"
        self.next_sample += 1
        try:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("RGB-D frame is incomplete")
            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            sample_dir = self.output / f"sample_{sample_id}"
            sample_dir.mkdir(parents=False, exist_ok=False)
            np.save(sample_dir / "rgb.npy", color)
            np.save(sample_dir / "depth.npy", depth)
            metadata = {
                "sample_id": sample_id,
                "side": self.args.side,
                "camera_serial": self.profile.get_device().get_info(self.rs.camera_info.serial_number),
                "stream": {"width": TOP_CAMERA_WIDTH, "height": TOP_CAMERA_HEIGHT, "fps": TOP_CAMERA_FPS},
                "captured_unix_s": time.time(),
                "robot_state_received_unix_s": self.robot_state_received_unix_s,
                "color_timestamp_ms": float(color_frame.get_timestamp()),
                "depth_timestamp_ms": float(depth_frame.get_timestamp()),
                "color_intrinsics": intrinsics(self.color_profile),
                "depth_intrinsics": intrinsics(self.depth_profile),
                "depth_scale_m": float(self.profile.get_device().first_depth_sensor().get_depth_scale()),
                "saved_depth_frame": "depth_aligned_to_color",
                "B_T_E": pose_to_matrix(self.robot_state.o_t_ee.pose),
                "F_T_E": pose_to_matrix(self.robot_state.f_t_ee.pose),
                "joint_names": [str(name) for name in self.joint_state.name],
                "joint_position": [float(value) for value in self.joint_state.position],
                "joint_velocity": [float(value) for value in self.joint_state.velocity],
            }
            (sample_dir / "sample.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            self.get_logger().info(f"saved {sample_dir}")
        except Exception as exc:
            self.get_logger().error(str(exc))

    def get_logger(self) -> Any:
        return self.node.get_logger()

    def close(self) -> None:
        self.pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    args = parser.parse_args()
    try:
        import pyrealsense2 as rs
        from franka_msgs.msg import FrankaRobotState
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise SystemExit(f"camera host needs pyrealsense2, rclpy, franka_msgs, sensor_msgs, and numpy: missing {exc.name}") from exc
    rclpy.init()
    node = Node("real_exp_camera_calibration_collector")
    collector = CameraCollector(args, rs, node, JointState, FrankaRobotState)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if ready:
                sys.stdin.readline()
                collector.capture()
    finally:
        collector.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
