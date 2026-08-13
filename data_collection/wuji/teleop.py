#!/usr/bin/env python3
"""Publish Wuji glove samples as FR3/GELLO-compatible ROS 2 commands."""

from __future__ import annotations

import argparse
import importlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

try:  # Works both as ``python teleop.py`` and ``python -m data_collection.wuji.teleop``.
    from .mapping import WujiJointMapper
except ImportError:  # pragma: no cover - direct script execution
    from mapping import WujiJointMapper


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _first_value(obj: Any, names: tuple[str, ...]) -> Any:
    """Look through a sample and its common nested state containers."""
    value = _value(obj, *names)
    if value is not None:
        return value
    for container_name in ("state", "data", "sample", "hand", "glove"):
        container = _value(obj, container_name)
        if container is not None:
            value = _value(container, *names)
            if value is not None:
                return value
    return None


class WujiGloveDevice:
    """Small compatibility adapter around the vendor-provided Wuji SDK."""

    def __init__(self, module_name: str, class_name: str, device_id: str | None) -> None:
        module = None
        import_errors: list[str] = []
        module_candidates = [module_name]
        if module_name == "auto":
            module_candidates = ["wuji", "wuji_glove", "wuji_sdk"]
        for candidate in module_candidates:
            try:
                module = importlib.import_module(candidate)
                break
            except ModuleNotFoundError as exc:
                import_errors.append(f"{candidate}: {exc}")
        if module is None:
            raise RuntimeError(
                "Could not import a Wuji SDK module. Install the vendor SDK or pass --device-module. "
                + "; ".join(import_errors)
            )
        cls = getattr(module, class_name, None)
        if cls is None:
            for candidate in ("WujiGlove", "Glove", "Wuji"):
                cls = getattr(module, candidate, None)
                if cls is not None:
                    break
        if cls is None:
            raise RuntimeError(
                f"{module.__name__!r} exposes no {class_name!r}, WujiGlove, Glove, or Wuji class"
            )
        self._device = self._construct(cls, device_id)

    @staticmethod
    def _construct(cls: Any, device_id: str | None) -> Any:
        attempts = []
        if device_id is not None:
            attempts.extend((lambda: cls(device_id=device_id), lambda: cls(device_id)))
        attempts.append(cls)
        last: Exception | None = None
        for attempt in attempts:
            try:
                device = attempt()
                for method in ("connect", "open", "start"):
                    fn = getattr(device, method, None)
                    if callable(fn):
                        try:
                            result = fn()
                        except TypeError:
                            if device_id is None:
                                raise
                            result = fn(device_id)
                        if result is False:
                            raise RuntimeError(f"Wuji SDK {method}() returned False")
                        break
                return device
            except (TypeError, OSError, RuntimeError) as exc:
                last = exc
        raise RuntimeError("could not construct/connect Wuji glove device") from last

    def read(self) -> tuple[list[float], float]:
        raw = None
        for method in (
            "read",
            "poll",
            "get_state",
            "get_data",
            "read_state",
            "get_joint_angles",
        ):
            fn = getattr(self._device, method, None)
            if callable(fn):
                raw = fn()
                break
        if raw is None:
            joint_fn = next(
                (getattr(self._device, name, None) for name in ("get_joint_angles", "joint_angles")),
                None,
            )
            if callable(joint_fn):
                raw = {
                    "joints": joint_fn(),
                    "gripper": next(
                        (
                            fn()
                            for fn in (
                                getattr(self._device, "get_gripper_position", None),
                                getattr(self._device, "gripper_position", None),
                            )
                            if callable(fn)
                        ),
                        0.0,
                    ),
                }
        if raw is None:
            raise RuntimeError("Wuji device has no read/poll/get_state method")
        joints = _first_value(raw, ("joints", "joint_angles", "angles", "q", "positions"))
        gripper = _first_value(raw, ("gripper", "gripper_position", "grip", "thumb"))
        if joints is None and isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            joints, gripper = raw[:-1], raw[-1]
        if joints is None:
            # Some SDKs return a numpy array rather than a Python Sequence.
            try:
                values = list(raw)
            except TypeError:
                values = []
            if len(values) >= 8:
                joints, gripper = values[:-1], values[-1]
        if joints is None:
            raise ValueError("Wuji sample does not contain joint angles")
        if gripper is None:
            gripper = 0.0
        return [float(v) for v in joints], float(gripper)

    def close(self) -> None:
        for method in ("close", "disconnect", "stop"):
            fn = getattr(self._device, method, None)
            if callable(fn):
                fn()
                return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--namespace", default="left")
    p.add_argument(
        "--device-module",
        default="auto",
        help="SDK module to import (default: probe wuji, wuji_glove, then wuji_sdk)",
    )
    p.add_argument("--device-class", default="WujiGlove")
    p.add_argument("--device-id", default=None)
    p.add_argument("--rate", type=float, default=50.0)
    p.add_argument("--max-step-rad", type=float, default=0.15)
    p.add_argument("--joint-indices", nargs=7, type=int, default=list(range(7)))
    p.add_argument("--joint-signs", nargs=7, type=float, default=[1, -1, 1, 1, 1, -1, 1])
    p.add_argument("--joint-offsets", nargs=7, type=float, default=[0.0] * 7)
    p.add_argument("--joint-min", nargs=7, type=float, default=None)
    p.add_argument("--joint-max", nargs=7, type=float, default=None)
    p.add_argument(
        "--input-unit",
        choices=("radians", "degrees"),
        default="radians",
        help="unit returned by the SDK (converted to radians before publishing)",
    )
    p.add_argument("--stdin", action="store_true", help="read 8-value samples from stdin instead of the glove")
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32

    rclpy.init()
    node = Node("wuji_glove_teleop")
    ns = args.namespace.strip("/")
    arm_pub = node.create_publisher(JointState, f"/{ns}/gello/joint_states", 10)
    grip_pub = node.create_publisher(Float32, f"/{ns}/gripper/gripper_client/target_gripper_width_percent", 10)
    mapper = WujiJointMapper(
        joint_indices=args.joint_indices,
        joint_signs=args.joint_signs,
        joint_offsets=args.joint_offsets,
        joint_min=args.joint_min,
        joint_max=args.joint_max,
        max_step_rad=args.max_step_rad,
    )
    device = None if args.stdin else WujiGloveDevice(args.device_module, args.device_class, args.device_id)
    try:
        period = 1.0 / args.rate
        while rclpy.ok():
            if args.stdin:
                line = input().strip()
                if not line:
                    continue
                values = [float(v) for v in line.split()]
                if len(values) != 8:
                    raise ValueError("stdin samples require 7 joint angles and one gripper value")
                joints, gripper = values[:7], values[7]
            else:
                joints, gripper = device.read()
            if args.input_unit == "degrees":
                joints = [value * 3.141592653589793 / 180.0 for value in joints]
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = [f"fr3_joint{i}" for i in range(1, 8)]
            msg.header.frame_id = "fr3_link0"
            msg.position = mapper.map(joints).tolist()
            arm_pub.publish(msg)
            grip = Float32()
            grip.data = mapper.gripper(gripper)
            grip_pub.publish(grip)
            rclpy.spin_once(node, timeout_sec=0.0)
            if not args.stdin:
                time.sleep(period)
    finally:
        if device is not None:
            device.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    run(parse_args())
