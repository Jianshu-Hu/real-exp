#!/usr/bin/env python3
"""Map GELLO gripper width commands to pneumatic Softhand voltage."""

import argparse
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def width_to_voltage(width: float) -> float:
    """Map GELLO width (0 closed, 1 open) to Softhand voltage (5 closed, 0 open)."""
    width = float(width)
    if not math.isfinite(width):
        raise ValueError("GELLO gripper width must be finite")
    return 5.0 * (1.0 - max(0.0, min(1.0, width)))


def publish_gripper_width(publisher, width: float, channel_count: int) -> float:
    voltage = width_to_voltage(width)
    publisher.publish_all([voltage] * channel_count)
    return voltage


def detect_f407driver_port(ports) -> str:
    ports = list(ports)
    matches = []
    for port in ports:
        description = " ".join(
            str(value or "")
            for value in (
                getattr(port, "device", ""),
                getattr(port, "description", ""),
                getattr(port, "manufacturer", ""),
                getattr(port, "product", ""),
                getattr(port, "hwid", ""),
            )
        ).lower()
        if "f407driver" in description:
            matches.append(port.device)

    if not matches:
        available = ", ".join(
            f"{port.device} ({getattr(port, 'description', None) or 'unknown'})"
            for port in ports
        ) or "none"
        raise RuntimeError(
            "F407Driver USB CDC device was not found. "
            f"Detected serial ports: {available}. "
            "Check the controller power and USB data connection, then retry. "
            "For a controller using a different USB adapter, select its port with --serial-port."
        )
    if len(matches) > 1:
        raise RuntimeError(
            "multiple F407Driver USB CDC devices were found: "
            f"{', '.join(matches)}; pass --serial-port to select one"
        )
    return matches[0]


def resolve_serial_port(port_override: str | None = None) -> str:
    if port_override:
        return port_override
    try:
        from serial.tools import list_ports
    except ImportError as error:
        raise RuntimeError(
            "pyserial is required to auto-detect the F407Driver USB port; "
            "install pyserial or pass --serial-port"
        ) from error
    return detect_f407driver_port(list_ports.comports())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sides = parser.add_mutually_exclusive_group(required=True)
    sides.add_argument("--left", action="store_true")
    sides.add_argument("--right", action="store_true")
    sides.add_argument("--duo", action="store_true")
    parser.add_argument(
        "--serial-port",
        default=os.environ.get("SOFTHAND_SERIAL_PORT"),
        help="PneumaticSystem serial port (default: auto-detect F407Driver USB CDC)",
    )
    parser.add_argument(
        "--detect-port-only", action="store_true",
        help="Print the selected serial port without opening it or starting ROS",
    )
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        serial_port = resolve_serial_port(args.serial_port)
    except RuntimeError as error:
        parser.error(str(error))
    if args.detect_port_only:
        print(serial_port)
        return

    sys.path.insert(0, str(ROOT / "PneumaticSystem"))
    from Service.publisher import CHANNEL_COUNT, Publisher

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32

    class SofthandGripperTeleop(Node):
        def __init__(self, publisher):
            super().__init__("softhand_gripper_teleop")
            self._publisher = publisher
            sides = ("left", "right") if args.duo else (("left",) if args.left else ("right",))
            self._subscriptions = [
                self.create_subscription(
                    Float32,
                    f"/{side}/gripper/gripper_client/target_gripper_width_percent",
                    self._command_callback,
                    10,
                )
                for side in sides
            ]
            self.get_logger().info(
                f"Subscribing to GELLO gripper width for {', '.join(sides)}; "
                f"publishing {CHANNEL_COUNT} Softhand channels."
            )

        def _command_callback(self, message):
            try:
                voltage = publish_gripper_width(
                    self._publisher, message.data, CHANNEL_COUNT
                )
            except ValueError as error:
                self.get_logger().warning(str(error))
                return
            self.get_logger().debug(f"Softhand command: {voltage:.3f} V per channel")

    rclpy.init()
    try:
        with Publisher(port=serial_port) as publisher:
            node = SofthandGripperTeleop(publisher)
            try:
                rclpy.spin(node)
            finally:
                node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
