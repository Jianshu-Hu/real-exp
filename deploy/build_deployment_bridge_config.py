"""Build a complete ROS parameter file for one dataset deployment contract."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return normalized == "true"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-rate-hz", required=True, type=float)
    parser.add_argument("--publish-host", required=True)
    parser.add_argument("--publish-port", required=True, type=int)
    parser.add_argument("--command-host", required=True)
    parser.add_argument("--command-port", required=True, type=int)
    parser.add_argument("--camera-cache-host", required=True)
    parser.add_argument("--camera-cache-port", required=True, type=int)
    parser.add_argument("--include-right-arm", required=True, type=parse_bool)
    parser.add_argument("--arm-mode", required=True, choices=("left", "right", "duo"))
    parser.add_argument("--include-gripper", required=True, type=parse_bool)
    parser.add_argument("--include-hand", required=True, type=parse_bool)
    parser.add_argument("--hand-telemetry-host", required=True)
    parser.add_argument("--hand-telemetry-port", required=True, type=int)
    for index in range(1, 4):
        parser.add_argument(f"--camera-{index}-enabled", required=True, type=parse_bool)
    return parser.parse_args(argv)


def build_config(base_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    try:
        parameters = base_config["lerobot_data_bridge"]["ros__parameters"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Bridge YAML must contain lerobot_data_bridge.ros__parameters."
        ) from exc
    if not isinstance(parameters, dict):
        raise ValueError("Bridge ros__parameters must be a mapping.")
    if args.sample_rate_hz <= 0.0:
        raise ValueError("Bridge sample rate must be positive.")
    if args.include_gripper and args.include_hand:
        raise ValueError("Bridge cannot include both a gripper and a Wuji hand.")
    if (args.arm_mode == "duo") != args.include_right_arm:
        raise ValueError("include_right_arm must be true exactly when arm_mode is duo.")

    parameters.update(
        {
            "sample_rate_hz": args.sample_rate_hz,
            "publish_host": args.publish_host,
            "publish_port": args.publish_port,
            "command_host": args.command_host,
            "command_port": args.command_port,
            "camera_cache_host": args.camera_cache_host,
            "camera_cache_port": args.camera_cache_port,
            "include_right_arm": args.include_right_arm,
            "arm_mode": args.arm_mode,
            "include_gripper": args.include_gripper,
            "include_hand": args.include_hand,
            "hand_telemetry_host": args.hand_telemetry_host,
            "hand_telemetry_port": args.hand_telemetry_port,
            **{
                f"camera_{index}_enabled": getattr(args, f"camera_{index}_enabled")
                for index in range(1, 4)
            },
        }
    )
    return base_config


def main() -> None:
    args = parse_args()
    base_config = yaml.safe_load(args.base_config.read_text())
    if not isinstance(base_config, dict):
        raise ValueError(f"Bridge YAML must be a mapping: {args.base_config}")
    config = build_config(base_config, args)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
