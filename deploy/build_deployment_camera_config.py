"""Build a RealSense camera parameter file for a deployment camera contract."""

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
    for index in range(1, 4):
        parser.add_argument(f"--camera-{index}-enabled", required=True, type=parse_bool)
    return parser.parse_args(argv)


def build_config(
    base_config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    try:
        parameters = base_config["realsense_camera_publisher"]["ros__parameters"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Camera YAML must contain realsense_camera_publisher.ros__parameters."
        ) from exc
    if not isinstance(parameters, dict):
        raise ValueError("Camera ros__parameters must be a mapping.")

    parameters.update(
        {
            f"camera_{index}_enabled": getattr(args, f"camera_{index}_enabled")
            for index in range(1, 4)
        }
    )
    if not any(parameters[f"camera_{index}_enabled"] for index in range(1, 4)):
        raise ValueError("Deployment must enable at least one camera.")
    return base_config


def main() -> None:
    args = parse_args()
    base_config = yaml.safe_load(args.base_config.read_text())
    if not isinstance(base_config, dict):
        raise ValueError(f"Camera YAML must be a mapping: {args.base_config}")
    config = build_config(base_config, args)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
