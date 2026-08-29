#!/usr/bin/env python3
"""Serve trigger-based D435 grasp inference to the robot-control computer."""

from __future__ import annotations

import argparse
import collections
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any
import uuid

from grasp.camera_inference import (
    add_camera_inference_arguments,
    run_camera_inference,
    validate_camera_inference_args,
)
from grasp.common import (
    COMMAND_FORMAT,
    INFERENCE_RESPONSE_FORMAT,
    WUJI_COMMAND_CONVERSION,
    WUJI_COMMAND_HAND_MODEL,
    WUJI_COMMAND_JOINT_CONVENTION,
    WUJI_COMMAND_SOURCE_MODEL,
    validate_command,
    validate_inference_request,
    wuji_v1_model_to_hand2_firmware,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bind",
        default="tcp://192.168.50.13:5571",
        help="ZMQ endpoint on the camera server (default: tcp://192.168.50.13:5571).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs",
        help="Parent directory for one output directory per request.",
    )
    parser.add_argument("--side", choices=("right",), default="right")
    parser.add_argument("--max-request-age-s", type=float, default=30.0)
    add_camera_inference_arguments(parser, include_output_dir=False)
    return parser


def _trial_directory(runs_dir: Path) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    base = runs_dir / f"camera_trial_{timestamp}"
    if not base.exists():
        return base
    for suffix in range(1, 1000):
        candidate = runs_dir / f"camera_trial_{timestamp}_{suffix:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"too many trial directories for timestamp {timestamp}")


def build_grasp_command(
    record: dict[str, Any], *, side: str, output_dir: Path
) -> dict[str, Any]:
    poses = record["poses"]
    firmware_joints = wuji_v1_model_to_hand2_firmware(poses["hand_joints_rad"])
    command = {
        "format": COMMAND_FORMAT,
        "command_id": str(uuid.uuid4()),
        "created_unix_s": time.time(),
        "side": side,
        # Execution permission is deliberately controlled only by the robot client.
        "execute": False,
        "base_T_ee": poses["base_T_ee"],
        "ee_pose_xyz_rpy": poses["base_T_ee_xyz_rpy"],
        "hand_joints": firmware_joints.tolist(),
        "hand_joint_names": poses["hand_joint_names"],
        "hand_model": WUJI_COMMAND_HAND_MODEL,
        "hand_joint_convention": WUJI_COMMAND_JOINT_CONVENTION,
        "hand_joint_source_model": WUJI_COMMAND_SOURCE_MODEL,
        "hand_joint_conversion": WUJI_COMMAND_CONVERSION,
        "world_T_hand": poses["world_T_hand"],
        "base_T_world": record["calibration"]["base_T_world"],
        "ee_T_hand": record["calibration"]["ee_T_hand"],
        "inference": record["inference"],
        "camera_observation": record["camera"].get("observation"),
        "server_output_dir": str(output_dir.resolve()),
    }
    # Refuse to transmit a malformed or out-of-workspace target even if a future
    # inference change accidentally violates the command contract.
    validate_command(command, max_age_s=5.0, expected_side=side)
    return command


def _write_server_record(
    output_dir: Path,
    record: dict[str, Any],
    request: dict[str, Any],
    command: dict[str, Any],
) -> None:
    record["remote_request"] = {
        "request_id": request["request_id"],
        "created_unix_s": request["created_unix_s"],
        "side": request["side"],
    }
    record["command"] = command
    (output_dir / "result.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_request_age_s <= 0:
        parser.error("max-request-age-s must be positive")
    validate_camera_inference_args(args, parser)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    try:
        import zmq
    except ImportError as exc:
        raise SystemExit(f"camera inference server requires pyzmq: missing {exc.name}") from exc

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    seen: collections.deque[str] = collections.deque(maxlen=256)
    print(f"Camera grasp inference server listening at {args.bind}", flush=True)
    print(f"Trial outputs will be stored below {args.runs_dir.resolve()}", flush=True)
    print("Waiting for the robot-control computer to request a grasp...", flush=True)
    try:
        while True:
            raw_request = socket.recv_json()
            request_id = raw_request.get("request_id") if isinstance(raw_request, dict) else None
            output_dir: Path | None = None
            try:
                request = validate_inference_request(
                    raw_request,
                    max_age_s=args.max_request_age_s,
                    expected_side=args.side,
                )
                if request["request_id"] in seen:
                    raise ValueError(f"duplicate request_id {request['request_id']!r}")
                seen.append(request["request_id"])
                output_dir = _trial_directory(args.runs_dir)
                args.output_dir = output_dir
                print(
                    f"Accepted request {request['request_id']}; capturing and inferring...",
                    flush=True,
                )
                record = run_camera_inference(args)
                command = build_grasp_command(record, side=args.side, output_dir=output_dir)
                _write_server_record(output_dir, record, request, command)
                response = {
                    "format": INFERENCE_RESPONSE_FORMAT,
                    "ok": True,
                    "request_id": request["request_id"],
                    "command": command,
                    "server_output_dir": str(output_dir.resolve()),
                }
                print(
                    f"Completed request {request['request_id']}; returning target to control host",
                    flush=True,
                )
            except Exception as exc:
                response = {
                    "format": INFERENCE_RESPONSE_FORMAT,
                    "ok": False,
                    "request_id": request_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if output_dir is not None:
                    response["server_output_dir"] = str(output_dir.resolve())
                print(f"Grasp request failed: {response['error']}", flush=True)
            socket.send_json(response)
    except KeyboardInterrupt:
        return 0
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
