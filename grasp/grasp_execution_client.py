#!/usr/bin/env python3
"""Request a camera grasp and execute it locally through move_to_target_ee.sh."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import uuid

from grasp.common import (
    INFERENCE_REQUEST_FORMAT,
    INFERENCE_RESPONSE_FORMAT,
    validate_command,
)
from grasp.control_server import DEFAULT_MOVE_SCRIPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-address",
        default="tcp://192.168.50.13:5571",
        help="Camera inference endpoint (default: tcp://192.168.50.13:5571).",
    )
    parser.add_argument("--side", choices=("right",), default="right")
    parser.add_argument("--move-script", type=Path, default=DEFAULT_MOVE_SCRIPT)
    parser.add_argument(
        "--return-ee-pose",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        help=(
            "After a completed grasp, return to this xyzrpy pose. In "
            "arm-with-hand mode, also reset all hand joints to zero. The "
            "launcher supplies its configured initial pose."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--arm-only",
        dest="control_mode",
        action="store_const",
        const="arm_only",
        help="Move only the FR3 arm; do not start or command a Wuji hand.",
    )
    mode.add_argument(
        "--arm-with-hand",
        dest="control_mode",
        action="store_const",
        const="arm_with_hand",
        help="Move the FR3 arm and command the Wuji hand joints.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--max-command-age-s", type=float, default=120.0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Permit real hardware execution. The local move script still displays "
            "state and requires y/yes confirmation. Default is dry-run."
        ),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Send one request immediately and exit; otherwise use an interactive loop.",
    )
    return parser


def request_grasp(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError(f"grasp execution client requires pyzmq: missing {exc.name}") from exc

    request_id = str(uuid.uuid4())
    request = {
        "format": INFERENCE_REQUEST_FORMAT,
        "request_id": request_id,
        "created_unix_s": time.time(),
        "action": "infer_grasp",
        "side": args.side,
    }
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    timeout_ms = int(args.request_timeout_s * 1000)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, min(timeout_ms, 10000))
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.server_address)
    try:
        socket.send_json(request)
        response = socket.recv_json()
    finally:
        socket.close(0)
        context.term()

    if not isinstance(response, dict) or response.get("format") != INFERENCE_RESPONSE_FORMAT:
        raise RuntimeError(f"invalid inference response: {response!r}")
    if response.get("request_id") != request_id:
        raise RuntimeError("inference response request_id does not match the request")
    if not response.get("ok", False):
        detail = response.get("error", "unknown server error")
        output_dir = response.get("server_output_dir")
        suffix = f"; partial output: {output_dir}" if output_dir else ""
        raise RuntimeError(f"camera inference failed: {detail}{suffix}")
    return validate_command(
        response.get("command"),
        max_age_s=args.max_command_age_s,
        expected_side=args.side,
    )


def reset_to_initial_pose(args: argparse.Namespace) -> int:
    """Run the guarded local move that restores the configured initial state."""
    if args.return_ee_pose is None:
        raise RuntimeError(
            "reset is unavailable because no initial EE pose was configured; "
            "start this client through start_grasp_execution_client.sh"
        )

    reset_pose = [f"{value:.12g}" for value in args.return_ee_pose]
    reset_command = [
        str(args.move_script),
        f"--{args.side}",
        "--hand" if args.control_mode == "arm_with_hand" else "--arm",
        "--target-ee-pose",
        *reset_pose,
    ]
    if args.control_mode == "arm_with_hand":
        reset_command.extend(("--target-ee-joint", *(["0"] * 20)))
        reset_command.extend(
            (
                f"--{args.side}-hand-ip",
                os.environ["GRASP_FIXED_RIGHT_HAND_IP"],
            )
        )
    if not args.execute:
        reset_command.append("--dry-run")

    reset_detail = (
        " and resetting all hand joints to zero"
        if args.control_mode == "arm_with_hand"
        else ""
    )
    mode = "execute with local confirmation" if args.execute else "dry-run"
    print(
        f"Resetting the arm to the initial pose{reset_detail} in {mode} mode...",
        flush=True,
    )
    return subprocess.run(reset_command, check=False).returncode


def execute_grasp(args: argparse.Namespace) -> int:
    print("Request sent; keep the camera and object still during observation.", flush=True)
    command = request_grasp(args)
    print(f"Received validated command {command['command_id']}", flush=True)
    print(
        "base_T_ee_xyz_rpy: "
        + " ".join(f"{float(value):.9g}" for value in command["ee_pose_xyz_rpy"]),
        flush=True,
    )
    print(
        f"Server trial: {command.get('server_output_dir', 'not reported')}",
        flush=True,
    )

    # Remote data can never grant execution permission. This field is replaced
    # exclusively from the local --execute option after full target validation.
    command["execute"] = bool(args.execute)
    pose = [f"{value:.12g}" for value in command["ee_pose_xyz_rpy"]]
    move_command = [
        str(args.move_script),
        f"--{args.side}",
        "--hand" if args.control_mode == "arm_with_hand" else "--arm",
        "--target-ee-pose",
        *pose,
    ]
    if args.control_mode == "arm_with_hand":
        joints = [f"{value:.12g}" for value in command["hand_joints"]]
        move_command.extend(("--target-ee-joint", *joints))
        move_command.extend(
            (
                f"--{args.side}-hand-ip",
                os.environ["GRASP_FIXED_RIGHT_HAND_IP"],
            )
        )
    if not args.execute:
        move_command.append("--dry-run")
    mode = "execute with local confirmation" if args.execute else "dry-run"
    print(f"Starting local move utility in {mode} mode...", flush=True)
    grasp_returncode = subprocess.run(move_command, check=False).returncode
    if grasp_returncode != 0 or args.return_ee_pose is None:
        return grasp_returncode

    print("Grasp completed; starting the configured automatic reset.", flush=True)
    return reset_to_initial_pose(args)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.request_timeout_s <= 0 or args.max_command_age_s <= 0:
        parser.error("timeouts and maximum command age must be positive")
    if not args.move_script.is_file():
        parser.error(f"move script does not exist: {args.move_script}")
    if args.control_mode == "arm_with_hand" and not os.environ.get(
        "GRASP_FIXED_RIGHT_HAND_IP"
    ):
        parser.error(
            "arm-with-hand mode must be started through "
            "start_grasp_execution_client.sh"
        )

    if args.once:
        return execute_grasp(args)

    execution_mode = "EXECUTE (local confirmation required)" if args.execute else "DRY-RUN"
    print(
        f"Grasp execution client connected to {args.server_address}; "
        f"control={args.control_mode.replace('_', '-')}; mode={execution_mode}"
    )
    print(
        "Press Enter or type 'g' to request a grasp; "
        "type 'r' to reset to the initial pose; type 'q' to quit."
    )
    while True:
        try:
            action = input("grasp> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if action in {"q", "quit", "exit"}:
            return 0
        if action in {"r", "reset", "home"}:
            try:
                returncode = reset_to_initial_pose(args)
                if returncode != 0:
                    print(f"Local reset utility failed with exit code {returncode}")
                else:
                    print("Reset completed.")
            except Exception as exc:
                print(f"Reset failed: {type(exc).__name__}: {exc}")
            continue
        if action not in {"", "g", "grasp", "infer"}:
            print(
                "Unknown command. Press Enter/type 'g' to infer, "
                "'r' to reset, or 'q' to quit."
            )
            continue
        try:
            returncode = execute_grasp(args)
            if returncode != 0:
                print(f"Local move utility failed with exit code {returncode}")
            else:
                print("Request completed.")
        except Exception as exc:
            print(f"Request failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
