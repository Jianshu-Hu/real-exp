#!/usr/bin/env python3
"""Receive one-shot Wuji grasp targets and invoke the existing FR3 controller."""

from __future__ import annotations

import argparse
import collections
import subprocess
from pathlib import Path
from typing import Any

from grasp.common import validate_command


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOVE_SCRIPT = REPOSITORY_ROOT / "scripts" / "move_to_target_ee.sh"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://127.0.0.1:5570")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--max-command-age-s", type=float, default=120.0)
    parser.add_argument("--move-script", type=Path, default=DEFAULT_MOVE_SCRIPT)
    parser.add_argument("--hand-ip", default="", help="Wuji SDK IP:port passed to the move script")
    parser.add_argument(
        "--allow-execute",
        action="store_true",
        help="Allow requests marked execute=true to reach the hardware confirmation prompt.",
    )
    return parser


def build_move_command(args: argparse.Namespace, command: dict[str, Any]) -> list[str]:
    pose = [f"{value:.12g}" for value in command["ee_pose_xyz_rpy"]]
    joints = [f"{value:.12g}" for value in command["hand_joints"]]
    result = [
        str(args.move_script),
        f"--{args.side}",
        "--hand",
        "--target-ee-pose",
        *pose,
        "--target-ee-joint",
        *joints,
    ]
    if args.hand_ip:
        result.extend((f"--{args.side}-hand-ip", args.hand_ip))
    if not (args.allow_execute and bool(command.get("execute", False))):
        result.append("--dry-run")
    return result


def main() -> int:
    args = build_parser().parse_args()
    if args.max_command_age_s <= 0:
        raise SystemExit("--max-command-age-s must be positive")
    if not args.move_script.is_file():
        raise SystemExit(f"move script does not exist: {args.move_script}")
    try:
        import zmq
    except ImportError as exc:
        raise SystemExit(f"control server requires pyzmq: missing {exc.name}") from exc

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    seen: collections.deque[str] = collections.deque(maxlen=256)
    print(
        f"Grasp control server listening at {args.bind}; "
        f"hardware execution={'enabled' if args.allow_execute else 'disabled'}",
        flush=True,
    )
    try:
        while True:
            request = socket.recv_json()
            try:
                command = validate_command(
                    request, max_age_s=args.max_command_age_s, expected_side=args.side
                )
                if command["command_id"] in seen:
                    raise ValueError(f"duplicate command_id {command['command_id']!r}")
                seen.append(command["command_id"])
                move_command = build_move_command(args, command)
                mode = "execute" if "--dry-run" not in move_command else "dry-run"
                print(f"Accepted {command['command_id']} in {mode} mode", flush=True)
                # In execute mode stdin/stdout stay attached so the local operator sees
                # and answers move_to_target_ee.py's final hardware confirmation.
                completed = subprocess.run(move_command, check=False)
                response = {
                    "ok": completed.returncode == 0,
                    "command_id": command["command_id"],
                    "mode": mode,
                    "returncode": completed.returncode,
                }
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                print(f"Rejected grasp request: {response['error']}", flush=True)
            socket.send_json(response)
    except KeyboardInterrupt:
        return 0
    finally:
        socket.close(0)
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
