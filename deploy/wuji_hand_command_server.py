"""Drive one Wuji Hand 2 from local policy targets and publish measured state.

This process runs on the robot-control computer. The policy executor connects
to the local PULL endpoint, while hand telemetry is pushed to the deployment
bridge on the inference server.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.wuji_hand_control import HAND_JOINT_COUNT, make_smoothed_backend_class


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robot-local Wuji Hand 2 command and deployment-telemetry worker."
    )
    parser.add_argument("--side", required=True, choices=("left", "right"))
    parser.add_argument(
        "--hand-ip",
        default="",
        help="Wuji Hand 2 SDK address (IP:port). Empty enables SDK discovery.",
    )
    parser.add_argument(
        "--command-address",
        required=True,
        help="Local ZMQ PULL address to bind for 20-joint policy targets.",
    )
    parser.add_argument(
        "--telemetry-address",
        required=True,
        help="Server-side deployment bridge ZMQ endpoint for measured hand telemetry.",
    )
    parser.add_argument("--telemetry-rate", required=True, type=float)
    args = parser.parse_args(argv)
    if not np.isfinite(args.telemetry_rate) or args.telemetry_rate <= 0.0:
        parser.error("--telemetry-rate must be positive and finite")
    return args


def hand_target(payload: Any) -> np.ndarray | None:
    if isinstance(payload, dict):
        payload = payload.get("target")
    target = np.asarray(payload, dtype=float)
    if target.shape != (HAND_JOINT_COUNT,) or not np.all(np.isfinite(target)):
        return None
    return target


def telemetry_payload(
    side: str, current: np.ndarray, target: np.ndarray, stamp_s: float
) -> dict[str, Any]:
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != (HAND_JOINT_COUNT,) or target.shape != (HAND_JOINT_COUNT,):
        raise ValueError("Wuji deployment telemetry requires two 20-joint vectors.")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
        raise ValueError("Wuji deployment telemetry values must be finite.")
    return {
        "side": side,
        "current": current.tolist(),
        "target": target.tolist(),
        "stamp_s": float(stamp_s),
    }


def load_backend_class() -> type:
    example_dir = REPO_ROOT / "libs/wuji-retargeting/example"
    os.chdir(example_dir)
    # teleop_real imports a submodule-local package also named ``utils``.
    for module_name in list(sys.modules):
        if module_name == "utils" or module_name.startswith("utils."):
            del sys.modules[module_name]
    example_path = str(example_dir)
    if example_path in sys.path:
        sys.path.remove(example_path)
    sys.path.insert(0, example_path)
    from teleop_real import WujiHand2Backend as OriginalWujiHand2Backend

    return make_smoothed_backend_class(OriginalWujiHand2Backend)


def run(args: argparse.Namespace) -> None:
    backend_class = load_backend_class()
    backend = backend_class(
        ip=args.hand_ip,
        kp=3.0,
        kd=0.1,
        current_limit=1.5,
        handedness=args.side,
    )
    context = zmq.Context()
    command_socket = context.socket(zmq.PULL)
    command_socket.setsockopt(zmq.RCVHWM, 2)
    command_socket.setsockopt(zmq.LINGER, 0)
    command_socket.bind(args.command_address)
    telemetry_socket = context.socket(zmq.PUSH)
    telemetry_socket.setsockopt(zmq.SNDHWM, 2)
    telemetry_socket.setsockopt(zmq.LINGER, 0)
    telemetry_socket.connect(args.telemetry_address)
    poller = zmq.Poller()
    poller.register(command_socket, zmq.POLLIN)
    target = backend.target_position
    telemetry_period_s = 1.0 / args.telemetry_rate
    next_telemetry_s = time.monotonic()
    invalid_targets = 0
    print(
        f"{args.side.title()} Wuji deployment worker: commands={args.command_address}, "
        f"telemetry={args.telemetry_address}, rate={args.telemetry_rate:g} Hz",
        flush=True,
    )
    try:
        while True:
            timeout_ms = max(0, min(100, int(1000.0 * (next_telemetry_s - time.monotonic()))))
            if command_socket in dict(poller.poll(timeout_ms)):
                latest_payload = command_socket.recv_pyobj()
                while True:
                    try:
                        latest_payload = command_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                next_target = hand_target(latest_payload)
                if next_target is None:
                    invalid_targets += 1
                    if invalid_targets == 1 or invalid_targets % 100 == 0:
                        print(
                            f"Ignoring invalid {args.side} hand policy target "
                            f"(count={invalid_targets}).",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    backend.send(next_target)
                    target = next_target

            now_s = time.monotonic()
            if now_s < next_telemetry_s:
                continue
            current = backend.actual_position()
            if current is not None:
                try:
                    telemetry_socket.send_pyobj(
                        telemetry_payload(args.side, current, target, time.time()),
                        flags=zmq.NOBLOCK,
                    )
                except zmq.Again:
                    # Startup and transient network gaps must not stop hand control.
                    pass
            # If the backend/network stalls, schedule one full period from the
            # current time rather than immediately emitting a duplicate packet.
            next_telemetry_s = max(next_telemetry_s + telemetry_period_s, now_s + telemetry_period_s)
    except KeyboardInterrupt:
        print(f"\nStopping {args.side} Wuji deployment worker...", flush=True)
    finally:
        command_socket.close(0)
        telemetry_socket.close(0)
        context.term()
        backend.close()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
