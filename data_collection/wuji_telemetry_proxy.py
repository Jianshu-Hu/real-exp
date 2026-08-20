#!/usr/bin/env python3
"""Run the existing Wuji teleop with optional hand telemetry.

The Wuji retargeting repository remains untouched. This adapter wraps the
already-connected Wuji Hand 2 backend in-process, so the measured position and
the exact target sent by the teleop loop share one SDK session.
"""

from __future__ import annotations

import atexit
import os
import sys
import time
from pathlib import Path

import zmq

DEFAULT_TELEMETRY_HOST = "192.168.50.13"
DEFAULT_TELEMETRY_PORT = 5558


def _take_option(argv: list[str], name: str, default: str) -> tuple[list[str], str]:
    remaining: list[str] = []
    value = default
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            value = argv[index + 1]
            index += 2
            continue
        remaining.append(argument)
        index += 1
    return remaining, value


def main() -> None:
    argv, telemetry_host = _take_option(
        sys.argv[1:],
        "--telemetry-host",
        os.environ.get("DATA_COLLECTION_SERVER_IP", DEFAULT_TELEMETRY_HOST),
    )
    argv, telemetry_port_text = _take_option(
        argv,
        "--telemetry-port",
        os.environ.get("HAND_TELEMETRY_PORT", str(DEFAULT_TELEMETRY_PORT)),
    )
    try:
        telemetry_port = int(telemetry_port_text)
    except ValueError as exc:
        raise SystemExit("--telemetry-port must be an integer") from exc

    example_dir = Path(__file__).resolve().parents[1] / "libs" / "wuji-retargeting" / "example"
    os.chdir(example_dir)
    sys.path.insert(0, str(example_dir))
    sys.argv = [str(example_dir / "teleop_real.py"), *argv]

    import teleop_real

    telemetry_socket = None
    if telemetry_host and telemetry_port > 0:
        telemetry_socket = zmq.Context.instance().socket(zmq.PUSH)
        telemetry_socket.setsockopt(zmq.SNDHWM, 2)
        telemetry_socket.setsockopt(zmq.LINGER, 0)
        telemetry_socket.connect(f"tcp://{telemetry_host}:{telemetry_port}")
        atexit.register(telemetry_socket.close, 0)
        print(
            f"Wuji hand telemetry: pushing to tcp://{telemetry_host}:{telemetry_port}",
            flush=True,
        )
    else:
        print("Wuji hand telemetry: disabled", flush=True)

    original_send = teleop_real.WujiHand2Backend.send
    telemetry_sent = 0
    telemetry_error_count = 0
    telemetry_last_error_time = 0.0

    def send_with_telemetry(backend, qpos):
        nonlocal telemetry_sent, telemetry_error_count, telemetry_last_error_time
        original_send(backend, qpos)
        if telemetry_socket is None:
            return
        try:
            current = backend._hand.read_joint_state().position
            current_values = [float(value) for value in current]
            target_values = [float(value) for value in qpos]
            if len(current_values) != 20 or len(target_values) != 20:
                return
            telemetry_socket.send_pyobj(
                {
                    "side": getattr(backend, "_telemetry_side", "") or _hand_side(argv),
                    "current": current_values,
                    "target": target_values,
                    "stamp_s": time.time(),
                },
                flags=zmq.NOBLOCK,
            )
            telemetry_sent += 1
            if telemetry_sent == 1:
                print("Wuji hand telemetry: first packet queued", flush=True)
        except Exception as exc:
            telemetry_error_count += 1
            now = time.monotonic()
            if telemetry_error_count == 1 or now - telemetry_last_error_time >= 5.0:
                print(
                    "Wuji hand telemetry: unable to queue packet "
                    f"({type(exc).__name__}: {exc}); failures={telemetry_error_count}",
                    file=sys.stderr,
                    flush=True,
                )
                telemetry_last_error_time = now

    teleop_real.WujiHand2Backend.send = send_with_telemetry
    teleop_real.main()


def _hand_side(argv: list[str]) -> str:
    for index, argument in enumerate(argv[:-1]):
        if argument == "--hand":
            return argv[index + 1]
    return "right"


if __name__ == "__main__":
    main()
