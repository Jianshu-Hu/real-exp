#!/usr/bin/env python3
"""Return the real right FR3 and Wuji Hand 2 to a rollout snapshot pose."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import zmq

from policy_contract import JOINT_NAMES, hardware_command_limits, load_joint_limits
from policy_executor import bridge_state


def interpolate(start: np.ndarray, goal: np.ndarray, elapsed: float, duration: float) -> np.ndarray:
    fraction = float(np.clip(elapsed / duration, 0.0, 1.0))
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    return start + fraction * (goal - start)


def snapshot_target(snapshot: Path) -> np.ndarray:
    if not snapshot.is_file():
        raise FileNotFoundError(f"missing snapshot: {snapshot}")
    value = np.asarray(json.loads(snapshot.read_text())["state"]["joint_position_27"], dtype=float)
    if value.shape != (27,) or not np.all(np.isfinite(value)):
        raise ValueError("snapshot joint_position_27 must be 27 finite values")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshot", type=Path, help="current approved real initial-pose snapshot")
    source.add_argument("--rollout", type=Path, help="legacy: use ROLLOUT/initial_snapshot.json")
    parser.add_argument("--state-connect", default="tcp://127.0.0.1:5555")
    parser.add_argument("--arm-command-connect", default="tcp://127.0.0.1:5556")
    parser.add_argument("--hand-command-address", default="tcp://127.0.0.1:5562")
    parser.add_argument("--robot-urdf", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--hold", type=float, default=5.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--arm-tolerance", type=float, default=0.02)
    parser.add_argument("--hand-tolerance", type=float, default=0.06)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute == args.dry_run:
        parser.error("choose exactly one of --dry-run or --execute")
    if args.execute and not args.confirm:
        parser.error("--execute requires --confirm")
    if min(args.duration, args.rate, args.arm_tolerance, args.hand_tolerance) <= 0 or args.hold < 0:
        parser.error("duration/rate/tolerances must be positive; hold must be nonnegative")
    if not args.robot_urdf.is_file():
        parser.error(f"robot URDF not found: {args.robot_urdf}")
    return args


def main() -> int:
    args = parse_args()
    snapshot = args.snapshot if args.snapshot is not None else args.rollout / "initial_snapshot.json"
    target = snapshot_target(snapshot)
    lower, upper = load_joint_limits(args.robot_urdf)
    command_lower, command_upper = hardware_command_limits(lower, upper)
    if np.any(target < command_lower) or np.any(target > command_upper):
        violations = [
            f"{name}: target={value:.6f}, allowed=[{lo:.6f}, {hi:.6f}]"
            for name, value, lo, hi in zip(JOINT_NAMES, target, command_lower, command_upper)
            if value < lo or value > hi
        ]
        raise SystemExit("snapshot target exceeds hardware command limits:\n  " + "\n  ".join(violations))

    context = zmq.Context()
    state = context.socket(zmq.SUB)
    state.setsockopt(zmq.SUBSCRIBE, b"")
    state.setsockopt(zmq.CONFLATE, 1)
    state.setsockopt(zmq.LINGER, 0)
    state.connect(args.state_connect)
    arm = hand = None
    if args.execute:
        arm = context.socket(zmq.PUSH)
        arm.setsockopt(zmq.SNDHWM, 1)
        arm.setsockopt(zmq.LINGER, 0)
        arm.connect(args.arm_command_connect)
        hand = context.socket(zmq.PUSH)
        hand.setsockopt(zmq.SNDHWM, 1)
        hand.setsockopt(zmq.LINGER, 0)
        hand.connect(args.hand_command_address)
    stopped = [False]
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))

    def read_state(timeout: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if state.poll(100, zmq.POLLIN):
                try:
                    q, _, _ = bridge_state(state.recv_pyobj())
                    return q
                except (TypeError, ValueError):
                    continue
        raise TimeoutError("state stream timeout")

    try:
        current = read_state()
        print("mode=" + ("EXECUTE" if args.execute else "DRY-RUN"))
        print("initial_arm_max_delta_rad=" + f"{np.max(np.abs(current[:7] - target[:7])):.6f}")
        print("initial_hand_max_delta_rad=" + f"{np.max(np.abs(current[7:] - target[7:])):.6f}")
        if args.dry_run:
            print("DRY-RUN: no FR3 or Wuji command was sent")
            return 0
        print("WARNING: returning FR3 and Wuji Hand 2 to the rollout initial pose; press Ctrl-C to stop", flush=True)
        started = time.monotonic()
        period = 1.0 / args.rate
        while not stopped[0] and time.monotonic() - started < args.duration + args.hold:
            elapsed = time.monotonic() - started
            arm_target = interpolate(current[:7], target[:7], elapsed, args.duration)
            arm.send_pyobj({"timestamp": time.time(), "right_joint_target": arm_target.tolist()})
            hand.send_pyobj(target[7:].tolist())
            time.sleep(period)
        if stopped[0]:
            return 130
        final = read_state(timeout=2.0)
        arm_error = float(np.max(np.abs(final[:7] - target[:7])))
        hand_error = float(np.max(np.abs(final[7:] - target[7:])))
        print("final_arm_max_error_rad=" + f"{arm_error:.6f}")
        print("final_hand_max_error_rad=" + f"{hand_error:.6f}")
        if arm_error > args.arm_tolerance or hand_error > args.hand_tolerance:
            raise SystemExit("return did not reach tolerance; do not start another replay")
        print("return_to_snapshot: PASS")
        return 0
    finally:
        for socket in (state, arm, hand):
            if socket is not None:
                socket.close(0)
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
