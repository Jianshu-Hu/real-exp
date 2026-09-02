#!/usr/bin/env python3
"""Move the real right FR3 to the SimToolReal training start pose.

This is deliberately independent of policy inference.  Without ``--execute``
it only reports the measured state, target, and FK palm poses.  With
``--execute`` it sends a slow joint-space interpolation to the existing local
deployment bridge and never sends Wuji hand commands.
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from kinematics import PolicyKinematics
from observation import checked_transform
from policy_contract import load_joint_limits


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf"
DEFAULT_WORLD_FROM_ROBOT = np.asarray(
    [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.45],
     [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
)
# IK solution for the simulation start palm pose translated 20 cm away from
# the center/forward camera pole.  Palm orientation and height are unchanged.
# FK target in Wp: xyz=(0.0651, -0.1222, 0.7069) m.
SIM_INITIAL_ARM = np.asarray(
    [0.85498374, 1.34871996, -0.27068954, -1.72059770,
     -2.43997698, 1.92438901, 0.94507977], dtype=np.float64
)


def parse_csv(value: str, count: int, label: str) -> np.ndarray:
    try:
        result = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must contain comma-separated numbers") from exc
    if result.shape != (count,) or not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError(f"{label} must contain {count} finite numbers")
    return result


def bridge_state(packet: Any) -> np.ndarray:
    if not isinstance(packet, dict) or packet.get("arm_mode") != "right" or not packet.get("include_hand"):
        raise ValueError("expected right-arm bridge state with include_hand=true")
    values = np.asarray(packet.get("joint_state"), dtype=np.float64)
    if values.shape != (27,) or not np.all(np.isfinite(values)):
        raise ValueError("bridge joint_state must be a finite 27-vector")
    return values


def interpolate(start: np.ndarray, goal: np.ndarray, elapsed: float, duration: float) -> np.ndarray:
    if duration <= 0.0:
        return goal.copy()
    fraction = float(np.clip(elapsed / duration, 0.0, 1.0))
    # Cubic easing gives zero velocity at both ends of the move.
    fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    return start + fraction * (goal - start)


def pose_summary(kinematics: PolicyKinematics, q: np.ndarray, world_from_robot: np.ndarray) -> str:
    palm_pos, palm_quat, _ = kinematics.evaluate(q, world_from_robot)
    return f"palm_xyz={np.array2string(palm_pos, precision=5)} palm_quat_xyzw={np.array2string(palm_quat, precision=5)}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-connect", default="tcp://127.0.0.1:5555")
    parser.add_argument("--command-connect", default="tcp://127.0.0.1:5556")
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--world-from-robot", default=None, help="16 comma-separated values; default is the trained policy pose")
    parser.add_argument("--duration", type=float, default=30.0, help="interpolation duration in seconds")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--hold", type=float, default=5.0, help="seconds to hold the final target")
    parser.add_argument("--tolerance", type=float, default=0.02, help="completion error in radians")
    parser.add_argument("--execute", action="store_true", help="send the slow interpolation to the real FR3")
    parser.add_argument("--target-arm", default=None, help="override the seven-joint target with comma-separated radians")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.duration <= 0 or args.rate <= 0 or args.hold < 0 or args.tolerance <= 0:
        raise SystemExit("duration/rate/tolerance must be positive; hold must be non-negative")
    target_arm = SIM_INITIAL_ARM if args.target_arm is None else parse_csv(args.target_arm, 7, "--target-arm")
    world_from_robot = DEFAULT_WORLD_FROM_ROBOT if args.world_from_robot is None else parse_csv(args.world_from_robot, 16, "--world-from-robot").reshape(4, 4)
    world_from_robot = checked_transform(world_from_robot, name="world-from-robot")
    lower, upper = load_joint_limits(args.robot_urdf)
    if np.any(target_arm < lower[:7]) or np.any(target_arm > upper[:7]):
        raise SystemExit("target arm contains a joint outside the selected URDF limits")
    context = zmq.Context()
    state_socket = context.socket(zmq.SUB)
    state_socket.setsockopt(zmq.SUBSCRIBE, b"")
    state_socket.setsockopt(zmq.RCVHWM, 1)
    state_socket.connect(args.state_connect)
    command_socket = context.socket(zmq.PUSH) if args.execute else None
    if command_socket is not None:
        command_socket.setsockopt(zmq.SNDHWM, 1)
        command_socket.connect(args.command_connect)
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    try:
        print(f"Waiting for right-arm state at {args.state_connect}; mode={'EXECUTE' if args.execute else 'DRY-RUN'}", flush=True)
        current = None
        deadline = time.monotonic() + 15.0
        while not stop[0] and time.monotonic() < deadline:
            if not state_socket.poll(200, zmq.POLLIN):
                continue
            try:
                current = bridge_state(state_socket.recv_pyobj())
                break
            except (ValueError, TypeError) as exc:
                print(f"rejected state: {exc}", flush=True)
        if current is None:
            raise SystemExit("timed out waiting for a valid right-arm state")
        goal = current.copy()
        goal[:7] = target_arm
        kinematics = PolicyKinematics(args.robot_urdf)
        print("current_arm=" + np.array2string(current[:7], precision=6), flush=True)
        print("target_arm=" + np.array2string(target_arm, precision=6), flush=True)
        print("current " + pose_summary(kinematics, current, world_from_robot), flush=True)
        print("target  " + pose_summary(kinematics, goal, world_from_robot), flush=True)
        print("max_joint_delta_rad=" + f"{np.max(np.abs(target_arm - current[:7])):.6f}", flush=True)
        if not args.execute:
            print("DRY-RUN: no joint command was sent", flush=True)
            return 0
        print("WARNING: sending a slow joint-space move to the real FR3; press Ctrl-C to stop", flush=True)
        start = time.monotonic()
        period = 1.0 / args.rate
        while not stop[0] and time.monotonic() - start < args.duration + args.hold:
            elapsed = time.monotonic() - start
            arm = interpolate(current[:7], target_arm, elapsed, args.duration)
            command_socket.send_pyobj({"timestamp": time.time(), "right_joint_target": arm.tolist()})
            if elapsed >= args.duration and int(elapsed / max(period, 1e-6)) % max(1, int(args.rate)) == 0:
                print("holding target; arm=" + np.array2string(arm, precision=5), flush=True)
            time.sleep(period)
        if stop[0]:
            return 130
        # Read a fresh state after the hold and report the actual tracking error.
        latest = current
        end = time.monotonic() + 2.0
        while time.monotonic() < end and state_socket.poll(100, zmq.POLLIN):
            try:
                latest = bridge_state(state_socket.recv_pyobj())
            except (ValueError, TypeError):
                continue
        error = latest[:7] - target_arm
        print("final_arm=" + np.array2string(latest[:7], precision=6), flush=True)
        print("final_max_error_rad=" + f"{np.max(np.abs(error)):.6f}", flush=True)
        if np.max(np.abs(error)) > args.tolerance:
            print(f"WARNING: final error exceeds tolerance {args.tolerance:.6f} rad", flush=True)
        return 0
    finally:
        state_socket.close(0)
        if command_socket is not None:
            command_socket.close(0)
        context.term()


if __name__ == "__main__":
    raise SystemExit(main())
