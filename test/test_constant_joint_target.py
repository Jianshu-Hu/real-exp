"""Test a constant joint target through the deployment bridge without a policy.

The script enters through the bridge's production ZMQ command socket. The
bridge sends one absolute waypoint per 15 Hz cycle; the robot-side controller
generates the constrained reference at 1 kHz before applying the unchanged
impedance torque law.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle  # nosec
import signal
import subprocess  # nosec
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import zmq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_IP = os.environ.get("DEPLOYMENT_SERVER_IP", "192.168.50.13")
MAX_DIAGNOSTIC_OFFSET_RAD = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one small constant joint offset through the deployment bridge."
    )
    parser.add_argument("--execute", action="store_true", help="Required to command real hardware.")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--joint", type=int, choices=range(1, 8), default=1)
    parser.add_argument("--offset-rad", type=float, default=0.02)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--zmq-host", default=DEFAULT_SERVER_IP)
    parser.add_argument("--observation-port", type=int, default=5555)
    parser.add_argument("--command-port", type=int, default=5556)
    parser.add_argument("--bridge-activation-service", default="/set_deployment_active")
    parser.add_argument(
        "--log-dir", type=Path, default=REPO_ROOT / "outputs" / "controller_tests"
    )
    args = parser.parse_args()
    if args.duration_s <= 0.0 or args.settle_s < 0.0 or args.fps <= 0.0:
        parser.error("duration-s and fps must be positive; settle-s must be non-negative")
    if abs(args.offset_rad) > MAX_DIAGNOSTIC_OFFSET_RAD:
        parser.error(
            "For this diagnostic, --offset-rad must be within "
            f"[-{MAX_DIAGNOSTIC_OFFSET_RAD}, {MAX_DIAGNOSTIC_OFFSET_RAD}]."
        )
    return args


def ros_environment() -> dict[str, str]:
    ros_env = os.environ.copy()
    ros_env.setdefault("ROS_DOMAIN_ID", "0")
    ros_env["ROS_LOCALHOST_ONLY"] = "0"
    ros_env["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "SUBNET"
    return ros_env


def set_bridge_active(service: str, active: bool) -> None:
    state = "true" if active else "false"
    ros_env = ros_environment()
    try:
        result = subprocess.run(  # nosec B603
            [
                "ros2",
                "service",
                "call",
                service,
                "std_srvs/srv/SetBool",
                f"{{data: {state}}}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
            env=ros_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out discovering {service}. Verify that start_deployment_server.sh is running, "
            "that this shell and the server use the same ROS_DOMAIN_ID, and that DDS discovery "
            "is allowed between the computers."
        ) from exc
    if result.returncode != 0 or "success=True" not in result.stdout:
        raise RuntimeError(
            f"Could not set bridge active={active}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def latest_packet(socket: zmq.Socket) -> dict[str, Any]:
    packet = socket.recv_pyobj()
    while True:
        try:
            packet = socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return packet


def topic_recorder(topics: list[str], output_dir: Path, stop_event: threading.Event) -> None:
    command = ["ros2", "bag", "record", "-o", str(output_dir), *topics]
    process = subprocess.Popen(  # nosec B603
        command,
        start_new_session=True,
        env=ros_environment(),
    )
    try:
        while process.poll() is None and not stop_event.wait(0.1):
            pass
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2.0)


def main() -> None:
    args = parse_args()
    if not args.execute:
        raise SystemExit(
            "Dry run only. Add --execute after checking arm, joint, offset, and workspace clearance."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.log_dir / f"{timestamp}_{args.arm}_j{args.joint}_{args.offset_rad:+.3f}rad"
    run_dir.mkdir(parents=True, exist_ok=False)
    jsonl = (run_dir / "samples.jsonl").open("w", encoding="utf-8")
    stop_recording = threading.Event()

    context = zmq.Context()
    observations = context.socket(zmq.SUB)
    observations.setsockopt(zmq.RCVHWM, 1)
    observations.setsockopt(zmq.CONFLATE, 1)
    observations.setsockopt(zmq.RCVTIMEO, 1000)
    observations.connect(f"tcp://{args.zmq_host}:{args.observation_port}")
    observations.setsockopt_string(zmq.SUBSCRIBE, "")
    commands = context.socket(zmq.PUSH)
    commands.setsockopt(zmq.SNDHWM, 1)
    commands.connect(f"tcp://{args.zmq_host}:{args.command_port}")

    topics = [
        f"/{args.arm}/deployment/joint_states",
        f"/{args.arm}/franka/commanded_joint_states",
        f"/{args.arm}/joint_states",
    ]
    recorder = threading.Thread(
        target=topic_recorder,
        args=(topics, run_dir / "rosbag", stop_recording),
        daemon=True,
    )

    activated = False
    try:
        set_bridge_active(args.bridge_activation_service, True)
        activated = True
        initial_packet = latest_packet(observations)
        initial_state = np.asarray(initial_packet["state"], dtype=float)
        if initial_state.shape != (16,) or not np.all(np.isfinite(initial_state)):
            raise ValueError(f"Expected a finite 16-D bridge state, got {initial_state.shape}.")

        arm_slice = slice(0, 7) if args.arm == "left" else slice(8, 15)
        left_target = initial_state[0:7].copy()
        right_target = initial_state[8:15].copy()
        selected_target = left_target if args.arm == "left" else right_target
        selected_target[args.joint - 1] += args.offset_rad
        payload = {
            "timestamp": time.time(),
            "left_joint_target": left_target.tolist(),
            "right_joint_target": right_target.tolist(),
            "left_gripper_command": None,
            "right_gripper_command": None,
        }
        metadata = {
            "arm": args.arm,
            "joint": args.joint,
            "offset_rad": args.offset_rad,
            "duration_s": args.duration_s,
            "settle_s": args.settle_s,
            "fps": args.fps,
            "initial_state": initial_state.tolist(),
            "requested_target": payload,
            "recorded_topics": topics,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Initial {args.arm} J{args.joint}: {initial_state[arm_slice][args.joint - 1]:+.6f} rad")
        print(f"Requested offset: {args.offset_rad:+.6f} rad")
        print(f"Logs: {run_dir}")
        recorder.start()
        # rosbag discovery is asynchronous. Give it time to subscribe before
        # the first short limiter ramp begins.
        time.sleep(1.0)

        end_time = time.monotonic() + args.duration_s
        period = 1.0 / args.fps
        while time.monotonic() < end_time:
            loop_start = time.monotonic()
            payload["timestamp"] = time.time()
            commands.send_pyobj(payload)
            try:
                packet = latest_packet(observations)
                state = np.asarray(packet["state"], dtype=float)
                record = {
                    "wall_time": time.time(),
                    "elapsed_s": args.duration_s - max(end_time - time.monotonic(), 0.0),
                    "requested_target": payload,
                    "robot_state": state.tolist(),
                    "selected_error_rad": float(
                        selected_target[args.joint - 1] - state[arm_slice][args.joint - 1]
                    ),
                }
                jsonl.write(json.dumps(record, separators=(",", ":")) + "\n")
                jsonl.flush()
            except zmq.Again:
                pass
            time.sleep(max(0.0, period - (time.monotonic() - loop_start)))

        hold_end = time.monotonic() + args.settle_s
        while time.monotonic() < hold_end:
            payload["timestamp"] = time.time()
            commands.send_pyobj(payload)
            time.sleep(period)
    except KeyboardInterrupt:
        print("Interrupted; returning bridge to standby.")
    finally:
        stop_recording.set()
        if recorder.is_alive():
            recorder.join(timeout=7.0)
        if activated:
            try:
                set_bridge_active(args.bridge_activation_service, False)
            except Exception as exc:
                print(f"WARNING: failed to return bridge to standby: {exc}")
        jsonl.close()
        observations.close(0)
        commands.close(0)
        context.term()


if __name__ == "__main__":
    main()
