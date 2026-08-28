"""Publish robot joint state and optionally forward policy hand targets."""

from __future__ import annotations

import argparse
import signal
import time
from typing import Any

import numpy as np
import zmq

from policy_contract import ARM_JOINT_NAMES, JOINT_NAMES
from transport import make_joint_state, validate_packet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", default="tcp://127.0.0.1:5565", help="ZMQ PUSH endpoint")
    parser.add_argument("--topic", action="append", default=[], help="ROS JointState topic; repeat for multiple groups")
    parser.add_argument("--rate", type=float, default=30.0, help="Mock/publish rate in Hz")
    parser.add_argument("--mock", action="store_true", help="Publish deterministic angles without ROS")
    parser.add_argument("--mock-joints", type=int, default=7)
    parser.add_argument("--command-connect", help="Policy-server PULL endpoint for 27-joint targets")
    parser.add_argument("--hand-command-address", default="tcp://127.0.0.1:5562", help="Robot-local Wuji worker PUSH endpoint")
    parser.add_argument("--execute", action="store_true", help="Forward received hand targets to the Wuji worker")
    parser.add_argument("--receive-only", action="store_true", help="Only receive policy targets; robot state is supplied through the deployment bridge")
    return parser.parse_args()


def drain_commands(command_socket: Any, hand_socket: Any, execute: bool) -> None:
    if command_socket is None:
        return
    latest = None
    while command_socket.poll(0, zmq.POLLIN):
        latest = command_socket.recv_json()
    if latest is None:
        return
    try:
        packet = validate_packet(latest)
    except (ValueError, TypeError) as exc:
        print(f"simtoolreal client rejected target: {exc}", flush=True)
        return
    if packet["kind"] != "joint_target":
        return
    hand = packet["target"][len(ARM_JOINT_NAMES) :]
    print("received policy hand target=" + np.array2string(np.asarray(hand), precision=5), flush=True)
    if execute:
        hand_socket.send_pyobj({"target": hand})


def run_mock(socket: Any, command_socket: Any, hand_socket: Any, args: argparse.Namespace, stop: list[bool]) -> None:
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")
    period = 1.0 / args.rate
    sequence = 0
    while not stop[0]:
        joints = [0.1 * (index + 1) + 0.01 * sequence for index in range(args.mock_joints)]
        names = list(JOINT_NAMES) if args.mock_joints == len(JOINT_NAMES) else [f"mock_joint_{index + 1}" for index in range(args.mock_joints)]
        socket.send_json(make_joint_state(joints, names, velocities=[0.0] * len(joints), source="mock-client"))
        drain_commands(command_socket, hand_socket, args.execute)
        sequence += 1
        time.sleep(period)


def run_receive_only(command_socket: Any, hand_socket: Any, args: argparse.Namespace, stop: list[bool]) -> None:
    if command_socket is None:
        raise SystemExit("--receive-only requires --command-connect")
    while not stop[0]:
        drain_commands(command_socket, hand_socket, args.execute)
        time.sleep(0.002)


def run_ros(socket: Any, command_socket: Any, hand_socket: Any, args: argparse.Namespace, stop: list[bool]) -> None:
    if not args.topic:
        args.topic = ["/left/franka/joint_states"]
    try:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise SystemExit("ROS mode requires rclpy and sensor_msgs; use --mock for a transport-only test") from exc

    rclpy.init()
    node = rclpy.create_node("simtoolreal_joint_client")
    latest: dict[str, tuple[list[str], list[float], list[float]]] = {}

    def callback(message: Any, topic: str) -> None:
        names = [str(name) for name in message.name]
        values = [float(value) for value in message.position]
        velocities = [float(value) for value in message.velocity]
        if len(names) != len(values):
            return
        if len(velocities) != len(values):
            velocities = [0.0] * len(values)
        latest[topic] = (names, values, velocities)

    for topic in args.topic:
        node.create_subscription(JointState, topic, lambda message, topic=topic: callback(message, topic), 10)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    next_publish = time.monotonic()
    try:
        while rclpy.ok() and not stop[0]:
            executor.spin_once(timeout_sec=0.05)
            now = time.monotonic()
            if any(topic not in latest for topic in args.topic) or now < next_publish:
                continue
            next_publish = now + 1.0 / args.rate
            drain_commands(command_socket, hand_socket, args.execute)
            names = [name for topic in args.topic for name in latest.get(topic, ([], [], []))[0]]
            joints = [value for topic in args.topic for value in latest.get(topic, ([], [], []))[1]]
            velocities = [value for topic in args.topic for value in latest.get(topic, ([], [], []))[2]]
            if joints:
                socket.send_json(make_joint_state(joints, names, velocities=velocities, source="ros2"))
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def main() -> int:
    args = parse_args()
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")
    if args.mock_joints <= 0:
        raise SystemExit("--mock-joints must be positive")
    if args.execute and not args.command_connect:
        raise SystemExit("--execute requires --command-connect")
    context = zmq.Context()
    socket = None
    if not args.receive_only:
        socket = context.socket(zmq.PUSH)
        socket.setsockopt(zmq.SNDHWM, 2)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(args.connect)
    command_socket = None
    hand_socket = None
    if args.command_connect:
        command_socket = context.socket(zmq.PULL)
        command_socket.setsockopt(zmq.RCVHWM, 1)
        command_socket.setsockopt(zmq.LINGER, 0)
        command_socket.connect(args.command_connect)
        hand_socket = context.socket(zmq.PUSH)
        hand_socket.setsockopt(zmq.SNDHWM, 1)
        hand_socket.setsockopt(zmq.LINGER, 0)
        hand_socket.connect(args.hand_command_address)
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    print(f"simtoolreal client {'receiving targets only' if args.receive_only else 'publishing on ' + args.connect}", flush=True)
    try:
        if args.receive_only:
            run_receive_only(command_socket, hand_socket, args, stop)
        elif args.mock:
            assert socket is not None
            run_mock(socket, command_socket, hand_socket, args, stop)
        else:
            assert socket is not None
            run_ros(socket, command_socket, hand_socket, args, stop)
    finally:
        if socket is not None:
            socket.close(0)
        if command_socket is not None:
            command_socket.close(0)
        if hand_socket is not None:
            hand_socket.close(0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
