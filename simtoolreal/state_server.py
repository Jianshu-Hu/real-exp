"""Run the FR3/Wuji SimToolReal inference loop on the server computer.

The preferred state source is the repository's existing deployment bridge.
The lightweight robot client remains available for a two-script test. Commands
are opt-in: without ``--execute`` this process only prints inferred targets.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from action import ActionPipeline
from kinematics import PolicyKinematics
from observation import ObservationBuilder, checked_transform
from policy_contract import ACTION_DIM, JOINT_NAMES, OBS_DIM, OBS_FIELDS, POLICY_RATE_HZ, hardware_command_limits, load_joint_limits, reorder_joint_vector
from pose_publisher import read_pose
from rl_policy import MockPolicy, RlGamesPolicy
from transport import make_joint_target, validate_packet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT / "libs/SimToolReal-Franka-Wuji2"
DEFAULT_ROBOT_URDF = DEFAULT_UPSTREAM / "assets/urdf/franka_wuji_right_slanted/fr3v2_wuji_hand2_right_slanted.urdf"
DEFAULT_WORLD_FROM_ROBOT = np.asarray(
    (
        (0.0, 1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.45),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


def csv_floats(value: str, count: int, label: str) -> np.ndarray:
    try:
        result = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated numbers") from exc
    if result.shape != (count,) or not np.all(np.isfinite(result)):
        raise argparse.ArgumentTypeError(f"{label} must contain {count} finite numbers")
    return result


def transform_arg(value: str) -> np.ndarray:
    path = Path(value).expanduser()
    raw = read_pose(path) if path.is_file() else csv_floats(value, 16, "transform")
    return checked_transform(raw, name=str(path) if path.is_file() else "transform")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    state = parser.add_mutually_exclusive_group()
    state.add_argument("--bridge-address", help="Existing deploy bridge PUB endpoint, e.g. tcp://SERVER_IP:5555")
    state.add_argument("--bind", default="tcp://*:5565", help="Lightweight joint/object PULL endpoint")
    parser.add_argument("--command-address", default="tcp://127.0.0.1:5556", help="Existing deploy bridge command endpoint")
    parser.add_argument("--activate-bridge", action="store_true", help="Call the ROS2 deployment bridge activation service on startup")
    parser.add_argument("--bridge-activation-service", default="/set_deployment_active")
    parser.add_argument("--client-command-bind", help="Optional target PUSH bind for the lightweight robot client")
    parser.add_argument("--pose-address", help="FoundationPose++ JSON/PUB endpoint (default: disabled when --pose-file/--mock-pose is used)")
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF, help="Exact combined URDF used for training limits and FK")
    parser.add_argument("--config", type=Path, help="rl_games config.yaml")
    parser.add_argument("--checkpoint", type=Path, help="rl_games .pth checkpoint")
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mock-policy", action="store_true", help="Use a deterministic zero-action policy")
    parser.add_argument("--execute", action="store_true", help="Actually send arm/hand targets")
    parser.add_argument("--rate", type=float, default=POLICY_RATE_HZ)
    parser.add_argument("--max-state-age", type=float, default=0.25)
    parser.add_argument("--max-pose-age", type=float, default=0.25)
    parser.add_argument("--pose-file", type=Path, help="Live FoundationPose++ transform")
    parser.add_argument("--mock-pose", action="store_true")
    parser.add_argument("--pose-frame", choices=("robot", "world", "camera"), default="camera")
    parser.add_argument("--world-from-camera", type=transform_arg, help="Required calibration when --pose-frame=camera")
    parser.add_argument("--world-from-robot", type=transform_arg, default=DEFAULT_WORLD_FROM_ROBOT, help="Policy world-from-robot transform (default: bundled slanted checkpoint profile)")
    parser.add_argument("--goal-pose", required=False, type=transform_arg, help="Goal object pose in policy-world coordinates")
    parser.add_argument("--object-scales", default="1,1,1", help="Training-scale triplet, not raw metres")
    parser.add_argument("--arm-moving-average", type=float, default=0.1)
    parser.add_argument("--hand-moving-average", type=float, default=0.1)
    parser.add_argument("--dof-speed-scale", type=float, default=1.5)
    parser.add_argument("--print-period", type=float, default=1.0)
    parser.add_argument("--wait-only", action="store_true", help="Wait for and report the first valid state packet without running policy inference")
    args = parser.parse_args(argv)
    if not args.wait_only and args.goal_pose is None:
        parser.error("--goal-pose is required unless --wait-only is used")
    try:
        args.object_scales = csv_floats(args.object_scales, 3, "--object-scales")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.rate <= 0.0 or args.max_state_age <= 0.0 or args.max_pose_age <= 0.0:
        parser.error("rate and freshness limits must be positive")
    if args.print_period < 0.0:
        parser.error("--print-period must be non-negative")
    if not args.wait_only and args.pose_frame == "camera" and args.world_from_camera is None and not args.mock_pose:
        parser.error("--world-from-camera is required when FoundationPose++ output is in camera coordinates")
    if args.execute and not args.bridge_address and not args.client_command_bind:
        parser.error("--execute requires --bridge-address or --client-command-bind")
    if args.execute and args.bridge_address and not args.client_command_bind:
        parser.error("--execute with the FR3/Wuji bridge also requires --client-command-bind for the robot-local hand worker")
    if not args.wait_only and not args.mock_policy and (args.config is None or args.checkpoint is None):
        parser.error("provide --config and --checkpoint, or use --mock-policy")
    return args


def wait_for_first_observation(args: argparse.Namespace) -> int:
    """Server readiness mode used when policy inference is hosted separately."""
    context = zmq.Context()
    socket = context.socket(zmq.SUB if args.bridge_address else zmq.PULL)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.LINGER, 0)
    if args.bridge_address:
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.connect(args.bridge_address)
    else:
        socket.bind(args.bind)
    stop = [False]
    announced = False
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    print("SimToolReal server waiting for first valid right-arm/Wuji observation", flush=True)
    try:
        while not stop[0]:
            if not socket.poll(100, zmq.POLLIN):
                continue
            raw = socket.recv_pyobj() if args.bridge_address else socket.recv_json()
            try:
                if args.bridge_address:
                    bridge_state(raw)
                else:
                    client_state(raw)
                if not announced:
                    print("received first valid right-arm/Wuji robot observation; server is ready", flush=True)
                    announced = True
            except (ValueError, TypeError) as exc:
                print(f"simtoolreal rejected state packet: {exc}", flush=True)
    finally:
        socket.close(0)
        context.term()
    return 0


def pose_in_world(pose: np.ndarray, frame: str, world_from_camera: np.ndarray | None, world_from_robot: np.ndarray) -> np.ndarray:
    source_pose = checked_transform(pose, name="FoundationPose++ pose")
    if frame == "world":
        return source_pose
    if frame == "robot":
        return world_from_robot @ source_pose
    if world_from_camera is None:
        raise ValueError("camera-frame pose received without world-from-camera calibration")
    return world_from_camera @ source_pose


def bridge_state(packet: Any) -> tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(packet, dict):
        raise ValueError("deployment bridge packet must be a dictionary")
    if packet.get("arm_mode") != "right" or not packet.get("include_hand"):
        raise ValueError("SimToolReal requires bridge arm_mode=right and include_hand=true")
    position = np.asarray(packet.get("joint_state"), dtype=np.float64)
    if position.shape != (27,) or not np.all(np.isfinite(position)):
        raise ValueError(f"bridge joint_state must be a finite 27-vector, got {position.shape}")
    velocity = np.zeros(27, dtype=np.float64)
    stamp_s = float(packet.get("robot_state_stamp_s", packet.get("bridge_publish_s", time.time())))
    return position, velocity, int(stamp_s * 1e9)


def client_state(packet: Any) -> tuple[np.ndarray, np.ndarray, int]:
    validated = validate_packet(packet)
    if validated["kind"] != "joint_state":
        raise ValueError("not a joint-state packet")
    if not validated["names"]:
        if len(validated["joints"]) != 27:
            raise ValueError("unnamed joint state must contain exactly 27 values")
        position = np.asarray(validated["joints"], dtype=np.float64)
        velocity = np.asarray(validated["velocities"], dtype=np.float64)
    else:
        position = reorder_joint_vector(validated["joints"], validated["names"])
        velocity = reorder_joint_vector(validated["velocities"], validated["names"])
    return position, velocity, int(validated["timestamp_ns"])


def format_cycle(q: np.ndarray, pose: np.ndarray, observation: np.ndarray, action: np.ndarray, targets: np.ndarray, execute: bool) -> str:
    return (
        f"mode={'EXECUTE' if execute else 'DRY-RUN'} state[27]={np.array2string(q, precision=4)}\n"
        f"object_xyz={np.array2string(pose[:3, 3], precision=5)} observation[{observation.size}] "
        f"min/max=({observation.min():+.4f},{observation.max():+.4f})\n"
        f"action[27]={np.array2string(action, precision=4)}\n"
        f"target[27]={np.array2string(targets, precision=4)}"
    )


def set_bridge_active(service: str, active: bool) -> None:
    """Switch the existing ROS2 deployment bridge between active and standby."""
    command = [
        "ros2", "service", "call", service, "std_srvs/srv/SetBool",
        f"{{data: {'true' if active else 'false'}}}",
    ]
    last_error = ""
    for attempt in range(5):
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            last_error = str(exc)
        else:
            if result.returncode == 0 and "success=True" in result.stdout:
                return
            last_error = f"stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}"
        if attempt < 4:
            time.sleep(1.0)
    raise RuntimeError(f"deployment bridge service failed ({service}, active={active}): {last_error}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.wait_only:
        return wait_for_first_observation(args)
    lower, upper = load_joint_limits(args.robot_urdf)
    command_lower, command_upper = hardware_command_limits(lower, upper)
    builder = ObservationBuilder(PolicyKinematics(args.robot_urdf), lower, upper, fields=OBS_FIELDS)
    pipeline = ActionPipeline(
        lower, upper, dt=1.0 / args.rate, dof_speed_scale=args.dof_speed_scale,
        arm_moving_average=args.arm_moving_average, hand_moving_average=args.hand_moving_average,
        command_lower_limits=command_lower, command_upper_limits=command_upper,
    )
    policy = MockPolicy(ACTION_DIM) if args.mock_policy else RlGamesPolicy(
        args.upstream_root, args.config, args.checkpoint, OBS_DIM, ACTION_DIM, args.device
    )
    context = zmq.Context()
    state_socket = context.socket(zmq.SUB if args.bridge_address else zmq.PULL)
    state_socket.setsockopt(zmq.RCVHWM, 2)
    state_socket.setsockopt(zmq.LINGER, 0)
    if args.bridge_address:
        state_socket.setsockopt(zmq.SUBSCRIBE, b"")
        state_socket.connect(args.bridge_address)
    else:
        state_socket.bind(args.bind)
    pose_socket = None
    if args.pose_address:
        pose_socket = context.socket(zmq.SUB)
        pose_socket.setsockopt(zmq.SUBSCRIBE, b"")
        pose_socket.setsockopt(zmq.RCVHWM, 2)
        pose_socket.setsockopt(zmq.LINGER, 0)
        pose_socket.connect(args.pose_address)
    arm_socket = client_command_socket = None
    if args.execute and args.bridge_address:
        arm_socket = context.socket(zmq.PUSH)
        arm_socket.setsockopt(zmq.SNDHWM, 1)
        arm_socket.setsockopt(zmq.LINGER, 0)
        arm_socket.connect(args.command_address)
    if args.execute and args.client_command_bind:
        client_command_socket = context.socket(zmq.PUSH)
        client_command_socket.setsockopt(zmq.SNDHWM, 1)
        client_command_socket.setsockopt(zmq.LINGER, 0)
        client_command_socket.bind(args.client_command_bind)
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    latest_q = latest_qd = latest_pose = None
    previous_measured_q = None
    previous_state_stamp_ns = 0
    state_stamp_ns = pose_stamp_ns = 0
    pose_signature = None
    announced_state = False
    last_cycle = last_print = 0.0
    print(
        f"SimToolReal server: state={'bridge ' + args.bridge_address if args.bridge_address else args.bind}, "
        f"policy={'mock' if args.mock_policy else args.checkpoint}, observation/action={OBS_DIM}/{ACTION_DIM}, "
        f"mode={'EXECUTE' if args.execute else 'DRY-RUN'}", flush=True,
    )
    print("SimToolReal server waiting for first valid robot observation and object pose", flush=True)
    bridge_activated = False
    if args.activate_bridge:
        if not args.bridge_address:
            raise SystemExit("--activate-bridge requires --bridge-address")
        set_bridge_active(args.bridge_activation_service, True)
        bridge_activated = True
    try:
        while not stop[0]:
            sockets = [state_socket] + ([pose_socket] if pose_socket is not None else [])
            poller = zmq.Poller()
            for sock in sockets:
                poller.register(sock, zmq.POLLIN)
            for ready, _ in poller.poll(5):
                raw = ready.recv_pyobj() if ready is state_socket and args.bridge_address else ready.recv_json()
                try:
                    if isinstance(raw, dict) and raw.get("kind") == "object_pose":
                        packet = validate_packet(raw)
                        packet_frame = str(packet["frame_id"]).strip().lower()
                        frame = packet_frame if packet_frame in {"world", "robot", "camera"} else args.pose_frame
                        latest_pose = pose_in_world(np.asarray(packet["pose"]).reshape(4, 4), frame, args.world_from_camera, args.world_from_robot)
                        pose_stamp_ns = int(packet["timestamp_ns"])
                    elif ready is state_socket:
                        next_q, next_qd, next_stamp_ns = bridge_state(raw) if args.bridge_address else client_state(raw)
                        # The deploy bridge currently does not include measured
                        # velocities and the Wuji SDK telemetry is position-only.
                        # Estimate the complete vector from consecutive samples,
                        # preserving explicit client velocities when non-zero.
                        if previous_measured_q is not None and next_stamp_ns > previous_state_stamp_ns:
                            estimated = (next_q - previous_measured_q) / ((next_stamp_ns - previous_state_stamp_ns) * 1e-9)
                            if args.bridge_address or not np.any(next_qd):
                                next_qd = estimated
                        previous_measured_q = next_q.copy()
                        previous_state_stamp_ns = next_stamp_ns
                        latest_q, latest_qd, state_stamp_ns = next_q, next_qd, next_stamp_ns
                        if not announced_state:
                            print("received first valid right-arm/Wuji robot observation", flush=True)
                            announced_state = True
                except (ValueError, TypeError) as exc:
                    print(f"simtoolreal rejected state packet: {exc}", flush=True)
            if args.pose_file is not None:
                try:
                    stat = args.pose_file.stat()
                    signature = (stat.st_mtime_ns, stat.st_size)
                    if signature != pose_signature:
                        latest_pose = pose_in_world(np.asarray(read_pose(args.pose_file)).reshape(4, 4), args.pose_frame, args.world_from_camera, args.world_from_robot)
                        pose_stamp_ns, pose_signature = stat.st_mtime_ns, signature
                except FileNotFoundError:
                    pass
            elif args.mock_pose:
                latest_pose = np.eye(4)
                latest_pose[:3, 3] = (0.0, 0.0, 0.65)
                pose_stamp_ns = time.time_ns()
            now, now_ns = time.monotonic(), time.time_ns()
            if latest_q is None or latest_pose is None or now - last_cycle < 1.0 / args.rate:
                continue
            if (now_ns - state_stamp_ns) * 1e-9 > args.max_state_age or (now_ns - pose_stamp_ns) * 1e-9 > args.max_pose_age:
                if now - last_print >= max(args.print_period, 1.0):
                    print("waiting for fresh robot state and FoundationPose++ pose", flush=True)
                    last_print = now
                continue
            if pipeline.previous is None:
                pipeline.reset(latest_q)
                policy.reset()
            result = builder.build(latest_q, latest_qd, pipeline.previous, latest_pose, args.goal_pose, args.object_scales, args.world_from_robot)
            action = policy.act(result.vector)
            target = pipeline.targets(action)
            last_cycle = now
            if args.print_period == 0.0 or now - last_print >= args.print_period:
                print(format_cycle(latest_q, latest_pose, result.vector, action, target.full, args.execute), flush=True)
                last_print = now
            if args.execute:
                if arm_socket is not None:
                    arm_socket.send_pyobj({"timestamp": time.time(), "right_joint_target": target.arm.tolist()})
                if client_command_socket is not None:
                    client_command_socket.send_json(make_joint_target(target.full.tolist(), names=list(JOINT_NAMES)))
    finally:
        if bridge_activated:
            try:
                set_bridge_active(args.bridge_activation_service, False)
            except RuntimeError as exc:
                print(f"warning: {exc}", flush=True)
        state_socket.close(0)
        if pose_socket is not None:
            pose_socket.close(0)
        if arm_socket is not None:
            arm_socket.close(0)
        if client_command_socket is not None:
            client_command_socket.close(0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
