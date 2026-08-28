"""Robot-side SimToolReal observation, policy RPC, and command executor.

The checkpoint stays on the server.  This process consumes the existing
deployment bridge's 27-joint right-arm/Wuji state and the server-local
FoundationPose++ pose stream, requests a 27-value normalized action, converts
it to safe targets, and routes seven arm values plus twenty hand values.
"""

from __future__ import annotations

import argparse
import json
import os
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
from policy_contract import (
    ACTION_DIM,
    OBS_FIELDS,
    POLICY_RATE_HZ,
    hardware_command_limits,
    load_joint_limits,
)
from pose_publisher import read_pose
from rl_policy import MockPolicy
from transport import make_policy_observation, validate_packet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "libs/SimToolReal-Franka-Wuji2/assets/urdf/franka_wuji_right_slanted/fr3v2_wuji_hand2_right_slanted.urdf"
DEFAULT_WORLD_FROM_ROBOT = np.asarray(
    (
        (0.0, 1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.45),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)


def parse_matrix(value: str) -> np.ndarray:
    path = Path(value).expanduser()
    if path.is_file():
        raw = read_pose(path) if path.suffix.lower() != ".json" else json.loads(path.read_text())
    else:
        raw = [float(item) for item in value.split(",")]
    return checked_transform(raw, name=str(path) if path.is_file() else "transform")


def bridge_state(packet: Any) -> tuple[np.ndarray, np.ndarray, int]:
    if not isinstance(packet, dict) or packet.get("arm_mode") != "right" or not packet.get("include_hand"):
        raise ValueError("expected right-arm bridge state with include_hand=true")
    q = np.asarray(packet.get("joint_state"), dtype=np.float64)
    if q.shape != (27,) or not np.all(np.isfinite(q)):
        raise ValueError("bridge joint_state must be a finite 27-vector")
    stamp_s = float(packet.get("robot_state_stamp_s", packet.get("bridge_publish_s", time.time())))
    return q, np.zeros(27, dtype=np.float64), int(stamp_s * 1e9)


def _state(packet: Any) -> tuple[np.ndarray, np.ndarray, int]:
    """Compatibility alias used by diagnostics and transport tests."""
    if isinstance(packet, dict) and packet.get("arm_mode") == "right":
        return bridge_state(packet)
    if isinstance(packet, dict) and "arm_mode" in packet:
        raise ValueError("expected right-arm bridge state with include_hand=true")
    validated = validate_packet(packet)
    if validated["kind"] != "joint_state":
        raise ValueError("expected joint_state packet")
    q = np.asarray(validated["joints"], dtype=np.float64)
    qd = np.asarray(validated["velocities"], dtype=np.float64)
    if q.shape != (27,):
        raise ValueError("unnamed state must contain 27 joints")
    return q, qd, int(validated["timestamp_ns"])


class PolicyExecutor:
    """Dependency-light local adapter for observation/action contract tests.

    Production execution uses :class:`PolicyRpcClient`; this class retains a
    pure local path for validating FK, limits, and action conversion without a
    checkpoint or network.
    """

    def __init__(self, *, robot_urdf: Path, upstream_root: Path, config: Path | None,
                 checkpoint: Path | None, device: str = "cpu", rate: float = POLICY_RATE_HZ,
                 dof_speed_scale: float = 1.5, arm_moving_average: float = 0.1,
                 hand_moving_average: float = 0.1, mock_policy: bool = False) -> None:
        del upstream_root, config, checkpoint, device
        lower, upper = load_joint_limits(robot_urdf)
        command_lower, command_upper = hardware_command_limits(lower, upper)
        self.builder = ObservationBuilder(PolicyKinematics(robot_urdf), lower, upper, fields=OBS_FIELDS)
        self.pipeline = ActionPipeline(lower, upper, dt=1.0 / rate,
                                        dof_speed_scale=dof_speed_scale,
                                        arm_moving_average=arm_moving_average,
                                        hand_moving_average=hand_moving_average,
                                        command_lower_limits=command_lower,
                                        command_upper_limits=command_upper)
        if not mock_policy:
            raise ValueError("local PolicyExecutor is diagnostic-only; use policy RPC in production")
        self.policy = MockPolicy(ACTION_DIM)

    def infer(self, measured: np.ndarray, velocity: np.ndarray, object_pose: np.ndarray,
              goal_pose: np.ndarray, object_scales: np.ndarray,
              world_from_robot: np.ndarray) -> tuple[np.ndarray, Any]:
        measured = np.asarray(measured, dtype=np.float64)
        if self.pipeline.previous is None:
            self.pipeline.reset(measured)
            self.policy.reset()
        observation = self.builder.build(measured, velocity, self.pipeline.previous,
                                         object_pose, goal_pose, object_scales,
                                         world_from_robot).vector
        action = self.policy.act(observation)
        return action, self.pipeline.targets(action)


class PolicyRpcClient:
    def __init__(self, context: zmq.Context, address: str) -> None:
        self.socket = context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(address)

    def act(self, observation: np.ndarray) -> np.ndarray:
        self.socket.send_json(make_policy_observation(observation.tolist()))
        if not self.socket.poll(5000, zmq.POLLIN):
            raise TimeoutError("policy server did not return an action within 5 seconds")
        response = validate_packet(self.socket.recv_json())
        if response["kind"] != "policy_action":
            raise ValueError(f"policy server returned {response.get('kind')!r}: {response.get('error', '')}")
        action = np.asarray(response["action"], dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"policy action must have shape ({ACTION_DIM},), got {action.shape}")
        return action

    def close(self) -> None:
        self.socket.close(0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_server_ip = os.environ.get("SIMTOOLREAL_SERVER_IP", os.environ.get("DEPLOYMENT_SERVER_IP", "192.168.50.13"))
    parser.add_argument("--server-ip", default=default_server_ip, help="Server computer address for default endpoints")
    parser.add_argument("--state-connect", default=None, help="Bridge PUB endpoint (default: tcp://SERVER_IP:5555)")
    parser.add_argument("--pose-connect", default=None, help="FoundationPose++ PUB endpoint (default: tcp://SERVER_IP:5570)")
    parser.add_argument("--policy-address", default=None, help="Policy RPC REP endpoint (default: tcp://SERVER_IP:5571)")
    parser.add_argument("--arm-command-connect", default=None, help="Bridge arm-command PULL endpoint (default: tcp://SERVER_IP:5556)")
    parser.add_argument("--hand-command-address", default="tcp://127.0.0.1:5562", help="Robot-local Wuji PULL endpoint")
    parser.add_argument("--goal-pose", required=True, type=parse_matrix)
    parser.add_argument("--pose-frame", choices=("camera", "world", "robot"), default="camera")
    parser.add_argument("--world-from-camera", type=parse_matrix)
    parser.add_argument("--world-from-robot", type=parse_matrix, default=DEFAULT_WORLD_FROM_ROBOT, help="Policy-world pose of the selected robot URDF root")
    parser.add_argument("--object-scales", default="1,1,1")
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--rate", type=float, default=POLICY_RATE_HZ)
    parser.add_argument("--max-state-age", type=float, default=0.25)
    parser.add_argument("--max-pose-age", type=float, default=0.25)
    parser.add_argument("--arm-moving-average", type=float, default=0.1)
    parser.add_argument("--hand-moving-average", type=float, default=0.1)
    parser.add_argument("--dof-speed-scale", type=float, default=1.5)
    parser.add_argument("--execute", action="store_true", help="Send commands to the bridge and Wuji worker")
    parser.add_argument("--bridge-activation-service", default="/set_deployment_active")
    parser.add_argument("--no-auto-activate-bridge", action="store_true")
    args = parser.parse_args(argv)
    args.state_connect = args.state_connect or f"tcp://{args.server_ip}:5555"
    args.pose_connect = args.pose_connect or f"tcp://{args.server_ip}:5570"
    args.policy_address = args.policy_address or f"tcp://{args.server_ip}:5571"
    args.arm_command_connect = args.arm_command_connect or f"tcp://{args.server_ip}:5556"
    if args.pose_frame == "camera" and args.world_from_camera is None:
        parser.error("--world-from-camera is required when --pose-frame=camera")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rate <= 0 or args.max_state_age <= 0 or args.max_pose_age <= 0:
        raise SystemExit("rate and freshness limits must be positive")
    scales = np.asarray([float(item) for item in args.object_scales.split(",")], dtype=np.float64)
    if scales.shape != (3,) or np.any(~np.isfinite(scales)) or np.any(scales <= 0):
        raise SystemExit("--object-scales must contain three positive finite values")
    lower, upper = load_joint_limits(args.robot_urdf)
    command_lower, command_upper = hardware_command_limits(lower, upper)
    builder = ObservationBuilder(PolicyKinematics(args.robot_urdf), lower, upper, fields=OBS_FIELDS)
    pipeline = ActionPipeline(lower, upper, dt=1.0 / args.rate,
                              dof_speed_scale=args.dof_speed_scale,
                              arm_moving_average=args.arm_moving_average,
                              hand_moving_average=args.hand_moving_average,
                              command_lower_limits=command_lower,
                              command_upper_limits=command_upper)
    context = zmq.Context()
    state_socket = context.socket(zmq.SUB); state_socket.setsockopt(zmq.SUBSCRIBE, b""); state_socket.setsockopt(zmq.CONFLATE, 1); state_socket.connect(args.state_connect)
    pose_socket = context.socket(zmq.SUB); pose_socket.setsockopt(zmq.SUBSCRIBE, b""); pose_socket.setsockopt(zmq.CONFLATE, 1); pose_socket.connect(args.pose_connect)
    arm_socket = context.socket(zmq.PUSH); arm_socket.setsockopt(zmq.SNDHWM, 1); arm_socket.connect(args.arm_command_connect)
    hand_socket = context.socket(zmq.PUSH); hand_socket.setsockopt(zmq.SNDHWM, 1); hand_socket.connect(args.hand_command_address)
    policy = PolicyRpcClient(context, args.policy_address)
    poller = zmq.Poller(); poller.register(state_socket, zmq.POLLIN); poller.register(pose_socket, zmq.POLLIN)
    latest_q = latest_qd = latest_pose = None; state_stamp_ns = pose_stamp_ns = 0
    stop = [False]; signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True)); signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    print(f"SimToolReal executor waiting for right-arm/Wuji state and pose; mode={'EXECUTE' if args.execute else 'DRY-RUN'}", flush=True)
    bridge_activated = False
    if args.execute and not args.no_auto_activate_bridge:
        command = ["ros2", "service", "call", args.bridge_activation_service, "std_srvs/srv/SetBool", "{data: true}"]
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=8.0)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SystemExit(f"could not activate deployment bridge: {exc}") from exc
        if result.returncode != 0 or "success=True" not in result.stdout:
            raise SystemExit(f"deployment bridge activation failed: {result.stdout.strip()} {result.stderr.strip()}")
        bridge_activated = True
    try:
        while not stop[0]:
            for socket, _ in poller.poll(100):
                raw = socket.recv_pyobj() if socket is state_socket else socket.recv_json()
                try:
                    if socket is state_socket:
                        latest_q, latest_qd, state_stamp_ns = bridge_state(raw)
                    else:
                        packet = validate_packet(raw)
                        if packet["kind"] != "object_pose": raise ValueError("expected object_pose")
                        source = checked_transform(np.asarray(packet["pose"]).reshape(4, 4), name="FoundationPose++ pose")
                        frame = str(packet.get("frame_id", args.pose_frame)).lower()
                        if frame == "camera":
                            if args.world_from_camera is None:
                                raise ValueError("camera pose received without --world-from-camera")
                            latest_pose = args.world_from_camera @ source
                        elif frame == "robot":
                            latest_pose = args.world_from_robot @ source
                        else:
                            latest_pose = source
                        pose_stamp_ns = int(packet["timestamp_ns"])
                except (ValueError, TypeError) as exc:
                    print(f"rejected packet: {exc}", flush=True)
            if latest_q is None or latest_pose is None:
                continue
            now_ns = time.time_ns()
            if (now_ns - state_stamp_ns) * 1e-9 > args.max_state_age or (now_ns - pose_stamp_ns) * 1e-9 > args.max_pose_age:
                continue
            if pipeline.previous is None:
                pipeline.reset(latest_q)
            observation = builder.build(latest_q, latest_qd, pipeline.previous, latest_pose,
                                        args.goal_pose, scales, args.world_from_robot).vector
            action = policy.act(observation)
            target = pipeline.targets(action)
            print(f"state[27]={np.array2string(latest_q, precision=4)} object_xyz={np.array2string(latest_pose[:3,3], precision=4)} target[27]={np.array2string(target.full, precision=4)}", flush=True)
            if args.execute:
                arm_socket.send_pyobj({"timestamp": time.time(), "right_joint_target": target.arm.tolist()})
                hand_socket.send_pyobj(target.hand.tolist())
            time.sleep(1.0 / args.rate)
    finally:
        if bridge_activated:
            subprocess.run(["ros2", "service", "call", args.bridge_activation_service, "std_srvs/srv/SetBool", "{data: false}"], check=False, timeout=8.0)
        policy.close(); state_socket.close(0); pose_socket.close(0); arm_socket.close(0); hand_socket.close(0); context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
