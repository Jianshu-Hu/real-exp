#!/usr/bin/env python3
"""Capture one real FR3/Wuji/object state for deterministic Isaac Sim replay.

This tool is read-only. It subscribes to the deployment bridge and
FoundationPose++ streams, applies the same frame transforms as the policy
executor, computes the policy palm/fingertip geometry, and writes one JSON
snapshot. It never opens a command socket or sends a robot target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from kinematics import PolicyKinematics
from observation import checked_transform
from policy_contract import JOINT_NAMES
from policy_executor import bridge_state, parse_matrix
from transport import validate_packet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf"
DEFAULT_MESH = "libs/FoundationPose-plus-plus/test/mesh/hammer.stl"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_scales(value: str) -> np.ndarray:
    result = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if result.shape != (3,) or np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise argparse.ArgumentTypeError("expected three positive finite values")
    return result


def _json_matrix(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).tolist()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def build_snapshot(
    *,
    state_packet: dict[str, Any],
    pose_packet: dict[str, Any],
    state_arrival_ns: int,
    pose_arrival_ns: int,
    world_from_camera: np.ndarray | None,
    world_from_robot: np.ndarray,
    goal_pose: np.ndarray,
    robot_urdf: Path,
    object_scales: np.ndarray,
    mesh_path: str,
    mesh_scale: float,
    camera_name: str,
    camera_serial: str,
    fallback_pose_frame: str,
) -> dict[str, Any]:
    """Validate two packets and build the portable snapshot payload."""
    joint_position, joint_velocity, state_stamp_ns = bridge_state(state_packet)
    validated_pose = validate_packet(pose_packet)
    if validated_pose["kind"] != "object_pose":
        raise ValueError("expected an object_pose packet")

    source_pose = checked_transform(
        np.asarray(validated_pose["pose"], dtype=np.float64).reshape(4, 4),
        name="FoundationPose++ pose",
    )
    source_frame = str(validated_pose.get("frame_id", fallback_pose_frame)).lower()
    if source_frame == "camera":
        if world_from_camera is None:
            raise ValueError("camera-frame pose requires --world-from-camera")
        world_from_object = world_from_camera @ source_pose
    elif source_frame == "robot":
        world_from_object = world_from_robot @ source_pose
    elif source_frame == "world":
        world_from_object = source_pose
    else:
        raise ValueError(f"unsupported object pose frame {source_frame!r}")
    world_from_object = checked_transform(world_from_object, name="policy world-from-object")

    kinematics = PolicyKinematics(robot_urdf)
    palm_pos, palm_quat_xyzw, fingertips = kinematics.evaluate(
        joint_position, world_from_robot
    )
    pose_stamp_ns = int(validated_pose["timestamp_ns"])
    captured_ns = time.time_ns()
    bridge_publish_s = state_packet.get("bridge_publish_s")
    bridge_publish_ns = (
        int(float(bridge_publish_s) * 1e9) if bridge_publish_s is not None else None
    )

    mesh = Path(mesh_path).expanduser()
    mesh_hash = _sha256(mesh.resolve()) if mesh.is_file() else None
    return {
        "format": "simtoolreal_real_snapshot_v1",
        "captured_at_unix_ns": captured_ns,
        "frame_convention": "A_T_B maps coordinates from frame B into frame A",
        "policy_world_frame": "Wp (the selected Isaac task env-local frame)",
        "synchronization": {
            "state_arrival_unix_ns": int(state_arrival_ns),
            "pose_arrival_unix_ns": int(pose_arrival_ns),
            "arrival_skew_ms": abs(state_arrival_ns - pose_arrival_ns) / 1e6,
            "source_stamp_skew_ms": abs(state_stamp_ns - pose_stamp_ns) / 1e6,
            "note": (
                "arrival_skew is authoritative when client/server wall clocks are not synchronized"
            ),
        },
        "state": {
            "joint_names": list(JOINT_NAMES),
            "joint_position_27": joint_position.tolist(),
            "joint_velocity_27": joint_velocity.tolist(),
            "robot_state_stamp_ns": int(state_stamp_ns),
            "bridge_publish_ns": bridge_publish_ns,
            "arm_mode": "right",
            "include_hand": True,
        },
        "object": {
            "object_id": str(validated_pose.get("object_id", "object")),
            "source": str(validated_pose.get("source", "foundationpose++")),
            "source_frame": source_frame,
            "source_timestamp_ns": pose_stamp_ns,
            "source_T_object_raw_mesh": _json_matrix(source_pose),
            "Wp_T_object_raw_mesh": _json_matrix(world_from_object),
            "mesh_repository_path": mesh_path,
            "mesh_sha256": mesh_hash,
            "mesh_scale_m_per_source_unit": float(mesh_scale),
            "mesh_frame": "raw_uncentered_stl",
        },
        "camera": {
            "name": camera_name,
            "serial": camera_serial,
            "Wp_T_camera": (
                _json_matrix(world_from_camera) if world_from_camera is not None else None
            ),
        },
        "robot": {
            "Wp_T_robot_root": _json_matrix(world_from_robot),
            "urdf": _file_record(robot_urdf),
            "policy_palm": {
                "position_xyz": palm_pos.tolist(),
                "quaternion_xyzw": palm_quat_xyzw.tolist(),
            },
            "policy_fingertips_xyz": fingertips.tolist(),
        },
        "goal": {"Wp_T_goal": _json_matrix(goal_pose)},
        "object_scales": np.asarray(object_scales, dtype=np.float64).tolist(),
        "limitations": [
            "This snapshot checks coordinate, joint-order, and mesh-frame consistency.",
            "It does not validate contact dynamics, latency, controller gains, or grasp success.",
            "The policy object_scales value is metadata; it must be validated against training separately.",
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-ip", default=os.environ.get("SIMTOOLREAL_SERVER_IP", "192.168.50.13"))
    parser.add_argument("--state-connect", default="tcp://127.0.0.1:5555")
    parser.add_argument("--pose-connect", default=None)
    parser.add_argument("--world-from-camera", type=parse_matrix, required=True)
    parser.add_argument("--world-from-robot", type=parse_matrix, required=True)
    parser.add_argument("--goal-pose", type=parse_matrix, required=True)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--object-scales", type=_parse_scales, default=_parse_scales("1,1,1"))
    parser.add_argument("--mesh-path", default=DEFAULT_MESH)
    parser.add_argument("--mesh-scale", type=float, default=0.001)
    parser.add_argument("--camera-name", default="l515")
    parser.add_argument("--camera-serial", default="f1480539")
    parser.add_argument("--pose-frame", choices=("camera", "world", "robot"), default="camera")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-arrival-skew", type=float, default=0.20)
    parser.add_argument("--max-stream-age", type=float, default=0.50)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "calibration/runs/real_snapshot.json",
    )
    args = parser.parse_args(argv)
    args.pose_connect = args.pose_connect or f"tcp://{args.server_ip}:5570"
    if args.timeout <= 0.0 or args.max_arrival_skew <= 0.0 or args.max_stream_age <= 0.0:
        parser.error("timeout and freshness limits must be positive")
    if not np.isfinite(args.mesh_scale) or args.mesh_scale <= 0.0:
        parser.error("--mesh-scale must be positive and finite")
    if not args.robot_urdf.is_file():
        parser.error(f"robot URDF does not exist: {args.robot_urdf}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = zmq.Context()
    state_socket = context.socket(zmq.SUB)
    pose_socket = context.socket(zmq.SUB)
    for socket in (state_socket, pose_socket):
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt(zmq.LINGER, 0)
    state_socket.connect(args.state_connect)
    pose_socket.connect(args.pose_connect)
    poller = zmq.Poller()
    poller.register(state_socket, zmq.POLLIN)
    poller.register(pose_socket, zmq.POLLIN)

    latest_state: tuple[dict[str, Any], int] | None = None
    latest_pose: tuple[dict[str, Any], int] | None = None
    deadline = time.monotonic() + args.timeout
    print(
        "Waiting for a fresh, synchronized right FR3/Wuji state and "
        f"FoundationPose++ pose ({args.state_connect}, {args.pose_connect})",
        flush=True,
    )
    try:
        while time.monotonic() < deadline:
            events = dict(poller.poll(100))
            now_ns = time.time_ns()
            if state_socket in events:
                packet = state_socket.recv_pyobj()
                bridge_state(packet)
                latest_state = (packet, now_ns)
            if pose_socket in events:
                packet = pose_socket.recv_json()
                validated = validate_packet(packet)
                if validated["kind"] != "object_pose":
                    raise ValueError("pose endpoint returned a non-object-pose packet")
                latest_pose = (packet, now_ns)
            if latest_state is None or latest_pose is None:
                continue
            state_packet, state_arrival_ns = latest_state
            pose_packet, pose_arrival_ns = latest_pose
            current_ns = time.time_ns()
            newest_age_s = (current_ns - min(state_arrival_ns, pose_arrival_ns)) * 1e-9
            arrival_skew_s = abs(state_arrival_ns - pose_arrival_ns) * 1e-9
            if newest_age_s > args.max_stream_age or arrival_skew_s > args.max_arrival_skew:
                continue
            snapshot = build_snapshot(
                state_packet=state_packet,
                pose_packet=pose_packet,
                state_arrival_ns=state_arrival_ns,
                pose_arrival_ns=pose_arrival_ns,
                world_from_camera=args.world_from_camera,
                world_from_robot=args.world_from_robot,
                goal_pose=args.goal_pose,
                robot_urdf=args.robot_urdf,
                object_scales=args.object_scales,
                mesh_path=args.mesh_path,
                mesh_scale=args.mesh_scale,
                camera_name=args.camera_name,
                camera_serial=args.camera_serial,
                fallback_pose_frame=args.pose_frame,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
            temporary.replace(args.output)
            print(f"Wrote read-only replay snapshot: {args.output.resolve()}", flush=True)
            print(
                f"arrival_skew_ms={snapshot['synchronization']['arrival_skew_ms']:.3f} "
                f"object_xyz={np.asarray(snapshot['object']['Wp_T_object_raw_mesh'])[:3, 3]} "
                f"palm_xyz={snapshot['robot']['policy_palm']['position_xyz']}",
                flush=True,
            )
            return 0
    finally:
        state_socket.close(0)
        pose_socket.close(0)
        context.term()
    raise SystemExit(
        f"no synchronized state/pose pair arrived within {args.timeout:.1f}s; "
        "verify the client bridge, FoundationPose++ stream, and endpoint addresses"
    )


if __name__ == "__main__":
    raise SystemExit(main())
