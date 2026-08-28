"""Versioned transport helpers for SimToolReal state and target streams."""

from __future__ import annotations

import math
import time
from typing import Any


PROTOCOL_VERSION = 1


def now_ns() -> int:
    return time.time_ns()


def _finite_numbers(values: Any, *, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite numbers")
    return result


def make_joint_state(
    joints: list[float],
    names: list[str] | None = None,
    *,
    velocities: list[float] | None = None,
    source: str = "robot",
    timestamp_ns: int | None = None,
) -> dict[str, Any]:
    values = _finite_numbers(joints, name="joints")
    if not values:
        raise ValueError("joints must not be empty")
    if names is not None and len(names) != len(values):
        raise ValueError("names and joints must have the same length")
    velocity_values = (
        [0.0] * len(values)
        if velocities is None
        else _finite_numbers(velocities, name="velocities")
    )
    if len(velocity_values) != len(values):
        raise ValueError("velocities and joints must have the same length")
    return {
        "protocol": PROTOCOL_VERSION,
        "kind": "joint_state",
        "timestamp_ns": int(timestamp_ns if timestamp_ns is not None else now_ns()),
        "source": str(source),
        "names": list(names) if names is not None else [],
        "joints": values,
        "velocities": velocity_values,
    }


def make_object_pose(
    pose: Any,
    *,
    object_id: str = "object",
    frame_id: str = "camera",
    source: str = "foundationpose++",
    timestamp_ns: int | None = None,
) -> dict[str, Any]:
    # Accept either a 4x4 nested matrix or a flat row-major 16-value matrix.
    if isinstance(pose, (list, tuple)) and len(pose) == 4 and all(
        isinstance(row, (list, tuple)) for row in pose
    ):
        flat = [value for row in pose for value in row]
    else:
        flat = pose
    values = _finite_numbers(flat, name="pose")
    if len(values) != 16:
        raise ValueError("pose must contain exactly 16 values (a 4x4 transform)")
    return {
        "protocol": PROTOCOL_VERSION,
        "kind": "object_pose",
        "timestamp_ns": int(timestamp_ns if timestamp_ns is not None else now_ns()),
        "source": str(source),
        "object_id": str(object_id),
        "frame_id": str(frame_id),
        "pose": values,
    }


def make_joint_target(
    target: list[float],
    *,
    names: list[str] | None = None,
    source: str = "policy-server",
    timestamp_ns: int | None = None,
) -> dict[str, Any]:
    values = _finite_numbers(target, name="target")
    if len(values) != 27:
        raise ValueError("joint target must contain exactly 27 values")
    if names is not None and len(names) != len(values):
        raise ValueError("target names and values must have the same length")
    return {
        "protocol": PROTOCOL_VERSION,
        "kind": "joint_target",
        "timestamp_ns": int(timestamp_ns if timestamp_ns is not None else now_ns()),
        "source": str(source),
        "names": list(names) if names is not None else [],
        "target": values,
    }


def make_policy_observation(
    observation: list[float], *, source: str = "policy-executor", timestamp_ns: int | None = None
) -> dict[str, Any]:
    values = _finite_numbers(observation, name="observation")
    if len(values) != 134:
        raise ValueError("SimToolReal observation must contain exactly 134 values")
    return {
        "protocol": PROTOCOL_VERSION,
        "kind": "policy_observation",
        "timestamp_ns": int(timestamp_ns if timestamp_ns is not None else now_ns()),
        "source": str(source),
        "observation": values,
    }


def make_policy_action(
    action: list[float], *, source: str = "policy-server", timestamp_ns: int | None = None
) -> dict[str, Any]:
    values = _finite_numbers(action, name="action")
    if len(values) != 27:
        raise ValueError("SimToolReal action must contain exactly 27 values")
    return {
        "protocol": PROTOCOL_VERSION,
        "kind": "policy_action",
        "timestamp_ns": int(timestamp_ns if timestamp_ns is not None else now_ns()),
        "source": str(source),
        "action": values,
    }


def make_error(message: str, *, source: str = "simtoolreal", timestamp_ns: int | None = None) -> dict[str, Any]:
    if not str(message).strip():
        raise ValueError("error message must not be empty")
    return {
        "protocol": PROTOCOL_VERSION,
        "kind": "error",
        "timestamp_ns": int(timestamp_ns if timestamp_ns is not None else now_ns()),
        "source": str(source),
        "error": str(message),
    }


def validate_packet(packet: Any) -> dict[str, Any]:
    if not isinstance(packet, dict):
        raise ValueError("packet must be a JSON object")
    if packet.get("protocol") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {packet.get('protocol')!r}")
    kind = packet.get("kind")
    if kind == "joint_state":
        return make_joint_state(
            packet.get("joints"),
            packet.get("names") or None,
            velocities=packet.get("velocities"),
            source=str(packet.get("source", "unknown")),
            timestamp_ns=int(packet.get("timestamp_ns", now_ns())),
        )
    if kind == "joint_target":
        return make_joint_target(
            packet.get("target"),
            names=packet.get("names") or None,
            source=str(packet.get("source", "unknown")),
            timestamp_ns=int(packet.get("timestamp_ns", now_ns())),
        )
    if kind == "object_pose":
        return make_object_pose(
            packet.get("pose"),
            object_id=str(packet.get("object_id", "object")),
            frame_id=str(packet.get("frame_id", "camera")),
            source=str(packet.get("source", "unknown")),
            timestamp_ns=int(packet.get("timestamp_ns", now_ns())),
        )
    if kind == "policy_observation":
        return make_policy_observation(
            packet.get("observation"),
            source=str(packet.get("source", "unknown")),
            timestamp_ns=int(packet.get("timestamp_ns", now_ns())),
        )
    if kind == "policy_action":
        return make_policy_action(
            packet.get("action"),
            source=str(packet.get("source", "unknown")),
            timestamp_ns=int(packet.get("timestamp_ns", now_ns())),
        )
    if kind == "error":
        message = packet.get("error")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("error packet must contain a non-empty error string")
        return make_error(
            message,
            source=str(packet.get("source", "unknown")),
            timestamp_ns=int(packet.get("timestamp_ns", now_ns())),
        )
    raise ValueError(f"unsupported packet kind: {kind!r}")
