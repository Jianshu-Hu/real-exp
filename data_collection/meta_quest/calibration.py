"""Pose calibration primitives for the future Quest teleoperation mapper."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


def _vector3(values: Sequence[float]) -> Vector3:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError("position must contain three finite values")
    return result  # type: ignore[return-value]


def _quaternion(values: Sequence[float]) -> Quaternion:
    result = tuple(float(value) for value in values)
    if len(result) != 4 or not all(math.isfinite(value) for value in result):
        raise ValueError("quaternion must contain four finite values")
    norm = math.sqrt(sum(value * value for value in result))
    if norm < 1e-8:
        raise ValueError("quaternion has zero length")
    return tuple(value / norm for value in result)  # type: ignore[return-value]


def quaternion_multiply(first: Quaternion, second: Quaternion) -> Quaternion:
    """Compose two xyzw quaternions."""

    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return _quaternion(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def quaternion_inverse(quaternion: Quaternion) -> Quaternion:
    qx, qy, qz, qw = _quaternion(quaternion)
    return (-qx, -qy, -qz, qw)


def rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    """Rotate a vector by an xyzw quaternion."""

    qx, qy, qz, qw = _quaternion(quaternion)
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


@dataclass(frozen=True, slots=True)
class RigidPose:
    """Position and orientation in one declared, common coordinate frame."""

    position: Vector3
    quaternion_xyzw: Quaternion

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _vector3(self.position))
        object.__setattr__(self, "quaternion_xyzw", _quaternion(self.quaternion_xyzw))


def compose(first: RigidPose, second: RigidPose) -> RigidPose:
    """Return ``first * second`` for poses in the same coordinate convention."""

    rotated_second = rotate_vector(first.quaternion_xyzw, second.position)
    return RigidPose(
        tuple(first.position[index] + rotated_second[index] for index in range(3)),
        quaternion_multiply(first.quaternion_xyzw, second.quaternion_xyzw),
    )


def inverse(pose: RigidPose) -> RigidPose:
    inverse_rotation = quaternion_inverse(pose.quaternion_xyzw)
    inverse_position = rotate_vector(
        inverse_rotation,
        tuple(-value for value in pose.position),
    )
    return RigidPose(inverse_position, inverse_rotation)


@dataclass(frozen=True, slots=True)
class PoseCalibration:
    """Map a live source pose using a captured source and predefined target pose."""

    source_anchor: RigidPose
    target_anchor: RigidPose

    @classmethod
    def from_anchor(cls, source_anchor: RigidPose, target_anchor: RigidPose) -> "PoseCalibration":
        return cls(source_anchor=source_anchor, target_anchor=target_anchor)

    def apply(self, source_pose: RigidPose) -> RigidPose:
        """Map source motion relative to the calibration anchor to the target frame."""

        relative_motion = compose(inverse(self.source_anchor), source_pose)
        return compose(self.target_anchor, relative_motion)


class CalibrationButtonEdge:
    """Detect one calibration action from a held controller button."""

    def __init__(self) -> None:
        self._previous = False

    def update(self, pressed: bool) -> bool:
        pressed = bool(pressed)
        rising_edge = pressed and not self._previous
        self._previous = pressed
        return rising_edge


class PoseCalibrator:
    """Capture one tracked pose when the selected button is pressed."""

    def __init__(self, target_pose: RigidPose) -> None:
        self.target_pose = target_pose
        self.button_edge = CalibrationButtonEdge()
        self.calibration: PoseCalibration | None = None

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def update(self, source_pose: RigidPose, button_pressed: bool) -> RigidPose | None:
        if self.button_edge.update(button_pressed):
            self.calibration = PoseCalibration.from_anchor(source_pose, self.target_pose)
        if self.calibration is None:
            return None
        return self.calibration.apply(source_pose)

    def reset(self) -> None:
        self.calibration = None
