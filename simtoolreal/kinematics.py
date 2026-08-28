"""Dependency-light FK for the FR3 + right Wuji Hand 2 policy geometry."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from policy_contract import (
    FINGERTIP_LINK_NAMES,
    FINGERTIP_OFFSETS,
    JOINT_NAMES,
    PALM_CENTER_OFFSET,
    PALM_FRAME_QUAT_WXYZ,
)


def rpy_matrix(values: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = np.asarray(tuple(values), dtype=np.float64)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def transform(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rpy_matrix(rpy)
    result[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return result


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    result = np.eye(4)
    result[:3, :3] = (
        (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
        (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
        (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
    )
    return result


def quat_wxyz_to_matrix(quaternion: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(quaternion), dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a normalized xyzw quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x, y, z = 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s
        elif index == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x, y, z = (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x, y, z = (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s
    q = np.asarray((x, y, z, w))
    q /= np.linalg.norm(q)
    return q


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray


class UrdfForwardKinematics:
    """Compute link poses from the exact combined policy-training URDF."""

    def __init__(self, urdf_path: Path) -> None:
        root = ET.parse(urdf_path).getroot()
        self.child_joint: dict[str, Joint] = {}
        children: set[str] = set()
        parents: set[str] = set()
        for node in root.findall("joint"):
            parent_node, child_node = node.find("parent"), node.find("child")
            if parent_node is None or child_node is None:
                continue
            origin_node, axis_node = node.find("origin"), node.find("axis")
            xyz = (origin_node.attrib.get("xyz", "0 0 0") if origin_node is not None else "0 0 0")
            rpy = (origin_node.attrib.get("rpy", "0 0 0") if origin_node is not None else "0 0 0")
            axis = (axis_node.attrib.get("xyz", "1 0 0") if axis_node is not None else "1 0 0")
            joint = Joint(
                name=str(node.attrib.get("name", "")),
                parent=str(parent_node.attrib["link"]),
                child=str(child_node.attrib["link"]),
                kind=str(node.attrib.get("type", "fixed")),
                origin=transform(np.fromstring(xyz, sep=" "), np.fromstring(rpy, sep=" ")),
                axis=np.fromstring(axis, sep=" "),
            )
            self.child_joint[joint.child] = joint
            children.add(joint.child)
            parents.add(joint.parent)
        roots = parents - children
        if len(roots) != 1:
            raise ValueError(f"expected one policy-robot URDF root link, found {sorted(roots)}")
        self.root_link = next(iter(roots))
        for name in FINGERTIP_LINK_NAMES:
            self._chain(name)

    def _chain(self, link_name: str) -> list[Joint]:
        chain: list[Joint] = []
        current = link_name
        while current != self.root_link:
            try:
                joint = self.child_joint[current]
            except KeyError as exc:
                raise ValueError(f"policy URDF has no chain from {self.root_link} to {link_name}") from exc
            chain.append(joint)
            current = joint.parent
        return list(reversed(chain))

    def link_pose(self, link_name: str, q: np.ndarray) -> np.ndarray:
        positions = dict(zip(JOINT_NAMES, np.asarray(q), strict=True))
        result = np.eye(4)
        for joint in self._chain(link_name):
            result = result @ joint.origin
            if joint.kind in {"revolute", "continuous"}:
                if joint.name not in positions:
                    raise ValueError(f"moving joint {joint.name!r} is outside policy contract")
                result = result @ axis_rotation(joint.axis, float(positions[joint.name]))
        return result


class PolicyKinematics:
    """Return policy palm and fingertip geometry in the policy world frame."""

    def __init__(self, robot_urdf: Path) -> None:
        self.robot_fk = UrdfForwardKinematics(robot_urdf)
        self.robot_fk._chain("right_fr3_link7")

    def evaluate(self, q: np.ndarray, world_from_robot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (27,):
            raise ValueError(f"policy FK requires 27 joints, got {q.shape}")
        robot_from_link7 = self.robot_fk.link_pose("right_fr3_link7", q)
        world_from_link7 = world_from_robot @ robot_from_link7
        palm_pos = world_from_link7[:3, 3] + world_from_link7[:3, :3] @ PALM_CENTER_OFFSET
        palm_rotation = world_from_link7[:3, :3] @ quat_wxyz_to_matrix(PALM_FRAME_QUAT_WXYZ)
        fingertip_positions = []
        for link_name, offset in zip(FINGERTIP_LINK_NAMES, FINGERTIP_OFFSETS, strict=True):
            robot_from_tip = self.robot_fk.link_pose(link_name, q)
            world_from_tip = world_from_robot @ robot_from_tip
            fingertip_positions.append(world_from_tip[:3, 3] + world_from_tip[:3, :3] @ offset)
        return palm_pos, matrix_to_quat_xyzw(palm_rotation), np.asarray(fingertip_positions)
