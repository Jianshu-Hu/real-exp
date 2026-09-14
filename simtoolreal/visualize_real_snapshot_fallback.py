#!/usr/bin/env python3
"""Visualize a real SimToolReal snapshot without Isaac Sim or RTX.

The plot uses exactly the snapshot's policy-world (``Wp``) coordinates,
canonical 27-joint vector, deployment FK URDF, and ``Wp_T_object_raw_mesh``.
It is intended to answer one question: do the real robot and object occupy
the same coordinate frame as the simulated task?  It does not simulate
contacts, dynamics, materials, or the Isaac Sim camera.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ROOT / "simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf"


def _rpy(values: np.ndarray) -> np.ndarray:
    r, p, y = values
    sr, cr, sp, cp, sy, cy = np.sin(r), np.cos(r), np.sin(p), np.cos(p), np.sin(y), np.cos(y)
    return np.array(((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
                     (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
                     (-sp, cp * sr, cp * cr)), dtype=float)


def _transform(xyz: str = "0 0 0", rpy: str = "0 0 0") -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = _rpy(np.fromstring(rpy, sep=" "))
    out[:3, 3] = np.fromstring(xyz, sep=" ")
    return out


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, oc = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    out = np.eye(4)
    out[:3, :3] = ((c + x*x*oc, x*y*oc - z*s, x*z*oc + y*s),
                   (y*x*oc + z*s, c + y*y*oc, y*z*oc - x*s),
                   (z*x*oc - y*s, z*y*oc + x*s, c + z*z*oc))
    return out


class UrdfSkeleton:
    def __init__(self, path: Path) -> None:
        root = ET.parse(path).getroot()
        self.links = {str(node.attrib["name"]) for node in root.findall("link")}
        self.children: dict[str, list[dict]] = {name: [] for name in self.links}
        child_names: set[str] = set()
        for node in root.findall("joint"):
            parent, child = node.find("parent"), node.find("child")
            if parent is None or child is None:
                continue
            origin = node.find("origin")
            axis = node.find("axis")
            joint = {
                "name": str(node.attrib.get("name", "")),
                "child": str(child.attrib["link"]),
                "kind": str(node.attrib.get("type", "fixed")),
                "origin": _transform(
                    origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0",
                    origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0",
                ),
                "axis": np.fromstring(axis.attrib.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" "),
            }
            self.children[str(parent.attrib["link"])].append(joint)
            child_names.add(str(child.attrib["link"]))
        roots = self.links - child_names
        if len(roots) != 1:
            raise ValueError(f"expected one URDF root, found {sorted(roots)}")
        self.root = next(iter(roots))

    def poses(self, q: np.ndarray, joint_names: list[str], world_from_root: np.ndarray) -> dict[str, np.ndarray]:
        if len(joint_names) != len(q):
            raise ValueError(
                f"snapshot has {len(joint_names)} joint names but {len(q)} joint positions"
            )
        values = dict(zip(joint_names, q))
        result: dict[str, np.ndarray] = {}

        def visit(parent: str, parent_pose: np.ndarray) -> None:
            result[parent] = parent_pose
            for joint in self.children.get(parent, []):
                pose = parent_pose @ joint["origin"]
                if joint["kind"] in {"revolute", "continuous"}:
                    if joint["name"] not in values:
                        raise ValueError(f"URDF moving joint {joint['name']!r} missing from snapshot")
                    pose = pose @ _axis_angle(joint["axis"], float(values[joint["name"]]))
                visit(joint["child"], pose)

        visit(self.root, np.asarray(world_from_root, dtype=float))
        return result


def _read_stl(path: Path, scale: float) -> np.ndarray:
    data = path.read_bytes()
    vertices: np.ndarray
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + 50 * count
        if expected <= len(data):
            # Binary STL facets are 50 bytes: normal (12), vertices (36),
            # and a 2-byte attribute field.  A plain float32 view would drift
            # by two bytes after every facet, so use the facet record stride.
            facet_dtype = np.dtype([
                ("normal", "<f4", (3,)),
                ("vertices", "<f4", (9,)),
                ("attribute", "<u2"),
            ])
            facets = np.frombuffer(data, dtype=facet_dtype, count=count, offset=84)
            vertices = facets["vertices"].reshape(-1, 3)
            return np.asarray(vertices, dtype=float) * scale
    text = data.decode("utf-8", errors="ignore")
    values = [float(x) for x in re.findall(r"vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)", text) for x in x]
    if not values:
        raise ValueError(f"could not parse STL mesh: {path}")
    return np.asarray(values, dtype=float).reshape(-1, 3) * scale


def _checked_pose(value: object, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous row")
    return matrix


def _draw_frame(ax, pose: np.ndarray, length: float, label: str) -> None:
    origin = pose[:3, 3]
    colors = ("r", "g", "b")
    for i, color in enumerate(colors):
        end = origin + pose[:3, i] * length
        ax.plot((origin[0], end[0]), (origin[1], end[1]), (origin[2], end[2]), color=color, linewidth=2)
    ax.text(*origin, label, fontsize=8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("calibration/runs/real_pose_fallback.png"))
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)
    args.snapshot = args.snapshot.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    snapshot = __import__("json").loads(args.snapshot.read_text(encoding="utf-8"))
    if snapshot.get("format") != "simtoolreal_real_snapshot_v1":
        parser.error(f"unsupported snapshot format: {snapshot.get('format')!r}")
    urdf = (args.urdf or Path(snapshot["robot"]["urdf"]["path"])).expanduser()
    if not urdf.is_absolute():
        urdf = args.repo_root / urdf
    urdf = urdf.resolve()
    if not urdf.is_file():
        parser.error(f"robot URDF does not exist: {urdf}")
    mesh = args.mesh
    if mesh is None:
        mesh = args.repo_root / snapshot["object"]["mesh_repository_path"]
    mesh = mesh.expanduser().resolve()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(snapshot["state"]["joint_names"])
    q = np.asarray(snapshot["state"]["joint_position_27"], dtype=float)
    world_from_root = _checked_pose(snapshot["robot"]["Wp_T_robot_root"], "Wp_T_robot_root")
    world_from_object = _checked_pose(snapshot["object"]["Wp_T_object_raw_mesh"], "Wp_T_object_raw_mesh")
    skeleton = UrdfSkeleton(urdf)
    link_poses = skeleton.poses(q, names, world_from_root)

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    # Every URDF parent-child edge is drawn, so the plot includes FR3 and Wuji.
    for parent, children in skeleton.children.items():
        if parent not in link_poses:
            continue
        p = link_poses[parent][:3, 3]
        for joint in children:
            child = link_poses[joint["child"]][:3, 3]
            color = "#2367a8" if "fr3" in joint["child"] else "#9b59b6"
            ax.plot((p[0], child[0]), (p[1], child[1]), (p[2], child[2]), color=color, linewidth=2.2, alpha=0.9)
            ax.scatter(*child, color=color, s=8)

    palm = np.asarray(snapshot["robot"]["policy_palm"]["position_xyz"], dtype=float)
    fingertips = np.asarray(snapshot["robot"]["policy_fingertips_xyz"], dtype=float)
    ax.scatter(*palm, color="orange", s=45, label="policy palm")
    ax.scatter(fingertips[:, 0], fingertips[:, 1], fingertips[:, 2], color="gold", s=25, label="policy fingertips")
    ax.plot(fingertips[:, 0], fingertips[:, 1], fingertips[:, 2], color="gold", linewidth=1)
    ax.scatter(*world_from_object[:3, 3], color="black", marker="x", s=70, label="object origin")
    _draw_frame(ax, world_from_root, 0.12, "robot root")
    _draw_frame(ax, world_from_object, 0.10, "object")

    if mesh.is_file() and mesh.suffix.lower() == ".stl":
        try:
            local = _read_stl(mesh, float(snapshot["object"]["mesh_scale_m_per_source_unit"]))
            # Keep rendering responsive for very dense meshes.
            if len(local) > 12000:
                local = local[:: max(1, len(local) // 12000)]
            world = (world_from_object[:3, :3] @ local.T).T + world_from_object[:3, 3]
            ax.plot_trisurf(world[:, 0], world[:, 1], world[:, 2], color="#60656b", alpha=0.72, linewidth=0.1)
            mesh_note = f"mesh: {mesh.name}"
        except Exception as exc:
            mesh_note = f"mesh unavailable ({exc})"
    else:
        # A pose-only fallback box keeps the object frame visible if the STL is absent.
        half = np.array((0.10, 0.025, 0.025))
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float) * half
        corners = (world_from_object[:3, :3] @ corners.T).T + world_from_object[:3, 3]
        for i, j in ((0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)):
            ax.plot(corners[[i,j],0], corners[[i,j],1], corners[[i,j],2], color="black", linewidth=1.4)
        mesh_note = f"STL not found: {mesh}"

    points = np.vstack([np.asarray(list(link_poses.values()))[:, :3, 3], palm[None], fingertips, world_from_object[:3, 3][None]])
    center = points.mean(axis=0)
    radius = max(float(np.max(np.ptp(points, axis=0))) * 0.65, 0.55)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
    ax.set_xlabel("Wp X (m)"); ax.set_ylabel("Wp Y (m)"); ax.set_zlabel("Wp Z (m)")
    ax.set_title("SimToolReal real snapshot in policy world Wp")
    ax.legend(loc="upper left")
    fig.text(0.01, 0.01, mesh_note + " | blue=FR3, purple=Wuji, orange/gold=policy geometry", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved fallback visualization: {args.output}")
    print(f"Snapshot joints: {len(q)} | object origin Wp: {world_from_object[:3, 3].round(4).tolist()}")
    print(f"Policy palm Wp: {palm.round(4).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
