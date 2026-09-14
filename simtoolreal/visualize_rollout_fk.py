#!/usr/bin/env python3
"""Render every recorded rollout state as a URDF-FK diagnostic MP4.

This renderer is intentionally separate from Isaac Sim.  It is for use only
when Isaac's RTX renderer cannot launch; it visualizes the same 27-DoF states
recorded by the Isaac rollout, not any command stream or live robot state.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from kinematics import UrdfForwardKinematics
from policy_contract import JOINT_NAMES


FINGERS = (
    ("r_thumb_cmc_flex", "r_thumb_cmc_abd", "r_thumb_mcp", "r_thumb_ip"),
    ("r_index_finger_mcp_flex", "r_index_finger_mcp_abd", "r_index_finger_pip", "r_index_finger_dip"),
    ("r_middle_finger_mcp_flex", "r_middle_finger_mcp_abd", "r_middle_finger_pip", "r_middle_finger_dip"),
    ("r_ring_finger_mcp_flex", "r_ring_finger_mcp_abd", "r_ring_finger_pip", "r_ring_finger_dip"),
    ("r_pinky_mcp_flex", "r_pinky_mcp_abd", "r_pinky_pip", "r_pinky_dip"),
)
FINGER_COLORS = ((0, 190, 255), (75, 205, 75), (50, 150, 255), (70, 70, 255), (200, 90, 225))


def project(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fixed orthographic camera, chosen to make arm height and hand motion clear."""
    # Camera basis: view from world (1.8, -3.1, 2.1) toward (0, 0.3, 0.7).
    eye = np.asarray((1.8, -3.1, 2.1)); target = np.asarray((0.0, 0.3, 0.7))
    forward = target - eye; forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0)); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    view = np.column_stack((right, up, forward))
    xy = (np.asarray(points) - target) @ view[:, :2]
    scale = min(width / 2.7, height / 1.65)
    screen = np.empty_like(xy)
    screen[:, 0] = width * .5 + scale * xy[:, 0]
    screen[:, 1] = height * .54 - scale * xy[:, 1]
    return np.rint(screen).astype(np.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--robot-urdf", type=Path, default=Path(__file__).resolve().parent / "assets/fr3v2_wuji_hand2_right_slanted.urdf")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--state-key", choices=("joint_position", "target"), default="joint_position")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()
    rollout, urdf, out = args.rollout.resolve(), args.robot_urdf.resolve(), args.out.resolve()
    data = np.load(rollout / "rollout.npz")
    states = np.asarray(data[args.state_key], dtype=float)
    time_s = np.asarray(data["time_s"], dtype=float)
    if states.ndim != 2 or states.shape[1] != 27 or not np.isfinite(states).all():
        parser.error(f"{args.state_key} must be finite with shape [N,27]")
    snapshot = json.loads((rollout / "initial_snapshot.json").read_text(encoding="utf-8"))
    world_from_robot = np.asarray(snapshot["robot"]["Wp_T_robot_root"], dtype=float)
    object_xyz = np.asarray(snapshot["object"]["Wp_T_object_raw_mesh"], dtype=float)[:3, 3]
    goal_xyz = np.asarray(snapshot["goal"]["Wp_T_goal"], dtype=float)[:3, 3]
    fk = UrdfForwardKinematics(urdf)
    joint_by_name = {joint.name: joint for joint in fk.child_joint.values()}
    arm_links = [joint_by_name[f"right_fr3_joint{i}"].child for i in range(1, 8)]
    finger_links = [[joint_by_name[name].child for name in finger] for finger in FINGERS]
    fps = float(args.fps or (1.0 / np.median(np.diff(time_s))))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.width, args.height))
    if not writer.isOpened(): raise RuntimeError(f"could not open video writer: {out}")
    try:
        for i, q in enumerate(states):
            def positions(links):
                return np.asarray([(world_from_robot @ fk.link_pose(link, q))[:3, 3] for link in links])
            arm = np.vstack((world_from_robot[:3, 3], positions(arm_links)))
            fingers = [np.vstack((arm[-1], positions(links))) for links in finger_links]
            canvas = np.full((args.height, args.width, 3), 245, dtype=np.uint8)
            all_points = np.vstack((arm, *fingers, object_xyz, goal_xyz))
            screen = project(all_points, args.width, args.height)
            arm_screen = screen[:len(arm)]; cursor = len(arm)
            cv2.polylines(canvas, [arm_screen], False, (35, 35, 35), 10, cv2.LINE_AA)
            cv2.polylines(canvas, [arm_screen], False, (90, 90, 90), 5, cv2.LINE_AA)
            for p in arm_screen: cv2.circle(canvas, tuple(p), 8, (25, 25, 25), -1, cv2.LINE_AA)
            for finger, color in zip(fingers, FINGER_COLORS):
                pts = screen[cursor:cursor + len(finger)]; cursor += len(finger)
                cv2.polylines(canvas, [pts], False, color, 5, cv2.LINE_AA)
                for p in pts[1:]: cv2.circle(canvas, tuple(p), 5, color, -1, cv2.LINE_AA)
            object_p, goal_p = screen[-2], screen[-1]
            cv2.circle(canvas, tuple(object_p), 13, (45, 100, 180), -1, cv2.LINE_AA)
            cv2.drawMarker(canvas, tuple(goal_p), (35, 155, 35), cv2.MARKER_CROSS, 26, 3, cv2.LINE_AA)
            label = f"URDF FK fallback | Isaac recorded {args.state_key} | frame {i + 1}/{len(states)} | t={time_s[i]:.3f}s"
            cv2.rectangle(canvas, (0, 0), (args.width, 42), (28, 28, 28), -1)
            cv2.putText(canvas, label, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, .62, (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(canvas, "blue: object  green cross: goal", (18, args.height - 18), cv2.FONT_HERSHEY_SIMPLEX, .52, (60, 60, 60), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    report = {"format": "simtoolreal_rollout_fk_video_v1", "rollout": str(rollout), "state_key": args.state_key,
              "frames": int(len(states)), "fps": fps, "video": str(out),
              "note": "URDF FK visualization of states recorded in Isaac; not an Isaac RTX render."}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
