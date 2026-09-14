#!/usr/bin/env python3
"""Render every frame of an Isaac SimToolReal rollout to an MP4 video.

This is deliberately read-only with respect to real hardware.  It loads the
recorded rollout state and writes those joint poses into Isaac Sim before each
render, so the video shows the trajectory that was actually simulated.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import traceback
from pathlib import Path

TASK = "Isaacsimenvs-SimToolReal-FrankaWujiRightSlanted-Direct-v0"


def matrix_to_quat_wxyz(rotation):
    import numpy as np
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s,
             (rotation[0, 2] - rotation[2, 0]) / s,
             (rotation[1, 0] - rotation[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(rotation)))
        if i == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            q = ((rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s,
                 (rotation[0, 1] + rotation[1, 0]) / s,
                 (rotation[0, 2] + rotation[2, 0]) / s)
        elif i == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            q = ((rotation[0, 2] - rotation[2, 0]) / s,
                 (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                 (rotation[1, 2] + rotation[2, 1]) / s)
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            q = ((rotation[1, 0] - rotation[0, 1]) / s,
                 (rotation[0, 2] + rotation[2, 0]) / s,
                 (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s)
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def write_object_urdf(path: Path, mesh: Path, scale: float) -> None:
    s = f"{scale:.12g} {scale:.12g} {scale:.12g}"
    path.write_text(f'''<?xml version="1.0"?>
<robot name="rollout_object"><link name="object">
<visual><geometry><mesh filename="{html.escape(str(mesh.resolve()), quote=True)}" scale="{s}"/></geometry></visual>
<collision><geometry><mesh filename="{html.escape(str(mesh.resolve()), quote=True)}" scale="{s}"/></geometry></collision>
<inertial><mass value="0.20"/><inertia ixx="0.00025" ixy="0" ixz="0" iyy="0.0012" iyz="0" izz="0.0012"/></inertial>
</link></robot>\n''', encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True, help="output .mp4 path")
    parser.add_argument("--state-key", choices=("joint_position", "target"), default="joint_position")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=None)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.rollout = args.rollout.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    args.headless = True
    args.enable_cameras = True
    data = __import__("numpy").load(args.rollout / "rollout.npz")
    snapshot = json.loads((args.rollout / "initial_snapshot.json").read_text(encoding="utf-8"))
    q_frames = __import__("numpy").asarray(data[args.state_key], dtype=float)
    if q_frames.ndim != 2 or q_frames.shape[1] != 27 or not __import__("numpy").isfinite(q_frames).all():
        parser.error(f"{args.state_key} must be finite with shape [N,27]")
    upstream = args.repo_root / "libs/SimToolReal-Franka-Wuji2"
    mesh = args.repo_root / snapshot["object"]["mesh_repository_path"]
    robot_urdf = args.repo_root / "simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf"
    for path, label in ((upstream, "upstream"), (mesh, "mesh"), (robot_urdf, "robot URDF")):
        if not path.exists(): parser.error(f"missing {label}: {path}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    wrapper = args.out.with_suffix(".object.urdf")
    write_object_urdf(wrapper, mesh, float(snapshot["object"]["mesh_scale_m_per_source_unit"]))
    sys.path.insert(0, str(upstream)); os.chdir(upstream)
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import yaml
        import isaacsimenvs  # noqa: F401
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        from PIL import Image, ImageDraw
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        yp = gym.spec(TASK).kwargs.get("env_cfg_yaml_entry_point")
        if yp:
            with open(yp, encoding="utf-8") as f: cfg.from_dict(yaml.safe_load(f) or {})
        cfg.scene.num_envs = 1; cfg.assets.num_assets_per_type = 1
        cfg.assets.object_urdf = str(wrapper); cfg.assets.object_scale = tuple(snapshot["object_scales"])
        cfg.reset.reset_dof_pos_random_interval_arm = 0.; cfg.reset.reset_dof_pos_random_interval_fingers = 0.; cfg.reset.reset_dof_vel_random_interval = 0.
        cfg.sim.device = str(args.device)
        env = gym.make(TASK, cfg=cfg); inner = env.unwrapped; env.reset()
        camera = Camera(cfg=CameraCfg(prim_path="/World/RolloutCamera", update_period=0,
            height=args.height, width=args.width, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 100.0)),
            offset=CameraCfg.OffsetCfg(pos=(0, 0, 10), rot=(1, 0, 0, 0), convention="opengl")))
        inner.sim.reset(); env.reset(); origin = inner.scene.env_origins[0]
        q0 = torch.tensor(snapshot["state"]["joint_position_27"], device=inner.device, dtype=torch.float32)
        zero_qd = torch.zeros((1, 27), device=inner.device)
        def pose(m):
            m = np.asarray(m, dtype=float)
            return torch.cat((origin + torch.tensor(m[:3, 3], device=inner.device, dtype=torch.float32),
                              torch.tensor(matrix_to_quat_wxyz(m[:3, :3]), device=inner.device, dtype=torch.float32))).unsqueeze(0)
        op, gp = pose(snapshot["object"]["Wp_T_object_raw_mesh"]), pose(snapshot["goal"]["Wp_T_goal"])
        camera.set_world_poses_from_view((origin + torch.tensor((0.8, -3.0, 2.0), device=inner.device)).unsqueeze(0),
                                         (origin + torch.tensor((-0.1, 0.1, 0.8), device=inner.device)).unsqueeze(0))
        def put(q):
            qlab = q[inner._perm_canon_to_lab].unsqueeze(0)
            inner.robot.write_joint_state_to_sim(qlab, zero_qd); inner.robot.set_joint_position_target(qlab)
            inner.object.write_root_pose_to_sim(op); inner.object.write_root_velocity_to_sim(torch.zeros((1, 6), device=inner.device))
            inner.goal_viz.write_root_pose_to_sim(gp); inner.goal_viz.write_root_velocity_to_sim(torch.zeros((1, 6), device=inner.device))
            inner._prev_targets[:] = qlab; inner._cur_targets[:] = qlab; inner.scene.write_data_to_sim()
        put(q0)
        try:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(args.out), fps=float(args.fps or data["time_s"][1] and 1.0 / np.median(np.diff(data["time_s"])) or 60.0), codec="libx264", quality=8)
        except Exception as exc:
            raise RuntimeError("imageio/ffmpeg is required in the Isaac Python environment") from exc
        dt = float(inner.sim.get_physics_dt()); written = 0
        try:
            for i, q_np in enumerate(q_frames):
                put(torch.tensor(q_np, device=inner.device, dtype=torch.float32))
                inner.sim.step(render=True); inner.scene.update(dt=dt); camera.update(dt)
                rgb = camera.data.output.get("rgb")
                if rgb is None or rgb.shape[0] == 0: raise RuntimeError(f"no RGB frame at index {i}")
                frame = rgb[0].detach().cpu().numpy()[..., :3].astype(np.uint8)
                # Small diagnostic overlay makes timing and the displayed stream unambiguous.
                image = Image.fromarray(frame); draw = ImageDraw.Draw(image)
                draw.rectangle((8, 8, 300, 50), fill=(0, 0, 0)); draw.text((16, 16), f"frame {i+1}/{len(q_frames)}  t={float(data['time_s'][i]):.3f}s", fill=(255, 255, 255))
                writer.append_data(np.asarray(image)); written += 1
        finally:
            writer.close()
        report = {"format": "simtoolreal_rollout_video_v1", "rollout": str(args.rollout), "state_key": args.state_key,
                  "frames": written, "requested_frames": int(len(q_frames)), "fps": float(args.fps or 60.0), "video": str(args.out)}
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True); env.close(); return 0
    except BaseException:
        traceback.print_exc(); return 1
    finally:
        del app


if __name__ == "__main__": raise SystemExit(main())
