#!/usr/bin/env python3
"""Render only the real robot joint pose and FoundationPose object pose."""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path


TASK = "Isaacsimenvs-SimToolReal-FrankaWujiRightSlanted-Direct-v0"


def write_object_urdf(output: Path, mesh: Path, scale: float) -> None:
    mesh_name = html.escape(str(mesh.resolve()), quote=True)
    mesh_scale = f"{scale:.12g} {scale:.12g} {scale:.12g}"
    output.write_text(
        f"""<?xml version="1.0"?>
<robot name="foundationpose_object">
  <link name="object">
    <visual>
      <geometry><mesh filename="{mesh_name}" scale="{mesh_scale}"/></geometry>
      <material name="object"><color rgba="0.18 0.20 0.22 1"/></material>
    </visual>
    <collision>
      <geometry><mesh filename="{mesh_name}" scale="{mesh_scale}"/></geometry>
    </collision>
    <inertial>
      <mass value="0.20"/>
      <inertia ixx="0.00025" ixy="0" ixz="0"
               iyy="0.0012" iyz="0" izz="0.0012"/>
    </inertial>
  </link>
</robot>
""",
        encoding="utf-8",
    )


def rotation_matrix_to_quat_wxyz(rotation):
    import numpy as np

    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.asarray((
            0.25 * s,
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
        ))
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.asarray(((rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s,
                               (rotation[0, 1] + rotation[1, 0]) / s,
                               (rotation[0, 2] + rotation[2, 0]) / s))
        elif index == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.asarray(((rotation[0, 2] - rotation[2, 0]) / s,
                               (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                               (rotation[1, 2] + rotation[2, 1]) / s))
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.asarray(((rotation[1, 0] - rotation[0, 1]) / s,
                               (rotation[0, 2] + rotation[2, 0]) / s,
                               (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s))
    return quat / np.linalg.norm(quat)


def main() -> None:
    ET.XMLParser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("/tmp/simtoolreal_real_pose.png"))
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.snapshot = args.snapshot.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.headless = True
    args.enable_cameras = True

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    upstream = args.repo_root / "libs/SimToolReal-Franka-Wuji2"
    mesh = args.repo_root / snapshot["object"]["mesh_repository_path"]
    if not upstream.is_dir():
        parser.error(f"missing Isaac task source: {upstream}")
    if not mesh.is_file():
        parser.error(f"missing FoundationPose mesh: {mesh}")
    if len(snapshot["state"]["joint_position_27"]) != 27:
        parser.error("snapshot must contain 27 joint positions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    object_urdf = args.output.with_suffix(".object.urdf")
    write_object_urdf(
        object_urdf,
        mesh,
        float(snapshot["object"]["mesh_scale_m_per_source_unit"]),
    )
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    app = AppLauncher(args).app
    exit_code = 0
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import yaml
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import Camera, CameraCfg
        from isaaclab.sim.utils import get_current_stage
        from PIL import Image
        from pxr import UsdGeom

        import isaacsimenvs  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        yaml_path = gym.spec(TASK).kwargs.get("env_cfg_yaml_entry_point")
        if yaml_path:
            with open(yaml_path, encoding="utf-8") as stream:
                env_cfg.from_dict(yaml.safe_load(stream) or {})
        env_cfg.scene.num_envs = 1
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.assets.object_urdf = str(object_urdf)
        env_cfg.assets.object_scale = tuple(float(v) for v in snapshot["object_scales"])
        env_cfg.reset.reset_dof_pos_random_interval_arm = 0.0
        env_cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
        env_cfg.reset.reset_dof_vel_random_interval = 0.0
        env_cfg.sim.device = str(args.device)

        env = gym.make(TASK, cfg=env_cfg)
        try:
            env.reset()
            inner = env.unwrapped
            camera = Camera(cfg=CameraCfg(
                prim_path="/World/RealPoseCamera",
                update_period=0,
                height=args.height,
                width=args.width,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0,
                    focus_distance=400.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.1, 100.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 10.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    convention="opengl",
                ),
            ))
            inner.sim.reset()
            env.reset()
            origin = inner.scene.env_origins[0]

            q_canonical = torch.tensor(
                snapshot["state"]["joint_position_27"],
                device=inner.device,
                dtype=torch.float32,
            )
            q_lab = q_canonical[inner._perm_canon_to_lab].unsqueeze(0)
            qd_lab = torch.zeros_like(q_lab)
            world_from_object = np.asarray(
                snapshot["object"]["Wp_T_object_raw_mesh"], dtype=np.float64
            )
            object_position = origin + torch.tensor(
                world_from_object[:3, 3], device=inner.device, dtype=torch.float32
            )
            object_quaternion = torch.tensor(
                rotation_matrix_to_quat_wxyz(world_from_object[:3, :3]),
                device=inner.device,
                dtype=torch.float32,
            )
            object_pose = torch.cat((object_position, object_quaternion)).unsqueeze(0)
            zero_velocity = torch.zeros((1, 6), device=inner.device)

            goal_prim = get_current_stage().GetPrimAtPath("/World/envs/env_0/GoalViz")
            if goal_prim.IsValid():
                UsdGeom.Imageable(goal_prim).MakeInvisible()

            camera.set_world_poses_from_view(
                (origin + torch.tensor((0.8, -3.0, 2.0), device=inner.device)).unsqueeze(0),
                (origin + torch.tensor((-0.1, 0.1, 0.8), device=inner.device)).unsqueeze(0),
            )

            def restore_pose() -> None:
                inner.robot.write_joint_state_to_sim(q_lab, qd_lab)
                inner.robot.set_joint_position_target(q_lab)
                inner.object.write_root_pose_to_sim(object_pose)
                inner.object.write_root_velocity_to_sim(zero_velocity)
                inner._prev_targets[:] = q_lab
                inner._cur_targets[:] = q_lab
                inner.scene.write_data_to_sim()

            restore_pose()
            frame = None
            best_mean = -1.0
            dt = float(inner.sim.get_physics_dt())
            for _ in range(30):
                inner.sim.step(render=True)
                restore_pose()
                inner.scene.update(dt=dt)
                camera.update(dt)
                rgb = camera.data.output.get("rgb")
                if rgb is None or rgb.shape[0] == 0:
                    continue
                candidate = rgb[0].detach().cpu().numpy()[..., :3]
                mean = float(candidate.mean())
                if mean > best_mean:
                    best_mean = mean
                    frame = candidate.copy()

            if frame is None or best_mean <= 0.5:
                raise RuntimeError(f"Isaac Sim did not return a usable frame: mean={best_mean}")
            Image.fromarray(frame.astype(np.uint8)).save(args.output)
            print(f"Rendered robot + object pose: {args.output}", flush=True)
            print(f"RGB mean: {best_mean:.3f}", flush=True)
        finally:
            env.close()
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        del app
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
