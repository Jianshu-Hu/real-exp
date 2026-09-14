#!/usr/bin/env python3
"""Debug script to print Isaac Sim's robot base pose and first observation."""
import argparse
import json
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, default=Path(__file__).resolve().parents[1]/"libs/SimToolReal-Franka-Wuji2")
    p.add_argument("--robot-urdf", type=Path, required=True)

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p)
    a = p.parse_args()
    a.headless = True

    for name in ("snapshot", "config", "checkpoint", "upstream_root", "robot_urdf"):
        setattr(a, name, getattr(a, name).expanduser().resolve())

    snap = json.loads(a.snapshot.read_text())

    sys.path.insert(0, str(a.upstream_root))
    import os
    os.chdir(a.upstream_root)

    app = AppLauncher(a).app

    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import yaml
        import isaacsimenvs
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        TASK = "Isaacsimenvs-SimToolReal-FrankaWujiRightSlanted-Direct-v0"
        cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        yp = gym.spec(TASK).kwargs.get("env_cfg_yaml_entry_point")
        if yp:
            with open(yp, encoding="utf-8") as f:
                cfg.from_dict(yaml.safe_load(f) or {})

        cfg.scene.num_envs = 1
        cfg.sim.device = str(a.device)

        env = gym.make(TASK, cfg=cfg)
        inner = env.unwrapped
        env.reset()

        print("\n=== SNAPSHOT ROBOT BASE ===")
        robot_base_snap = np.array(snap["robot"]["Wp_T_robot_root"])
        print("Translation:", robot_base_snap[:3, 3])
        print("Rotation:\n", robot_base_snap[:3, :3])

        print("\n=== ISAAC ROBOT BASE ===")
        # Get the robot's root pose in Isaac's world frame
        robot_root_pos = inner.robot.data.root_pos_w[0].detach().cpu().numpy()
        robot_root_quat = inner.robot.data.root_quat_w[0].detach().cpu().numpy()  # wxyz
        print("Position:", robot_root_pos)
        print("Quaternion (wxyz):", robot_root_quat)

        print("\n=== ISAAC SCENE INFO ===")
        print("Environment origin:", inner.scene.env_origins[0].detach().cpu().numpy())

        print("\n=== SNAPSHOT OBJECT/GOAL (in snapshot's robot frame) ===")
        obj_pose = np.array(snap["object"]["Wp_T_object_raw_mesh"])
        goal_pose = np.array(snap["goal"]["Wp_T_goal"])
        print("Object position:", obj_pose[:3, 3])
        print("Goal position:", goal_pose[:3, 3])

        # Now write snapshot joint positions and poses
        qcan = torch.tensor(snap["state"]["joint_position_27"], device=inner.device, dtype=torch.float32)
        qlab = qcan[inner._perm_canon_to_lab].unsqueeze(0)
        zq = torch.zeros_like(qlab)
        origin = inner.scene.env_origins[0]

        def matrix_to_quat_wxyz(R):
            tr = float(np.trace(R))
            if tr > 0:
                s = (tr + 1.0) ** 0.5 * 2
                return np.array([0.25 * s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
            else:
                i = int(np.argmax(np.diag(R)))
                if i == 0:
                    s = (1 + R[0,0] - R[1,1] - R[2,2]) ** 0.5 * 2
                    return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
                elif i == 1:
                    s = (1 + R[1,1] - R[0,0] - R[2,2]) ** 0.5 * 2
                    return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
                else:
                    s = (1 + R[2,2] - R[0,0] - R[1,1]) ** 0.5 * 2
                    return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])

        def root_pose(m):
            m = np.asarray(m, float)
            pos = origin + torch.tensor(m[:3, 3], device=inner.device, dtype=torch.float32)
            quat = torch.tensor(matrix_to_quat_wxyz(m[:3, :3]), device=inner.device, dtype=torch.float32)
            return torch.cat((pos, quat)).unsqueeze(0)

        op = root_pose(snap["object"]["Wp_T_object_raw_mesh"])
        gp = root_pose(snap["goal"]["Wp_T_goal"])

        # Restore snapshot
        inner.robot.write_joint_state_to_sim(qlab, zq)
        inner.robot.set_joint_position_target(qlab)
        inner.object.write_root_pose_to_sim(op)
        inner.goal_viz.write_root_pose_to_sim(gp)
        inner._prev_targets[:] = qlab
        inner._cur_targets[:] = qlab
        inner.scene.write_data_to_sim()

        physics_dt = float(inner.sim.get_physics_dt())
        inner.sim.step(render=False)
        inner.scene.update(dt=physics_dt)

        print("\n=== AFTER RESTORE: ISAAC OBJECT/GOAL POSITIONS ===")
        obj_pos_isaac = (inner.object.data.root_pos_w[0] - origin).detach().cpu().numpy()
        goal_pos_isaac = (inner.goal_viz.data.root_pos_w[0] - origin).detach().cpu().numpy()
        print("Object position in Isaac:", obj_pos_isaac)
        print("Goal position in Isaac:", goal_pos_isaac)

        # Get first observation
        obs = inner._get_observations()
        po = obs["policy"][0].detach().cpu().numpy()

        print("\n=== FIRST POLICY OBSERVATION (134-dim) ===")
        print("Full observation:", po)
        print("\nObservation breakdown:")
        print("  joint_pos (27):", po[:27])
        print("  joint_vel (27):", po[27:54])
        print("  target (27):", po[54:81])
        print("  palm_pos (3):", po[81:84])
        print("  object_rot_6d (6):", po[84:90])
        print("  object_pos (3):", po[90:93])
        print("  fingertip_state (20):", po[93:113])
        print("  relative_goal_pos (3):", po[113:116])
        print("  relative_goal_rot_6d (6):", po[116:122])
        print("  actions_prev (12):", po[122:134])

        env.close()
        return 0
    finally:
        del app

if __name__ == "__main__":
    raise SystemExit(main())
