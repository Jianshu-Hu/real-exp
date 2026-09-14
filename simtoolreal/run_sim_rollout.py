#!/usr/bin/env python3
"""Run the SimToolReal checkpoint in Isaac Sim from a captured snapshot."""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time, html
from pathlib import Path

TASK = "Isaacsimenvs-SimToolReal-FrankaWujiRightSlanted-Direct-v0"
ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write_mesh_urdf(path: Path, mesh: Path, scale: float) -> None:
    s = f"{scale:.12g} {scale:.12g} {scale:.12g}"
    path.write_text(f'''<?xml version="1.0"?>
<robot name="snapshot_object"><link name="object"><inertial><mass value="0.20"/><inertia ixx="0.00025" ixy="0" ixz="0" iyy="0.0012" iyz="0" izz="0.0012"/></inertial><visual><geometry><mesh filename="{html.escape(str(mesh.resolve()), quote=True)}" scale="{s}"/></geometry></visual><collision><geometry><mesh filename="{html.escape(str(mesh.resolve()), quote=True)}" scale="{s}"/></geometry></collision></link></robot>\n''')

def read_initial_joints(path: Path):
    """Read a finite 27-value canonical joint vector from JSON or text."""
    import numpy as np
    if not path.is_file():
        raise ValueError(f"initial joints file not found: {path}")
    try:
        value = json.loads(path.read_text()) if path.suffix.lower() == ".json" else np.loadtxt(path, delimiter=",")
        if isinstance(value, dict):
            value = value.get("joint_position_27", value.get("state", {}).get("joint_position_27"))
        joints = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise ValueError(f"could not parse initial joints file {path}: {exc}") from exc
    if joints.shape != (27,) or not np.all(np.isfinite(joints)):
        raise ValueError(f"initial joints must be a finite 27-vector in canonical order, got {joints.shape}")
    return joints

def matrix_quat_wxyz(rotation):
    import numpy as np
    tr = float(np.trace(rotation))
    if tr > 0:
        s = (tr + 1.0) ** .5 * 2
        q = ((tr + 1.0) ** .5 * .5, (rotation[2,1]-rotation[1,2])/s, (rotation[0,2]-rotation[2,0])/s, (rotation[1,0]-rotation[0,1])/s)
    else:
        i = int(rotation.diagonal().argmax())
        if i == 0:
            s = (1+rotation[0,0]-rotation[1,1]-rotation[2,2]) ** .5 * 2
            q = ((rotation[2,1]-rotation[1,2])/s, .25*s, (rotation[0,1]+rotation[1,0])/s, (rotation[0,2]+rotation[2,0])/s)
        elif i == 1:
            s = (1+rotation[1,1]-rotation[0,0]-rotation[2,2]) ** .5 * 2
            q = ((rotation[0,2]-rotation[2,0])/s, (rotation[0,1]+rotation[1,0])/s, .25*s, (rotation[1,2]+rotation[2,1])/s)
        else:
            s = (1+rotation[2,2]-rotation[0,0]-rotation[1,1]) ** .5 * 2
            q = ((rotation[1,0]-rotation[0,1])/s, (rotation[0,2]+rotation[2,0])/s, (rotation[1,2]+rotation[2,1])/s, .25*s)
    q = np.asarray(q, dtype=float); return q / np.linalg.norm(q)

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument(
        "--initial-pose",
        choices=("snapshot", "policy-default"),
        default="snapshot",
        help="initialize joints from the snapshot (default) or Isaac's configured policy default pose",
    )
    p.add_argument("--initial-joints-file", type=Path, help="27-value canonical joint vector (JSON list/object or comma-separated text); overrides --initial-pose")
    p.add_argument("--config", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, default=ROOT/"libs/SimToolReal-Franka-Wuji2"); p.add_argument("--robot-urdf", type=Path, required=True)
    p.add_argument("--mesh", type=Path, help="override snapshot repository-relative object mesh")
    p.add_argument("--steps", type=int, default=600); p.add_argument("--rate", type=float, default=60.0); p.add_argument("--seed", type=int, default=0); p.add_argument("--out", type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(p); a = p.parse_args()
    # Rollouts must be reproducible and should not open a viewer on the robot host.
    a.headless = True
    for name in ("snapshot","config","checkpoint","upstream_root","robot_urdf","out"): setattr(a, name, getattr(a,name).expanduser().resolve())
    if a.initial_joints_file is not None:
        a.initial_joints_file = a.initial_joints_file.expanduser().resolve()
    if a.mesh is not None:
        a.mesh = a.mesh.expanduser().resolve()
    if a.steps <= 0 or a.rate <= 0: p.error("steps and rate must be positive")
    snap = json.loads(a.snapshot.read_text());
    if snap.get("format") != "simtoolreal_real_snapshot_v1": p.error("unsupported snapshot format")
    mesh = a.mesh if a.mesh is not None else (ROOT / snap["object"]["mesh_repository_path"]).resolve()
    for path,label in ((a.config,"config"),(a.checkpoint,"checkpoint"),(a.robot_urdf,"URDF"),(mesh,"mesh")):
        if not path.is_file(): p.error(f"{label} not found: {path}")
    a.out.mkdir(parents=True, exist_ok=True)
    object_urdf = a.out / "snapshot_object.urdf"
    write_mesh_urdf(object_urdf, mesh, float(snap["object"]["mesh_scale_m_per_source_unit"]))
    sys.path.insert(0, str(a.upstream_root)); os.chdir(a.upstream_root); app = AppLauncher(a).app
    try:
        import gymnasium as gym, numpy as np, torch, yaml, isaacsimenvs
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        from deployment.rl_player import RlPlayer
        cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); yp = gym.spec(TASK).kwargs.get("env_cfg_yaml_entry_point")
        if yp:
            with open(yp, encoding="utf-8") as f: cfg.from_dict(yaml.safe_load(f) or {})
        random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
        if hasattr(cfg, "seed"):
            cfg.seed = a.seed
        cfg.scene.num_envs=1; cfg.assets.robot_profile="franka_wuji_right_slanted"; cfg.assets.num_assets_per_type=1; cfg.assets.object_urdf=str(object_urdf); cfg.assets.object_scale=tuple(snap["object_scales"])
        cfg.reset.reset_dof_pos_random_interval_arm=0.; cfg.reset.reset_dof_pos_random_interval_fingers=0.; cfg.reset.reset_dof_vel_random_interval=0.; cfg.sim.device=str(a.device)
        env=gym.make(TASK,cfg=cfg); inner=env.unwrapped; player=RlPlayer(num_observations=134,num_actions=27,config_path=str(a.config),checkpoint_path=str(a.checkpoint),device=a.device,num_envs=1)
        if hasattr(player,"player") and hasattr(player.player,"init_rnn"): player.player.init_rnn()
        if a.initial_joints_file is not None:
            from policy_contract import hardware_command_limits, load_joint_limits
            qcan_np = read_initial_joints(a.initial_joints_file)
            lower, upper = load_joint_limits(a.robot_urdf); command_lower, command_upper = hardware_command_limits(lower, upper)
            if np.any(qcan_np < lower - 1e-8) or np.any(qcan_np > upper + 1e-8):
                raise ValueError("initial joints exceed training URDF limits")
            if np.any(qcan_np < command_lower - 1e-8) or np.any(qcan_np > command_upper + 1e-8):
                raise ValueError("initial joints exceed FR3 hardware command limits")
            qcan = torch.tensor(qcan_np, device=inner.device, dtype=torch.float32)
            rollout_snapshot = json.loads(json.dumps(snap)); rollout_snapshot["state"]["joint_position_27"] = qcan_np.tolist(); rollout_snapshot["state"]["joint_velocity_27"] = [0.0] * 27
        elif a.initial_pose == "policy-default":
            # Read the actual configured Isaac articulation default in Lab order,
            # then convert it to the policy's canonical order.  This keeps the
            # comparison tied to the task profile/config rather than duplicated
            # constants in this launcher.
            qlab_default = inner.robot.data.default_joint_pos[0].detach().clone()
            qcan = qlab_default[inner._perm_lab_to_canon]
            rollout_snapshot = json.loads(json.dumps(snap))
            rollout_snapshot["state"]["joint_position_27"] = qcan.cpu().tolist()
            rollout_snapshot["state"]["joint_velocity_27"] = [0.0] * 27
        else:
            qcan = torch.tensor(snap["state"]["joint_position_27"],device=inner.device,dtype=torch.float32)
            rollout_snapshot = snap
        qlab=qcan[inner._perm_canon_to_lab].unsqueeze(0); zq=torch.zeros_like(qlab); origin=inner.scene.env_origins[0]
        def root_pose(m):
            m=np.asarray(m,float); pos=origin+torch.tensor(m[:3,3],device=inner.device,dtype=torch.float32); quat=torch.tensor(matrix_quat_wxyz(m[:3,:3]),device=inner.device,dtype=torch.float32); return torch.cat((pos,quat)).unsqueeze(0)
        robot_base=root_pose(rollout_snapshot["robot"]["Wp_T_robot_root"]); op=root_pose(rollout_snapshot["object"]["Wp_T_object_raw_mesh"]); gp=root_pose(rollout_snapshot["goal"]["Wp_T_goal"])
        env.reset()
        def restore_snapshot():
            inner.robot.write_root_pose_to_sim(robot_base); inner.robot.write_root_velocity_to_sim(torch.zeros((1,6),device=inner.device))
            inner.robot.write_joint_state_to_sim(qlab,zq); inner.robot.set_joint_position_target(qlab)
            inner.object.write_root_pose_to_sim(op); inner.object.write_root_velocity_to_sim(torch.zeros((1,6),device=inner.device))
            inner.goal_viz.write_root_pose_to_sim(gp); inner.goal_viz.write_root_velocity_to_sim(torch.zeros((1,6),device=inner.device))
            inner._prev_targets[:]=qlab; inner._cur_targets[:]=qlab; inner.scene.write_data_to_sim()
        restore_snapshot()
        physics_dt=float(inner.sim.get_physics_dt()); inner.sim.step(render=False); inner.scene.update(dt=physics_dt)
        if not hasattr(inner, "_get_observations"):
            raise RuntimeError("selected Isaac task does not expose _get_observations(); cannot guarantee snapshot-consistent policy input")
        obs = inner._get_observations()
        rows={k:[] for k in ("time_s","observation","action","target","joint_position","joint_velocity","object_position","object_quaternion_wxyz","goal_position","goal_quaternion_wxyz","reward","terminated","truncated")}
        for step in range(a.steps):
            po=obs["policy"].to(a.device); act=player.get_normalized_action(po,deterministic_actions=True); obs,rew,term,trunc,info=env.step(act.to(inner.device))
            if not torch.isfinite(po).all() or not torch.isfinite(act).all(): raise RuntimeError(f"non-finite policy data at step {step}")
            ql=inner.robot.data.joint_pos[0].detach().cpu().numpy(); qd=inner.robot.data.joint_vel[0].detach().cpu().numpy(); perm=inner._perm_lab_to_canon.cpu().numpy(); rows["time_s"].append(step/a.rate); rows["observation"].append(po[0].detach().cpu().numpy()); rows["action"].append(act[0].detach().cpu().numpy()); rows["target"].append(inner._cur_targets[0].detach().cpu().numpy()[perm]); rows["joint_position"].append(ql[perm]); rows["joint_velocity"].append(qd[perm]); rows["object_position"].append((inner.object.data.root_pos_w[0]-origin).detach().cpu().numpy()); rows["object_quaternion_wxyz"].append(inner.object.data.root_quat_w[0].detach().cpu().numpy()); rows["goal_position"].append((inner.goal_viz.data.root_pos_w[0]-origin).detach().cpu().numpy()); rows["goal_quaternion_wxyz"].append(inner.goal_viz.data.root_quat_w[0].detach().cpu().numpy()); rows["reward"].append(float(rew[0].detach().cpu()) if torch.is_tensor(rew) else float(rew[0])); rows["terminated"].append(bool(term[0])); rows["truncated"].append(bool(trunc[0]))
            if bool(term[0]) or bool(trunc[0]): break
        a.out.mkdir(parents=True,exist_ok=True); arrays={k:np.asarray(v) for k,v in rows.items()}; np.savez_compressed(a.out/"rollout.npz",**arrays); meta={"format":"simtoolreal_rollout_v1","simulation_backend":"isaac","task":TASK,"initial_pose":("custom-file" if a.initial_joints_file is not None else a.initial_pose),"initial_joints_file":str(a.initial_joints_file) if a.initial_joints_file is not None else None,"snapshot_format":rollout_snapshot["format"],"snapshot_sha256":sha256(a.snapshot),"robot_urdf_sha256":sha256(a.robot_urdf),"observation_dim":134,"action_dim":27,"rate_hz":a.rate,"requested_steps":a.steps,"recorded_steps":len(arrays["time_s"]),"seed":a.seed,"device":a.device,"checkpoint":str(a.checkpoint),"checkpoint_sha256":sha256(a.checkpoint),"config":str(a.config),"initial_joint_position":rollout_snapshot["state"]["joint_position_27"],"object_pose":rollout_snapshot["object"]["Wp_T_object_raw_mesh"],"goal_pose":rollout_snapshot["goal"]["Wp_T_goal"],"world_from_robot":rollout_snapshot["robot"]["Wp_T_robot_root"],"object_scales":rollout_snapshot["object_scales"],"created_unix_ns":time.time_ns()}; (a.out/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n"); (a.out/"initial_snapshot.json").write_text(json.dumps(rollout_snapshot,indent=2)+"\n"); print(json.dumps({"steps":len(arrays["time_s"]),"out":str(a.out),"backend":"isaac","initial_pose":("custom-file" if a.initial_joints_file is not None else a.initial_pose)},indent=2)); env.close(); return 0
    finally: del app
if __name__ == "__main__": raise SystemExit(main())
