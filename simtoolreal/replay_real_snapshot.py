#!/usr/bin/env python3
"""Replay a captured real SimToolReal snapshot in one Isaac Sim environment."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


DEFAULT_TASK = "Isaacsimenvs-SimToolReal-FrankaWujiRightSlanted-Direct-v0"
TASK_ROBOT_URDF = "assets/urdf/franka_wuji_right_slanted/fr3v2_wuji_hand2_right_slanted.urdf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_mesh_urdf(path: Path, mesh_path: Path, mesh_scale: float) -> None:
    """Wrap the original, uncentered FoundationPose STL without changing its origin."""
    escaped_mesh = html.escape(str(mesh_path.resolve()), quote=True)
    scale = f"{mesh_scale:.12g} {mesh_scale:.12g} {mesh_scale:.12g}"
    contents = f"""<?xml version="1.0"?>
<robot name="foundationpose_hammer_raw">
  <link name="hammer_raw_mesh">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.20"/>
      <inertia ixx="0.00025" ixy="0" ixz="0" iyy="0.0012" iyz="0" izz="0.0012"/>
    </inertial>
    <visual>
      <geometry><mesh filename="{escaped_mesh}" scale="{scale}"/></geometry>
      <material name="hammer"><color rgba="0.16 0.18 0.20 1"/></material>
    </visual>
    <collision>
      <geometry><mesh filename="{escaped_mesh}" scale="{scale}"/></geometry>
    </collision>
  </link>
</robot>
"""
    path.write_text(contents, encoding="utf-8")


def main() -> None:
    try:
        ET.XMLParser()
    except ImportError as exc:
        raise RuntimeError(
            "Python's XML parser cannot load libexpat. Use the same LD_PRELOAD "
            "setting as scripts/visualize_initial_scene.py."
        ) from exc

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--task", default=DEFAULT_TASK, choices=(DEFAULT_TASK,))
    parser.add_argument("--out", type=Path, default=Path("outputs/real_snapshot_replay"))
    parser.add_argument(
        "--mesh",
        type=Path,
        help="Override the snapshot's repository-relative FoundationPose mesh path",
    )
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.snapshot = args.snapshot.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.out = args.out.expanduser().resolve()
    if not args.snapshot.is_file():
        parser.error(f"snapshot does not exist: {args.snapshot}")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if snapshot.get("format") != "simtoolreal_real_snapshot_v1":
        parser.error(f"unsupported snapshot format: {snapshot.get('format')!r}")
    upstream_root = args.repo_root / "libs/SimToolReal-Franka-Wuji2"
    if not upstream_root.is_dir():
        parser.error(f"SimToolReal Isaac source does not exist: {upstream_root}")
    sys.path.insert(0, str(upstream_root))
    task_robot_urdf = upstream_root / TASK_ROBOT_URDF
    if not task_robot_urdf.is_file():
        parser.error(f"right-slanted task URDF does not exist: {task_robot_urdf}")
    expected_robot_hash = snapshot["robot"].get("urdf", {}).get("sha256")
    actual_robot_hash = _sha256(task_robot_urdf)
    if expected_robot_hash and actual_robot_hash != expected_robot_hash:
        parser.error(
            "Isaac task robot URDF differs from the deployment FK URDF: "
            f"expected {expected_robot_hash}, got {actual_robot_hash}"
        )

    mesh_path = args.mesh
    if mesh_path is None:
        mesh_path = args.repo_root / snapshot["object"]["mesh_repository_path"]
    mesh_path = mesh_path.expanduser().resolve()
    if not mesh_path.is_file():
        parser.error(f"FoundationPose mesh does not exist: {mesh_path}")
    expected_mesh_hash = snapshot["object"].get("mesh_sha256")
    actual_mesh_hash = _sha256(mesh_path)
    if expected_mesh_hash and actual_mesh_hash != expected_mesh_hash:
        parser.error(
            "FoundationPose mesh hash differs from the captured mesh: "
            f"expected {expected_mesh_hash}, got {actual_mesh_hash}"
        )
    mesh_scale = float(snapshot["object"]["mesh_scale_m_per_source_unit"])
    if not math.isfinite(mesh_scale) or mesh_scale <= 0.0:
        parser.error("snapshot mesh scale must be positive and finite")

    args.out.mkdir(parents=True, exist_ok=True)
    wrapper_urdf = args.out / "foundationpose_hammer_raw.urdf"
    _write_mesh_urdf(wrapper_urdf, mesh_path, mesh_scale)
    args.enable_cameras = True
    args.headless = True
    os.chdir(upstream_root)
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
        from isaaclab.utils.math import quat_apply, quat_mul
        from PIL import Image
        from pxr import Gf, UsdGeom

        import isaacsimenvs  # noqa: F401; registers the task
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        def checked_matrix(value: Any, name: str) -> np.ndarray:
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be a finite 4x4 matrix")
            if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-6):
                raise ValueError(f"{name} has an invalid homogeneous bottom row")
            rotation = matrix[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
                raise ValueError(f"{name} rotation is not orthonormal")
            return matrix

        def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
            trace = float(np.trace(rotation))
            if trace > 0.0:
                s = math.sqrt(trace + 1.0) * 2.0
                values = (
                    0.25 * s,
                    (rotation[2, 1] - rotation[1, 2]) / s,
                    (rotation[0, 2] - rotation[2, 0]) / s,
                    (rotation[1, 0] - rotation[0, 1]) / s,
                )
            else:
                index = int(np.argmax(np.diag(rotation)))
                if index == 0:
                    s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                    values = ((rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s,
                              (rotation[0, 1] + rotation[1, 0]) / s,
                              (rotation[0, 2] + rotation[2, 0]) / s)
                elif index == 1:
                    s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                    values = ((rotation[0, 2] - rotation[2, 0]) / s,
                              (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                              (rotation[1, 2] + rotation[2, 1]) / s)
                else:
                    s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                    values = ((rotation[1, 0] - rotation[0, 1]) / s,
                              (rotation[0, 2] + rotation[2, 0]) / s,
                              (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s)
            result = np.asarray(values, dtype=np.float64)
            return result / np.linalg.norm(result)

        def quat_wxyz_to_matrix(quaternion: Any) -> np.ndarray:
            q = np.asarray(quaternion, dtype=np.float64)
            q /= np.linalg.norm(q)
            w, x, y, z = q
            return np.asarray((
                (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
            ))

        def pose_matrix(position: Any, quat_wxyz: Any) -> np.ndarray:
            result = np.eye(4)
            result[:3, :3] = quat_wxyz_to_matrix(quat_wxyz)
            result[:3, 3] = np.asarray(position, dtype=np.float64)
            return result

        def pose_error(expected: np.ndarray, actual: np.ndarray) -> dict[str, float]:
            translation = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
            relative = expected[:3, :3].T @ actual[:3, :3]
            cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
            return {
                "translation_m": translation,
                "rotation_deg": math.degrees(math.acos(cosine)),
            }

        world_from_object = checked_matrix(snapshot["object"]["Wp_T_object_raw_mesh"], "Wp_T_object")
        world_from_goal = checked_matrix(snapshot["goal"]["Wp_T_goal"], "Wp_T_goal")
        world_from_robot = checked_matrix(snapshot["robot"]["Wp_T_robot_root"], "Wp_T_robot_root")
        q_canonical_np = np.asarray(snapshot["state"]["joint_position_27"], dtype=np.float64)
        if q_canonical_np.shape != (27,) or not np.all(np.isfinite(q_canonical_np)):
            raise ValueError("snapshot joint_position_27 must be a finite 27-vector")
        object_scales = tuple(float(value) for value in snapshot["object_scales"])

        task = args.task.split(":")[-1]
        env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        spec = gym.spec(task)
        yaml_path = spec.kwargs.get("env_cfg_yaml_entry_point")
        if yaml_path:
            with open(yaml_path, encoding="utf-8") as stream:
                env_cfg.from_dict(yaml.safe_load(stream) or {})
        env_cfg.scene.num_envs = 1
        env_cfg.assets.object_urdf = str(wrapper_urdf)
        env_cfg.assets.object_scale = object_scales
        env_cfg.assets.num_assets_per_type = 1
        env_cfg.reset.reset_dof_pos_random_interval_arm = 0.0
        env_cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
        env_cfg.reset.reset_dof_vel_random_interval = 0.0
        env_cfg.sim.device = str(args.device)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (0.8, -3.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.4, 0.9)

        env = gym.make(task, cfg=env_cfg)
        try:
            env.reset()
            inner = env.unwrapped
            camera = Camera(cfg=CameraCfg(
                prim_path="/World/RealSnapshotCamera",
                update_period=0,
                height=1080,
                width=1920,
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
            camera.set_world_poses_from_view(
                (origin + torch.tensor((0.8, -3.0, 2.0), device=inner.device)).unsqueeze(0),
                (origin + torch.tensor((0.0, 0.4, 0.9), device=inner.device)).unsqueeze(0),
            )

            q_canonical = torch.tensor(q_canonical_np, device=inner.device, dtype=torch.float32)
            q_lab = q_canonical[inner._perm_canon_to_lab].unsqueeze(0)
            qd_lab = torch.zeros_like(q_lab)

            def root_pose(matrix: np.ndarray) -> torch.Tensor:
                position = origin + torch.tensor(matrix[:3, 3], device=inner.device, dtype=torch.float32)
                quaternion = torch.tensor(matrix_to_quat_wxyz(matrix[:3, :3]), device=inner.device, dtype=torch.float32)
                return torch.cat((position, quaternion)).unsqueeze(0)

            object_pose = root_pose(world_from_object)
            goal_pose = root_pose(world_from_goal)
            zero_velocity = torch.zeros((1, 6), device=inner.device)

            def restore_snapshot() -> None:
                inner.robot.write_joint_state_to_sim(q_lab, qd_lab)
                inner.robot.set_joint_position_target(q_lab)
                inner.object.write_root_pose_to_sim(object_pose)
                inner.object.write_root_velocity_to_sim(zero_velocity)
                inner.goal_viz.write_root_pose_to_sim(goal_pose)
                inner.goal_viz.write_root_velocity_to_sim(zero_velocity)
                inner._prev_targets[:] = q_lab
                inner._cur_targets[:] = q_lab
                inner.scene.write_data_to_sim()

            restore_snapshot()
            physics_dt = float(inner.sim.get_physics_dt())
            for _ in range(3):
                inner.sim.step(render=False)
                restore_snapshot()
                inner.scene.update(dt=physics_dt)

            profile = inner._robot_profile
            if list(profile.joint_names_canonical) != snapshot["state"]["joint_names"]:
                raise ValueError(
                    "snapshot canonical joint names do not match the selected Isaac robot profile"
                )
            palm_state = inner.robot.data.body_state_w[0, inner._palm_body_id]
            palm_body_pos = palm_state[0:3]
            palm_body_quat = palm_state[3:7]
            palm_frame_quat = torch.tensor(
                profile.palm_frame_quat_wxyz,
                device=inner.device,
                dtype=palm_body_quat.dtype,
            )
            sim_palm_pos_w = palm_body_pos + quat_apply(
                palm_body_quat,
                torch.tensor(profile.palm_center_offset, device=inner.device),
            )
            sim_palm_quat = quat_mul(palm_body_quat, palm_frame_quat)
            sim_palm_pos = (sim_palm_pos_w - origin).detach().cpu().numpy()
            sim_palm_quat_np = sim_palm_quat.detach().cpu().numpy()

            body_names = list(inner.robot.data.body_names)
            tip_offsets = profile.fingertip_offsets or tuple(
                profile.fingertip_offset for _ in profile.fingertip_link_names
            )
            sim_tips = []
            for name, offset in zip(profile.fingertip_link_names, tip_offsets):
                body_index = body_names.index(name)
                body_state = inner.robot.data.body_state_w[0, body_index]
                tip_world = body_state[0:3] + quat_apply(
                    body_state[3:7],
                    torch.tensor(offset, device=inner.device, dtype=body_state.dtype),
                )
                sim_tips.append((tip_world - origin).detach().cpu().numpy())
            sim_tips_np = np.asarray(sim_tips)

            stage = get_current_stage()

            def add_marker(path: str, position: Any, color: tuple[float, float, float], radius: float) -> None:
                sphere = UsdGeom.Sphere.Define(stage, path)
                sphere.GetRadiusAttr().Set(radius)
                sphere.AddTranslateOp().Set(Gf.Vec3d(*np.asarray(position, dtype=float).tolist()))
                sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

            add_marker("/World/ReplayMarkers/expected_palm", origin.cpu().numpy() + np.asarray(
                snapshot["robot"]["policy_palm"]["position_xyz"]
            ), (1.0, 0.05, 0.05), 0.014)
            add_marker("/World/ReplayMarkers/sim_palm", sim_palm_pos_w.detach().cpu().numpy(),
                       (0.05, 1.0, 1.0), 0.010)
            axis_colors = ((1.0, 0.05, 0.05), (0.05, 1.0, 0.05), (0.05, 0.25, 1.0))
            for label, axis, color in zip(("x", "y", "z"), world_from_object[:3, :3].T, axis_colors):
                add_marker(
                    f"/World/ReplayMarkers/object_axis_{label}",
                    origin.detach().cpu().numpy() + world_from_object[:3, 3] + 0.10 * axis,
                    color,
                    0.009,
                )

            best_frame = None
            best_mean = -1.0
            for frame_index in range(30):
                inner.sim.step(render=True)
                restore_snapshot()
                inner.scene.update(dt=physics_dt)
                camera.update(physics_dt)
                rgb = camera.data.output.get("rgb")
                if rgb is None or rgb.shape[0] == 0:
                    continue
                candidate = rgb[0].detach().cpu().numpy()[..., :3]
                candidate_mean = float(candidate.mean()) if candidate.size else 0.0
                if candidate_mean > best_mean:
                    best_frame = candidate.copy()
                    best_mean = candidate_mean

            actual_q_lab = inner.robot.data.joint_pos[0].detach().cpu()
            actual_q_canonical = actual_q_lab[inner._perm_lab_to_canon.cpu()].numpy()

            def local_asset_matrix(asset: Any) -> np.ndarray:
                position = (asset.data.root_pos_w[0] - origin).detach().cpu().numpy()
                quaternion = asset.data.root_quat_w[0].detach().cpu().numpy()
                return pose_matrix(position, quaternion)

            actual_robot = local_asset_matrix(inner.robot)
            actual_object = local_asset_matrix(inner.object)
            actual_goal = local_asset_matrix(inner.goal_viz)
            expected_palm_pos = np.asarray(snapshot["robot"]["policy_palm"]["position_xyz"])
            expected_palm_xyzw = np.asarray(snapshot["robot"]["policy_palm"]["quaternion_xyzw"])
            expected_palm_wxyz = expected_palm_xyzw[[3, 0, 1, 2]]
            expected_palm = pose_matrix(expected_palm_pos, expected_palm_wxyz)
            actual_palm = pose_matrix(sim_palm_pos, sim_palm_quat_np)
            expected_tips = np.asarray(snapshot["robot"]["policy_fingertips_xyz"])

            checks = {
                "joint_max_abs_error_rad": float(np.max(np.abs(actual_q_canonical - q_canonical_np))),
                "robot_root": pose_error(world_from_robot, actual_robot),
                "object_pose": pose_error(world_from_object, actual_object),
                "goal_pose": pose_error(world_from_goal, actual_goal),
                "policy_palm": pose_error(expected_palm, actual_palm),
                "fingertip_max_position_error_m": float(np.max(np.linalg.norm(sim_tips_np - expected_tips, axis=1))),
            }
            thresholds = {
                "joint_max_abs_error_rad": 1e-5,
                "root_translation_m": 1e-5,
                "root_rotation_deg": 1e-3,
                "written_pose_translation_m": 1e-5,
                "written_pose_rotation_deg": 1e-3,
                "fk_position_m": 0.002,
                "fk_rotation_deg": 1.0,
            }
            passed = (
                checks["joint_max_abs_error_rad"] <= thresholds["joint_max_abs_error_rad"]
                and checks["robot_root"]["translation_m"] <= thresholds["root_translation_m"]
                and checks["robot_root"]["rotation_deg"] <= thresholds["root_rotation_deg"]
                and checks["object_pose"]["translation_m"] <= thresholds["written_pose_translation_m"]
                and checks["object_pose"]["rotation_deg"] <= thresholds["written_pose_rotation_deg"]
                and checks["goal_pose"]["translation_m"] <= thresholds["written_pose_translation_m"]
                and checks["goal_pose"]["rotation_deg"] <= thresholds["written_pose_rotation_deg"]
                and checks["policy_palm"]["translation_m"] <= thresholds["fk_position_m"]
                and checks["policy_palm"]["rotation_deg"] <= thresholds["fk_rotation_deg"]
                and checks["fingertip_max_position_error_m"] <= thresholds["fk_position_m"]
            )
            report = {
                "format": "simtoolreal_replay_report_v1",
                "snapshot": str(args.snapshot),
                "task": task,
                "passed_internal_replay_checks": bool(passed),
                "checks": checks,
                "thresholds": thresholds,
                "mesh": {
                    "path": str(mesh_path),
                    "sha256": actual_mesh_hash,
                    "scale_m_per_source_unit": mesh_scale,
                    "frame": "raw_uncentered_stl",
                },
                "robot_urdf": {
                    "path": str(task_robot_urdf),
                    "sha256": actual_robot_hash,
                    "matches_snapshot": actual_robot_hash == expected_robot_hash,
                },
                "object_scales_policy_metadata": list(object_scales),
                "expected": {
                    "joint_names_canonical": snapshot["state"]["joint_names"],
                    "joint_position_canonical": q_canonical_np.tolist(),
                    "Wp_T_robot_root": world_from_robot.tolist(),
                    "Wp_T_object_raw_mesh": world_from_object.tolist(),
                    "Wp_T_goal": world_from_goal.tolist(),
                },
                "actual": {
                    "joint_position_canonical": actual_q_canonical.tolist(),
                    "Wp_T_robot_root": actual_robot.tolist(),
                    "Wp_T_object_raw_mesh": actual_object.tolist(),
                    "Wp_T_goal": actual_goal.tolist(),
                    "policy_palm_position_xyz": sim_palm_pos.tolist(),
                    "policy_palm_quaternion_wxyz": sim_palm_quat_np.tolist(),
                    "policy_fingertips_xyz": sim_tips_np.tolist(),
                },
                "interpretation": [
                    "Pass means the snapshot was reproduced consistently inside the selected task.",
                    "The PNG must still be compared with the stationary physical scene.",
                    "This does not validate contact dynamics, latency, gains, or policy success.",
                    "object_scales is policy metadata and is not inferred from this STL replay.",
                ],
            }
            report_path = args.out / "replay_report.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if best_frame is None or best_mean <= 0.5:
                raise RuntimeError(
                    f"Isaac Sim returned no usable RGB frame (best mean={best_mean:.3f})"
                )
            image_path = args.out / "replay.png"
            Image.fromarray(best_frame.astype(np.uint8)).save(image_path)
            print(json.dumps({"passed": passed, "checks": checks}, indent=2), flush=True)
            print(f"Wrote {image_path}", flush=True)
            print(f"Wrote {report_path}", flush=True)
            if not passed:
                exit_code = 2
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
