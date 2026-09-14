from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = ROOT / "simtoolreal"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from capture_real_snapshot import build_snapshot
from transport import make_object_pose


def test_snapshot_uses_policy_executor_camera_composition() -> None:
    world_from_camera = np.eye(4)
    world_from_camera[:3, 3] = (0.1, -0.2, 0.3)
    camera_from_object = np.eye(4)
    camera_from_object[:3, 3] = (0.4, 0.5, 0.6)
    state_packet = {
        "arm_mode": "right",
        "include_hand": True,
        "joint_state": [0.0] * 27,
        "robot_state_stamp_s": 10.0,
        "bridge_publish_s": 10.01,
    }
    pose_packet = make_object_pose(
        camera_from_object.tolist(),
        frame_id="camera",
        timestamp_ns=10_020_000_000,
    )
    urdf = ROOT / "simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf"

    snapshot = build_snapshot(
        state_packet=state_packet,
        pose_packet=pose_packet,
        state_arrival_ns=20_000_000_000,
        pose_arrival_ns=20_005_000_000,
        world_from_camera=world_from_camera,
        world_from_robot=np.eye(4),
        goal_pose=np.eye(4),
        robot_urdf=urdf,
        object_scales=np.ones(3),
        mesh_path="libs/FoundationPose-plus-plus/test/mesh/hammer.stl",
        mesh_scale=0.001,
        camera_name="l515",
        camera_serial="f1480539",
        fallback_pose_frame="camera",
    )

    assert snapshot["format"] == "simtoolreal_real_snapshot_v1"
    np.testing.assert_allclose(
        snapshot["object"]["Wp_T_object_raw_mesh"],
        world_from_camera @ camera_from_object,
    )
    assert snapshot["state"]["joint_names"][0] == "right_fr3_joint1"
    assert snapshot["state"]["joint_names"][-1] == "r_pinky_dip"
    assert len(snapshot["robot"]["policy_fingertips_xyz"]) == 5
    assert snapshot["synchronization"]["arrival_skew_ms"] == 5.0
    assert snapshot["object"]["mesh_frame"] == "raw_uncentered_stl"
