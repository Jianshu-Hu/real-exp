from __future__ import annotations

import numpy as np

from data_collection.lerobot_collection import compute_recorded_action, packet_pair_to_frame


def make_packet(
    *,
    state_action_mode: str,
    state: np.ndarray,
    action: np.ndarray,
    joint_state: np.ndarray,
    target_joint: np.ndarray,
    ee_pose: np.ndarray | None = None,
    target_ee_pose: np.ndarray | None = None,
) -> dict[str, object]:
    state = np.asarray(state, dtype=np.float32)
    return {
        "robot_state_dim": state.size,
        "action_dim": state.size,
        "joint_state_dim": 16,
        "target_joint_dim": 16,
        "state_action_mode": state_action_mode,
        "arm_mode": "duo",
        "include_right_arm": True,
        "include_gripper": True,
        "include_hand": False,
        "state": state,
        "action": np.asarray(action, dtype=np.float32),
        "joint_state": np.asarray(joint_state, dtype=np.float32),
        "target_joint": np.asarray(target_joint, dtype=np.float32),
        "ee_pose": np.zeros(12, dtype=np.float32) if ee_pose is None else np.asarray(ee_pose, dtype=np.float32),
        "target_ee_pose": (
            np.zeros(12, dtype=np.float32)
            if target_ee_pose is None
            else np.asarray(target_ee_pose, dtype=np.float32)
        ),
        "delta_ee_pose": np.zeros(12, dtype=np.float32),
        "camera_names": [],
        "cameras": {},
    }


def test_joint_primary_gripper_action_uses_the_next_target_width() -> None:
    current = make_packet(
        state_action_mode="joint",
        state=np.arange(16),
        action=np.arange(16),
        joint_state=np.arange(16),
        target_joint=np.arange(16),
    )
    next_packet = make_packet(
        state_action_mode="joint",
        state=np.arange(16),
        action=np.arange(100, 116),
        joint_state=np.arange(16),
        target_joint=np.arange(100, 116),
    )

    action = compute_recorded_action(current, next_packet)

    np.testing.assert_array_equal(action, np.arange(100, 116, dtype=np.float32))


def test_ee_primary_gripper_fields_are_clamped_at_ee_block_indices() -> None:
    current_state = np.arange(14, dtype=np.float32)
    current_state[[6, 13]] = [-0.2, 1.2]
    next_action = np.arange(14, dtype=np.float32)
    next_action[[6, 13]] = [-0.4, 1.4]
    current = make_packet(
        state_action_mode="end_effector",
        state=current_state,
        action=np.zeros(14, dtype=np.float32),
        joint_state=np.arange(16),
        target_joint=np.arange(16),
        ee_pose=np.zeros(12),
    )
    next_packet = make_packet(
        state_action_mode="end_effector",
        state=np.zeros(14, dtype=np.float32),
        action=next_action,
        joint_state=np.arange(16),
        target_joint=np.arange(16),
        target_ee_pose=np.zeros(12),
    )

    frame = packet_pair_to_frame(current, next_packet, [], "test")

    np.testing.assert_array_equal(frame["observation.state"][[6, 13]], np.asarray([0.0, 1.0]))
    np.testing.assert_array_equal(frame["action"][[6, 13]], np.asarray([0.0, 1.0]))
