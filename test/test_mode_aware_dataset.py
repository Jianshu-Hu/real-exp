from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from utils.mode_aware_dataset import ModeAwareDataset, mode_action_config, mode_trajectory_config


def _source_dataset() -> SimpleNamespace:
    features = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [16], "names": ["action"]},
        "observation.ee_pose": {"dtype": "float32", "shape": [14], "names": ["ee_pose"]},
        "action.delta_ee_pose": {"dtype": "float32", "shape": [14], "names": ["delta_ee_pose"]},
        "observation.joint_state": {"dtype": "float32", "shape": [16], "names": ["joint_state"]},
        "action.target_joint": {"dtype": "float32", "shape": [16], "names": ["target_joint"]},
    }
    stats = {key: {"mean": [0.0], "std": [1.0], "min": [0.0], "max": [1.0]} for key in features}
    meta = SimpleNamespace(info={"features": features}, stats=stats)

    class Dataset:
        def __init__(self):
            self.meta = meta

        def __len__(self):
            return 1

        def __getitem__(self, index):
            del index
            return {
                "observation.state": torch.zeros(16),
                "action": torch.ones(16),
                "observation.ee_pose": torch.arange(14),
                "action.delta_ee_pose": torch.arange(14) + 1,
                "observation.joint_state": torch.arange(16) + 2,
                "action.target_joint": torch.arange(16) + 3,
            }

    return Dataset()


def _arm_only_ee_dataset() -> SimpleNamespace:
    features = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [16], "names": ["action"]},
        "observation.ee_pose": {"dtype": "float32", "shape": [12], "names": ["ee_pose"]},
        "action.delta_ee_pose": {"dtype": "float32", "shape": [12], "names": ["delta_ee_pose"]},
        "observation.joint_state": {"dtype": "float32", "shape": [16], "names": ["joint_state"]},
        "action.target_joint": {"dtype": "float32", "shape": [16], "names": ["target_joint"]},
    }
    stats = {
        key: {
            "mean": np.arange(feature["shape"][0], dtype=np.float32),
            "std": np.ones(feature["shape"][0], dtype=np.float32),
            "min": np.zeros(feature["shape"][0], dtype=np.float32),
            "max": np.ones(feature["shape"][0], dtype=np.float32),
            "count": np.array([2]),
        }
        for key, feature in features.items()
    }
    meta = SimpleNamespace(info={"features": features}, stats=stats)

    class Dataset:
        def __init__(self):
            self.meta = meta
            self.delta_timestamps = {
                "action": [0.0, 1.0],
                "observation.ee_pose": [-1.0, 0.0],
                "observation.joint_state": [-1.0, 0.0],
            }
            self.reader = SimpleNamespace(
                delta_indices={
                    "action": [0, 1],
                    "observation.ee_pose": [-1, 0],
                    "observation.joint_state": [-1, 0],
                }
            )

        def __len__(self):
            return 1

        def __getitem__(self, index):
            del index
            joint_state = torch.arange(32, dtype=torch.float32).reshape(2, 16)
            target_joint = torch.arange(100, 132, dtype=torch.float32).reshape(2, 16)
            return {
                "observation.state": joint_state[-1],
                "action": target_joint,
                "observation.ee_pose": torch.arange(24, dtype=torch.float32).reshape(2, 12),
                "action.delta_ee_pose": torch.arange(200, 224, dtype=torch.float32).reshape(2, 12),
                "observation.joint_state": joint_state,
                "action.target_joint": target_joint,
            }

    return Dataset()


def test_ee_dataset_view_replaces_policy_fields() -> None:
    dataset = ModeAwareDataset(_source_dataset(), "end_effector")
    sample = dataset[0]

    assert tuple(dataset.meta.info["features"]["observation.state"]["shape"]) == (14,)
    assert tuple(sample["observation.state"].shape) == (14,)
    assert tuple(sample["action"].shape) == (14,)
    assert torch.equal(sample["action"], torch.arange(14) + 1)


def test_joint_dataset_view_preserves_primary_fields() -> None:
    dataset = ModeAwareDataset(_source_dataset(), "joint")
    sample = dataset[0]

    assert tuple(dataset.meta.info["features"]["observation.state"]["shape"]) == (16,)
    assert tuple(sample["observation.state"].shape) == (16,)
    assert torch.equal(sample["observation.state"], torch.arange(16) + 2)
    assert torch.equal(sample["action"], torch.arange(16) + 3)


def test_ee_dataset_view_preserves_current_and_target_gripper_widths() -> None:
    dataset = ModeAwareDataset(_arm_only_ee_dataset(), "end_effector")
    sample = dataset[0]

    assert tuple(dataset.meta.info["features"]["observation.state"]["shape"]) == (14,)
    assert tuple(dataset.meta.info["features"]["action"]["shape"]) == (14,)
    assert tuple(sample["observation.state"].shape) == (2, 14)
    assert tuple(sample["action"].shape) == (2, 14)
    # Each arm switches from 7 joint values to 6 EE values, while the state
    # keeps current gripper width (joint-state indices 7/15) and the action
    # keeps target width (target-joint indices 7/15).
    assert torch.equal(sample["observation.state"][0], torch.tensor([0, 1, 2, 3, 4, 5, 7, 6, 7, 8, 9, 10, 11, 15]))
    assert torch.equal(sample["action"][0], torch.tensor([200, 201, 202, 203, 204, 205, 107, 206, 207, 208, 209, 210, 211, 115]))
    assert dataset.dataset.reader.delta_indices["action.delta_ee_pose"] == [0, 1]
    assert dataset.dataset.reader.delta_indices["action.target_joint"] == [0, 1]
    assert dataset.meta.stats["action"]["mean"].shape == (14,)


def test_arm_only_ee_fields_form_complete_gripper_contract() -> None:
    source = {
        "schema_version": 1,
        "end_effector": "gripper",
        "arm_mode": "duo",
        "arms": ["left", "right"],
        "include_gripper": True,
        "include_hand": False,
        "robot_state_dim": 16,
        "action_dim": 16,
        "state_action_mode": "joint",
        "state_representation": "joint",
        "action_representation": "target_joint",
    }

    trajectory = mode_trajectory_config(source, "end_effector", state_dim=12, action_dim=12)

    assert trajectory["robot_state_dim"] == 14
    assert trajectory["action_dim"] == 14


def test_arm_only_ee_fields_preserve_current_and_target_hand_joints() -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": [54], "names": ["state"]},
        "action": {"dtype": "float32", "shape": [54], "names": ["action"]},
        "observation.ee_pose": {"dtype": "float32", "shape": [12], "names": ["ee_pose"]},
        "action.delta_ee_pose": {"dtype": "float32", "shape": [12], "names": ["delta_ee_pose"]},
        "observation.joint_state": {"dtype": "float32", "shape": [54], "names": ["joint_state"]},
        "action.target_joint": {"dtype": "float32", "shape": [54], "names": ["target_joint"]},
    }
    stats = {
        key: {
            "mean": np.arange(feature["shape"][0], dtype=np.float32),
            "std": np.ones(feature["shape"][0], dtype=np.float32),
            "min": np.zeros(feature["shape"][0], dtype=np.float32),
            "max": np.ones(feature["shape"][0], dtype=np.float32),
            "count": np.array([2]),
        }
        for key, feature in features.items()
    }
    meta = SimpleNamespace(info={"features": features}, stats=stats)

    class Dataset:
        def __init__(self):
            self.meta = meta
            self.delta_timestamps = {"action": [0.0, 1.0]}
            self.reader = SimpleNamespace(delta_indices={"action": [0, 1]})

        def __len__(self):
            return 1

        def __getitem__(self, index):
            del index
            return {
                "observation.state": torch.arange(54, dtype=torch.float32),
                "action": torch.arange(100, 208, dtype=torch.float32).reshape(2, 54),
                "observation.ee_pose": torch.arange(24, dtype=torch.float32).reshape(2, 12),
                "action.delta_ee_pose": torch.arange(200, 224, dtype=torch.float32).reshape(2, 12),
                "observation.joint_state": torch.arange(108, dtype=torch.float32).reshape(2, 54),
                "action.target_joint": torch.arange(300, 408, dtype=torch.float32).reshape(2, 54),
            }

    dataset = ModeAwareDataset(Dataset(), "end_effector")
    sample = dataset[0]

    assert tuple(dataset.meta.info["features"]["observation.state"]["shape"]) == (52,)
    assert tuple(dataset.meta.info["features"]["action"]["shape"]) == (52,)
    assert tuple(sample["observation.state"].shape) == (2, 52)
    assert tuple(sample["action"].shape) == (2, 52)
    assert torch.equal(sample["observation.state"][0, 6:26], torch.arange(7, 27))
    assert torch.equal(sample["observation.state"][0, 32:52], torch.arange(34, 54))
    assert torch.equal(sample["action"][0, 6:26], torch.arange(307, 327))
    assert torch.equal(sample["action"][0, 32:52], torch.arange(334, 354))

    source = {
        "schema_version": 1,
        "end_effector": "hand",
        "arm_mode": "duo",
        "arms": ["left", "right"],
        "include_gripper": False,
        "include_hand": True,
        "robot_state_dim": 54,
        "action_dim": 54,
        "state_action_mode": "joint",
        "state_representation": "joint",
        "action_representation": "target_joint",
    }
    trajectory = mode_trajectory_config(source, "end_effector", state_dim=12, action_dim=12)
    assert trajectory["robot_state_dim"] == 52
    assert trajectory["action_dim"] == 52


def test_selected_ee_contract_updates_action_metadata() -> None:
    source = {
        "schema_version": 1,
        "end_effector": "gripper",
        "arm_mode": "duo",
        "arms": ["left", "right"],
        "include_gripper": True,
        "include_hand": False,
        "robot_state_dim": 16,
        "action_dim": 16,
        "state_action_mode": "joint",
        "state_representation": "joint",
        "action_representation": "target_joint",
    }
    trajectory = mode_trajectory_config(source, "end_effector", state_dim=14, action_dim=14)
    action = mode_action_config(
        {"action_dim": 16, "arm_action_representation": "absolute_joint_position"},
        "end_effector",
        trajectory,
    )

    assert trajectory["state_action_mode"] == "end_effector"
    assert trajectory["action_dim"] == 14
    assert action["arm_action_representation"] == "delta_end_effector_pose"
