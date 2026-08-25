from __future__ import annotations

from types import SimpleNamespace

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
