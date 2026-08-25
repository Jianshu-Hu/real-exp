"""Mode-specific views over datasets that store joint and EE telemetry."""

from __future__ import annotations

import copy
from typing import Any

import torch

from utils.trajectory_metadata import (
    ARM_JOINT_DIM,
    EE_POSE_DIM,
    validate_trajectory_config,
)


def normalize_training_mode(mode: str | None, default: str) -> str:
    value = default if mode is None else str(mode).strip().lower()
    if value in {"ee", "pose", "end_effector_pose", "end-effector"}:
        value = "end_effector"
    if value not in {"joint", "end_effector"}:
        raise ValueError(f"Unsupported training state/action mode {value!r}.")
    return value


def mode_trajectory_config(
    source_config: dict[str, Any],
    mode: str,
    *,
    state_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    """Return the checkpoint-facing trajectory contract for ``mode``."""
    mode = normalize_training_mode(mode, str(source_config.get("state_action_mode", "joint")))
    expected_arm_dim = ARM_JOINT_DIM if mode == "joint" else EE_POSE_DIM
    expected_dim = expected_arm_dim * len(source_config["arms"])
    if source_config["end_effector"] == "gripper":
        expected_dim += len(source_config["arms"])
    elif source_config["end_effector"] == "hand":
        expected_dim += 20 * len(source_config["arms"])
    if state_dim != expected_dim or action_dim != expected_dim:
        raise ValueError(
            f"Cannot train {mode} mode from {state_dim}/{action_dim}-value primary vectors; "
            f"the selected layout requires {expected_dim}/{expected_dim}."
        )
    config = dict(source_config)
    config.update(
        {
            "robot_state_dim": expected_dim,
            "action_dim": expected_dim,
            "state_action_mode": mode,
            "state_representation": "joint" if mode == "joint" else "end_effector_pose",
            "action_representation": "target_joint" if mode == "joint" else "delta_end_effector_pose",
        }
    )
    return validate_trajectory_config(config, expected_dim, expected_dim, source="selected training mode")


def mode_action_config(source_config: dict[str, Any], mode: str, trajectory: dict[str, Any]) -> dict[str, Any]:
    """Return action metadata matching the selected policy-facing mode."""
    result = dict(source_config)
    result.update(
        {
            "action_dim": int(trajectory["action_dim"]),
            "state_action_mode": trajectory["state_action_mode"],
            "state_representation": trajectory["state_representation"],
            "action_representation": trajectory["action_representation"],
            "arm_action_representation": (
                "absolute_joint_position"
                if mode == "joint"
                else "delta_end_effector_pose"
            ),
            "arm_action_definition": (
                "q_target[t+1]" if mode == "joint" else "ee_target[t+1]-ee_current[t]"
            ),
        }
    )
    return result


def _feature_dim(info: dict[str, Any], key: str) -> int:
    try:
        return int(info["features"][key]["shape"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Dataset is missing a valid {key} feature.") from exc


class ModeAwareDataset(torch.utils.data.Dataset):
    """A LeRobot dataset view exposing the selected policy-facing representation."""

    def __init__(self, dataset: Any, mode: str):
        self.dataset = dataset
        self.mode = normalize_training_mode(mode, "joint")
        self.meta = copy.copy(dataset.meta)
        self.meta.info = copy.deepcopy(dataset.meta.info)
        self.meta.stats = copy.deepcopy(dataset.meta.stats)

        if self.mode == "joint":
            state_key = (
                "observation.joint_state"
                if "observation.joint_state" in self.meta.info["features"]
                else "observation.state"
            )
            action_key = (
                "action.target_joint"
                if "action.target_joint" in self.meta.info["features"]
                else "action"
            )
        else:
            state_key, action_key = "observation.ee_pose", "action.delta_ee_pose"
        state_dim = _feature_dim(self.meta.info, state_key)
        action_dim = _feature_dim(self.meta.info, action_key)
        if state_dim != action_dim:
            raise ValueError(
                f"Selected dataset fields have different dimensions: {state_key}={state_dim}, {action_key}={action_dim}."
            )
        for key in (state_key, action_key):
            if key not in self.meta.stats:
                raise ValueError(f"Dataset statistics are missing the selected feature {key!r}.")

        features = self.meta.info["features"]
        features["observation.state"] = copy.deepcopy(features[state_key])
        features["action"] = copy.deepcopy(features[action_key])
        features["observation.state"]["names"] = ["state"]
        features["action"]["names"] = ["action"]
        self.meta.stats["observation.state"] = copy.deepcopy(self.meta.stats[state_key])
        self.meta.stats["action"] = copy.deepcopy(self.meta.stats[action_key])
        self._state_key = state_key
        self._action_key = action_key
        self._configure_selected_action_window()

    def _configure_selected_action_window(self) -> None:
        """Give the selected action feature the policy's canonical action offsets."""
        if self._action_key == "action":
            return

        delta_timestamps = getattr(self.dataset, "delta_timestamps", None)
        if delta_timestamps is None:
            return

        action_offsets = delta_timestamps.get("action")
        if action_offsets is None:
            return

        # LeRobot only gives the canonical action feature a temporal window when
        # constructing the dataset. The mode-specific action fields must use the
        # same offsets so policy training receives [time, action_dim] samples.
        updated_timestamps = dict(delta_timestamps)
        updated_timestamps[self._action_key] = list(action_offsets)
        self.dataset.delta_timestamps = updated_timestamps

        # LeRobot eagerly creates and loads its reader for local datasets. Update
        # its already-built index map in place so episode filtering and the
        # reader's relative-index mapping remain intact.
        reader = getattr(self.dataset, "reader", None)
        reader_delta_indices = getattr(reader, "delta_indices", None)
        if reader_delta_indices is not None and "action" in reader_delta_indices:
            reader_delta_indices = dict(reader_delta_indices)
            reader_delta_indices[self._action_key] = list(reader_delta_indices["action"])
            reader.delta_indices = reader_delta_indices

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.dataset[index])
        item["observation.state"] = item[self._state_key]
        item["action"] = item[self._action_key]
        return item

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def adapt_dataset_for_mode(dataset: Any, mode: str) -> ModeAwareDataset:
    return ModeAwareDataset(dataset, mode)
