"""Migrate a real-exp LeRobot dataset to the schema-v2 delta contract.

The migration rewrites only parquet data and JSON/statistics metadata. Videos are
left byte-for-byte untouched. A sibling ``<dataset>_state_action_v1_backup`` is
created before the first write so the operation is recoverable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
for path in (REPO_ROOT, LOCAL_LEROBOT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(REPO_ROOT / ".hf-cache" / "datasets")

from utils.dataset_stats import ensure_dataset_stats
from utils.fr3_kinematics import (
    EE_ACTION_DIM,
    EE_STATE_DIM,
    ee_delta,
    matrix_to_ee_state,
    pose_vector_to_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate a dataset to state/action schema v2.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def joint_layout(trajectory: dict[str, Any]) -> tuple[int, int]:
    end_effector = trajectory["end_effector"]
    stride = 7 + (1 if end_effector == "gripper" else 20 if end_effector == "hand" else 0)
    return stride, len(trajectory["arms"])


def arm_joint_delta(state: Any, target: Any, trajectory: dict[str, Any]) -> np.ndarray:
    state_array = np.asarray(state, dtype=float)
    target_array = np.asarray(target, dtype=float)
    stride, arm_count = joint_layout(trajectory)
    return np.concatenate([
        target_array[index * stride : index * stride + 7]
        - state_array[index * stride : index * stride + 7]
        for index in range(arm_count)
    ]).astype(np.float32)


def convert_legacy_ee(values: Any, arm_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == (EE_STATE_DIM * arm_count,):
        return array.astype(np.float32)
    if array.shape != (6 * arm_count,):
        raise ValueError(f"Expected {6 * arm_count} legacy EE values, got {array.shape}.")
    return np.concatenate([
        matrix_to_ee_state(pose_vector_to_matrix(array[index * 6 : (index + 1) * 6]))
        for index in range(arm_count)
    ]).astype(np.float32)


def ee_action(current: Any, target: Any, arm_count: int) -> np.ndarray:
    current_array = np.asarray(current, dtype=float)
    target_array = np.asarray(target, dtype=float)
    return np.concatenate([
        ee_delta(
            _ee_matrix(current_array[index * EE_STATE_DIM : (index + 1) * EE_STATE_DIM]),
            _ee_matrix(target_array[index * EE_STATE_DIM : (index + 1) * EE_STATE_DIM]),
        )
        for index in range(arm_count)
    ]).astype(np.float32)


def _ee_matrix(values: np.ndarray) -> np.ndarray:
    from utils.fr3_kinematics import ee_state_to_matrix

    return ee_state_to_matrix(values)


def compose_primary(
    joint_state: Any,
    target_joint: Any,
    ee_state: np.ndarray,
    ee_delta_values: np.ndarray,
    trajectory: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    joint_state_array = np.asarray(joint_state, dtype=float)
    target_joint_array = np.asarray(target_joint, dtype=float)
    stride, arm_count = joint_layout(trajectory)
    mode = trajectory["state_action_mode"]
    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    for index in range(arm_count):
        joint_start = index * stride
        if mode == "joint":
            state_parts.append(joint_state_array[joint_start : joint_start + 7])
            action_parts.append(
                target_joint_array[joint_start : joint_start + 7]
                - joint_state_array[joint_start : joint_start + 7]
            )
        else:
            state_parts.append(ee_state[index * EE_STATE_DIM : (index + 1) * EE_STATE_DIM])
            action_parts.append(ee_delta_values[index * EE_ACTION_DIM : (index + 1) * EE_ACTION_DIM])
        if trajectory["end_effector"] == "gripper":
            state_parts.append(joint_state_array[joint_start + 7 : joint_start + 8])
            action_parts.append(target_joint_array[joint_start + 7 : joint_start + 8])
        elif trajectory["end_effector"] == "hand":
            state_parts.append(joint_state_array[joint_start + 7 : joint_start + 27])
            action_parts.append(target_joint_array[joint_start + 7 : joint_start + 27])
    return np.concatenate(state_parts).astype(np.float32), np.concatenate(action_parts).astype(np.float32)


def migrate(dataset_root: Path, dry_run: bool = False) -> None:
    dataset_root = dataset_root.expanduser().resolve()
    cache_root = REPO_ROOT / ".hf-cache"
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(cache_root / "datasets")
    info_path = dataset_root / "meta/info.json"
    trajectory_path = dataset_root / "meta/real_exp_trajectory_config.json"
    action_path = dataset_root / "meta/real_exp_action_config.json"
    info = load_json(info_path)
    trajectory = load_json(trajectory_path)
    action_config = load_json(action_path)
    if int(trajectory.get("schema_version", 1)) >= 2:
        print(f"{dataset_root} already uses state/action schema v2; no migration needed.")
        return
    arm_count = len(trajectory["arms"])
    state_arm_dim = 7 if trajectory["state_action_mode"] == "joint" else EE_STATE_DIM
    action_arm_dim = 7 if trajectory["state_action_mode"] == "joint" else EE_ACTION_DIM
    end_dim = 1 if trajectory["end_effector"] == "gripper" else 20 if trajectory["end_effector"] == "hand" else 0
    state_dim = (state_arm_dim + end_dim) * arm_count
    action_dim = (action_arm_dim + end_dim) * arm_count
    backup_root = dataset_root.with_name(dataset_root.name + "_state_action_v1_backup")
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    print("State/action schema-v2 migration")
    print(f"  dataset: {dataset_root}")
    print(f"  backup: {backup_root}")
    print(f"  parquet files: {len(parquet_files)}")
    print(f"  primary state/action dimensions: {state_dim}/{action_dim}")
    print("  videos: unchanged")
    if dry_run:
        return
    if backup_root.exists():
        raise FileExistsError(f"Migration backup already exists: {backup_root}")
    shutil.copytree(dataset_root, backup_root, copy_function=shutil.copy2)
    try:
        for parquet_file in parquet_files:
            frame = pd.read_parquet(parquet_file)
            new_ee_states = []
            new_ee_targets = []
            new_ee_deltas = []
            new_joint_deltas = []
            new_states = []
            new_actions = []
            for row in frame.to_dict(orient="records"):
                ee_state = convert_legacy_ee(row["observation.ee_pose"], arm_count)
                ee_target = convert_legacy_ee(row["action.target_ee_pose"], arm_count)
                delta_ee = ee_action(ee_state, ee_target, arm_count)
                delta_joint = arm_joint_delta(
                    row["observation.joint_state"], row["action.target_joint"], trajectory
                )
                state, action = compose_primary(
                    row["observation.joint_state"], row["action.target_joint"],
                    ee_state, delta_ee, trajectory,
                )
                new_ee_states.append(ee_state)
                new_ee_targets.append(ee_target)
                new_ee_deltas.append(delta_ee)
                new_joint_deltas.append(delta_joint)
                new_states.append(state)
                new_actions.append(action)
            frame["observation.ee_pose"] = new_ee_states
            frame["action.target_ee_pose"] = new_ee_targets
            frame["action.delta_ee_pose"] = new_ee_deltas
            frame["action.delta_joint"] = new_joint_deltas
            frame["observation.state"] = new_states
            frame["action"] = new_actions
            temporary = parquet_file.with_suffix(".parquet.tmp")
            frame.to_parquet(temporary, index=False)
            temporary.replace(parquet_file)

        features = info["features"]
        features["observation.state"].update(shape=[state_dim], names=["state"])
        features["action"].update(shape=[action_dim], names=["action"])
        features["observation.ee_pose"].update(
            shape=[EE_STATE_DIM * arm_count], names=["ee_position_rotation_6d"]
        )
        features["action.target_ee_pose"].update(
            shape=[EE_STATE_DIM * arm_count], names=["target_ee_position_rotation_6d"]
        )
        features["action.delta_ee_pose"].update(
            shape=[EE_ACTION_DIM * arm_count], names=["delta_position_rotation_vector"]
        )
        features["action.delta_joint"] = {
            "dtype": "float32", "shape": [7 * arm_count], "names": ["delta_joint"]
        }
        info["state_action_schema_version"] = 2
        info["processed"] = True
        trajectory.update(
            schema_version=2,
            robot_state_dim=state_dim,
            action_dim=action_dim,
            state_representation=(
                "joint" if trajectory["state_action_mode"] == "joint"
                else "end_effector_position_rotation_6d"
            ),
            action_representation=(
                "delta_joint_position" if trajectory["state_action_mode"] == "joint"
                else "delta_end_effector_position_rotation_vector"
            ),
            delta_alignment="one_step",
        )
        action_config.update(
            action_dim=action_dim,
            state_action_contract_version=2,
            state_representation=trajectory["state_representation"],
            action_representation=trajectory["action_representation"],
            arm_action_representation=(
                "delta_joint_position" if trajectory["state_action_mode"] == "joint"
                else "delta_end_effector_position_rotation_vector"
            ),
            arm_action_definition=(
                "q_target[t+1]-q_measured[t]" if trajectory["state_action_mode"] == "joint"
                else "base_translation_delta+Log(R_target@R_current.T)"
            ),
            delta_alignment="one_step",
            ee_state_rotation_representation="rotation_6d_first_two_columns",
            ee_action_rotation_representation="rotation_vector",
            ee_action_rotation_frame="robot_base_spatial",
            ee_rotation_composition="R_target=Exp(rotvec)@R_anchor",
        )
        atomic_json(info_path, info)
        atomic_json(trajectory_path, trajectory)
        atomic_json(action_path, action_config)
        ensure_dataset_stats(f"local/{dataset_root.name}", dataset_root, force_recompute=True)
    except Exception:
        shutil.rmtree(dataset_root)
        backup_root.rename(dataset_root)
        raise
    print("Migration complete.")


if __name__ == "__main__":
    arguments = parse_args()
    migrate(arguments.dataset_root, arguments.dry_run)
