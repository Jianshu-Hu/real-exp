"""Validate local LeRobot datasets.

Usage:
    python data_collection/validate_dataset.py \
        --dataset-root data/my_dataset
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from utils.trajectory_metadata import (  # noqa: E402
    require_dataset_trajectory_config,
    validate_action_trajectory_contract,
)
from utils.limit import (  # noqa: E402
    TrajectoryViolationCounts,
    arm_joint_slices,
    validate_joint_trajectory,
)

INFO_PATH = Path("meta/info.json")
ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
PROCESSED_FLAG = "processed"
EE_FK_POSITION_TOLERANCE_M = 0.01
EE_FK_ORIENTATION_TOLERANCE_RAD = 0.03
EE_DELTA_POSITION_TOLERANCE_M = 1e-5
EE_DELTA_ORIENTATION_TOLERANCE_RAD = 1e-5


def trajectory_vector_layout(
    trajectory_config: dict[str, Any],
    vector_size: int,
) -> tuple[dict[str, slice], tuple[int, ...]]:
    """Return arm slices and gripper indices from trajectory metadata.

    A trajectory consists of one block per active arm. Each block starts with
    seven Franka joints and is followed by no values, one gripper value, or
    twenty hand-joint values. In particular, a 27-D hand trajectory must not be
    interpreted as a 16-D dual-arm/gripper trajectory merely because it is long
    enough to contain the old hard-coded indices.
    """
    raw_arms = trajectory_config.get("arms")
    if not isinstance(raw_arms, list) or not raw_arms:
        arm_mode = str(trajectory_config.get("arm_mode", "")).strip().lower()
        raw_arms = ["left", "right"] if arm_mode == "duo" else [arm_mode]
    arms = [str(arm).strip().lower() for arm in raw_arms]
    if not arms or len(set(arms)) != len(arms) or any(
        arm not in {"left", "right"} for arm in arms
    ):
        raise ValueError(f"invalid active arms in trajectory metadata: {raw_arms!r}")

    end_effector = str(trajectory_config.get("end_effector", "arm")).strip().lower()
    end_effector_size = {"arm": 0, "gripper": 1, "hand": 20}.get(end_effector)
    if end_effector_size is None:
        raise ValueError(
            f"unsupported end-effector mode {end_effector!r}; expected arm, gripper, or hand"
        )

    is_ee = str(trajectory_config.get("state_action_mode", "joint")).strip().lower() == "end_effector"
    arm_value_size = 6 if is_ee else 7
    block_size = arm_value_size + end_effector_size
    expected_size = block_size * len(arms)
    if vector_size != expected_size:
        raise ValueError(
            f"trajectory metadata describes {len(arms)} {end_effector} arm block(s) "
            f"({expected_size} values), but vector has {vector_size} dimensions"
        )

    arm_layout = {
        arm: slice(block_index * block_size, block_index * block_size + arm_value_size)
        for block_index, arm in enumerate(arms)
    }
    gripper_indices = (
        tuple(block_index * block_size + 7 for block_index in range(len(arms)))
        if end_effector == "gripper"
        else ()
    )
    return arm_layout, gripper_indices


def legacy_vector_layout(vector_size: int) -> tuple[dict[str, slice], tuple[int, ...]]:
    """Return the historical arm/gripper layout when metadata is unavailable."""
    arm_layout = dict(arm_joint_slices(vector_size))
    gripper_indices = {8: (7,), 16: (7, 15)}.get(vector_size, ())
    return arm_layout, gripper_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a local LeRobot dataset.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--skip-video-frames",
        action="store_true",
        help="Skip full MP4 decode, physical frame checks, and TorchCodec random-access checks.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one summary row per episode.",
    )
    parser.add_argument(
        "--delta-action-tolerance",
        type=float,
        default=1e-4,
        help=(
            "Maximum allowed absolute error for arm delta-action consistency checks. "
            "Default: 1e-4."
        ),
    )
    parser.add_argument(
        "--action-outlier-threshold",
        type=float,
        default=0.3,
        help="Flag arm action deltas with absolute value above this threshold in rad/frame. Default: 0.3.",
    )
    parser.add_argument(
        "--gripper-min",
        type=float,
        default=0.0,
        help="Minimum valid gripper state/action value. Default: 0.0.",
    )
    parser.add_argument(
        "--gripper-max",
        type=float,
        default=1.0,
        help="Maximum valid gripper state/action value. Default: 1.0.",
    )
    parser.add_argument(
        "--gripper-tolerance",
        type=float,
        default=1e-5,
        help="Tolerance around gripper min/max bounds for floating-point sensor noise. Default: 1e-5.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_action_config(dataset_root: Path) -> dict[str, Any] | None:
    action_config_path = dataset_root / ACTION_CONFIG_PATH
    if not action_config_path.exists():
        return None
    return load_json(action_config_path)


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required to validate LeRobot parquet files. "
            "Install it in the active Python environment."
        ) from exc

    return pa, pq


def load_parquet_rows(parquet_dir: Path) -> list[dict[str, Any]]:
    _, pq = require_pyarrow()
    rows: list[dict[str, Any]] = []
    for parquet_file in sorted(parquet_dir.glob("chunk-*/*.parquet")):
        table = pq.read_table(parquet_file)
        for row in table.to_pylist():
            row["_source_file"] = str(parquet_file)
            rows.append(row)
    return rows


def safe_len(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


def flatten_numeric(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[float] = []
        for item in value:
            flattened.extend(flatten_numeric(item))
        return flattened
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def has_non_finite(values: list[float]) -> bool:
    return any(not math.isfinite(value) for value in values)


def format_indices(indices: list[int]) -> str:
    if not indices:
        return "none"
    if len(indices) <= 20:
        return ", ".join(str(index) for index in indices)
    head = ", ".join(str(index) for index in indices[:10])
    tail = ", ".join(str(index) for index in indices[-5:])
    return f"{head}, ..., {tail}"


def get_feature_dim(info: dict[str, Any], feature_name: str) -> int | None:
    feature = info.get("features", {}).get(feature_name)
    if not feature:
        return None
    shape = feature.get("shape")
    if not shape:
        return None
    return int(shape[0])


def get_video_keys(info: dict[str, Any]) -> list[str]:
    return [
        feature_name
        for feature_name, feature_spec in info.get("features", {}).items()
        if feature_spec.get("dtype") == "video"
    ]


def build_data_index(data_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in data_rows:
        episode_index = int(row["episode_index"])
        by_episode.setdefault(episode_index, []).append(row)
    return by_episode


def check_cross_camera_ranges(
    episode: dict[str, Any],
    video_keys: list[str],
    fps: float,
    timestamp_tolerance: float = 1e-6,
) -> list[str]:
    if len(video_keys) <= 1:
        return []

    ranges: list[tuple[str, int, float, float]] = []
    for video_key in video_keys:
        start = float(episode[f"videos/{video_key}/from_timestamp"])
        end = float(episode[f"videos/{video_key}/to_timestamp"])
        frame_count = round((end - start) * fps)
        ranges.append((video_key, frame_count, start, end))

    reference_key, reference_frames, reference_start, reference_end = ranges[0]
    issues: list[str] = []
    for video_key, frame_count, start, end in ranges[1:]:
        if frame_count != reference_frames:
            issues.append(
                f"cross-camera frame count mismatch: {video_key} has {frame_count}, "
                f"{reference_key} has {reference_frames}"
            )
        if abs(start - reference_start) > timestamp_tolerance or abs(end - reference_end) > timestamp_tolerance:
            issues.append(
                f"cross-camera timestamp mismatch: {video_key} [{start:.6f},{end:.6f}) vs "
                f"{reference_key} [{reference_start:.6f},{reference_end:.6f})"
            )

    return issues


def check_state_action_semantics(
    episode_index: int,
    rows: list[dict[str, Any]],
    arm_action_representation: str,
    delta_action_tolerance: float,
    action_outlier_threshold: float,
    gripper_min: float,
    gripper_max: float,
    gripper_tolerance: float,
    trajectory_config: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    metrics: dict[str, Any] = {
        "max_left_arm_delta": 0.0,
        "max_right_arm_delta": 0.0,
        "delta_action_max_error": 0.0,
        "delta_action_bad_frames": 0,
        "arm_action_outlier_frames": [],
        "gripper_outlier_frames": [],
        "gripper_checked": False,
        "non_finite_frames": [],
    }

    sorted_rows = sorted(rows, key=lambda row: int(row["frame_index"]))
    states: list[list[float]] = []
    actions: list[list[float]] = []
    frame_indices: list[int] = []

    state_layout: dict[str, slice] = {}
    action_layout: dict[str, slice] = {}
    state_gripper_indices: tuple[int, ...] = ()
    action_gripper_indices: tuple[int, ...] = ()
    if sorted_rows:
        first_state = flatten_numeric(sorted_rows[0].get("observation.state"))
        first_action = flatten_numeric(sorted_rows[0].get("action"))
        try:
            if trajectory_config is None:
                state_layout, state_gripper_indices = legacy_vector_layout(
                    len(first_state)
                )
                action_layout, action_gripper_indices = legacy_vector_layout(
                    len(first_action)
                )
            else:
                state_layout, state_gripper_indices = trajectory_vector_layout(
                    trajectory_config, len(first_state)
                )
                action_layout, action_gripper_indices = trajectory_vector_layout(
                    trajectory_config, len(first_action)
                )
        except ValueError as exc:
            issues.append(f"semantic layout check failed: {exc}")
        metrics["gripper_checked"] = bool(
            state_gripper_indices or action_gripper_indices
        )

    for row in sorted_rows:
        frame_index = int(row["frame_index"])
        state = flatten_numeric(row.get("observation.state"))
        action = flatten_numeric(row.get("action"))
        states.append(state)
        actions.append(action)
        frame_indices.append(frame_index)

        if has_non_finite(state) or has_non_finite(action):
            metrics["non_finite_frames"].append(frame_index)

        if state_gripper_indices and max(state_gripper_indices) < len(state):
            for value in (state[index] for index in state_gripper_indices):
                if (
                    value < gripper_min - gripper_tolerance
                    or value > gripper_max + gripper_tolerance
                ):
                    metrics["gripper_outlier_frames"].append(frame_index)
                    break

        if action_gripper_indices and max(action_gripper_indices) < len(action):
            for value in (action[index] for index in action_gripper_indices):
                if (
                    value < gripper_min - gripper_tolerance
                    or value > gripper_max + gripper_tolerance
                ):
                    metrics["gripper_outlier_frames"].append(frame_index)
                    break

        action_outlier: dict[str, Any] = {"frame_index": frame_index}
        for arm_name, arm_slice in action_layout.items():
            if arm_slice.stop is None or arm_slice.stop > len(action):
                continue
            arm_max = max((abs(value) for value in action[arm_slice]), default=0.0)
            metric_name = f"max_{arm_name}_arm_delta"
            metrics[metric_name] = max(metrics[metric_name], arm_max)
            if (
                arm_action_representation in {"delta_joint_position", "delta_end_effector_pose"}
                and arm_max > action_outlier_threshold
            ):
                action_outlier[f"{arm_name}_max"] = arm_max
        if len(action_outlier) > 1:
            metrics["arm_action_outlier_frames"].append(action_outlier)

    if metrics["non_finite_frames"]:
        issues.append(
            f"non-finite state/action values at frames {metrics['non_finite_frames'][:10]}"
            + (" ..." if len(metrics["non_finite_frames"]) > 10 else "")
        )

    if metrics["gripper_outlier_frames"]:
        unique_frames = sorted(set(metrics["gripper_outlier_frames"]))
        issues.append(
            f"gripper state/action values outside [{gripper_min}, {gripper_max}] at frames "
            f"{unique_frames[:10]}"
            + (" ..." if len(unique_frames) > 10 else "")
        )

    if metrics["arm_action_outlier_frames"]:
        sample = metrics["arm_action_outlier_frames"][:5]
        issues.append(
            f"{len(metrics['arm_action_outlier_frames'])} arm action outlier frame(s) above "
            f"{action_outlier_threshold}: {sample}"
        )

    if arm_action_representation == "delta_joint_position" and len(sorted_rows) >= 2:
        for idx in range(len(sorted_rows) - 1):
            state = states[idx]
            next_state = states[idx + 1]
            action = actions[idx]
            frame_index = frame_indices[idx]

            shared_arms = [arm for arm in state_layout if arm in action_layout]
            if not shared_arms:
                continue

            errors: list[float] = []
            for arm_name in shared_arms:
                state_slice = state_layout[arm_name]
                action_slice = action_layout[arm_name]
                if (
                    state_slice.stop is None
                    or action_slice.stop is None
                    or state_slice.stop > len(state)
                    or state_slice.stop > len(next_state)
                    or action_slice.stop > len(action)
                ):
                    continue
                expected = np.asarray(next_state[state_slice]) - np.asarray(
                    state[state_slice]
                )
                actual = np.asarray(action[action_slice])
                errors.extend(np.abs(actual - expected).tolist())
            frame_error = max(errors, default=0.0)
            metrics["delta_action_max_error"] = max(
                metrics["delta_action_max_error"], frame_error
            )
            if frame_error > delta_action_tolerance:
                metrics["delta_action_bad_frames"] += 1

        if metrics["delta_action_bad_frames"]:
            issues.append(
                f"delta-action check failed on {metrics['delta_action_bad_frames']} frame(s); "
                f"max error {metrics['delta_action_max_error']:.6g} > tolerance {delta_action_tolerance}"
            )
    elif arm_action_representation not in {"absolute_joint_position", "delta_end_effector_pose"}:
        issues.append(f"unsupported arm action representation '{arm_action_representation}'")

    return issues, metrics


def check_ee_state_action_semantics(
    rows: list[dict[str, Any]],
    trajectory_config: dict[str, Any],
    tolerance: float,
) -> tuple[list[str], dict[str, Any]]:
    """Validate persisted EE observations/actions alongside the primary vectors."""
    arms = list(trajectory_config["arms"])
    pose_dim = 6 * len(arms)
    state_action_mode = str(trajectory_config.get("state_action_mode", "joint")).strip().lower()
    issues: list[str] = []
    metrics = {
        "ee_state_missing_frames": [],
        "ee_action_missing_frames": [],
        "ee_layout_invalid_frames": [],
        "ee_state_mismatch_frames": [],
        "ee_action_mismatch_frames": [],
        "max_ee_state_error": 0.0,
        "max_ee_action_error": 0.0,
    }
    for row in sorted(rows, key=lambda item: int(item["frame_index"])):
        frame_index = int(row["frame_index"])
        ee_state = flatten_numeric(row.get("observation.ee_pose"))
        ee_action = flatten_numeric(row.get("action.delta_ee_pose"))
        if len(ee_state) != pose_dim or has_non_finite(ee_state):
            metrics["ee_state_missing_frames"].append(frame_index)
        if len(ee_action) != pose_dim or has_non_finite(ee_action):
            metrics["ee_action_missing_frames"].append(frame_index)
        if (
            len(ee_state) != pose_dim
            or has_non_finite(ee_state)
            or len(ee_action) != pose_dim
            or has_non_finite(ee_action)
        ):
            continue

        state = flatten_numeric(row.get("observation.state"))
        action = flatten_numeric(row.get("action"))
        if state_action_mode == "end_effector" and len(state) >= pose_dim:
            try:
                state_layout, _ = trajectory_vector_layout(trajectory_config, len(state))
                action_layout, _ = trajectory_vector_layout(trajectory_config, len(action))
            except ValueError:
                metrics["ee_layout_invalid_frames"].append(frame_index)
                continue
            state_pose = np.concatenate([np.asarray(state[state_layout[side]]) for side in arms])
            action_pose = np.concatenate([np.asarray(action[action_layout[side]]) for side in arms])
            state_error = float(np.max(np.abs(np.asarray(ee_state) - state_pose)))
            metrics["max_ee_state_error"] = max(metrics["max_ee_state_error"], state_error)
            if state_error > tolerance:
                metrics["ee_state_mismatch_frames"].append(frame_index)
            if len(action) >= pose_dim:
                action_error = float(np.max(np.abs(np.asarray(ee_action) - action_pose)))
                metrics["max_ee_action_error"] = max(metrics["max_ee_action_error"], action_error)
                if action_error > tolerance:
                    metrics["ee_action_mismatch_frames"].append(frame_index)

    if metrics["ee_state_missing_frames"]:
        issues.append(
            "missing or non-finite observation.ee_pose at frames "
            f"{metrics['ee_state_missing_frames'][:10]}"
        )
    if metrics["ee_action_missing_frames"]:
        issues.append(
            "missing or non-finite action.delta_ee_pose at frames "
            f"{metrics['ee_action_missing_frames'][:10]}"
        )
    if metrics["ee_layout_invalid_frames"]:
        issues.append(
            "observation.state/action layout is invalid for EE validation at frames "
            f"{metrics['ee_layout_invalid_frames'][:10]}"
        )
    if metrics["ee_state_mismatch_frames"]:
        issues.append(
            "observation.ee_pose disagrees with observation.state at frames "
            f"{metrics['ee_state_mismatch_frames'][:10]} (max error "
            f"{metrics['max_ee_state_error']:.6g})"
        )
    if metrics["ee_action_mismatch_frames"]:
        issues.append(
            "action.delta_ee_pose disagrees with action at frames "
            f"{metrics['ee_action_mismatch_frames'][:10]} (max error "
            f"{metrics['max_ee_action_error']:.6g})"
        )
    return issues, metrics


def check_ee_joint_kinematic_consistency(
    rows: list[dict[str, Any]],
    trajectory_config: dict[str, Any],
    kinematics: Any,
) -> tuple[list[str], dict[str, Any]]:
    """Cross-check both EE representations against the recorded joint fields."""
    from utils.fr3_kinematics import (
        infer_flange_to_ee,
        pose_error,
        pose_vector_to_matrix,
    )

    arms = list(trajectory_config["arms"])
    end_effector = str(trajectory_config["end_effector"])
    joint_stride = 7 + (1 if end_effector == "gripper" else 20 if end_effector == "hand" else 0)
    ordered_rows = sorted(rows, key=lambda item: int(item["frame_index"]))
    metrics: dict[str, Any] = {
        "target_ee_missing_frames": [],
        "ee_delta_mismatch_frames": [],
        "ee_state_fk_mismatch_frames": [],
        "ee_target_fk_mismatch_frames": [],
        "max_ee_delta_position_error_m": 0.0,
        "max_ee_delta_orientation_error_rad": 0.0,
        "max_ee_state_fk_position_error_m": 0.0,
        "max_ee_state_fk_orientation_error_rad": 0.0,
        "max_ee_target_fk_position_error_m": 0.0,
        "max_ee_target_fk_orientation_error_rad": 0.0,
    }
    valid_rows: list[tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    expected_pose_dim = 6 * len(arms)
    expected_joint_dim = joint_stride * len(arms)
    for row in ordered_rows:
        frame = int(row["frame_index"])
        observed_ee = np.asarray(flatten_numeric(row.get("observation.ee_pose")), dtype=float)
        delta_ee = np.asarray(flatten_numeric(row.get("action.delta_ee_pose")), dtype=float)
        target_ee = np.asarray(flatten_numeric(row.get("action.target_ee_pose")), dtype=float)
        observed_joint = np.asarray(flatten_numeric(row.get("observation.joint_state")), dtype=float)
        target_joint = np.asarray(flatten_numeric(row.get("action.target_joint")), dtype=float)
        if target_ee.shape != (expected_pose_dim,) or not np.all(np.isfinite(target_ee)):
            metrics["target_ee_missing_frames"].append(frame)
            continue
        if (
            observed_ee.shape != (expected_pose_dim,)
            or delta_ee.shape != (expected_pose_dim,)
            or observed_joint.shape != (expected_joint_dim,)
            or target_joint.shape != (expected_joint_dim,)
            or not all(
                np.all(np.isfinite(values))
                for values in (observed_ee, delta_ee, observed_joint, target_joint)
            )
        ):
            continue
        valid_rows.append((row, observed_ee, delta_ee, target_ee, observed_joint, target_joint))

    if not valid_rows:
        issues = []
        if metrics["target_ee_missing_frames"]:
            issues.append(
                "missing or non-finite action.target_ee_pose at frames "
                f"{metrics['target_ee_missing_frames'][:10]}"
            )
        return issues, metrics

    for arm_index, arm_name in enumerate(arms):
        pose_slice = slice(arm_index * 6, arm_index * 6 + 6)
        joint_offset = arm_index * joint_stride
        joint_slice = slice(joint_offset, joint_offset + 7)
        flange_to_ee = infer_flange_to_ee(
            kinematics,
            np.asarray([item[4][joint_slice] for item in valid_rows]),
            np.asarray([item[1][pose_slice] for item in valid_rows]),
        )
        for row, observed_ee, delta_ee, target_ee, observed_joint, target_joint in valid_rows:
            frame = int(row["frame_index"])
            observed_matrix = pose_vector_to_matrix(observed_ee[pose_slice])
            target_matrix = pose_vector_to_matrix(target_ee[pose_slice])
            reconstructed_matrix = pose_vector_to_matrix(
                observed_ee[pose_slice] + delta_ee[pose_slice]
            )
            delta_position, delta_orientation = pose_error(
                reconstructed_matrix, target_matrix
            )
            state_position, state_orientation = pose_error(
                kinematics.end_effector_pose(observed_joint[joint_slice], flange_to_ee),
                observed_matrix,
            )
            target_position, target_orientation = pose_error(
                kinematics.end_effector_pose(target_joint[joint_slice], flange_to_ee),
                target_matrix,
            )
            metrics["max_ee_delta_position_error_m"] = max(
                metrics["max_ee_delta_position_error_m"], delta_position
            )
            metrics["max_ee_delta_orientation_error_rad"] = max(
                metrics["max_ee_delta_orientation_error_rad"], delta_orientation
            )
            metrics["max_ee_state_fk_position_error_m"] = max(
                metrics["max_ee_state_fk_position_error_m"], state_position
            )
            metrics["max_ee_state_fk_orientation_error_rad"] = max(
                metrics["max_ee_state_fk_orientation_error_rad"], state_orientation
            )
            metrics["max_ee_target_fk_position_error_m"] = max(
                metrics["max_ee_target_fk_position_error_m"], target_position
            )
            metrics["max_ee_target_fk_orientation_error_rad"] = max(
                metrics["max_ee_target_fk_orientation_error_rad"], target_orientation
            )
            if (
                delta_position > EE_DELTA_POSITION_TOLERANCE_M
                or delta_orientation > EE_DELTA_ORIENTATION_TOLERANCE_RAD
            ):
                metrics["ee_delta_mismatch_frames"].append((frame, arm_name))
            if (
                state_position > EE_FK_POSITION_TOLERANCE_M
                or state_orientation > EE_FK_ORIENTATION_TOLERANCE_RAD
            ):
                metrics["ee_state_fk_mismatch_frames"].append((frame, arm_name))
            if (
                target_position > EE_FK_POSITION_TOLERANCE_M
                or target_orientation > EE_FK_ORIENTATION_TOLERANCE_RAD
            ):
                metrics["ee_target_fk_mismatch_frames"].append((frame, arm_name))

    issues: list[str] = []
    for key, label in (
        ("target_ee_missing_frames", "missing or non-finite action.target_ee_pose"),
        ("ee_delta_mismatch_frames", "EE delta does not reconstruct action.target_ee_pose"),
        ("ee_state_fk_mismatch_frames", "FK(observation.joint_state) disagrees with observation.ee_pose"),
        ("ee_target_fk_mismatch_frames", "FK(action.target_joint) disagrees with action.target_ee_pose"),
    ):
        if metrics[key]:
            issues.append(f"{label} at frames {metrics[key][:10]}")
    return issues, metrics


def check_joint_state_action_fields(
    rows: list[dict[str, Any]], trajectory_config: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    """Validate the representation-neutral measured/target joint fields."""
    arms = list(trajectory_config["arms"])
    joint_dim = 7 * len(arms)
    if trajectory_config["end_effector"] == "gripper":
        joint_dim += len(arms)
    elif trajectory_config["end_effector"] == "hand":
        joint_dim += 20 * len(arms)
    metrics = {
        "joint_state_missing_frames": [],
        "joint_action_missing_frames": [],
        "joint_state_mismatch_frames": [],
        "joint_action_mismatch_frames": [],
    }
    for row in sorted(rows, key=lambda item: int(item["frame_index"])):
        frame = int(row["frame_index"])
        joint_state = flatten_numeric(row.get("observation.joint_state"))
        target_joint = flatten_numeric(row.get("action.target_joint"))
        if len(joint_state) != joint_dim or has_non_finite(joint_state):
            metrics["joint_state_missing_frames"].append(frame)
        if len(target_joint) != joint_dim or has_non_finite(target_joint):
            metrics["joint_action_missing_frames"].append(frame)
        if len(joint_state) != joint_dim or len(target_joint) != joint_dim:
            continue
        if str(trajectory_config.get("state_action_mode", "joint")).strip().lower() == "joint":
            state = flatten_numeric(row.get("observation.state"))
            action = flatten_numeric(row.get("action"))
            if len(state) == joint_dim and len(action) == joint_dim:
                state_arm_values: list[float] = []
                action_arm_values: list[float] = []
                joint_state_arm_values: list[float] = []
                target_joint_arm_values: list[float] = []
                offset = 0
                for _ in arms:
                    state_arm_values.extend(state[offset : offset + 7])
                    action_arm_values.extend(action[offset : offset + 7])
                    joint_state_arm_values.extend(joint_state[offset : offset + 7])
                    target_joint_arm_values.extend(target_joint[offset : offset + 7])
                    offset += 7
                    offset += 1 if trajectory_config["end_effector"] == "gripper" else 20 if trajectory_config["end_effector"] == "hand" else 0
                if not np.allclose(state_arm_values, joint_state_arm_values, atol=1e-6, rtol=0.0):
                    metrics["joint_state_mismatch_frames"].append(frame)
                if not np.allclose(action_arm_values, target_joint_arm_values, atol=1e-6, rtol=0.0):
                    metrics["joint_action_mismatch_frames"].append(frame)
    issues: list[str] = []
    for key, label in (
        ("joint_state_missing_frames", "observation.joint_state"),
        ("joint_action_missing_frames", "action.target_joint"),
        ("joint_state_mismatch_frames", "observation.joint_state disagrees with observation.state"),
        ("joint_action_mismatch_frames", "action.target_joint disagrees with action"),
    ):
        if metrics[key]:
            issues.append(f"{label} at frames {metrics[key][:10]}")
    return issues, metrics


def format_sampled_state_warning(
    arm_name: str,
    counts: TrajectoryViolationCounts,
    frame_indices: list[int],
    *,
    motion: bool,
) -> str:
    """Format state validity or sampled finite-difference motion warnings."""

    def episode_frames(offsets: tuple[int, ...]) -> str:
        return format_indices([frame_indices[offset] for offset in offsets])

    if motion:
        warning_indices = tuple(
            sorted(set(counts.velocity_indices) | set(counts.acceleration_indices))
        )
        return (
            f"{arm_name} sampled state motion warnings: "
            f"total={len(warning_indices)} frames=[{episode_frames(warning_indices)}], "
            f"velocity={counts.velocity_steps} "
            f"frames=[{episode_frames(counts.velocity_indices)}], "
            f"acceleration={counts.acceleration_steps} "
            f"frames=[{episode_frames(counts.acceleration_indices)}]"
        )

    validity_indices = tuple(
        sorted(
            set(counts.non_finite_indices)
            | set(counts.timing_indices)
            | set(counts.position_indices)
        )
    )
    return (
        f"{arm_name} measured-state validity violations: "
        f"total={len(validity_indices)} frames=[{episode_frames(validity_indices)}], "
        f"position={counts.position_steps} "
        f"frames=[{episode_frames(counts.position_indices)}], "
        f"non_finite={counts.non_finite_steps} "
        f"frames=[{episode_frames(counts.non_finite_indices)}], "
        f"timing={counts.timing_steps} "
        f"frames=[{episode_frames(counts.timing_indices)}]"
    )


def format_accepted_target_warning(
    arm_name: str,
    counts: TrajectoryViolationCounts,
    frame_indices: list[int],
) -> str:
    """Format validity violations for accepted low-rate joint waypoints.

    Velocity and acceleration are deliberately omitted. Consecutive accepted
    targets are waypoints for the robot-side trajectory generator, not samples
    of the generated controller reference.
    """

    def episode_frames(offsets: tuple[int, ...]) -> str:
        return format_indices([frame_indices[offset] for offset in offsets])

    target_indices = tuple(
        sorted(
            set(counts.non_finite_indices)
            | set(counts.timing_indices)
            | set(counts.position_indices)
        )
    )
    return (
        f"{arm_name} accepted action-target validity violations: "
        f"total={len(target_indices)} frames=[{episode_frames(target_indices)}], "
        f"position={counts.position_steps} "
        f"frames=[{episode_frames(counts.position_indices)}], "
        f"non_finite={counts.non_finite_steps} "
        f"frames=[{episode_frames(counts.non_finite_indices)}], "
        f"timing={counts.timing_steps} "
        f"frames=[{episode_frames(counts.timing_indices)}]"
    )


def check_joint_safety_constraints(
    rows: list[dict[str, Any]],
    arm_action_representation: str,
    trajectory_config: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Validate measured states and accepted action targets.

    Measured states are sampled trajectories, so their finite differences are
    useful (but approximate) motion diagnostics. Absolute actions are accepted
    low-rate waypoints. Their finite differences are *not* the velocity or
    acceleration of the 1 kHz reference generated inside the robot controller.
    For actions, only finite values, timestamps, and the position envelope are
    safety-validity checks; waypoint derivatives are reported separately as a
    slew diagnostic.
    """
    empty_metrics = {
        "state_violation_steps": 0,
        "state_motion_warning_steps": 0,
        "action_violation_steps": 0,
        "action_waypoint_slew_steps": 0,
    }
    if not rows:
        return [], [], empty_metrics.copy()

    sorted_rows = sorted(rows, key=lambda row: int(row["frame_index"]))
    states = [flatten_numeric(row.get("observation.state")) for row in sorted_rows]
    actions = [flatten_numeric(row.get("action")) for row in sorted_rows]
    timestamps = np.asarray(
        [float(row["timestamp"]) for row in sorted_rows],
        dtype=np.float64,
    )
    issues: list[str] = []
    warnings: list[str] = []
    metrics = empty_metrics.copy()

    if trajectory_config is not None and str(trajectory_config.get("state_action_mode", "joint")).strip().lower() == "end_effector":
        non_finite = [
            int(row["frame_index"])
            for row in sorted_rows
            if has_non_finite(flatten_numeric(row.get("observation.state")))
            or has_non_finite(flatten_numeric(row.get("action")))
        ]
        if non_finite:
            return [f"non-finite end-effector state/action values at frames {non_finite[:10]}"], warnings, metrics
        return issues, warnings, metrics

    if not states or any(len(state) != len(states[0]) for state in states):
        return [
            "joint safety check skipped because state vector lengths are inconsistent"
        ], warnings, metrics
    if not actions or any(len(action) != len(actions[0]) for action in actions):
        return [
            "joint safety check skipped because action vector lengths are inconsistent"
        ], warnings, metrics

    try:
        if trajectory_config is None:
            state_layout, _ = legacy_vector_layout(len(states[0]))
            action_layout, _ = legacy_vector_layout(len(actions[0]))
        else:
            state_layout, _ = trajectory_vector_layout(
                trajectory_config, len(states[0])
            )
            action_layout, _ = trajectory_vector_layout(
                trajectory_config, len(actions[0])
            )
    except ValueError as exc:
        return [f"joint safety check skipped: {exc}"], warnings, metrics

    state_array = np.asarray(states, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    frame_indices = [int(row["frame_index"]) for row in sorted_rows]
    shared_arms = [arm_name for arm_name in state_layout if arm_name in action_layout]
    if not shared_arms:
        return [
            "joint safety check found no matching arms in state and action layouts"
        ], warnings, metrics

    for arm_name in shared_arms:
        state_trajectory = state_array[:, state_layout[arm_name]]
        state_counts = validate_joint_trajectory(state_trajectory, timestamps)
        state_validity_indices = (
            set(state_counts.non_finite_indices)
            | set(state_counts.timing_indices)
            | set(state_counts.position_indices)
        )
        state_motion_indices = (
            set(state_counts.velocity_indices)
            | set(state_counts.acceleration_indices)
        )
        metrics["state_violation_steps"] += len(state_validity_indices)
        metrics["state_motion_warning_steps"] += len(state_motion_indices)
        if state_validity_indices:
            warnings.append(
                format_sampled_state_warning(
                    arm_name,
                    state_counts,
                    frame_indices,
                    motion=False,
                )
            )
        if state_motion_indices:
            warnings.append(
                format_sampled_state_warning(
                    arm_name,
                    state_counts,
                    frame_indices,
                    motion=True,
                )
            )

        action_trajectory = action_array[:, action_layout[arm_name]]
        if arm_action_representation in {"delta_joint_position", "delta_end_effector_pose"}:
            action_trajectory = state_trajectory + action_trajectory
        elif arm_action_representation not in {"absolute_joint_position", "delta_end_effector_pose"}:
            issues.append(
                f"joint action safety check does not support representation "
                f"'{arm_action_representation}'"
            )
            continue

        action_counts = validate_joint_trajectory(action_trajectory, timestamps)
        target_violation_indices = (
            set(action_counts.non_finite_indices)
            | set(action_counts.timing_indices)
            | set(action_counts.position_indices)
        )
        waypoint_slew_indices = (
            set(action_counts.velocity_indices)
            | set(action_counts.acceleration_indices)
        )
        metrics["action_violation_steps"] += len(target_violation_indices)
        metrics["action_waypoint_slew_steps"] += len(waypoint_slew_indices)
        if target_violation_indices:
            warnings.append(
                format_accepted_target_warning(
                    arm_name,
                    action_counts,
                    frame_indices,
                )
            )

    return issues, warnings, metrics


def check_physical_video_frames(
    dataset_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    video_keys: list[str],
) -> list[str]:
    """Fully decode each referenced video and compare it with dataset metadata.

    Strict FFmpeg decoding rejects codec/packet errors, while ``ffprobe
    -count_frames`` counts decoded frames instead of trusting the MP4 header. Both
    run out of process so malformed native codec input cannot crash the validator.
    """
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        missing_tools = ", ".join(
            tool_name
            for tool_name, tool_path in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))
            if tool_path is None
        )
        return [
            f"physical video quality checks require {missing_tools}, but it is not installed"
        ]

    issues: list[str] = []
    video_path_template = info.get("video_path")
    if not video_path_template:
        return ["dataset info.json does not define video_path"]

    for video_key in video_keys:
        expected_by_file: dict[tuple[int, int], int] = {}
        for episode in episodes:
            file_key = (
                int(episode[f"videos/{video_key}/chunk_index"]),
                int(episode[f"videos/{video_key}/file_index"]),
            )
            expected_by_file[file_key] = expected_by_file.get(file_key, 0) + int(episode["length"])

        for (chunk_index, file_index), expected_frames in sorted(expected_by_file.items()):
            video_path = dataset_root / video_path_template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if not video_path.exists():
                issues.append(f"{video_key} file {chunk_index}/{file_index}: missing {video_path}")
                continue

            strict_decode = subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(video_path),
                    "-map",
                    "0:v:0",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if strict_decode.returncode != 0:
                issues.append(
                    f"{video_key} file {chunk_index}/{file_index}: full decode failed: "
                    f"{strict_decode.stderr.strip() or 'ffmpeg returned a non-zero status'}"
                )
                continue

            expected_spec = info.get("features", {}).get(video_key, {})
            expected_shape = expected_spec.get("shape")
            expected_height = int(expected_shape[-2]) if expected_shape and len(expected_shape) >= 2 else None
            expected_width = int(expected_shape[-1]) if expected_shape and len(expected_shape) >= 1 else None
            probe = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_read_frames",
                    "-of",
                    "json",
                    str(video_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode != 0:
                issues.append(
                    f"{video_key} file {chunk_index}/{file_index}: full decode failed: "
                    f"{probe.stderr.strip() or 'ffprobe returned a non-zero status'}"
                )
                continue

            try:
                streams = json.loads(probe.stdout).get("streams", [])
                stream = streams[0]
                actual_frames = int(stream["nb_read_frames"])
                actual_fps = float(stream["avg_frame_rate"].split("/")[0]) / float(
                    stream["avg_frame_rate"].split("/")[1]
                )
                actual_width = int(stream["width"])
                actual_height = int(stream["height"])
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
                issues.append(
                    f"{video_key} file {chunk_index}/{file_index}: invalid ffprobe output: {exc}"
                )
                continue

            if actual_frames != expected_frames:
                issues.append(
                    f"{video_key} file {chunk_index}/{file_index}: "
                    f"physical frames {actual_frames} != expected {expected_frames}"
                )
            if abs(actual_fps - float(info["fps"])) > 1e-3:
                issues.append(
                    f"{video_key} file {chunk_index}/{file_index}: "
                    f"physical fps {actual_fps:.6f} != dataset fps {float(info['fps']):.6f}"
                )
            if (
                expected_width is not None
                and expected_height is not None
                and (actual_width != expected_width or actual_height != expected_height)
            ):
                issues.append(
                    f"{video_key} file {chunk_index}/{file_index}: decoded frame shape "
                    f"{actual_width}x{actual_height} != expected "
                    f"{expected_width}x{expected_height}"
                )

    return issues


def check_torchcodec_random_access(
    dataset_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    video_keys: list[str],
) -> list[str]:
    """Verify the random MP4 access pattern used by LeRobot's default decoder."""
    try:
        from lerobot.datasets.video_utils import decode_video_frames
    except ModuleNotFoundError as exc:
        return [f"TorchCodec random-access checks require the lerobot environment: {exc}"]

    video_path_template = info.get("video_path")
    if not video_path_template:
        return ["dataset info.json does not define video_path"]

    fps = float(info["fps"])
    issues: list[str] = []
    for video_key in video_keys:
        expected_by_file: dict[tuple[int, int], int] = {}
        for episode in episodes:
            file_key = (
                int(episode[f"videos/{video_key}/chunk_index"]),
                int(episode[f"videos/{video_key}/file_index"]),
            )
            expected_by_file[file_key] = expected_by_file.get(file_key, 0) + int(episode["length"])

        for (chunk_index, file_index), frame_count in sorted(expected_by_file.items()):
            if frame_count <= 0:
                issues.append(f"{video_key} file {chunk_index}/{file_index}: no frames to seek")
                continue
            video_path = dataset_root / video_path_template.format(
                video_key=video_key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            frame_indices = sorted({0, frame_count // 2, frame_count - 1})
            timestamps = [frame_index / fps for frame_index in frame_indices]
            try:
                frames = decode_video_frames(
                    video_path, timestamps, tolerance_s=1e-3, backend="torchcodec"
                )
                if len(frames) != len(timestamps):
                    raise RuntimeError(
                        f"returned {len(frames)} frames for {len(timestamps)} requested timestamps"
                    )
            except Exception as exc:
                issues.append(
                    f"{video_key} file {chunk_index}/{file_index}: TorchCodec random seek failed: {exc}"
                )
    return issues


def validate_dataset(
    dataset_root: Path,
    skip_video_frames: bool,
    verbose: bool,
    delta_action_tolerance: float,
    action_outlier_threshold: float,
    gripper_min: float,
    gripper_max: float,
    gripper_tolerance: float,
) -> int:
    if not (dataset_root / INFO_PATH).exists():
        raise FileNotFoundError(
            f"{dataset_root} is not a LeRobot dataset root. Missing {INFO_PATH}."
        )

    info = load_json(dataset_root / INFO_PATH)
    processing_warning = None
    if info.get(PROCESSED_FLAG) is not True:
        processing_warning = (
            "Dataset is not marked as processed in meta/info.json. "
            "Run process_dataset.py trim-initial before using this dataset."
        )
        print(f"WARNING: {processing_warning}", file=sys.stderr)

    action_config = load_action_config(dataset_root)
    arm_action_representation = str(
        (action_config or {}).get("arm_action_representation", "absolute_joint_position")
    ).strip().lower()
    fps = float(info["fps"])
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    state_dim = get_feature_dim(info, "observation.state")
    action_dim = get_feature_dim(info, "action")
    if state_dim is None or action_dim is None:
        raise ValueError(
            "Dataset metadata must declare observation.state and action dimensions."
        )
    trajectory_config = require_dataset_trajectory_config(dataset_root)
    validate_action_trajectory_contract(
        action_config or {}, trajectory_config, source=str(dataset_root / "meta")
    )
    video_keys = get_video_keys(info)

    episodes = load_parquet_rows(dataset_root / "meta/episodes")
    episodes.sort(key=lambda row: int(row["episode_index"]))
    data_rows = load_parquet_rows(dataset_root / "data")
    data_by_episode = build_data_index(data_rows)

    issues: list[str] = []
    warning_issues: list[str] = [processing_warning] if processing_warning else []
    expected_ee_dim = 6 * len(trajectory_config["arms"])
    expected_joint_dim = 7 * len(trajectory_config["arms"])
    if trajectory_config["end_effector"] == "gripper":
        expected_joint_dim += len(trajectory_config["arms"])
    elif trajectory_config["end_effector"] == "hand":
        expected_joint_dim += 20 * len(trajectory_config["arms"])
    ee_state_dim = get_feature_dim(info, "observation.ee_pose")
    ee_action_dim = get_feature_dim(info, "action.delta_ee_pose")
    ee_target_dim = get_feature_dim(info, "action.target_ee_pose")
    joint_state_dim = get_feature_dim(info, "observation.joint_state")
    joint_action_dim = get_feature_dim(info, "action.target_joint")
    if ee_state_dim != expected_ee_dim:
        issues.append(
            f"observation.ee_pose feature dimension {ee_state_dim!r} != expected {expected_ee_dim}"
        )
    if ee_action_dim != expected_ee_dim:
        issues.append(
            f"action.delta_ee_pose feature dimension {ee_action_dim!r} != expected {expected_ee_dim}"
        )
    if ee_target_dim != expected_ee_dim:
        issues.append(
            f"action.target_ee_pose feature dimension {ee_target_dim!r} != expected {expected_ee_dim}"
        )
    if joint_state_dim != expected_joint_dim:
        issues.append(
            f"observation.joint_state feature dimension {joint_state_dim!r} != expected {expected_joint_dim}"
        )
    if joint_action_dim != expected_joint_dim:
        issues.append(
            f"action.target_joint feature dimension {joint_action_dim!r} != expected {expected_joint_dim}"
        )

    episode_indices = [int(row["episode_index"]) for row in episodes]
    data_episode_indices = sorted(data_by_episode)

    print("Dataset summary")
    print(f"  root: {dataset_root}")
    print(f"  fps: {fps:g}")
    print(f"  total episodes declared: {total_episodes}")
    print(f"  total frames declared: {total_frames}")
    print(f"  episode metadata rows: {len(episodes)}")
    print(f"  data rows: {len(data_rows)}")
    print(f"  state dim: {state_dim if state_dim is not None else 'missing'}")
    print(f"  action dim: {action_dim if action_dim is not None else 'missing'}")
    print(f"  arm action representation: {arm_action_representation}")
    print(f"  video keys: {', '.join(video_keys) if video_keys else 'none'}")
    print(f"  metadata episode indices: {format_indices(episode_indices)}")
    print(f"  data episode indices: {format_indices(data_episode_indices)}")

    if len(episodes) != total_episodes:
        issues.append(f"episode metadata rows {len(episodes)} != total_episodes {total_episodes}")
    if len(data_rows) != total_frames:
        issues.append(f"data rows {len(data_rows)} != total_frames {total_frames}")
    if episode_indices != list(range(len(episodes))):
        issues.append("episode metadata indices are not continuous from 0")
    if data_episode_indices != episode_indices:
        issues.append("data episode indices do not match metadata episode indices")

    all_global_indices: list[int] = []
    max_left_arm_delta = 0.0
    max_right_arm_delta = 0.0
    max_delta_action_error = 0.0
    total_delta_action_bad_frames = 0
    total_arm_action_outlier_frames = 0
    total_gripper_outlier_frames = 0
    gripper_checks_enabled = False
    total_non_finite_frames = 0
    total_ee_state_missing_frames = 0
    total_ee_action_missing_frames = 0
    total_ee_layout_invalid_frames = 0
    total_ee_state_mismatch_frames = 0
    total_ee_action_mismatch_frames = 0
    total_ee_target_missing_frames = 0
    total_ee_delta_kinematic_mismatch_frames = 0
    total_ee_state_fk_mismatch_frames = 0
    total_ee_target_fk_mismatch_frames = 0
    max_ee_delta_position_error_m = 0.0
    max_ee_delta_orientation_error_rad = 0.0
    max_ee_state_fk_position_error_m = 0.0
    max_ee_state_fk_orientation_error_rad = 0.0
    max_ee_target_fk_position_error_m = 0.0
    max_ee_target_fk_orientation_error_rad = 0.0
    total_joint_state_missing_frames = 0
    total_joint_action_missing_frames = 0
    total_joint_state_mismatch_frames = 0
    total_joint_action_mismatch_frames = 0
    total_state_safety_violation_steps = 0
    total_state_motion_warning_steps = 0
    total_action_safety_violation_steps = 0
    total_action_waypoint_slew_steps = 0

    try:
        from utils.fr3_kinematics import Fr3ForwardKinematics

        ee_kinematics = Fr3ForwardKinematics()
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FR3 kinematic validation requires pinocchio, xacro, ament_index_python, "
            "and franka_description. Run the validator with /usr/bin/python3 after "
            "sourcing the ROS workspace."
        ) from exc

    if verbose:
        print("\nEpisodes")
        print("  ep  length  data_rows  data_range        timestamp_range    video  max_left  max_right  delta_err")

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        rows = data_by_episode.get(episode_index, [])
        episode_issues: list[str] = []

        all_global_indices.extend(int(row["index"]) for row in rows)

        if len(rows) != length:
            episode_issues.append(f"data rows {len(rows)} != length {length}")

        if rows:
            indices = sorted(int(row["index"]) for row in rows)
            frame_indices = sorted(int(row["frame_index"]) for row in rows)
            timestamps = [float(row["timestamp"]) for row in rows]
            expected_indices = list(range(indices[0], indices[0] + len(indices)))
            expected_frames = list(range(length))
            expected_last_timestamp = (length - 1) / fps if length > 0 else 0.0

            if indices != expected_indices:
                episode_issues.append("global index is not contiguous within episode")
            if frame_indices != expected_frames:
                episode_issues.append("frame_index is not continuous 0..length-1")
            if abs(min(timestamps)) > 1e-4:
                episode_issues.append(f"timestamp starts at {min(timestamps):.6f}, expected 0")
            if abs(max(timestamps) - expected_last_timestamp) > 1e-3:
                episode_issues.append(
                    f"last timestamp {max(timestamps):.6f} != expected {expected_last_timestamp:.6f}"
                )

            if state_dim is not None:
                bad_state = sum(
                    1
                    for row in rows
                    if safe_len(row.get("observation.state")) != state_dim
                )
                if bad_state:
                    episode_issues.append(f"{bad_state} rows have invalid observation.state length")

            if action_dim is not None:
                bad_action = sum(1 for row in rows if safe_len(row.get("action")) != action_dim)
                if bad_action:
                    episode_issues.append(f"{bad_action} rows have invalid action length")

            data_range = f"[{indices[0]},{indices[-1] + 1})"
            timestamp_range = f"[{min(timestamps):.3f},{max(timestamps):.3f}]"
        else:
            data_range = "missing"
            timestamp_range = "missing"

        bad_video_ranges: list[str] = []
        for video_key in video_keys:
            required_keys = [
                f"videos/{video_key}/chunk_index",
                f"videos/{video_key}/file_index",
                f"videos/{video_key}/from_timestamp",
                f"videos/{video_key}/to_timestamp",
            ]
            missing_keys = [key for key in required_keys if key not in episode]
            if missing_keys:
                bad_video_ranges.append(f"{video_key}: missing metadata")
                continue

            start = float(episode[f"videos/{video_key}/from_timestamp"])
            end = float(episode[f"videos/{video_key}/to_timestamp"])
            video_frames = round((end - start) * fps)
            if start < -1e-6:
                bad_video_ranges.append(f"{video_key}: negative start {start:.6f}")
            if end <= start:
                bad_video_ranges.append(f"{video_key}: non-positive range [{start:.6f},{end:.6f})")
            if video_frames != length:
                bad_video_ranges.append(f"{video_key}: {video_frames} video frames != length {length}")

        if bad_video_ranges:
            episode_issues.extend(bad_video_ranges)

        if video_keys:
            episode_issues.extend(check_cross_camera_ranges(episode, video_keys, fps))

        semantic_issues, semantic_metrics = check_state_action_semantics(
            episode_index=episode_index,
            rows=rows,
            arm_action_representation=arm_action_representation,
            delta_action_tolerance=delta_action_tolerance,
            action_outlier_threshold=action_outlier_threshold,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
            gripper_tolerance=gripper_tolerance,
            trajectory_config=trajectory_config,
        )
        episode_issues.extend(semantic_issues)

        ee_issues, ee_metrics = check_ee_state_action_semantics(
            rows,
            trajectory_config,
            tolerance=delta_action_tolerance,
        )
        episode_issues.extend(ee_issues)
        total_ee_state_missing_frames += len(ee_metrics["ee_state_missing_frames"])
        total_ee_action_missing_frames += len(ee_metrics["ee_action_missing_frames"])
        total_ee_layout_invalid_frames += len(ee_metrics["ee_layout_invalid_frames"])
        total_ee_state_mismatch_frames += len(ee_metrics["ee_state_mismatch_frames"])
        total_ee_action_mismatch_frames += len(ee_metrics["ee_action_mismatch_frames"])
        kinematic_issues, kinematic_metrics = check_ee_joint_kinematic_consistency(
            rows,
            trajectory_config,
            ee_kinematics,
        )
        episode_issues.extend(kinematic_issues)
        total_ee_target_missing_frames += len(kinematic_metrics["target_ee_missing_frames"])
        total_ee_delta_kinematic_mismatch_frames += len(
            kinematic_metrics["ee_delta_mismatch_frames"]
        )
        total_ee_state_fk_mismatch_frames += len(
            kinematic_metrics["ee_state_fk_mismatch_frames"]
        )
        total_ee_target_fk_mismatch_frames += len(
            kinematic_metrics["ee_target_fk_mismatch_frames"]
        )
        max_ee_delta_position_error_m = max(
            max_ee_delta_position_error_m,
            float(kinematic_metrics["max_ee_delta_position_error_m"]),
        )
        max_ee_delta_orientation_error_rad = max(
            max_ee_delta_orientation_error_rad,
            float(kinematic_metrics["max_ee_delta_orientation_error_rad"]),
        )
        max_ee_state_fk_position_error_m = max(
            max_ee_state_fk_position_error_m,
            float(kinematic_metrics["max_ee_state_fk_position_error_m"]),
        )
        max_ee_state_fk_orientation_error_rad = max(
            max_ee_state_fk_orientation_error_rad,
            float(kinematic_metrics["max_ee_state_fk_orientation_error_rad"]),
        )
        max_ee_target_fk_position_error_m = max(
            max_ee_target_fk_position_error_m,
            float(kinematic_metrics["max_ee_target_fk_position_error_m"]),
        )
        max_ee_target_fk_orientation_error_rad = max(
            max_ee_target_fk_orientation_error_rad,
            float(kinematic_metrics["max_ee_target_fk_orientation_error_rad"]),
        )
        joint_issues, joint_metrics = check_joint_state_action_fields(rows, trajectory_config)
        episode_issues.extend(joint_issues)
        total_joint_state_missing_frames += len(joint_metrics["joint_state_missing_frames"])
        total_joint_action_missing_frames += len(joint_metrics["joint_action_missing_frames"])
        total_joint_state_mismatch_frames += len(joint_metrics["joint_state_mismatch_frames"])
        total_joint_action_mismatch_frames += len(joint_metrics["joint_action_mismatch_frames"])

        max_left_arm_delta = max(max_left_arm_delta, float(semantic_metrics["max_left_arm_delta"]))
        max_right_arm_delta = max(max_right_arm_delta, float(semantic_metrics["max_right_arm_delta"]))
        max_delta_action_error = max(max_delta_action_error, float(semantic_metrics["delta_action_max_error"]))
        total_delta_action_bad_frames += int(semantic_metrics["delta_action_bad_frames"])
        total_arm_action_outlier_frames += len(semantic_metrics["arm_action_outlier_frames"])
        total_gripper_outlier_frames += len(set(semantic_metrics["gripper_outlier_frames"]))
        gripper_checks_enabled = gripper_checks_enabled or bool(
            semantic_metrics["gripper_checked"]
        )
        total_non_finite_frames += len(set(semantic_metrics["non_finite_frames"]))

        safety_issues, safety_warnings, safety_metrics = check_joint_safety_constraints(
            rows,
            arm_action_representation,
            trajectory_config,
        )
        episode_issues.extend(safety_issues)
        total_state_safety_violation_steps += safety_metrics["state_violation_steps"]
        total_state_motion_warning_steps += safety_metrics["state_motion_warning_steps"]
        total_action_safety_violation_steps += safety_metrics["action_violation_steps"]
        total_action_waypoint_slew_steps += safety_metrics["action_waypoint_slew_steps"]
        warning_issues.extend(
            f"episode {episode_index}: {warning}" for warning in safety_warnings
        )

        if verbose:
            video_status = "ok" if not bad_video_ranges else "BAD"
            print(
                f"  {episode_index:2d}  {length:6d}  {len(rows):9d}  "
                f"{data_range:15s} {timestamp_range:18s} {video_status:5s} "
                f"{semantic_metrics['max_left_arm_delta']:.4f}    "
                f"{semantic_metrics['max_right_arm_delta']:.4f}     "
                f"{semantic_metrics['delta_action_max_error']:.2e}"
            )

        for issue in episode_issues:
            issues.append(f"episode {episode_index}: {issue}")

    if all_global_indices:
        sorted_indices = sorted(all_global_indices)
        if sorted_indices != list(range(len(sorted_indices))):
            issues.append("global data indices are not continuous from 0")

    print("\nSemantic checks")
    if arm_action_representation == "delta_joint_position":
        print(f"  max absolute left arm action delta: {max_left_arm_delta:.6g}")
        print(f"  max absolute right arm action delta: {max_right_arm_delta:.6g}")
        print(f"  arm action outlier threshold: {action_outlier_threshold:.6g}")
        print(f"  arm action outlier frames: {total_arm_action_outlier_frames}")
    else:
        print(f"  max absolute left arm target value: {max_left_arm_delta:.6g}")
        print(f"  max absolute right arm target value: {max_right_arm_delta:.6g}")
    if gripper_checks_enabled:
        print(
            f"  gripper valid range: [{gripper_min:.6g}, {gripper_max:.6g}] "
            f"+/- {gripper_tolerance:.6g}"
        )
        print(f"  gripper outlier frames: {total_gripper_outlier_frames}")
    else:
        print("  gripper range check: skipped (trajectory has no gripper)")
    print(f"  non-finite state/action frames: {total_non_finite_frames}")
    if str(trajectory_config.get("state_action_mode", "joint")) == "end_effector":
        print(
            "  EE state/action fields: "
            f"missing_state={total_ee_state_missing_frames}, "
            f"missing_action={total_ee_action_missing_frames}, "
            f"invalid_layout={total_ee_layout_invalid_frames}, "
            f"state_mismatch={total_ee_state_mismatch_frames}, "
            f"action_mismatch={total_ee_action_mismatch_frames}"
        )
    print(
        "  EE cross-representation fields: "
        f"missing_target={total_ee_target_missing_frames}, "
        f"delta_mismatch={total_ee_delta_kinematic_mismatch_frames}, "
        f"state_fk_mismatch={total_ee_state_fk_mismatch_frames}, "
        f"target_fk_mismatch={total_ee_target_fk_mismatch_frames}"
    )
    print(
        "  EE max errors: "
        f"delta={max_ee_delta_position_error_m:.6g} m/"
        f"{max_ee_delta_orientation_error_rad:.6g} rad, "
        f"state_fk={max_ee_state_fk_position_error_m:.6g} m/"
        f"{max_ee_state_fk_orientation_error_rad:.6g} rad, "
        f"target_fk={max_ee_target_fk_position_error_m:.6g} m/"
        f"{max_ee_target_fk_orientation_error_rad:.6g} rad"
    )
    print(
        "  joint state/action fields: "
        f"missing_state={total_joint_state_missing_frames}, "
        f"missing_action={total_joint_action_missing_frames}, "
        f"state_mismatch={total_joint_state_mismatch_frames}, "
        f"action_mismatch={total_joint_action_mismatch_frames}"
    )
    if arm_action_representation == "delta_joint_position":
        print(f"  delta-action tolerance: {delta_action_tolerance:.6g}")
        print(f"  max delta-action error: {max_delta_action_error:.6g}")
        print(f"  delta-action bad frames: {total_delta_action_bad_frames}")
    else:
        print("  delta-action consistency check: skipped for absolute_joint_position actions")

    print("\nJoint safety checks")
    print(
        "  measured-state validity violation steps: "
        f"{total_state_safety_violation_steps}"
    )
    print(
        "  sampled measured-state motion warning steps: "
        f"{total_state_motion_warning_steps}"
    )
    print(
        "  accepted action-target validity violation steps: "
        f"{total_action_safety_violation_steps}"
    )
    print(
        "  accepted waypoint slew diagnostic steps: "
        f"{total_action_waypoint_slew_steps}"
    )
    print(
        "  note: waypoint slew is not controller-reference velocity/acceleration; "
        "the constrained reference is generated internally at 1 kHz"
    )

    if not skip_video_frames and video_keys:
        print("\nPhysical video checks")
        physical_video_issues = check_physical_video_frames(dataset_root, info, episodes, video_keys)
        for issue in physical_video_issues:
            if issue.startswith("physical video quality checks skipped"):
                warning_issues.append(issue)
            else:
                issues.append(issue)
        if not physical_video_issues:
            print("  ok")
        else:
            for issue in physical_video_issues:
                prefix = "  warning:" if issue.startswith("physical video quality checks skipped") else "  issue:"
                print(f"{prefix} {issue}")
        print("\nTorchCodec random-access checks")
        torchcodec_issues = check_torchcodec_random_access(dataset_root, info, episodes, video_keys)
        issues.extend(torchcodec_issues)
        if not torchcodec_issues:
            print("  ok")
        else:
            for issue in torchcodec_issues:
                print(f"  issue: {issue}")
    elif skip_video_frames:
        print("\nPhysical video checks skipped by --skip-video-frames")

    print("\nValidation summary")
    if warning_issues:
        print(f"  warnings: {len(warning_issues)}")
        for warning in warning_issues:
            print(f"  - {warning}")
    if issues:
        print(f"  status: FAILED")
        print(f"  issues: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("  status: PASS")
    return 0


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    try:
        exit_code = validate_dataset(
            dataset_root=dataset_root,
            skip_video_frames=args.skip_video_frames,
            verbose=args.verbose,
            delta_action_tolerance=args.delta_action_tolerance,
            action_outlier_threshold=args.action_outlier_threshold,
            gripper_min=args.gripper_min,
            gripper_max=args.gripper_max,
            gripper_tolerance=args.gripper_tolerance,
        )
    except Exception as exc:
        print(f"Validation failed with error: {exc}", file=sys.stderr)
        raise

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
