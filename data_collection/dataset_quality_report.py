from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
INFO_PATH = Path("meta/info.json")
ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
DEFAULT_REPORT_ROOT = Path(
    os.environ.get(
        "REAL_EXP_DATASET_QUALITY_ROOT",
        "~/.local/share/real-exp/dataset-quality",
    )
).expanduser()

FR3_JOINT_LOWER_RAD = [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508]
FR3_JOINT_UPPER_RAD = [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508]
FR3_MAX_VELOCITY_RAD_S = [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]
FR3_MAX_ACCELERATION_RAD_S2 = [10.0] * 7

SEVERITY_ORDER = {"INFO": 0, "REVIEW": 1, "UNSAFE": 2, "ERROR": 3}
AUTO_REPAIR_ACTIONS = {"keep", "exclude_episode", "constrained_action_repair"}


@dataclass
class Thresholds:
    safety_factor: float = 0.8
    position_margin_rad: float = 0.02
    max_action_state_gap_rad: float = 0.25
    initial_configuration_l2_rad: float = 0.50
    gripper_min: float = 0.0
    gripper_max: float = 1.0
    gripper_state_tolerance: float = 0.002
    gripper_action_tolerance: float = 1e-5

    def validate(self) -> None:
        if not 0.0 < self.safety_factor <= 1.0:
            raise ValueError("safety_factor must be in (0, 1].")
        if self.position_margin_rad < 0.0:
            raise ValueError("position_margin_rad must be non-negative.")
        if self.max_action_state_gap_rad <= 0.0:
            raise ValueError("max_action_state_gap_rad must be positive.")
        if self.initial_configuration_l2_rad <= 0.0:
            raise ValueError("initial_configuration_l2_rad must be positive.")


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    recommendation: str
    episode_index: int | None = None
    frame_index: int | None = None
    arm: str | None = None
    joint: int | None = None
    value: float | None = None
    limit: float | None = None
    ratio: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit LeRobot datasets for structural and motion-quality problems, "
            "produce reports outside the dataset, and optionally create a repaired copy."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Read datasets and generate quality reports.")
    source_group = scan.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--dataset-root", help="One LeRobot dataset root.")
    source_group.add_argument(
        "--datasets-root",
        help="Directory whose immediate children are LeRobot dataset roots.",
    )
    scan.add_argument(
        "--name-pattern",
        default="*",
        help="Glob used with --datasets-root. Default: *.",
    )
    scan.add_argument(
        "--report-root",
        default=str(DEFAULT_REPORT_ROOT),
        help=(
            "External report directory. Default: "
            "~/.local/share/real-exp/dataset-quality."
        ),
    )
    scan.add_argument(
        "--check-video-frames",
        action="store_true",
        help="Decode video containers and compare physical frame counts (slower).",
    )
    add_threshold_arguments(scan)

    review = subparsers.add_parser(
        "review",
        help="Interactively select one repair choice for each flagged episode.",
    )
    review.add_argument("--plan", required=True, help="repair_plan.json produced by scan.")

    apply_parser = subparsers.add_parser(
        "apply",
        help="Apply selected repairs to a new dataset directory.",
    )
    apply_parser.add_argument("--plan", required=True, help="Reviewed repair_plan.json.")
    apply_parser.add_argument(
        "--output-dir",
        required=True,
        help="New derived dataset directory. It must not already exist or be inside the source.",
    )
    apply_parser.add_argument("--repo-id", default=None, help="Optional output LeRobot repo id.")
    apply_parser.add_argument("--video-workers", type=int, default=None)
    apply_parser.add_argument(
        "--max-action-correction-rad",
        type=float,
        default=0.05,
        help=(
            "Abort constrained action repair when any joint changes by more than this value. "
            "Default: 0.05 rad."
        ),
    )
    apply_parser.add_argument(
        "--allow-large-action-repair",
        action="store_true",
        help="Allow corrections above --max-action-correction-rad after printing the magnitude.",
    )
    apply_parser.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Allow ERROR/UNSAFE episodes whose selected_action is still null.",
    )
    return parser.parse_args()


def add_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--safety-factor", type=float, default=0.8)
    parser.add_argument("--position-margin-rad", type=float, default=0.02)
    parser.add_argument("--max-action-state-gap-rad", type=float, default=0.25)
    parser.add_argument("--initial-configuration-l2-rad", type=float, default=0.50)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=1.0)
    parser.add_argument(
        "--gripper-state-tolerance",
        type=float,
        default=0.002,
        help="Tolerance for measured gripper-width noise. Default: 0.002 m.",
    )
    parser.add_argument(
        "--gripper-action-tolerance",
        type=float,
        default=1e-5,
        help="Tolerance for normalized gripper action values. Default: 1e-5.",
    )


def thresholds_from_args(args: argparse.Namespace) -> Thresholds:
    thresholds = Thresholds(
        safety_factor=args.safety_factor,
        position_margin_rad=args.position_margin_rad,
        max_action_state_gap_rad=args.max_action_state_gap_rad,
        initial_configuration_l2_rad=args.initial_configuration_l2_rad,
        gripper_min=args.gripper_min,
        gripper_max=args.gripper_max,
        gripper_state_tolerance=args.gripper_state_tolerance,
        gripper_action_tolerance=args.gripper_action_tolerance,
    )
    thresholds.validate()
    return thresholds


def require_numeric_dependencies():
    try:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "dataset_quality_report.py requires numpy and pyarrow. Run it from the "
            "same Conda environment used for LeRobot training or collection."
        ) from exc
    return np, pa, pq


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def require_external_path(path: Path, dataset_root: Path, label: str) -> None:
    if path_is_within(path, dataset_root):
        raise ValueError(
            f"{label} must be outside the source dataset so training cannot ingest it: {path}"
        )


def require_dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not (path / INFO_PATH).is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root (missing {INFO_PATH}): {path}")
    if not any((path / "data").glob("chunk-*/*.parquet")):
        raise FileNotFoundError(f"No data/chunk-*/*.parquet files found under: {path}")
    return path


def dataset_fingerprint(dataset_root: Path) -> str:
    digest = hashlib.sha256()
    candidates = [dataset_root / INFO_PATH, dataset_root / ACTION_CONFIG_PATH]
    candidates.extend(sorted((dataset_root / "data").glob("chunk-*/*.parquet")))
    candidates.extend(sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet")))
    for path in candidates:
        if not path.exists():
            continue
        stat = path.stat()
        digest.update(str(path.relative_to(dataset_root)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        if path.suffix == ".json":
            digest.update(path.read_bytes())
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def arm_slices(vector_dim: int) -> list[tuple[str, slice]]:
    if vector_dim >= 16:
        return [("left", slice(0, 7)), ("right", slice(8, 15))]
    if vector_dim >= 14:
        return [("left", slice(0, 7)), ("right", slice(7, 14))]
    if vector_dim >= 7:
        return [("left", slice(0, 7))]
    return []


def gripper_indices(vector_dim: int) -> list[tuple[str, int]]:
    if vector_dim >= 16:
        return [("left", 7), ("right", 15)]
    if vector_dim == 8:
        return [("left", 7)]
    return []


def stack_arms(values: Any, specs: list[tuple[str, slice]], np: Any):
    return np.stack([values[:, item] for _, item in specs], axis=1)


def read_data_table(dataset_root: Path, pa: Any, pq: Any):
    paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    tables = [pq.read_table(path) for path in paths]
    return pa.concat_tables(tables), paths


def read_episode_metadata(dataset_root: Path, pa: Any, pq: Any):
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        return None
    return pa.concat_tables([pq.read_table(path) for path in paths])


def make_issue(
    issues: list[Issue],
    severity: str,
    code: str,
    message: str,
    recommendation: str,
    **details: Any,
) -> None:
    issues.append(
        Issue(
            severity=severity,
            code=code,
            message=message,
            recommendation=recommendation,
            **details,
        )
    )


def add_worst_joint_issues(
    issues: list[Issue],
    *,
    severity: str,
    code: str,
    values: Any,
    ratios: Any,
    episodes: Any,
    frames: Any,
    arm_names: list[str],
    limit_values: Any,
    message: str,
    recommendation: str,
    np: Any,
) -> None:
    if ratios.size == 0:
        return
    bad_rows = np.flatnonzero((ratios > 1.0).any(axis=(1, 2)))
    for row in bad_rows:
        flat_index = int(np.argmax(ratios[row]))
        arm_index, joint_index = np.unravel_index(flat_index, ratios[row].shape)
        limit = (
            float(limit_values[joint_index])
            if getattr(limit_values, "ndim", 0) > 0
            else float(limit_values)
        )
        make_issue(
            issues,
            severity,
            code,
            message,
            recommendation,
            episode_index=int(episodes[row]),
            frame_index=int(frames[row]),
            arm=arm_names[arm_index],
            joint=joint_index + 1,
            value=float(values[row, arm_index, joint_index]),
            limit=limit,
            ratio=float(ratios[row, arm_index, joint_index]),
        )


def add_position_issues(
    issues: list[Issue],
    *,
    values: Any,
    episodes: Any,
    frames: Any,
    arm_names: list[str],
    lower: Any,
    upper: Any,
    severity: str,
    code: str,
    message: str,
    recommendation: str,
    np: Any,
) -> None:
    below = lower[None, None, :] - values
    above = values - upper[None, None, :]
    distance = np.maximum(below, above)
    bad_rows = np.flatnonzero((distance > 0.0).any(axis=(1, 2)))
    for row in bad_rows:
        flat_index = int(np.argmax(distance[row]))
        arm_index, joint_index = np.unravel_index(flat_index, distance[row].shape)
        value = float(values[row, arm_index, joint_index])
        limit = float(lower[joint_index] if value < lower[joint_index] else upper[joint_index])
        make_issue(
            issues,
            severity,
            code,
            message,
            recommendation,
            episode_index=int(episodes[row]),
            frame_index=int(frames[row]),
            arm=arm_names[arm_index],
            joint=joint_index + 1,
            value=value,
            limit=limit,
            ratio=abs(value - limit) / max(abs(limit), 1e-12),
        )


def scan_dataset(
    dataset_root: Path,
    thresholds: Thresholds,
    *,
    check_video_frames: bool,
) -> dict[str, Any]:
    np, pa, pq = require_numeric_dependencies()
    info = load_json(dataset_root / INFO_PATH)
    action_config = (
        load_json(dataset_root / ACTION_CONFIG_PATH)
        if (dataset_root / ACTION_CONFIG_PATH).is_file()
        else {}
    )
    representation = str(
        action_config.get("arm_action_representation", "absolute_joint_position")
    ).strip().lower()
    fps = float(info.get("fps", 15.0))
    issues: list[Issue] = []

    table, parquet_paths = read_data_table(dataset_root, pa, pq)
    required_columns = {"episode_index", "frame_index", "observation.state", "action"}
    missing_columns = sorted(required_columns - set(table.column_names))
    if missing_columns:
        make_issue(
            issues,
            "ERROR",
            "missing_columns",
            f"Data parquet is missing required columns: {missing_columns}.",
            "Restore the dataset or re-record affected episodes; motion repair is not valid.",
        )
        return build_report(
            dataset_root,
            info,
            action_config,
            thresholds,
            issues,
            {},
            len(table),
            len(parquet_paths),
        )

    ep = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    frame = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    timestamp = (
        np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        if "timestamp" in table.column_names
        else frame.astype(np.float64) / fps
    )
    state_rows = table["observation.state"].to_pylist()
    action_rows = table["action"].to_pylist()
    try:
        state = np.asarray(state_rows, dtype=np.float64)
        action = np.asarray(action_rows, dtype=np.float64)
    except ValueError:
        make_issue(
            issues,
            "ERROR",
            "ragged_state_or_action",
            "State or action rows have inconsistent dimensions.",
            "Exclude malformed episodes or restore them from an intact source.",
        )
        return build_report(
            dataset_root,
            info,
            action_config,
            thresholds,
            issues,
            {},
            len(table),
            len(parquet_paths),
        )

    order = np.lexsort((frame, ep))
    ep, frame, timestamp = ep[order], frame[order], timestamp[order]
    state, action = state[order], action[order]

    declared_frames = int(info.get("total_frames", len(ep)))
    declared_episodes = int(info.get("total_episodes", len(np.unique(ep))))
    if declared_frames != len(ep):
        make_issue(
            issues,
            "ERROR",
            "total_frame_mismatch",
            f"info.json declares {declared_frames} frames but parquet contains {len(ep)}.",
            "Repair metadata from a trusted source before training.",
        )
    unique_episodes = sorted(int(value) for value in np.unique(ep))
    if unique_episodes != list(range(declared_episodes)):
        make_issue(
            issues,
            "ERROR",
            "episode_index_mismatch",
            "Parquet episode indices are not continuous or do not match info.json.",
            "Reindex the dataset or restore it before applying motion repairs.",
        )

    episode_metadata = read_episode_metadata(dataset_root, pa, pq)
    episode_lengths_meta: dict[int, int] = {}
    if episode_metadata is None:
        make_issue(
            issues,
            "ERROR",
            "missing_episode_metadata",
            "No meta/episodes/chunk-*/*.parquet files were found.",
            "Restore episode metadata before training.",
        )
    else:
        for row in episode_metadata.to_pylist():
            episode_lengths_meta[int(row["episode_index"])] = int(row["length"])
        if len(episode_lengths_meta) != declared_episodes:
            make_issue(
                issues,
                "ERROR",
                "episode_metadata_count_mismatch",
                "Episode metadata row count does not match info.json.",
                "Rebuild episode metadata before training.",
            )

    episode_metrics: dict[int, dict[str, Any]] = {}
    episode_hashes: dict[str, list[int]] = defaultdict(list)
    initial_vectors: list[tuple[int, Any]] = []
    for episode_index in unique_episodes:
        rows = np.flatnonzero(ep == episode_index)
        if rows.size == 0:
            continue
        local_frames = frame[rows]
        local_timestamps = timestamp[rows]
        expected_frames = np.arange(rows.size)
        if not np.array_equal(local_frames, expected_frames):
            make_issue(
                issues,
                "ERROR",
                "frame_index_discontinuity",
                "frame_index is not continuous from zero within the episode.",
                "Exclude or rebuild this episode together with its videos.",
                episode_index=episode_index,
            )
        expected_timestamps = expected_frames / fps
        if np.max(np.abs(local_timestamps - expected_timestamps)) > 1e-3:
            make_issue(
                issues,
                "ERROR",
                "timestamp_discontinuity",
                "Episode timestamps do not follow frame_index / fps.",
                "Exclude or rebuild this episode together with its videos.",
                episode_index=episode_index,
            )
        metadata_length = episode_lengths_meta.get(episode_index)
        if metadata_length is not None and metadata_length != rows.size:
            make_issue(
                issues,
                "ERROR",
                "episode_length_mismatch",
                f"Episode metadata length {metadata_length} differs from {rows.size} data rows.",
                "Exclude or rebuild this episode together with its videos.",
                episode_index=episode_index,
            )

        episode_digest = hashlib.sha256()
        episode_digest.update(state[rows].tobytes())
        episode_digest.update(action[rows].tobytes())
        episode_hashes[episode_digest.hexdigest()].append(episode_index)
        initial_vectors.append((episode_index, state[rows[0]].copy()))
        episode_metrics[episode_index] = {
            "frames": int(rows.size),
            "duration_s": float(rows.size / fps),
        }

    for duplicate_indices in episode_hashes.values():
        if len(duplicate_indices) < 2:
            continue
        for episode_index in duplicate_indices[1:]:
            make_issue(
                issues,
                "INFO",
                "duplicate_episode",
                f"State/action sequence duplicates episode {duplicate_indices[0]} exactly.",
                "Confirm that the duplicate is intentional before training.",
                episode_index=episode_index,
            )

    declared_state_dim = feature_dim(info, "observation.state")
    declared_action_dim = feature_dim(info, "action")
    if state.ndim != 2 or (declared_state_dim is not None and state.shape[1] != declared_state_dim):
        make_issue(
            issues,
            "ERROR",
            "state_dimension_mismatch",
            f"Observed state shape {state.shape} does not match metadata dimension {declared_state_dim}.",
            "Restore or exclude malformed data before training.",
        )
    if action.ndim != 2 or (declared_action_dim is not None and action.shape[1] != declared_action_dim):
        make_issue(
            issues,
            "ERROR",
            "action_dimension_mismatch",
            f"Observed action shape {action.shape} does not match metadata dimension {declared_action_dim}.",
            "Restore or exclude malformed data before training.",
        )

    finite_state_rows = np.isfinite(state).all(axis=1)
    finite_action_rows = np.isfinite(action).all(axis=1)
    for row in np.flatnonzero(~finite_state_rows):
        make_issue(
            issues,
            "ERROR",
            "non_finite_state",
            "observation.state contains NaN or infinity.",
            "Exclude this episode; numeric interpolation is not trustworthy.",
            episode_index=int(ep[row]),
            frame_index=int(frame[row]),
        )
    for row in np.flatnonzero(~finite_action_rows):
        make_issue(
            issues,
            "ERROR",
            "non_finite_action",
            "action contains NaN or infinity.",
            "Exclude this episode; numeric interpolation is not trustworthy.",
            episode_index=int(ep[row]),
            frame_index=int(frame[row]),
        )

    state_specs = arm_slices(state.shape[1]) if state.ndim == 2 else []
    action_specs = arm_slices(action.shape[1]) if action.ndim == 2 else []
    if not state_specs or not action_specs:
        make_issue(
            issues,
            "ERROR",
            "unsupported_robot_vector",
            f"Expected at least seven arm joints, got state/action shapes {state.shape}/{action.shape}.",
            "Add an explicit vector layout before auditing this dataset.",
        )
    elif finite_state_rows.all() and finite_action_rows.all():
        analyze_motion(
            issues,
            ep,
            frame,
            state,
            action,
            state_specs,
            action_specs,
            representation,
            fps,
            thresholds,
            np,
        )
        analyze_initial_configurations(
            issues,
            initial_vectors,
            state_specs,
            thresholds.initial_configuration_l2_rad,
            np,
        )

    analyze_grippers(issues, ep, frame, state, action, thresholds, np)
    analyze_episode_durations(issues, episode_metrics, np)
    analyze_video_metadata(
        issues,
        dataset_root,
        info,
        episode_metadata,
        episode_lengths_meta,
        check_video_frames,
    )

    return build_report(
        dataset_root,
        info,
        action_config,
        thresholds,
        issues,
        episode_metrics,
        len(table),
        len(parquet_paths),
    )


def feature_dim(info: dict[str, Any], name: str) -> int | None:
    shape = info.get("features", {}).get(name, {}).get("shape")
    if not shape:
        return None
    return int(shape[0])


def contiguous_pair_mask(ep: Any, frame: Any, np: Any):
    return (ep[1:] == ep[:-1]) & (frame[1:] == frame[:-1] + 1)


def contiguous_triple_mask(ep: Any, frame: Any, np: Any):
    return (
        (ep[2:] == ep[1:-1])
        & (ep[1:-1] == ep[:-2])
        & (frame[1:-1] == frame[:-2] + 1)
        & (frame[2:] == frame[1:-1] + 1)
    )


def analyze_motion(
    issues: list[Issue],
    ep: Any,
    frame: Any,
    state: Any,
    action: Any,
    state_specs: list[tuple[str, slice]],
    action_specs: list[tuple[str, slice]],
    representation: str,
    fps: float,
    thresholds: Thresholds,
    np: Any,
) -> None:
    lower = np.asarray(FR3_JOINT_LOWER_RAD, dtype=float)
    upper = np.asarray(FR3_JOINT_UPPER_RAD, dtype=float)
    hard_velocity = np.asarray(FR3_MAX_VELOCITY_RAD_S, dtype=float)
    safe_velocity = hard_velocity * thresholds.safety_factor
    hard_acceleration = np.asarray(FR3_MAX_ACCELERATION_RAD_S2, dtype=float)
    safe_acceleration = hard_acceleration * thresholds.safety_factor
    state_arms = stack_arms(state, state_specs, np)
    action_arms = stack_arms(action, action_specs, np)
    state_names = [name for name, _ in state_specs]
    action_names = [name for name, _ in action_specs]

    add_position_issues(
        issues,
        values=state_arms,
        episodes=ep,
        frames=frame,
        arm_names=state_names,
        lower=lower,
        upper=upper,
        severity="ERROR",
        code="state_position_hard_limit",
        message="Measured robot state is outside the FR3 hard joint range.",
        recommendation="Exclude the episode and inspect robot/controller logs.",
        np=np,
    )

    absolute_action = representation == "absolute_joint_position"
    if absolute_action:
        add_position_issues(
            issues,
            values=action_arms,
            episodes=ep,
            frames=frame,
            arm_names=action_names,
            lower=lower,
            upper=upper,
            severity="ERROR",
            code="action_position_hard_limit",
            message="Saved absolute action is outside the FR3 hard joint range.",
            recommendation="Exclude or re-record the episode; do not clip this target directly.",
            np=np,
        )
        safe_lower = lower + thresholds.position_margin_rad
        safe_upper = upper - thresholds.position_margin_rad
        margin_bad = (
            ((action_arms < safe_lower) | (action_arms > safe_upper))
            & (action_arms >= lower)
            & (action_arms <= upper)
        )
        margin_distance = np.maximum(safe_lower - action_arms, action_arms - safe_upper)
        margin_distance = np.where(margin_bad, margin_distance, 0.0)
        for row in np.flatnonzero(margin_bad.any(axis=(1, 2))):
            flat_index = int(np.argmax(margin_distance[row]))
            arm_index, joint_index = np.unravel_index(flat_index, margin_bad[row].shape)
            value = float(action_arms[row, arm_index, joint_index])
            limit = float(
                safe_lower[joint_index] if value < safe_lower[joint_index] else safe_upper[joint_index]
            )
            make_issue(
                issues,
                "UNSAFE",
                "action_position_safety_margin",
                "Saved action enters the deployment position safety margin.",
                "Prefer re-recording, episode exclusion, or constrained trajectory repair.",
                episode_index=int(ep[row]),
                frame_index=int(frame[row]),
                arm=action_names[arm_index],
                joint=joint_index + 1,
                value=value,
                limit=limit,
                ratio=abs(value - limit) / max(thresholds.position_margin_rad, 1e-12),
            )
    else:
        make_issue(
            issues,
            "REVIEW",
            "non_absolute_action_representation",
            f"Motion target checks are limited for action representation '{representation}'.",
            "Confirm the action semantics before applying any automatic repair.",
        )

    pair = contiguous_pair_mask(ep, frame, np)
    pair_ep, pair_frame = ep[1:][pair], frame[1:][pair]
    state_velocity = np.diff(state_arms, axis=0)[pair] * fps
    action_velocity = np.diff(action_arms, axis=0)[pair] * fps

    state_velocity_hard_ratio = np.abs(state_velocity) / hard_velocity
    add_worst_joint_issues(
        issues,
        severity="ERROR",
        code="state_velocity_hard_limit",
        values=state_velocity,
        ratios=state_velocity_hard_ratio,
        episodes=pair_ep,
        frames=pair_frame,
        arm_names=state_names,
        limit_values=hard_velocity,
        message="Measured 15 Hz joint velocity exceeds the FR3 limit.",
        recommendation="Exclude or re-record the episode and inspect high-rate controller logs.",
        np=np,
    )
    state_velocity_safe_ratio = np.abs(state_velocity) / safe_velocity
    state_velocity_safe_ratio = np.where(state_velocity_hard_ratio <= 1.0, state_velocity_safe_ratio, 0.0)
    add_worst_joint_issues(
        issues,
        severity="REVIEW",
        code="state_velocity_safety_limit",
        values=state_velocity,
        ratios=state_velocity_safe_ratio,
        episodes=pair_ep,
        frames=pair_frame,
        arm_names=state_names,
        limit_values=safe_velocity,
        message="Measured 15 Hz joint velocity exceeds the deployment safety envelope.",
        recommendation="Review the episode; retain only when the faster behavior is intentional.",
        np=np,
    )
    action_velocity_ratio = np.abs(action_velocity) / safe_velocity
    add_worst_joint_issues(
        issues,
        severity="UNSAFE",
        code="action_velocity_safety_limit",
        values=action_velocity,
        ratios=action_velocity_ratio,
        episodes=pair_ep,
        frames=pair_frame,
        arm_names=action_names,
        limit_values=safe_velocity,
        message="Saved action target velocity exceeds the deployment safety envelope.",
        recommendation="Use constrained action repair, exclude the episode, or re-record it.",
        np=np,
    )

    triple = contiguous_triple_mask(ep, frame, np)
    triple_ep, triple_frame = ep[2:][triple], frame[2:][triple]
    state_acceleration = np.diff(state_arms, n=2, axis=0)[triple] * fps**2
    action_acceleration = np.diff(action_arms, n=2, axis=0)[triple] * fps**2
    state_acceleration_ratio = np.abs(state_acceleration) / safe_acceleration
    add_worst_joint_issues(
        issues,
        severity="REVIEW",
        code="state_acceleration_diagnostic",
        values=state_acceleration,
        ratios=state_acceleration_ratio,
        episodes=triple_ep,
        frames=triple_frame,
        arm_names=state_names,
        limit_values=safe_acceleration,
        message="15 Hz state second difference exceeds the acceleration safety envelope.",
        recommendation=(
            "Treat this as diagnostic only; inspect video and high-rate controller data before exclusion."
        ),
        np=np,
    )
    action_acceleration_ratio = np.abs(action_acceleration) / safe_acceleration
    add_worst_joint_issues(
        issues,
        severity="UNSAFE",
        code="action_acceleration_safety_limit",
        values=action_acceleration,
        ratios=action_acceleration_ratio,
        episodes=triple_ep,
        frames=triple_frame,
        arm_names=action_names,
        limit_values=safe_acceleration,
        message="Saved action target acceleration exceeds the deployment safety envelope.",
        recommendation="Use constrained action repair, exclude the episode, or re-record it.",
        np=np,
    )

    if absolute_action:
        arm_count = min(state_arms.shape[1], action_arms.shape[1])
        gap = np.abs(action_arms[:, :arm_count] - state_arms[:, :arm_count])
        gap_ratio = gap / thresholds.max_action_state_gap_rad
        add_worst_joint_issues(
            issues,
            severity="REVIEW",
            code="action_state_tracking_gap",
            values=gap,
            ratios=gap_ratio,
            episodes=ep,
            frames=frame,
            arm_names=action_names[:arm_count],
            limit_values=np.full(7, thresholds.max_action_state_gap_rad),
            message="Absolute action target is far from the measured robot state.",
            recommendation="Inspect for GELLO jumps, contact, tracking lag, or initial configuration mismatch.",
            np=np,
        )


def analyze_initial_configurations(
    issues: list[Issue],
    initial_vectors: list[tuple[int, Any]],
    state_specs: list[tuple[str, slice]],
    threshold: float,
    np: Any,
) -> None:
    if len(initial_vectors) < 3:
        return
    episode_indices = [item[0] for item in initial_vectors]
    starts = np.stack(
        [np.concatenate([vector[sl] for _, sl in state_specs]) for _, vector in initial_vectors]
    )
    median = np.median(starts, axis=0)
    distances = np.linalg.norm(starts - median, axis=1)
    for row in np.flatnonzero(distances > threshold):
        make_issue(
            issues,
            "INFO",
            "initial_configuration_outlier",
            "Episode initial joint configuration is far from the task median.",
            "Confirm that this is intended task coverage rather than an incorrect reset.",
            episode_index=int(episode_indices[row]),
            frame_index=0,
            value=float(distances[row]),
            limit=float(threshold),
            ratio=float(distances[row] / threshold),
        )


def analyze_grippers(
    issues: list[Issue],
    ep: Any,
    frame: Any,
    state: Any,
    action: Any,
    thresholds: Thresholds,
    np: Any,
) -> None:
    for label, values in (("state", state), ("action", action)):
        tolerance = (
            thresholds.gripper_state_tolerance
            if label == "state"
            else thresholds.gripper_action_tolerance
        )
        for arm, index in gripper_indices(values.shape[1]):
            column = values[:, index]
            bad = (column < thresholds.gripper_min - tolerance) | (
                column > thresholds.gripper_max + tolerance
            )
            for row in np.flatnonzero(bad):
                make_issue(
                    issues,
                    "ERROR",
                    f"{label}_gripper_range",
                    f"{label} gripper value is outside the configured range.",
                    "Check gripper representation and exclude malformed episodes.",
                    episode_index=int(ep[row]),
                    frame_index=int(frame[row]),
                    arm=arm,
                    value=float(column[row]),
                    limit=float(
                        thresholds.gripper_min
                        if column[row] < thresholds.gripper_min
                        else thresholds.gripper_max
                    ),
                )


def analyze_episode_durations(
    issues: list[Issue], episode_metrics: dict[int, dict[str, Any]], np: Any
) -> None:
    if len(episode_metrics) < 5:
        return
    indices = sorted(episode_metrics)
    durations = np.asarray([episode_metrics[index]["duration_s"] for index in indices])
    median = float(np.median(durations))
    mad = float(np.median(np.abs(durations - median)))
    if mad <= 1e-12:
        return
    robust_z = np.abs(durations - median) / (1.4826 * mad)
    for row in np.flatnonzero(robust_z > 4.0):
        make_issue(
            issues,
            "INFO",
            "episode_duration_outlier",
            "Episode duration is an outlier for this task.",
            "Confirm that recording started and ended at the intended task boundaries.",
            episode_index=int(indices[row]),
            value=float(durations[row]),
            limit=median,
            ratio=float(robust_z[row]),
        )


def analyze_video_metadata(
    issues: list[Issue],
    dataset_root: Path,
    info: dict[str, Any],
    episode_metadata: Any,
    episode_lengths: dict[int, int],
    check_video_frames: bool,
) -> None:
    video_keys = [
        name
        for name, spec in info.get("features", {}).items()
        if name.startswith("observation.images.") and spec.get("dtype") in {"video", "image"}
    ]
    if not video_keys or episode_metadata is None:
        return
    fps = float(info.get("fps", 15.0))
    rows = episode_metadata.to_pylist()
    for row in rows:
        episode_index = int(row["episode_index"])
        ranges = []
        for video_key in video_keys:
            prefix = f"videos/{video_key}"
            required = [
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
                f"{prefix}/to_timestamp",
            ]
            if any(name not in row for name in required):
                make_issue(
                    issues,
                    "ERROR",
                    "video_metadata_missing",
                    f"Episode is missing metadata for {video_key}.",
                    "Exclude or rebuild the episode and its videos.",
                    episode_index=episode_index,
                )
                continue
            start = float(row[f"{prefix}/from_timestamp"])
            end = float(row[f"{prefix}/to_timestamp"])
            ranges.append((video_key, start, end))
            expected = episode_lengths.get(episode_index)
            if expected is not None and round((end - start) * fps) != expected:
                make_issue(
                    issues,
                    "ERROR",
                    "video_timestamp_length_mismatch",
                    f"{video_key} timestamp range does not match episode length.",
                    "Exclude or rebuild the episode and its videos.",
                    episode_index=episode_index,
                )
        if ranges:
            reference = ranges[0][1:]
            if any(abs(start - reference[0]) > 1e-6 or abs(end - reference[1]) > 1e-6 for _, start, end in ranges[1:]):
                make_issue(
                    issues,
                    "ERROR",
                    "cross_camera_timestamp_mismatch",
                    "Camera timestamp ranges disagree within the episode.",
                    "Exclude or rebuild the episode and synchronized videos.",
                    episode_index=episode_index,
                )

    if not check_video_frames:
        return
    try:
        import cv2
    except ModuleNotFoundError:
        make_issue(
            issues,
            "REVIEW",
            "video_frame_check_unavailable",
            "OpenCV is unavailable, so physical video frame counts were not checked.",
            "Install opencv-python-headless or rerun without relying on physical checks.",
        )
        return

    path_template = info.get("video_path")
    if not path_template:
        make_issue(
            issues,
            "ERROR",
            "video_path_template_missing",
            "info.json has video features but no video_path template.",
            "Restore dataset metadata.",
        )
        return
    expected_by_file: Counter[tuple[str, int, int]] = Counter()
    for row in rows:
        episode_index = int(row["episode_index"])
        for video_key in video_keys:
            prefix = f"videos/{video_key}"
            try:
                key = (
                    video_key,
                    int(row[f"{prefix}/chunk_index"]),
                    int(row[f"{prefix}/file_index"]),
                )
            except KeyError:
                continue
            expected_by_file[key] += episode_lengths.get(episode_index, 0)
    for (video_key, chunk_index, file_index), expected in expected_by_file.items():
        path = dataset_root / path_template.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            make_issue(
                issues,
                "ERROR",
                "video_file_unreadable",
                f"Could not open video file: {path}",
                "Restore the video or exclude affected episodes.",
            )
            continue
        actual = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        capture.release()
        if actual != expected:
            make_issue(
                issues,
                "ERROR",
                "video_physical_frame_mismatch",
                f"{path.name} contains {actual} frames; metadata expects {expected}.",
                "Rebuild the video container or exclude affected episodes.",
                value=float(actual),
                limit=float(expected),
            )


def build_report(
    dataset_root: Path,
    info: dict[str, Any],
    action_config: dict[str, Any],
    thresholds: Thresholds,
    issues: list[Issue],
    episode_metrics: dict[int, dict[str, Any]],
    actual_frames: int,
    parquet_files: int,
) -> dict[str, Any]:
    severity_counts = Counter(issue.severity for issue in issues)
    code_counts = Counter(issue.code for issue in issues)
    episode_counts: dict[int, Counter[str]] = defaultdict(Counter)
    episode_severity: dict[int, str] = {}
    for issue in issues:
        if issue.episode_index is None:
            continue
        episode_counts[issue.episode_index][issue.code] += 1
        previous = episode_severity.get(issue.episode_index, "INFO")
        if SEVERITY_ORDER[issue.severity] > SEVERITY_ORDER[previous]:
            episode_severity[issue.episode_index] = issue.severity
        else:
            episode_severity.setdefault(issue.episode_index, previous)
    for episode_index, metrics in episode_metrics.items():
        metrics["issue_counts"] = dict(sorted(episode_counts[episode_index].items()))
        metrics["max_severity"] = episode_severity.get(episode_index, "PASS")
    max_severity = max(
        (issue.severity for issue in issues),
        key=lambda value: SEVERITY_ORDER[value],
        default="PASS",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset_root),
        "source_fingerprint": dataset_fingerprint(dataset_root),
        "dataset": {
            "name": dataset_root.name,
            "fps": float(info.get("fps", 15.0)),
            "declared_frames": int(info.get("total_frames", actual_frames)),
            "actual_frames": actual_frames,
            "declared_episodes": int(info.get("total_episodes", len(episode_metrics))),
            "parquet_files": parquet_files,
            "action_config": action_config,
        },
        "thresholds": asdict(thresholds),
        "summary": {
            "status": max_severity,
            "severity_counts": dict(sorted(severity_counts.items())),
            "code_counts": dict(sorted(code_counts.items())),
        },
        "episodes": {str(key): value for key, value in sorted(episode_metrics.items())},
        "issues": [asdict(issue) for issue in issues],
    }


def repair_choice_for_codes(codes: set[str]) -> tuple[str, list[str]]:
    structural_or_state_error = any(
        code.startswith("state_position_hard")
        or code.startswith("state_velocity_hard")
        or code.startswith("non_finite")
        or code in {
            "action_position_hard_limit",
            "frame_index_discontinuity",
            "timestamp_discontinuity",
            "episode_length_mismatch",
            "video_metadata_missing",
            "video_timestamp_length_mismatch",
            "cross_camera_timestamp_mismatch",
            "state_gripper_range",
            "action_gripper_range",
        }
        for code in codes
    )
    if structural_or_state_error:
        return "exclude_episode", ["keep", "exclude_episode", "rerecord"]
    action_repairable = any(
        code in {
            "action_position_safety_margin",
            "action_velocity_safety_limit",
            "action_acceleration_safety_limit",
        }
        for code in codes
    )
    if action_repairable:
        state_motion_warning = bool(
            codes
            & {
                "state_velocity_safety_limit",
                "state_acceleration_diagnostic",
            }
        )
        if state_motion_warning:
            return "exclude_episode", ["keep", "exclude_episode", "rerecord"]
        return "constrained_action_repair", [
            "keep",
            "exclude_episode",
            "constrained_action_repair",
            "rerecord",
        ]
    return "keep", ["keep", "exclude_episode", "rerecord"]


def build_repair_plan(report: dict[str, Any]) -> dict[str, Any]:
    issues_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for issue in report["issues"]:
        if issue["episode_index"] is not None:
            issues_by_episode[int(issue["episode_index"])].append(issue)
    entries = []
    for episode_index, issues in sorted(issues_by_episode.items()):
        codes = {issue["code"] for issue in issues}
        recommended, choices = repair_choice_for_codes(codes)
        max_severity = max(
            (issue["severity"] for issue in issues), key=lambda value: SEVERITY_ORDER[value]
        )
        entries.append(
            {
                "episode_index": episode_index,
                "max_severity": max_severity,
                "finding_counts": dict(sorted(Counter(issue["code"] for issue in issues).items())),
                "recommended_action": recommended,
                "allowed_actions": choices,
                "selected_action": None,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_dataset": report["source_dataset"],
        "source_fingerprint": report["source_fingerprint"],
        "thresholds": report["thresholds"],
        "instructions": (
            "Run the review subcommand or set selected_action to one allowed action. "
            "Reports and plans must remain outside the dataset."
        ),
        "episodes": entries,
    }


def write_report(report: dict[str, Any], report_dir: Path) -> None:
    dataset_root = Path(report["source_dataset"])
    require_external_path(report_dir, dataset_root, "Report directory")
    if report_dir.exists():
        raise FileExistsError(f"Report directory already exists: {report_dir}")
    report_dir.mkdir(parents=True)

    with (report_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=True, default=json_default)
        handle.write("\n")
    plan = build_repair_plan(report)
    with (report_dir / "repair_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=True, default=json_default)
        handle.write("\n")
    write_events_csv(report["issues"], report_dir / "events.csv")
    (report_dir / "summary.md").write_text(render_markdown(report), encoding="utf-8")


def write_events_csv(issues: list[dict[str, Any]], path: Path) -> None:
    fields = [field.name for field in Issue.__dataclass_fields__.values()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(issues)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    dataset = report["dataset"]
    lines = [
        f"# Dataset Quality Report: {dataset['name']}",
        "",
        f"- Source: `{report['source_dataset']}`",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Status: **{summary['status']}**",
        f"- Episodes: `{dataset['declared_episodes']}`",
        f"- Frames: `{dataset['actual_frames']}`",
        f"- FPS: `{dataset['fps']:g}`",
        "",
        "Reports and repair plans are external artifacts. They are not stored in the source dataset.",
        "",
        "## Severity Counts",
        "",
        "| Severity | Count |",
        "|---|---:|",
    ]
    for severity in ("ERROR", "UNSAFE", "REVIEW", "INFO"):
        lines.append(f"| {severity} | {summary['severity_counts'].get(severity, 0)} |")
    lines.extend(["", "## Finding Counts", "", "| Code | Count |", "|---|---:|"])
    for code, count in summary["code_counts"].items():
        lines.append(f"| `{code}` | {count} |")
    lines.extend(
        [
            "",
            "## Episode Summary",
            "",
            "| Episode | Frames | Severity | Findings |",
            "|---:|---:|---|---|",
        ]
    )
    for episode_index, metrics in report["episodes"].items():
        findings = ", ".join(
            f"{code}={count}" for code, count in metrics.get("issue_counts", {}).items()
        )
        lines.append(
            f"| {episode_index} | {metrics['frames']} | {metrics['max_severity']} | {findings or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Highest-Risk Samples",
            "",
            "| Severity | Code | Episode | Frame | Joint | Value | Limit | Ratio |",
            "|---|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    sorted_issues = sorted(
        report["issues"],
        key=lambda issue: (
            SEVERITY_ORDER[issue["severity"]],
            issue["ratio"] if issue["ratio"] is not None else 0.0,
        ),
        reverse=True,
    )
    for issue in sorted_issues[:100]:
        joint = "-"
        if issue["arm"] is not None:
            joint = issue["arm"]
            if issue["joint"] is not None:
                joint += f".q{issue['joint']}"
        lines.append(
            "| {severity} | `{code}` | {episode} | {frame} | {joint} | {value} | {limit} | {ratio} |".format(
                severity=issue["severity"],
                code=issue["code"],
                episode=issue["episode_index"] if issue["episode_index"] is not None else "-",
                frame=issue["frame_index"] if issue["frame_index"] is not None else "-",
                joint=joint,
                value=format_optional(issue["value"]),
                limit=format_optional(issue["limit"]),
                ratio=format_optional(issue["ratio"]),
            )
        )
    repair_plan = build_repair_plan(report)
    lines.extend(
        [
            "",
            "## Repair Suggestions",
            "",
            "| Episode | Severity | Recommended | Available choices |",
            "|---:|---|---|---|",
        ]
    )
    for entry in repair_plan["episodes"]:
        lines.append(
            f"| {entry['episode_index']} | {entry['max_severity']} | "
            f"`{entry['recommended_action']}` | "
            f"{', '.join(f'`{choice}`' for choice in entry['allowed_actions'])} |"
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Review `repair_plan.json`. No repair is applied until `selected_action` is set and the `apply` subcommand is run with a new output directory.",
            "",
        ]
    )
    return "\n".join(lines)


def format_optional(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6g}"


def review_plan(plan_path: Path) -> None:
    plan = load_json(plan_path)
    changed = False
    for entry in plan.get("episodes", []):
        choices = entry["allowed_actions"]
        print("\n" + "-" * 72)
        print(f"Episode {entry['episode_index']} [{entry['max_severity']}]")
        print("Findings:")
        for code, count in entry["finding_counts"].items():
            print(f"  {code}: {count}")
        print(f"Recommended: {entry['recommended_action']}")
        for index, choice in enumerate(choices, start=1):
            print(f"  {index}. {choice}")
        print("  0. leave unresolved")
        while True:
            answer = input("Select action: ").strip()
            try:
                selected_index = int(answer)
            except ValueError:
                print("Enter one of the displayed numbers.")
                continue
            if selected_index == 0:
                selected = None
                break
            if 1 <= selected_index <= len(choices):
                selected = choices[selected_index - 1]
                break
            print("Enter one of the displayed numbers.")
        if selected != entry.get("selected_action"):
            entry["selected_action"] = selected
            changed = True
    if changed:
        with plan_path.open("w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, ensure_ascii=True, default=json_default)
            handle.write("\n")
        print(f"\nUpdated plan: {plan_path}")
    else:
        print("\nPlan unchanged.")


def rate_limit_arm_targets(
    raw_targets: Any,
    initial_state: Any,
    fps: float,
    thresholds: Thresholds,
    np: Any,
):
    lower = np.asarray(FR3_JOINT_LOWER_RAD) + thresholds.position_margin_rad
    upper = np.asarray(FR3_JOINT_UPPER_RAD) - thresholds.position_margin_rad
    max_velocity = np.asarray(FR3_MAX_VELOCITY_RAD_S) * thresholds.safety_factor
    max_acceleration = np.asarray(FR3_MAX_ACCELERATION_RAD_S2) * thresholds.safety_factor
    dt = 1.0 / fps
    safe = np.empty_like(raw_targets, dtype=np.float64)
    previous = np.clip(np.asarray(initial_state, dtype=np.float64), lower, upper)
    velocity = np.zeros(7, dtype=np.float64)
    for index, raw_target in enumerate(raw_targets):
        target = np.clip(raw_target, lower, upper)
        error = target - previous
        stopping_velocity = np.sign(error) * np.minimum(
            max_velocity,
            np.sqrt(np.maximum(0.0, 2.0 * max_acceleration * np.abs(error))),
        )
        velocity += np.clip(
            stopping_velocity - velocity,
            -max_acceleration * dt,
            max_acceleration * dt,
        )
        velocity = np.clip(velocity, -max_velocity, max_velocity)
        candidate = previous + velocity * dt
        actual_velocity = (candidate - previous) / dt
        previous = np.clip(candidate, lower, upper)
        velocity = actual_velocity
        safe[index] = previous
    return safe


def compute_repaired_actions(
    state: Any,
    action: Any,
    fps: float,
    thresholds: Thresholds,
    np: Any,
):
    repaired = action.copy()
    state_specs = arm_slices(state.shape[1])
    action_specs = arm_slices(action.shape[1])
    if len(state_specs) != len(action_specs) or not state_specs:
        raise ValueError(
            f"Cannot repair incompatible state/action dimensions: {state.shape[1]}/{action.shape[1]}"
        )
    for (_, state_slice), (_, action_slice) in zip(state_specs, action_specs, strict=True):
        repaired[:, action_slice] = rate_limit_arm_targets(
            action[:, action_slice],
            state[0, state_slice],
            fps,
            thresholds,
            np,
        )
        validate_repaired_arm_targets(
            repaired[:, action_slice],
            state[0, state_slice],
            fps,
            thresholds,
            np,
        )
    return repaired


def validate_repaired_arm_targets(
    targets: Any,
    initial_state: Any,
    fps: float,
    thresholds: Thresholds,
    np: Any,
) -> None:
    lower = np.asarray(FR3_JOINT_LOWER_RAD) + thresholds.position_margin_rad
    upper = np.asarray(FR3_JOINT_UPPER_RAD) - thresholds.position_margin_rad
    max_velocity = np.asarray(FR3_MAX_VELOCITY_RAD_S) * thresholds.safety_factor
    max_acceleration = np.asarray(FR3_MAX_ACCELERATION_RAD_S2) * thresholds.safety_factor
    tolerance = 1e-8
    if np.any(targets < lower - tolerance) or np.any(targets > upper + tolerance):
        raise RuntimeError("Constrained repair produced a target outside the position envelope.")
    points = np.vstack([np.asarray(initial_state, dtype=float), targets])
    velocities = np.diff(points, axis=0) * fps
    accelerations = np.diff(np.vstack([np.zeros((1, 7)), velocities]), axis=0) * fps
    if np.any(np.abs(velocities) > max_velocity + tolerance):
        raise RuntimeError("Constrained repair produced a target velocity above its limit.")
    if np.any(np.abs(accelerations) > max_acceleration + tolerance):
        raise RuntimeError("Constrained repair produced a target acceleration above its limit.")


def rewrite_selected_actions(
    dataset_root: Path,
    episode_indices: set[int],
    thresholds: Thresholds,
    max_correction_rad: float,
    allow_large_repair: bool,
) -> dict[int, float]:
    np, pa, pq = require_numeric_dependencies()
    info = load_json(dataset_root / INFO_PATH)
    fps = float(info.get("fps", 15.0))
    paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    table_by_path = {path: pq.read_table(path) for path in paths}
    rows_by_episode: dict[int, list[tuple[Path, int, int, Any, Any]]] = defaultdict(list)
    for path, table in table_by_path.items():
        episodes = table["episode_index"].to_pylist()
        frames = table["frame_index"].to_pylist()
        states = table["observation.state"].to_pylist()
        actions = table["action"].to_pylist()
        for row, (episode_index, frame_index, state, action) in enumerate(
            zip(episodes, frames, states, actions, strict=True)
        ):
            episode_index = int(episode_index)
            if episode_index in episode_indices:
                rows_by_episode[episode_index].append(
                    (path, row, int(frame_index), state, action)
                )

    replacements: dict[tuple[Path, int], Any] = {}
    corrections: dict[int, float] = {}
    for episode_index in sorted(episode_indices):
        rows = sorted(rows_by_episode.get(episode_index, []), key=lambda item: item[2])
        if not rows:
            raise ValueError(f"Episode {episode_index} was selected for repair but has no rows.")
        state = np.asarray([item[3] for item in rows], dtype=np.float64)
        action = np.asarray([item[4] for item in rows], dtype=np.float64)
        repaired = compute_repaired_actions(state, action, fps, thresholds, np)
        correction = float(np.max(np.abs(repaired - action)))
        corrections[episode_index] = correction
        if correction > max_correction_rad and not allow_large_repair:
            raise ValueError(
                f"Episode {episode_index} requires a {correction:.6f} rad action correction, "
                f"above the allowed {max_correction_rad:.6f} rad. Exclude/re-record it or "
                "rerun with --allow-large-action-repair after manual review."
            )
        for item, replacement in zip(rows, repaired, strict=True):
            replacements[(item[0], item[1])] = replacement

    for path, table in table_by_path.items():
        selected_rows = [row for (selected_path, row) in replacements if selected_path == path]
        if not selected_rows:
            continue
        actions = table["action"].to_pylist()
        for row in selected_rows:
            actions[row] = replacements[(path, row)].tolist()
        action_index = table.schema.get_field_index("action")
        action_type = table.schema.field(action_index).type
        action_column = pa.array(actions, type=action_type)
        rewritten = table.set_column(action_index, "action", action_column)
        temporary_path = path.with_suffix(path.suffix + ".quality-rewrite.tmp")
        pq.write_table(rewritten, temporary_path)
        os.replace(temporary_path, path)
    return corrections


def apply_plan(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan).expanduser().resolve()
    plan = load_json(plan_path)
    source_root = require_dataset_root(Path(plan["source_dataset"]))
    output_root = Path(args.output_dir).expanduser().resolve()
    require_external_path(output_root, source_root, "Output dataset")
    if path_is_within(source_root, output_root):
        raise ValueError("Output dataset cannot be a parent of the source dataset.")
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    if dataset_fingerprint(source_root) != plan.get("source_fingerprint"):
        raise RuntimeError(
            "The source dataset changed after the report was generated. Run scan again."
        )
    if args.max_action_correction_rad <= 0.0:
        raise ValueError("--max-action-correction-rad must be positive.")

    entries = plan.get("episodes", [])
    unresolved = [
        entry
        for entry in entries
        if entry.get("selected_action") is None
        and entry.get("max_severity") in {"ERROR", "UNSAFE"}
    ]
    if unresolved and not args.allow_unresolved:
        indices = [entry["episode_index"] for entry in unresolved]
        raise ValueError(
            f"ERROR/UNSAFE episodes remain unresolved: {indices}. Run review or pass "
            "--allow-unresolved after manual review."
        )

    selected: dict[int, str] = {}
    for entry in entries:
        action = entry.get("selected_action")
        if action is None:
            continue
        if action not in entry.get("allowed_actions", []):
            raise ValueError(
                f"Episode {entry['episode_index']} selected unsupported action: {action}"
            )
        if action == "rerecord":
            raise ValueError(
                f"Episode {entry['episode_index']} is marked rerecord. Replace or exclude it "
                "before applying a derived dataset."
            )
        if action not in AUTO_REPAIR_ACTIONS:
            raise ValueError(f"Automatic repair is not implemented for action: {action}")
        selected[int(entry["episode_index"])] = action

    exclude_original = sorted(
        episode for episode, action in selected.items() if action == "exclude_episode"
    )
    repair_original = sorted(
        episode for episode, action in selected.items() if action == "constrained_action_repair"
    )
    info = load_json(source_root / INFO_PATH)
    total_episodes = int(info["total_episodes"])
    if len(exclude_original) >= total_episodes:
        raise ValueError("Refusing to exclude every episode.")
    kept_original = [index for index in range(total_episodes) if index not in set(exclude_original)]
    episode_mapping = {old: new for new, old in enumerate(kept_original)}
    repair_output = {episode_mapping[index] for index in repair_original if index in episode_mapping}
    repo_id = args.repo_id or f"local/{output_root.name}"
    thresholds = Thresholds(**plan["thresholds"])
    thresholds.validate()

    print("Repair plan")
    print(f"  source: {source_root}")
    print(f"  output: {output_root}")
    print(f"  exclude episodes: {exclude_original or 'none'}")
    print(f"  constrained action repair: {repair_original or 'none'}")
    print("  source dataset will not be modified")

    try:
        if exclude_original:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from delete_lerobot_episode import (
                copy_optional_metadata,
                delete_episodes_local,
            )

            delete_episodes_local(
                source_root=source_root,
                output_root=output_root,
                repo_id=repo_id,
                episode_indices=exclude_original,
                dataset_info=info,
                video_workers=args.video_workers,
            )
            copy_optional_metadata(source_root, output_root)
        else:
            shutil.copytree(source_root, output_root)

        corrections = {}
        if repair_output:
            corrections = rewrite_selected_actions(
                output_root,
                repair_output,
                thresholds,
                args.max_action_correction_rad,
                args.allow_large_action_repair,
            )

        if exclude_original or repair_output:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from dataset_stats import ensure_dataset_stats

            ensure_dataset_stats(repo_id, output_root, force_recompute=True)
    except Exception:
        if output_root.exists() and not path_is_within(output_root, source_root):
            shutil.rmtree(output_root)
        raise

    post_report = scan_dataset(
        output_root,
        thresholds,
        check_video_frames=False,
    )
    post_report_dir = plan_path.parent / f"post_repair_{utc_stamp()}"
    require_external_path(post_report_dir, output_root, "Post-repair report directory")
    write_report(post_report, post_report_dir)
    print("\nRepair complete")
    print(f"  derived dataset: {output_root}")
    print(f"  post-repair report: {post_report_dir}")
    for episode_index, correction in sorted(corrections.items()):
        print(f"  repaired output episode {episode_index}: max correction {correction:.6f} rad")


def scan_command(args: argparse.Namespace) -> None:
    thresholds = thresholds_from_args(args)
    report_root = Path(args.report_root).expanduser().resolve()
    if args.dataset_root:
        datasets = [require_dataset_root(Path(args.dataset_root))]
        batch_root = report_root
    else:
        parent = Path(args.datasets_root).expanduser().resolve()
        datasets = []
        for candidate in sorted(parent.glob(args.name_pattern)):
            try:
                datasets.append(require_dataset_root(candidate))
            except (FileNotFoundError, PermissionError):
                continue
        if not datasets:
            raise FileNotFoundError(
                f"No readable LeRobot datasets matching {args.name_pattern!r} under {parent}"
            )
        batch_root = report_root / f"batch_{utc_stamp()}"

    for dataset_root in datasets:
        require_external_path(report_root, dataset_root, "Report root")
    for dataset_root in datasets:
        report = scan_dataset(
            dataset_root,
            thresholds,
            check_video_frames=args.check_video_frames,
        )
        report_dir = batch_root / dataset_root.name / utc_stamp()
        write_report(report, report_dir)
        print(f"{dataset_root.name}: {report['summary']['status']}")
        print(f"  report: {report_dir / 'summary.md'}")
        print(f"  repair plan: {report_dir / 'repair_plan.json'}")


def main() -> None:
    args = parse_args()
    if args.command == "scan":
        scan_command(args)
    elif args.command == "review":
        review_plan(Path(args.plan).expanduser().resolve())
    elif args.command == "apply":
        apply_plan(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
