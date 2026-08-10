"""Validate and process local LeRobot datasets.

Usage:
    python data_collection/process_dataset.py validate \
        --dataset-root data/my_dataset

Dataset-processing commands may create a new dataset or replace the input only
when explicitly requested. Review a dry run before modifying recorded data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INFO_PATH = Path("meta/info.json")
ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))


@dataclass(frozen=True)
class InitialTrim:
    episode_index: int
    old_length: int
    trim_frames: int

    @property
    def new_length(self) -> int:
        return self.old_length - self.trim_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and process a local LeRobot dataset."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["validate", "trim-initial"],
        default="validate",
        help="Dataset operation to run. Defaults to validate.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Sibling output dataset for trim-initial. Required only when not using the default.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Replace the source dataset for trim-initial after moving it to a backup.",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="Backup directory used with --in-place for trim-initial.",
    )
    parser.add_argument(
        "--episode-indices",
        nargs="+",
        default=None,
        help="Episodes for trim-initial, using comma/whitespace-separated values and ranges.",
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.002,
        help="Maximum arm-joint displacement in rad/frame considered static. Default: 0.002.",
    )
    parser.add_argument(
        "--min-static-frames",
        type=int,
        default=5,
        help="Minimum initial static frames required before trimming. Default: 5.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the trim plan without modifying files.",
    )
    parser.add_argument(
        "--skip-video-frames",
        action="store_true",
        help="Skip physical MP4 frame count checks.",
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
    if isinstance(value, list):
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


def parse_episode_selection(text: str) -> list[int]:
    tokens = [token for token in text.replace(",", " ").split() if token]
    if not tokens:
        raise ValueError("--episode-indices must contain at least one episode.")
    selected: set[int] = set()
    for token in tokens:
        parts = token.split("-")
        if token.isdigit():
            selected.add(int(token))
        elif len(parts) == 2 and all(part.isdigit() for part in parts):
            start, end = map(int, parts)
            if start > end:
                raise ValueError(f"Invalid episode range {token!r}: start must be <= end.")
            selected.update(range(start, end + 1))
        else:
            raise ValueError(
                f"Invalid episode token {token!r}. Use non-negative integers or ranges like 4-8."
            )
    return sorted(selected)


def detect_initial_trim(rows: list[dict[str, Any]], threshold: float, min_static_frames: int) -> int:
    if threshold < 0:
        raise ValueError("--motion-threshold must be non-negative.")
    if min_static_frames < 1:
        raise ValueError("--min-static-frames must be positive.")
    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    states = [flatten_numeric(row.get("observation.state")) for row in ordered]
    if not states or len(states[0]) != 16:
        return 0
    static_transitions = 0
    for previous, current in zip(states, states[1:], strict=True):
        if len(current) != 16 or len(previous) != 16:
            break
        arm_delta = max(
            [abs(current[index] - previous[index]) for index in range(0, 7)]
            + [abs(current[index] - previous[index]) for index in range(8, 15)]
        )
        if arm_delta > threshold:
            break
        static_transitions += 1
    candidate = static_transitions + 1
    return candidate if candidate >= min_static_frames and candidate < len(ordered) else 0


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
) -> tuple[list[str], dict[str, Any]]:
    issues: list[str] = []
    metrics: dict[str, Any] = {
        "max_left_arm_delta": 0.0,
        "max_right_arm_delta": 0.0,
        "delta_action_max_error": 0.0,
        "delta_action_bad_frames": 0,
        "arm_action_outlier_frames": [],
        "gripper_outlier_frames": [],
        "non_finite_frames": [],
    }

    sorted_rows = sorted(rows, key=lambda row: int(row["frame_index"]))
    states: list[list[float]] = []
    actions: list[list[float]] = []
    frame_indices: list[int] = []

    for row in sorted_rows:
        frame_index = int(row["frame_index"])
        state = flatten_numeric(row.get("observation.state"))
        action = flatten_numeric(row.get("action"))
        states.append(state)
        actions.append(action)
        frame_indices.append(frame_index)

        if has_non_finite(state) or has_non_finite(action):
            metrics["non_finite_frames"].append(frame_index)

        if len(state) >= 16:
            for value in (state[7], state[15]):
                if value < gripper_min - gripper_tolerance or value > gripper_max + gripper_tolerance:
                    metrics["gripper_outlier_frames"].append(frame_index)
                    break

        if len(action) >= 16:
            for value in (action[7], action[15]):
                if value < gripper_min - gripper_tolerance or value > gripper_max + gripper_tolerance:
                    metrics["gripper_outlier_frames"].append(frame_index)
                    break

            left_max = max((abs(value) for value in action[0:7]), default=0.0)
            right_max = max((abs(value) for value in action[8:15]), default=0.0)
            metrics["max_left_arm_delta"] = max(metrics["max_left_arm_delta"], left_max)
            metrics["max_right_arm_delta"] = max(metrics["max_right_arm_delta"], right_max)
            if (
                arm_action_representation == "delta_joint_position"
                and (left_max > action_outlier_threshold or right_max > action_outlier_threshold)
            ):
                metrics["arm_action_outlier_frames"].append(
                    {
                        "frame_index": frame_index,
                        "left_max": left_max,
                        "right_max": right_max,
                    }
                )

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

            if len(state) < 16 or len(next_state) < 16 or len(action) < 16:
                continue

            expected_left = [next_state[j] - state[j] for j in range(0, 7)]
            expected_right = [next_state[j] - state[j] for j in range(8, 15)]
            actual_left = action[0:7]
            actual_right = action[8:15]
            frame_error = max(
                [abs(actual_left[j] - expected_left[j]) for j in range(7)]
                + [abs(actual_right[j] - expected_right[j]) for j in range(7)]
            )
            metrics["delta_action_max_error"] = max(metrics["delta_action_max_error"], frame_error)
            if frame_error > delta_action_tolerance:
                metrics["delta_action_bad_frames"] += 1

        if metrics["delta_action_bad_frames"]:
            issues.append(
                f"delta-action check failed on {metrics['delta_action_bad_frames']} frame(s); "
                f"max error {metrics['delta_action_max_error']:.6g} > tolerance {delta_action_tolerance}"
            )
    elif arm_action_representation != "absolute_joint_position":
        issues.append(f"unsupported arm action representation '{arm_action_representation}'")

    return issues, metrics


def check_physical_video_frames(
    dataset_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    video_keys: list[str],
) -> list[str]:
    try:
        import cv2
    except ModuleNotFoundError:
        return [
            "physical video frame checks skipped because OpenCV (cv2) is not installed"
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

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                issues.append(f"{video_key} file {chunk_index}/{file_index}: failed to open {video_path}")
                continue

            actual_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            actual_fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()

            if actual_frames != expected_frames:
                issues.append(
                    f"{video_key} file {chunk_index}/{file_index}: "
                    f"physical frames {actual_frames} != expected {expected_frames}"
                )
            if actual_fps and abs(actual_fps - float(info["fps"])) > 1e-3:
                issues.append(
                    f"{video_key} file {chunk_index}/{file_index}: "
                    f"physical fps {actual_fps:.6f} != dataset fps {float(info['fps']):.6f}"
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
    action_config = load_action_config(dataset_root)
    arm_action_representation = str(
        (action_config or {}).get("arm_action_representation", "absolute_joint_position")
    ).strip().lower()
    fps = float(info["fps"])
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    state_dim = get_feature_dim(info, "observation.state")
    action_dim = get_feature_dim(info, "action")
    video_keys = get_video_keys(info)

    episodes = load_parquet_rows(dataset_root / "meta/episodes")
    episodes.sort(key=lambda row: int(row["episode_index"]))
    data_rows = load_parquet_rows(dataset_root / "data")
    data_by_episode = build_data_index(data_rows)

    issues: list[str] = []
    warning_issues: list[str] = []

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
    total_non_finite_frames = 0

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
        )
        episode_issues.extend(semantic_issues)

        max_left_arm_delta = max(max_left_arm_delta, float(semantic_metrics["max_left_arm_delta"]))
        max_right_arm_delta = max(max_right_arm_delta, float(semantic_metrics["max_right_arm_delta"]))
        max_delta_action_error = max(max_delta_action_error, float(semantic_metrics["delta_action_max_error"]))
        total_delta_action_bad_frames += int(semantic_metrics["delta_action_bad_frames"])
        total_arm_action_outlier_frames += len(semantic_metrics["arm_action_outlier_frames"])
        total_gripper_outlier_frames += len(set(semantic_metrics["gripper_outlier_frames"]))
        total_non_finite_frames += len(set(semantic_metrics["non_finite_frames"]))

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
    print(f"  gripper valid range: [{gripper_min:.6g}, {gripper_max:.6g}] +/- {gripper_tolerance:.6g}")
    print(f"  gripper outlier frames: {total_gripper_outlier_frames}")
    print(f"  non-finite state/action frames: {total_non_finite_frames}")
    if arm_action_representation == "delta_joint_position":
        print(f"  delta-action tolerance: {delta_action_tolerance:.6g}")
        print(f"  max delta-action error: {max_delta_action_error:.6g}")
        print(f"  delta-action bad frames: {total_delta_action_bad_frames}")
    else:
        print("  delta-action consistency check: skipped for absolute_joint_position actions")

    if not skip_video_frames and video_keys:
        print("\nPhysical video checks")
        physical_video_issues = check_physical_video_frames(dataset_root, info, episodes, video_keys)
        for issue in physical_video_issues:
            if issue.startswith("physical video frame checks skipped"):
                warning_issues.append(issue)
            else:
                issues.append(issue)
        if not physical_video_issues:
            print("  ok")
        else:
            for issue in physical_video_issues:
                prefix = "  warning:" if issue.startswith("physical video frame checks skipped") else "  issue:"
                print(f"{prefix} {issue}")
    elif skip_video_frames:
        print("\nPhysical video checks skipped by --skip-video-frames")

    print("\nValidation summary")
    if warning_issues:
        print(f"  warnings: {len(warning_issues)}")
    if issues:
        print(f"  status: FAILED")
        print(f"  issues: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("  status: PASS")
    return 0


def trim_initial_static_segments(args: argparse.Namespace) -> int:
    """Detect and rebuild datasets after removing initial static arm frames."""
    try:
        import pandas as pd
        from dataset_stats import ensure_dataset_stats
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.dataset_tools import _keep_episodes_from_video_with_av, _write_parquet
        from lerobot.datasets.io_utils import load_episodes, write_info
    except ModuleNotFoundError as exc:
        raise RuntimeError("LeRobot dataset-processing dependencies are required for trim-initial.") from exc

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    info = load_json(dataset_root / INFO_PATH)
    fps = float(info["fps"])
    source_meta = LeRobotDatasetMetadata(
        repo_id=f"local/{dataset_root.name}",
        root=dataset_root,
        revision=str(info.get("codebase_version", "v3.0")),
    )
    if source_meta.episodes is None:
        source_meta.episodes = load_episodes(dataset_root)
    episode_indices = (
        parse_episode_selection(" ".join(args.episode_indices))
        if args.episode_indices
        else [int(episode["episode_index"]) for episode in source_meta.episodes]
    )
    available = {int(episode["episode_index"]) for episode in source_meta.episodes}
    invalid = sorted(set(episode_indices) - available)
    if invalid:
        raise ValueError(f"Episode indices not found in dataset: {invalid}.")

    rows_by_episode: dict[int, list[dict[str, Any]]] = {index: [] for index in available}
    for parquet_file in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        for row in pd.read_parquet(parquet_file).to_dict(orient="records"):
            rows_by_episode[int(row["episode_index"])].append(row)
    trim_plan: dict[int, InitialTrim] = {}
    for episode in source_meta.episodes:
        index = int(episode["episode_index"])
        old_length = int(episode["length"])
        trim_frames = detect_initial_trim(rows_by_episode[index], args.motion_threshold, args.min_static_frames) if index in episode_indices else 0
        trim_plan[index] = InitialTrim(index, old_length, trim_frames)

    print("Initial static-segment trim plan")
    for index in episode_indices:
        trim = trim_plan[index]
        print(f"  episode {index}: remove {trim.trim_frames} frame(s), {trim.old_length} -> {trim.new_length}")
    if args.dry_run:
        print("Dry run complete. No files were changed.")
        return 0

    if args.in_place and args.output_dir:
        raise ValueError("--output-dir cannot be used with --in-place.")
    if args.backup_dir and not args.in_place:
        raise ValueError("--backup-dir requires --in-place.")
    output_root = dataset_root if args.in_place else Path(args.output_dir).expanduser().resolve() if args.output_dir else dataset_root.with_name(f"{dataset_root.name}_trimmed")
    backup_root = Path(args.backup_dir).expanduser().resolve() if args.backup_dir else dataset_root.with_name(f"{dataset_root.name}_backup_before_trim")
    if output_root.exists() and output_root != dataset_root:
        raise FileExistsError(f"Output directory already exists: {output_root}")
    source_root = dataset_root
    moved = False
    try:
        if args.in_place:
            if backup_root.exists():
                raise FileExistsError(f"Backup directory already exists: {backup_root}")
            shutil.move(str(dataset_root), str(backup_root))
            source_root = backup_root
            moved = True
        source_meta = LeRobotDatasetMetadata(repo_id=f"local/{source_root.name}", root=source_root, revision=str(info.get("codebase_version", "v3.0")))
        if source_meta.episodes is None:
            source_meta.episodes = load_episodes(source_root)
        new_meta = LeRobotDatasetMetadata.create(repo_id=f"local/{output_root.name}", fps=source_meta.fps, features=source_meta.features, robot_type=source_meta.robot_type, root=output_root, use_videos=bool(source_meta.video_keys), chunks_size=source_meta.chunks_size, data_files_size_in_mb=source_meta.data_files_size_in_mb, video_files_size_in_mb=source_meta.video_files_size_in_mb)
        if source_meta.tasks is not None:
            new_meta.save_episode_tasks(list(source_meta.tasks.index))
        video_metadata: dict[int, dict[str, Any]] = {int(ep["episode_index"]): {} for ep in source_meta.episodes}
        for video_key in source_meta.video_keys:
            files: dict[tuple[int, int], list[int]] = {}
            for episode in source_meta.episodes:
                index = int(episode["episode_index"])
                chunk = int(episode[f"videos/{video_key}/chunk_index"])
                file_index = int(episode[f"videos/{video_key}/file_index"])
                files.setdefault((chunk, file_index), []).append(index)
            for (chunk, file_index), episode_ids in files.items():
                source_video = source_root / source_meta.video_path.format(video_key=video_key, chunk_index=chunk, file_index=file_index)
                destination_video = output_root / new_meta.video_path.format(video_key=video_key, chunk_index=chunk, file_index=file_index)
                destination_video.parent.mkdir(parents=True, exist_ok=True)
                ranges: list[tuple[int, int]] = []
                cumulative = 0.0
                for index in sorted(episode_ids):
                    episode = source_meta.episodes[index]
                    start = round(float(episode[f"videos/{video_key}/from_timestamp"]) * source_meta.fps)
                    end = round(float(episode[f"videos/{video_key}/to_timestamp"]) * source_meta.fps)
                    trim = trim_plan[index].trim_frames
                    ranges.append((start + trim, end))
                _keep_episodes_from_video_with_av(source_video, destination_video, ranges, source_meta.fps)
                for index in sorted(episode_ids):
                    trim = trim_plan[index]
                    duration = trim.new_length / source_meta.fps
                    video_metadata[index].update({
                        f"videos/{video_key}/chunk_index": chunk,
                        f"videos/{video_key}/file_index": file_index,
                        f"videos/{video_key}/from_timestamp": cumulative,
                        f"videos/{video_key}/to_timestamp": cumulative + duration,
                    })
                    cumulative += duration
        global_index = 0
        data_metadata: dict[int, dict[str, Any]] = {}
        for data_file in sorted((source_root / "data").glob("chunk-*/*.parquet")):
            frame_df = pd.read_parquet(data_file)
            kept_parts = []
            for index, group in frame_df.groupby("episode_index", sort=True):
                trim = trim_plan[int(index)]
                group = group.sort_values("frame_index").iloc[trim.trim_frames:].copy()
                group["frame_index"] = range(len(group))
                group["timestamp"] = [frame / source_meta.fps for frame in range(len(group))]
                group["index"] = range(global_index, global_index + len(group))
                kept_parts.append(group)
                data_metadata[int(index)] = {"data/chunk_index": int(data_file.parent.name.split("-")[-1]), "data/file_index": int(data_file.stem.split("-")[-1]), "dataset_from_index": int(group["index"].min()), "dataset_to_index": int(group["index"].max() + 1)}
                global_index += len(group)
            if kept_parts:
                destination = output_root / "data" / data_file.parent.name / data_file.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                _write_parquet(pd.concat(kept_parts, ignore_index=True), destination, new_meta)
        for episode in sorted(source_meta.episodes, key=lambda item: int(item["episode_index"])):
            index = int(episode["episode_index"])
            trim = trim_plan[index]
            metadata = {
                "episode_index": index,
                "tasks": episode["tasks"],
                "length": trim.new_length,
                **data_metadata[index],
                **video_metadata[index],
            }
            new_meta._save_episode_metadata(metadata)
        new_meta.finalize()
        new_meta.info.update({"total_episodes": len(trim_plan), "total_frames": sum(trim.new_length for trim in trim_plan.values()), "splits": {"train": f"0:{len(trim_plan)}"}})
        write_info(new_meta.info, output_root)
        source_action_config = source_root / ACTION_CONFIG_PATH
        if source_action_config.exists():
            target_action_config = output_root / ACTION_CONFIG_PATH
            target_action_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_action_config, target_action_config)
        ensure_dataset_stats(f"local/{output_root.name}", output_root, force_recompute=True)
    except Exception:
        if output_root.exists() and output_root != dataset_root:
            shutil.rmtree(output_root)
        if moved and backup_root.exists():
            if dataset_root.exists():
                shutil.rmtree(dataset_root)
            shutil.move(str(backup_root), str(dataset_root))
        raise
    print(f"Finished trimming initial static segments. Output: {output_root}")
    return 0


def main() -> None:
    args = parse_args()
    if args.command == "trim-initial":
        raise SystemExit(trim_initial_static_segments(args))
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
