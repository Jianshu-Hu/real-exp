"""Trim initial static segments from local LeRobot datasets.

Usage:
    python data_collection/process_dataset.py trim-initial \
        --dataset-root data/my_dataset

The source is renamed to a sibling `<dataset>_backup` directory and the
processed dataset is rebuilt at the source's original path. Review a dry run
before modifying recorded data.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


INFO_PATH = Path("meta/info.json")
ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
TRAJECTORY_CONFIG_PATH = Path("meta/real_exp_trajectory_config.json")
PROCESSED_FLAG = "processed"
REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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
        description="Trim initial static segments from a local LeRobot dataset."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["trim-initial"],
        default="trim-initial",
        help="Dataset operation to run. Defaults to trim-initial.",
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--episode-indices",
        nargs="+",
        default=None,
        help="Episodes to process, using comma/whitespace-separated values and ranges.",
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
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def move_dataset_to_backup(dataset_root: Path) -> Path:
    backup_root = dataset_root.with_name(f"{dataset_root.name}_backup")
    if backup_root.exists():
        raise FileExistsError(
            f"Backup directory already exists: {backup_root}. "
            "Move or remove it before processing the dataset again."
        )
    dataset_root.rename(backup_root)
    return backup_root


def restore_dataset_from_backup(dataset_root: Path, backup_root: Path) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    backup_root.rename(dataset_root)


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


def motion_indices_from_metadata(
    dataset_root: Path, info: dict[str, Any]
) -> tuple[int, ...]:
    """Return state indices belonging to the active arm joints.

    Real-exp trajectories are laid out as one block per active arm. Each block
    starts with seven arm joints, followed by an optional gripper (one value)
    or Wuji hand (twenty values). The explicit trajectory metadata is
    authoritative for the block layout.
    """
    features = info.get("features", {})
    try:
        state_dim = int(features["observation.state"]["shape"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not determine observation.state dimension from {dataset_root / INFO_PATH}."
        ) from exc

    try:
        from utils.trajectory_metadata import require_dataset_trajectory_config

        trajectory_config = require_dataset_trajectory_config(dataset_root)
    except (FileNotFoundError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid trajectory metadata for {dataset_root}; cannot determine state layout."
        ) from exc

    arms = trajectory_config.get("arms")
    if not isinstance(arms, list) or not arms:
        arm_mode = str(trajectory_config.get("arm_mode", "")).strip().lower()
        arms = ["left", "right"] if arm_mode == "duo" else [arm_mode]
    if any(str(arm).strip().lower() not in {"left", "right"} for arm in arms):
        raise ValueError(f"Invalid active arm combination in trajectory metadata: {arms!r}.")
    arms = [str(arm).strip().lower() for arm in arms]

    end_effector = str(trajectory_config.get("end_effector", "arm")).strip().lower()
    block_size = 7 + {"arm": 0, "gripper": 1, "hand": 20}.get(end_effector, -1)
    if block_size < 7:
        raise ValueError(
            f"Invalid end-effector mode {end_effector!r} in trajectory metadata; "
            "expected arm, gripper, or hand."
        )
    expected_dim = block_size * len(arms)
    if state_dim != expected_dim:
        raise ValueError(
            "Trajectory metadata does not match observation.state: "
            f"{len(arms)} {end_effector} arm block(s) require {expected_dim} values, "
            f"but metadata declares {state_dim}."
        )
    return tuple(
        arm_index * block_size + joint_index
        for arm_index in range(len(arms))
        for joint_index in range(7)
    )


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


def detect_initial_trim(
    rows: list[dict[str, Any]],
    threshold: float,
    min_static_frames: int,
    motion_indices: Sequence[int] | None = None,
) -> int:
    if threshold < 0:
        raise ValueError("--motion-threshold must be non-negative.")
    if min_static_frames < 1:
        raise ValueError("--min-static-frames must be positive.")
    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    states = [flatten_numeric(row.get("observation.state")) for row in ordered]
    if not states:
        raise ValueError("Static detection requires observation.state for every frame.")
    if motion_indices is None:
        # Preserve the old helper behavior for callers that do not have dataset
        # metadata. Dataset processing always supplies metadata-derived indices.
        if any(len(state) != 16 for state in states):
            raise ValueError(
                "Static detection requires trajectory metadata for non-16-D observation.state."
            )
        motion_indices = tuple(range(7)) + tuple(range(8, 15))
    motion_indices = tuple(motion_indices)
    if not motion_indices or min(motion_indices) < 0:
        raise ValueError("Static detection requires at least one valid arm-joint index.")
    state_dim = len(states[0])
    if any(len(state) != state_dim for state in states):
        raise ValueError("Static detection requires a consistent observation.state dimension.")
    if max(motion_indices) >= state_dim:
        raise ValueError(
            f"Static detection arm-joint index {max(motion_indices)} exceeds "
            f"observation.state dimension {state_dim}."
        )
    if any(has_non_finite(state) for state in states):
        raise ValueError("Static detection requires finite observation.state values.")

    static_transitions = 0
    for previous, current in zip(states, states[1:], strict=True):
        arm_delta = max(abs(current[index] - previous[index]) for index in motion_indices)
        if arm_delta > threshold:
            break
        static_transitions += 1
    candidate = static_transitions + 1
    return candidate if candidate >= min_static_frames and candidate < len(ordered) else 0


def trim_initial_static_segments(args: argparse.Namespace) -> int:
    """Replace a dataset with a trimmed copy and retain the original as a backup."""
    try:
        import pandas as pd
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.dataset_tools import (
            _keep_episodes_from_video_with_av,
            _write_parquet,
        )
        from lerobot.datasets.io_utils import load_episodes, write_info
        from utils.dataset_stats import ensure_dataset_stats
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LeRobot dataset-processing dependencies are required for trim-initial."
        ) from exc

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not (dataset_root / INFO_PATH).exists():
        raise FileNotFoundError(
            f"{dataset_root} is not a LeRobot dataset root. Missing {INFO_PATH}."
        )
    info = load_json(dataset_root / INFO_PATH)
    motion_indices = motion_indices_from_metadata(dataset_root, info)
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
        if len(rows_by_episode[index]) != old_length:
            raise ValueError(
                f"Episode {index} metadata declares {old_length} frames, "
                f"but {len(rows_by_episode[index])} data rows were found."
            )
        trim_frames = (
            detect_initial_trim(
                rows_by_episode[index],
                args.motion_threshold,
                args.min_static_frames,
                motion_indices,
            )
            if index in episode_indices
            else 0
        )
        trim_plan[index] = InitialTrim(index, old_length, trim_frames)

    print("Initial static-segment trim plan")
    for index in episode_indices:
        trim = trim_plan[index]
        print(
            f"  episode {index}: remove {trim.trim_frames} frame(s), "
            f"{trim.old_length} -> {trim.new_length}"
        )
    output_root = dataset_root
    backup_root = dataset_root.with_name(f"{dataset_root.name}_backup")
    print(f"  original dataset backup: {backup_root}")
    print(f"  processed dataset output: {output_root}")
    if args.dry_run:
        print("Dry run complete. No files were changed.")
        return 0

    source_root = backup_root
    moved = False
    try:
        move_dataset_to_backup(dataset_root)
        moved = True
        source_meta = LeRobotDatasetMetadata(
            repo_id=f"local/{source_root.name}",
            root=source_root,
            revision=str(info.get("codebase_version", "v3.0")),
        )
        if source_meta.episodes is None:
            source_meta.episodes = load_episodes(source_root)
        new_meta = LeRobotDatasetMetadata.create(
            repo_id=f"local/{output_root.name}",
            fps=source_meta.fps,
            features=source_meta.features,
            robot_type=source_meta.robot_type,
            root=output_root,
            use_videos=bool(source_meta.video_keys),
            chunks_size=source_meta.chunks_size,
            data_files_size_in_mb=source_meta.data_files_size_in_mb,
            video_files_size_in_mb=source_meta.video_files_size_in_mb,
        )
        if source_meta.tasks is not None:
            new_meta.save_episode_tasks(list(source_meta.tasks.index))

        video_metadata: dict[int, dict[str, Any]] = {
            int(episode["episode_index"]): {} for episode in source_meta.episodes
        }
        for video_key in source_meta.video_keys:
            files: dict[tuple[int, int], list[int]] = {}
            for episode in source_meta.episodes:
                index = int(episode["episode_index"])
                chunk = int(episode[f"videos/{video_key}/chunk_index"])
                file_index = int(episode[f"videos/{video_key}/file_index"])
                files.setdefault((chunk, file_index), []).append(index)
            for (chunk, file_index), episode_ids in files.items():
                source_video = source_root / source_meta.video_path.format(
                    video_key=video_key,
                    chunk_index=chunk,
                    file_index=file_index,
                )
                destination_video = output_root / new_meta.video_path.format(
                    video_key=video_key,
                    chunk_index=chunk,
                    file_index=file_index,
                )
                destination_video.parent.mkdir(parents=True, exist_ok=True)
                ranges: list[tuple[int, int]] = []
                cumulative = 0.0
                for index in sorted(episode_ids):
                    episode = source_meta.episodes[index]
                    start = round(
                        float(episode[f"videos/{video_key}/from_timestamp"])
                        * source_meta.fps
                    )
                    end = round(
                        float(episode[f"videos/{video_key}/to_timestamp"])
                        * source_meta.fps
                    )
                    ranges.append((start + trim_plan[index].trim_frames, end))
                _keep_episodes_from_video_with_av(
                    source_video, destination_video, ranges, source_meta.fps
                )
                for index in sorted(episode_ids):
                    trim = trim_plan[index]
                    duration = trim.new_length / source_meta.fps
                    video_metadata[index].update(
                        {
                            f"videos/{video_key}/chunk_index": chunk,
                            f"videos/{video_key}/file_index": file_index,
                            f"videos/{video_key}/from_timestamp": cumulative,
                            f"videos/{video_key}/to_timestamp": cumulative + duration,
                        }
                    )
                    cumulative += duration

        global_index = 0
        data_metadata: dict[int, dict[str, Any]] = {}
        for data_file in sorted((source_root / "data").glob("chunk-*/*.parquet")):
            frame_df = pd.read_parquet(data_file)
            kept_parts = []
            for index, group in frame_df.groupby("episode_index", sort=True):
                episode_index = int(index)
                trim = trim_plan[episode_index]
                group = group.sort_values("frame_index").iloc[trim.trim_frames:].copy()
                group["frame_index"] = range(len(group))
                group["timestamp"] = [frame / source_meta.fps for frame in range(len(group))]
                group["index"] = range(global_index, global_index + len(group))
                kept_parts.append(group)
                data_metadata[episode_index] = {
                    "data/chunk_index": int(data_file.parent.name.split("-")[-1]),
                    "data/file_index": int(data_file.stem.split("-")[-1]),
                    "dataset_from_index": int(group["index"].min()),
                    "dataset_to_index": int(group["index"].max() + 1),
                }
                global_index += len(group)
            if kept_parts:
                destination = output_root / "data" / data_file.parent.name / data_file.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                _write_parquet(pd.concat(kept_parts, ignore_index=True), destination, new_meta)

        for episode in sorted(
            source_meta.episodes, key=lambda item: int(item["episode_index"])
        ):
            index = int(episode["episode_index"])
            trim = trim_plan[index]
            new_meta._save_episode_metadata(
                {
                    "episode_index": index,
                    "tasks": episode["tasks"],
                    "length": trim.new_length,
                    **data_metadata[index],
                    **video_metadata[index],
                }
            )
        new_meta.finalize()
        new_meta.info.update(
            {
                "total_episodes": len(trim_plan),
                "total_frames": sum(trim.new_length for trim in trim_plan.values()),
                "splits": {"train": f"0:{len(trim_plan)}"},
                PROCESSED_FLAG: True,
            }
        )
        write_info(new_meta.info, output_root)
        source_action_config = source_root / ACTION_CONFIG_PATH
        if source_action_config.exists():
            target_action_config = output_root / ACTION_CONFIG_PATH
            target_action_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_action_config, target_action_config)
        source_trajectory_config = source_root / TRAJECTORY_CONFIG_PATH
        if source_trajectory_config.exists():
            target_trajectory_config = output_root / TRAJECTORY_CONFIG_PATH
            target_trajectory_config.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_trajectory_config, target_trajectory_config)
        ensure_dataset_stats(f"local/{output_root.name}", output_root, force_recompute=True)
    except Exception:
        if moved and backup_root.exists():
            restore_dataset_from_backup(dataset_root, backup_root)
        raise

    print("Finished trimming initial static segments.")
    print(f"  original dataset backup: {backup_root}")
    print(f"  processed dataset output: {output_root}")
    return 0


def main() -> None:
    raise SystemExit(trim_initial_static_segments(parse_args()))


if __name__ == "__main__":
    main()
