from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
DATA_COLLECTION_DIR = REPO_ROOT / "data_collection"
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))

INFO_PATH = Path("meta/info.json")
ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")


def configure_huggingface_cache() -> None:
    """Keep Hugging Face parquet cache writes out of read-only home directories."""
    cache_root = Path(os.environ.get("REAL_EXP_HF_CACHE", "/tmp/real_exp_hf_cache"))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf_home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    for env_name in ("HF_HOME", "HF_DATASETS_CACHE", "HUGGINGFACE_HUB_CACHE"):
        Path(os.environ[env_name]).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EpisodeTrim:
    episode_index: int
    old_length: int
    trim_start_frames: int
    trim_end_frames: int

    @property
    def new_length(self) -> int:
        return self.old_length - self.trim_start_frames - self.trim_end_frames

    @property
    def is_trimmed(self) -> bool:
        return self.trim_start_frames > 0 or self.trim_end_frames > 0


@dataclass
class LocalDatasetView:
    repo_id: str
    root: Path
    meta: object
    image_transforms: object = None
    delta_timestamps: object = None
    tolerance_s: float = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim frames from the start and/or end of selected episodes in a local "
            "LeRobot dataset, then rebuild parquet files, videos, metadata, and stats."
        )
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
        help=(
            "Optional comma/whitespace-separated episode indices to trim, supporting "
            "inclusive ranges like 0,1,4-8,12. If omitted, all episodes are trimmed."
        ),
    )

    start_group = parser.add_mutually_exclusive_group()
    start_group.add_argument(
        "--trim-start-seconds",
        type=float,
        default=None,
        help="Seconds to remove from the beginning of each selected episode.",
    )
    start_group.add_argument(
        "--trim-start-frames",
        type=int,
        default=None,
        help="Frames to remove from the beginning of each selected episode.",
    )

    end_group = parser.add_mutually_exclusive_group()
    end_group.add_argument(
        "--trim-end-seconds",
        type=float,
        default=None,
        help="Seconds to remove from the end of each selected episode.",
    )
    end_group.add_argument(
        "--trim-end-frames",
        type=int,
        default=None,
        help="Frames to remove from the end of each selected episode.",
    )

    parser.add_argument(
        "--repo-id",
        default=None,
        help=(
            "Optional LeRobot repo id used when loading the rebuilt dataset. "
            "Defaults to local/<output-folder-name>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for the trimmed dataset. If omitted, a sibling directory "
            "will be created automatically unless --in-place is set."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Replace the original dataset in place. The original dataset is moved "
            "to a backup directory first and kept after success."
        ),
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help="Optional backup directory used only with --in-place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned trim and exit without modifying anything.",
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        default=None,
        help=(
            "Number of worker threads used for copying or re-encoding video files. "
            "Defaults to min(number of affected video files, CPU count)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow replacing an existing --output-dir in non-in-place mode. "
            "Never applies to --in-place or backup directories."
        ),
    )
    return parser.parse_args()


def require_runtime_dependencies():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is required to rewrite LeRobot parquet files. "
            "Run this script inside the project's conda/devcontainer environment."
        ) from exc

    try:
        from dataset_stats import ensure_dataset_stats
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.dataset_tools import _keep_episodes_from_video_with_av, _write_parquet
        from lerobot.datasets.io_utils import load_episodes, write_info
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LeRobot and project dataset utilities are required. Run this script from "
            "the repository root inside the configured environment."
        ) from exc

    return {
        "pd": pd,
        "ensure_dataset_stats": ensure_dataset_stats,
        "LeRobotDatasetMetadata": LeRobotDatasetMetadata,
        "keep_episodes_from_video_with_av": _keep_episodes_from_video_with_av,
        "write_parquet": _write_parquet,
        "load_episodes": load_episodes,
        "write_info": write_info,
    }


def is_lerobot_dataset_root(root: Path) -> bool:
    return (root / INFO_PATH).exists()


def load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    with (dataset_root / INFO_PATH).open() as f:
        return json.load(f)


def parse_episode_selection(text: str, source: str) -> list[int]:
    normalized_text = text.replace(",", " ")
    tokens = [token.strip() for token in normalized_text.split() if token.strip()]
    if not tokens:
        raise ValueError(f"No episode indices found in {source}.")

    episode_indices: set[int] = set()
    for token in tokens:
        if token.isdigit():
            episode_indices.add(int(token))
            continue

        if token.startswith("-") and token[1:].isdigit():
            raise ValueError(f"Episode indices must be non-negative in {source}: {token!r}.")

        if "-" in token:
            range_parts = token.split("-")
            if len(range_parts) == 2 and range_parts[0].isdigit() and range_parts[1].isdigit():
                start = int(range_parts[0])
                end = int(range_parts[1])
                if start > end:
                    raise ValueError(
                        f"Invalid episode range {token!r} in {source}: start must be <= end."
                    )
                episode_indices.update(range(start, end + 1))
                continue

        raise ValueError(
            f"Invalid episode token {token!r} in {source}. "
            "Use non-negative integers or inclusive ranges like 4-8."
        )

    return sorted(episode_indices)


def resolve_selected_episodes(args: argparse.Namespace, total_episodes: int) -> list[int]:
    if args.episode_indices:
        selected = parse_episode_selection(" ".join(args.episode_indices), "--episode-indices")
    else:
        selected = list(range(total_episodes))

    invalid = [episode_index for episode_index in selected if episode_index >= total_episodes]
    if invalid:
        raise ValueError(
            f"Episode indices out of range: {invalid}. "
            f"Dataset contains {total_episodes} episodes indexed 0 to {total_episodes - 1}."
        )

    return selected


def seconds_to_frames(seconds: float | None, fps: float, label: str) -> int | None:
    if seconds is None:
        return None
    if seconds < 0:
        raise ValueError(f"{label} must be non-negative, received {seconds}.")
    return int(round(seconds * fps))


def resolve_trim_frames(args: argparse.Namespace, fps: float) -> tuple[int, int]:
    trim_start_frames = (
        args.trim_start_frames
        if args.trim_start_frames is not None
        else seconds_to_frames(args.trim_start_seconds, fps, "--trim-start-seconds")
    )
    trim_end_frames = (
        args.trim_end_frames
        if args.trim_end_frames is not None
        else seconds_to_frames(args.trim_end_seconds, fps, "--trim-end-seconds")
    )

    trim_start_frames = int(trim_start_frames or 0)
    trim_end_frames = int(trim_end_frames or 0)

    if trim_start_frames < 0:
        raise ValueError(f"--trim-start-frames must be non-negative, received {trim_start_frames}.")
    if trim_end_frames < 0:
        raise ValueError(f"--trim-end-frames must be non-negative, received {trim_end_frames}.")
    if trim_start_frames == 0 and trim_end_frames == 0:
        raise ValueError(
            "At least one trim amount must be greater than 0. "
            "Use --trim-start-seconds/frames or --trim-end-seconds/frames."
        )

    return trim_start_frames, trim_end_frames


def format_episode_indices(indices: list[int]) -> str:
    if not indices:
        return "none"
    if len(indices) <= 20:
        return ", ".join(str(index) for index in indices)
    head = ", ".join(str(index) for index in indices[:10])
    tail = ", ".join(str(index) for index in indices[-5:])
    return f"{head}, ..., {tail}"


def derive_default_output_dir(
    dataset_root: Path,
    trim_start_frames: int,
    trim_end_frames: int,
) -> Path:
    parts = ["trim"]
    if trim_start_frames:
        parts.append(f"start_{trim_start_frames}f")
    if trim_end_frames:
        parts.append(f"end_{trim_end_frames}f")
    suffix_name = "_".join(parts)

    candidate = dataset_root.with_name(f"{dataset_root.name}_{suffix_name}")
    suffix = 1
    while candidate.exists():
        candidate = dataset_root.with_name(f"{dataset_root.name}_{suffix_name}_{suffix}")
        suffix += 1
    return candidate


def resolve_repo_id(output_root: Path, repo_id: str | None) -> str:
    if repo_id:
        return repo_id
    return f"local/{output_root.name}"


def copy_optional_metadata(source_root: Path, target_root: Path) -> None:
    source_action_config = source_root / ACTION_CONFIG_PATH
    if not source_action_config.exists():
        return

    target_action_config = target_root / ACTION_CONFIG_PATH
    target_action_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_action_config, target_action_config)


def build_trim_plan(
    episodes: list[dict[str, Any]],
    selected_episodes: list[int],
    trim_start_frames: int,
    trim_end_frames: int,
) -> dict[int, EpisodeTrim]:
    selected_set = set(selected_episodes)
    trim_plan: dict[int, EpisodeTrim] = {}

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        old_length = int(episode["length"])
        start_frames = trim_start_frames if episode_index in selected_set else 0
        end_frames = trim_end_frames if episode_index in selected_set else 0
        trim = EpisodeTrim(
            episode_index=episode_index,
            old_length=old_length,
            trim_start_frames=start_frames,
            trim_end_frames=end_frames,
        )
        if trim.new_length <= 0:
            raise ValueError(
                f"Episode {episode_index} would become empty: old length={old_length}, "
                f"trim start={start_frames}, trim end={end_frames}."
            )
        trim_plan[episode_index] = trim

    return trim_plan


def summarize_trim_plan(trim_plan: dict[int, EpisodeTrim], selected_episodes: list[int]) -> None:
    old_total_frames = sum(trim.old_length for trim in trim_plan.values())
    new_total_frames = sum(trim.new_length for trim in trim_plan.values())

    print("\nTrim plan")
    print(f"  selected episodes: {format_episode_indices(selected_episodes)}")
    print(f"  selected episode count: {len(selected_episodes)}")
    print(f"  old total frames: {old_total_frames}")
    print(f"  new total frames: {new_total_frames}")
    print(f"  removed frames: {old_total_frames - new_total_frames}")

    print("  selected episode length changes:")
    preview = selected_episodes[:20]
    for episode_index in preview:
        trim = trim_plan[episode_index]
        print(
            f"    episode {episode_index}: {trim.old_length} -> {trim.new_length} "
            f"(start -{trim.trim_start_frames}, end -{trim.trim_end_frames})"
        )
    if len(selected_episodes) > len(preview):
        print(f"    ... {len(selected_episodes) - len(preview)} more selected episode(s)")


def resolve_video_workers(requested_workers: int | None, task_count: int) -> int:
    if task_count <= 0:
        return 1

    if requested_workers is not None:
        if requested_workers <= 0:
            raise ValueError(f"--video-workers must be positive, received: {requested_workers}")
        return min(requested_workers, task_count)

    return max(1, min(task_count, os.cpu_count() or 1))


def estimate_reencoded_video_files(source_meta: object, trim_plan: dict[int, EpisodeTrim]) -> int:
    if not source_meta.video_keys:
        return 0

    affected: set[tuple[str, int, int]] = set()
    for video_key in source_meta.video_keys:
        for episode in source_meta.episodes:
            episode_index = int(episode["episode_index"])
            if not trim_plan[episode_index].is_trimmed:
                continue

            chunk_idx = episode.get(f"videos/{video_key}/chunk_index")
            file_idx = episode.get(f"videos/{video_key}/file_index")
            if chunk_idx is None or file_idx is None:
                continue
            affected.add((video_key, int(chunk_idx), int(file_idx)))

    return len(affected)


def keep_video_ranges(
    src_video_path: Path,
    dst_video_path: Path,
    episode_frame_ranges: list[tuple[int, int]],
    fps: float,
    keep_episodes_from_video_with_av,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
) -> None:
    try:
        keep_episodes_from_video_with_av(
            src_video_path,
            dst_video_path,
            episode_frame_ranges,
            fps,
            vcodec,
            pix_fmt,
        )
        return
    except ModuleNotFoundError as exc:
        if exc.name != "av":
            raise

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyAV is not installed and OpenCV fallback is unavailable. "
            "Install av or opencv-python in the active environment."
        ) from exc

    capture = cv2.VideoCapture(str(src_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source video: {src_video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Failed to read video dimensions from: {src_video_path}")

    writer = cv2.VideoWriter(
        str(dst_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open destination video writer: {dst_video_path}")

    try:
        for start_frame, end_frame in episode_frame_ranges:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_idx in range(start_frame, end_frame):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to read frame {frame_idx} from {src_video_path}")
                writer.write(frame)
    finally:
        writer.release()
        capture.release()


def copy_and_trim_videos_parallel(
    source_dataset: LocalDatasetView,
    output_meta: object,
    trim_plan: dict[int, EpisodeTrim],
    keep_episodes_from_video_with_av,
    video_workers: int | None,
) -> dict[int, dict[str, Any]]:
    if output_meta.video_path is None:
        raise ValueError("Destination metadata has no video_path defined.")

    episodes_video_metadata: dict[int, dict[str, Any]] = {
        int(episode["episode_index"]): {} for episode in source_dataset.meta.episodes
    }
    file_tasks: list[tuple[str, int, int, list[int], bool]] = []

    for video_key in source_dataset.meta.video_keys:
        file_to_episodes: dict[tuple[int, int], list[int]] = {}
        file_to_has_trimmed_episode: dict[tuple[int, int], bool] = {}

        for episode in source_dataset.meta.episodes:
            episode_index = int(episode["episode_index"])
            chunk_idx = episode.get(f"videos/{video_key}/chunk_index")
            file_idx = episode.get(f"videos/{video_key}/file_index")
            if chunk_idx is None or file_idx is None:
                continue

            file_key = (int(chunk_idx), int(file_idx))
            file_to_episodes.setdefault(file_key, []).append(episode_index)
            file_to_has_trimmed_episode[file_key] = (
                file_to_has_trimmed_episode.get(file_key, False)
                or trim_plan[episode_index].is_trimmed
            )

        for (chunk_idx, file_idx), episode_indices in sorted(file_to_episodes.items()):
            file_tasks.append(
                (
                    video_key,
                    chunk_idx,
                    file_idx,
                    sorted(episode_indices),
                    file_to_has_trimmed_episode[(chunk_idx, file_idx)],
                )
            )

    if not file_tasks:
        return episodes_video_metadata

    max_workers = resolve_video_workers(video_workers, len(file_tasks))
    print(f"  video workers: {max_workers}")
    print(f"  video files to copy/re-encode: {len(file_tasks)}")

    episode_by_index = {
        int(episode["episode_index"]): episode for episode in source_dataset.meta.episodes
    }

    def process_video_file(
        video_key: str,
        src_chunk_idx: int,
        src_file_idx: int,
        episode_indices: list[int],
        has_trimmed_episode: bool,
    ) -> tuple[tuple[str, int, int], dict[int, dict[str, Any]]]:
        file_metadata: dict[int, dict[str, Any]] = {}

        assert source_dataset.meta.video_path is not None
        src_video_path = source_dataset.root / source_dataset.meta.video_path.format(
            video_key=video_key,
            chunk_index=src_chunk_idx,
            file_index=src_file_idx,
        )
        dst_video_path = output_meta.root / output_meta.video_path.format(
            video_key=video_key,
            chunk_index=src_chunk_idx,
            file_index=src_file_idx,
        )
        dst_video_path.parent.mkdir(parents=True, exist_ok=True)

        if not has_trimmed_episode:
            shutil.copy2(src_video_path, dst_video_path)
            for episode_index in episode_indices:
                source_episode = episode_by_index[episode_index]
                file_metadata[episode_index] = {
                    f"videos/{video_key}/chunk_index": src_chunk_idx,
                    f"videos/{video_key}/file_index": src_file_idx,
                    f"videos/{video_key}/from_timestamp": source_episode[
                        f"videos/{video_key}/from_timestamp"
                    ],
                    f"videos/{video_key}/to_timestamp": source_episode[
                        f"videos/{video_key}/to_timestamp"
                    ],
                }
            return (video_key, src_chunk_idx, src_file_idx), file_metadata

        episode_frame_ranges: list[tuple[int, int]] = []
        for episode_index in episode_indices:
            source_episode = episode_by_index[episode_index]
            trim = trim_plan[episode_index]
            from_frame = round(
                float(source_episode[f"videos/{video_key}/from_timestamp"]) * source_dataset.meta.fps
            )
            to_frame = round(
                float(source_episode[f"videos/{video_key}/to_timestamp"]) * source_dataset.meta.fps
            )
            expected_old_length = to_frame - from_frame
            if expected_old_length != trim.old_length:
                raise ValueError(
                    f"Episode {episode_index} video length mismatch for {video_key}: "
                    f"metadata length={trim.old_length}, video timestamp frames={expected_old_length}."
                )

            kept_start = from_frame + trim.trim_start_frames
            kept_end = to_frame - trim.trim_end_frames
            if kept_end <= kept_start:
                raise ValueError(
                    f"Episode {episode_index} would have an empty video range for {video_key}: "
                    f"[{kept_start}, {kept_end})."
                )
            if kept_end - kept_start != trim.new_length:
                raise ValueError(
                    f"Episode {episode_index} new video range length mismatch for {video_key}: "
                    f"{kept_end - kept_start} vs {trim.new_length}."
                )
            episode_frame_ranges.append((kept_start, kept_end))

        keep_video_ranges(
            src_video_path=src_video_path,
            dst_video_path=dst_video_path,
            episode_frame_ranges=episode_frame_ranges,
            fps=source_dataset.meta.fps,
            keep_episodes_from_video_with_av=keep_episodes_from_video_with_av,
        )

        cumulative_ts = 0.0
        for episode_index in episode_indices:
            trim = trim_plan[episode_index]
            episode_duration = trim.new_length / source_dataset.meta.fps
            file_metadata[episode_index] = {
                f"videos/{video_key}/chunk_index": src_chunk_idx,
                f"videos/{video_key}/file_index": src_file_idx,
                f"videos/{video_key}/from_timestamp": cumulative_ts,
                f"videos/{video_key}/to_timestamp": cumulative_ts + episode_duration,
            }
            cumulative_ts += episode_duration

        return (video_key, src_chunk_idx, src_file_idx), file_metadata

    task_results: list[tuple[tuple[str, int, int], dict[int, dict[str, Any]]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                process_video_file,
                video_key,
                chunk_idx,
                file_idx,
                episode_indices,
                has_trimmed_episode,
            ): (video_key, chunk_idx, file_idx)
            for video_key, chunk_idx, file_idx, episode_indices, has_trimmed_episode in file_tasks
        }

        completed = 0
        for future in as_completed(future_map):
            completed += 1
            task_descriptor = future_map[future]
            try:
                task_results.append(future.result())
            except Exception as exc:
                video_key, chunk_idx, file_idx = task_descriptor
                raise RuntimeError(
                    f"Failed while processing video file for {video_key} "
                    f"(chunk {chunk_idx}, file {file_idx})."
                ) from exc

            if completed == len(file_tasks) or completed % 10 == 0:
                print(f"  processed video files: {completed}/{len(file_tasks)}")

    for _, file_metadata in sorted(task_results, key=lambda item: item[0]):
        for episode_index, video_meta in file_metadata.items():
            episodes_video_metadata[episode_index].update(video_meta)

    return episodes_video_metadata


def rewrite_data_parquets(
    source_dataset: LocalDatasetView,
    output_meta: object,
    trim_plan: dict[int, EpisodeTrim],
    write_parquet,
    pd,
) -> dict[int, dict[str, Any]]:
    episode_by_index = {
        int(episode["episode_index"]): episode for episode in source_dataset.meta.episodes
    }
    file_to_episodes: dict[Path, list[int]] = {}
    for episode_index, episode in episode_by_index.items():
        chunk_idx = int(episode["data/chunk_index"])
        file_idx = int(episode["data/file_index"])
        src_path = Path(
            source_dataset.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        )
        file_to_episodes.setdefault(src_path, []).append(episode_index)

    global_index = 0
    episode_data_metadata: dict[int, dict[str, Any]] = {}

    for src_rel_path in sorted(file_to_episodes):
        episode_indices = sorted(file_to_episodes[src_rel_path])
        df = pd.read_parquet(source_dataset.root / src_rel_path)
        trimmed_dfs = []

        for episode_index in episode_indices:
            trim = trim_plan[episode_index]
            ep_df = df[df["episode_index"] == episode_index].copy()
            ep_df = ep_df.sort_values("frame_index").reset_index(drop=True)

            if len(ep_df) != trim.old_length:
                raise ValueError(
                    f"Episode {episode_index} data row count mismatch: "
                    f"metadata length={trim.old_length}, parquet rows={len(ep_df)}."
                )

            keep_start = trim.trim_start_frames
            keep_end = trim.old_length - trim.trim_end_frames
            ep_df = ep_df.iloc[keep_start:keep_end].copy().reset_index(drop=True)
            if len(ep_df) != trim.new_length:
                raise ValueError(
                    f"Episode {episode_index} trimmed row count mismatch: "
                    f"expected {trim.new_length}, got {len(ep_df)}."
                )

            ep_df["frame_index"] = range(trim.new_length)
            ep_df["timestamp"] = [frame_index / source_dataset.meta.fps for frame_index in range(trim.new_length)]
            ep_df["episode_index"] = episode_index
            trimmed_dfs.append(ep_df)

        if not trimmed_dfs:
            continue

        file_df = pd.concat(trimmed_dfs, ignore_index=True)
        file_df["index"] = range(global_index, global_index + len(file_df))

        first_episode = episode_by_index[episode_indices[0]]
        chunk_idx = int(first_episode["data/chunk_index"])
        file_idx = int(first_episode["data/file_index"])
        dst_path = output_meta.root / output_meta.data_path.format(
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        write_parquet(file_df, dst_path, output_meta)

        for episode_index in episode_indices:
            ep_df = file_df[file_df["episode_index"] == episode_index]
            episode_data_metadata[episode_index] = {
                "data/chunk_index": chunk_idx,
                "data/file_index": file_idx,
                "dataset_from_index": int(ep_df["index"].min()),
                "dataset_to_index": int(ep_df["index"].max() + 1),
            }

        global_index += len(file_df)

    return episode_data_metadata


def save_trimmed_episode_metadata(
    source_dataset: LocalDatasetView,
    output_meta: object,
    trim_plan: dict[int, EpisodeTrim],
    data_metadata: dict[int, dict[str, Any]],
    video_metadata: dict[int, dict[str, Any]] | None,
    write_info,
) -> None:
    source_episodes = sorted(
        (dict(source_episode) for source_episode in source_dataset.meta.episodes),
        key=lambda source_episode: int(source_episode["episode_index"]),
    )
    for source_episode in source_episodes:
        episode_index = int(source_episode["episode_index"])
        trim = trim_plan[episode_index]
        episode_dict: dict[str, Any] = {
            "episode_index": episode_index,
            "tasks": source_episode["tasks"],
            "length": trim.new_length,
        }
        episode_dict.update(data_metadata[episode_index])
        if video_metadata is not None:
            episode_dict.update(video_metadata[episode_index])

        # Do not copy old stats/* fields. They describe pre-trim data and would be stale.
        output_meta._save_episode_metadata(episode_dict)

    output_meta.finalize()
    output_meta.info.update(
        {
            "total_episodes": len(trim_plan),
            "total_frames": sum(trim.new_length for trim in trim_plan.values()),
            "total_tasks": len(output_meta.tasks) if output_meta.tasks is not None else 0,
            "splits": {"train": f"0:{len(trim_plan)}"},
        }
    )
    write_info(output_meta.info, output_meta.root)


def trim_dataset_local(
    source_root: Path,
    output_root: Path,
    repo_id: str,
    dataset_info: dict[str, Any],
    trim_plan: dict[int, EpisodeTrim],
    video_workers: int | None,
) -> None:
    deps = require_runtime_dependencies()
    pd = deps["pd"]
    LeRobotDatasetMetadata = deps["LeRobotDatasetMetadata"]
    keep_episodes_from_video_with_av = deps["keep_episodes_from_video_with_av"]
    write_parquet = deps["write_parquet"]
    load_episodes = deps["load_episodes"]
    write_info = deps["write_info"]

    source_meta = LeRobotDatasetMetadata(
        repo_id=repo_id,
        root=source_root,
        revision=str(dataset_info["codebase_version"]),
    )
    if source_meta.episodes is None:
        source_meta.episodes = load_episodes(source_meta.root)

    source_dataset = LocalDatasetView(repo_id=repo_id, root=source_root, meta=source_meta)

    new_meta = LeRobotDatasetMetadata.create(
        repo_id=repo_id,
        fps=source_meta.fps,
        features=source_meta.features,
        robot_type=source_meta.robot_type,
        root=output_root,
        use_videos=len(source_meta.video_keys) > 0,
        chunks_size=source_meta.chunks_size,
        data_files_size_in_mb=source_meta.data_files_size_in_mb,
        video_files_size_in_mb=source_meta.video_files_size_in_mb,
    )

    if source_meta.tasks is not None:
        source_tasks = source_meta.tasks.sort_values("task_index")
        new_meta.save_episode_tasks(list(source_tasks.index))

    video_metadata = None
    if source_meta.video_keys:
        video_metadata = copy_and_trim_videos_parallel(
            source_dataset=source_dataset,
            output_meta=new_meta,
            trim_plan=trim_plan,
            keep_episodes_from_video_with_av=keep_episodes_from_video_with_av,
            video_workers=video_workers,
        )

    data_metadata = rewrite_data_parquets(
        source_dataset=source_dataset,
        output_meta=new_meta,
        trim_plan=trim_plan,
        write_parquet=write_parquet,
        pd=pd,
    )
    save_trimmed_episode_metadata(
        source_dataset=source_dataset,
        output_meta=new_meta,
        trim_plan=trim_plan,
        data_metadata=data_metadata,
        video_metadata=video_metadata,
        write_info=write_info,
    )

    for video_key in new_meta.video_keys:
        new_meta.update_video_info(video_key)
    if new_meta.video_keys:
        write_info(new_meta.info, new_meta.root)


def load_source_metadata_for_plan(
    dataset_root: Path,
    repo_id: str,
    dataset_info: dict[str, Any],
) -> object:
    deps = require_runtime_dependencies()
    LeRobotDatasetMetadata = deps["LeRobotDatasetMetadata"]
    load_episodes = deps["load_episodes"]
    source_meta = LeRobotDatasetMetadata(
        repo_id=repo_id,
        root=dataset_root,
        revision=str(dataset_info["codebase_version"]),
    )
    if source_meta.episodes is None:
        source_meta.episodes = load_episodes(source_meta.root)
    return source_meta


def validate_args(args: argparse.Namespace) -> None:
    if args.backup_dir and not args.in_place:
        raise ValueError("--backup-dir can only be used together with --in-place.")
    if args.in_place and args.output_dir:
        raise ValueError("--output-dir cannot be used together with --in-place.")
    if args.in_place and args.force:
        raise ValueError("--force is only supported for non-in-place --output-dir replacement.")


def main() -> None:
    configure_huggingface_cache()
    args = parse_args()
    validate_args(args)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not is_lerobot_dataset_root(dataset_root):
        raise FileNotFoundError(
            f"'{dataset_root}' is not a LeRobot dataset root. "
            f"Expected to find '{INFO_PATH}' underneath it."
        )

    dataset_info = load_dataset_info(dataset_root)
    fps = float(dataset_info["fps"])
    total_episodes = int(dataset_info["total_episodes"])
    trim_start_frames, trim_end_frames = resolve_trim_frames(args, fps)
    selected_episodes = resolve_selected_episodes(args, total_episodes)

    output_root = (
        dataset_root
        if args.in_place
        else (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else derive_default_output_dir(dataset_root, trim_start_frames, trim_end_frames)
        )
    )
    repo_id = resolve_repo_id(output_root, args.repo_id)

    source_meta = load_source_metadata_for_plan(dataset_root, repo_id, dataset_info)
    episodes = [dict(episode) for episode in source_meta.episodes]
    episodes.sort(key=lambda episode: int(episode["episode_index"]))
    trim_plan = build_trim_plan(episodes, selected_episodes, trim_start_frames, trim_end_frames)

    if args.in_place:
        backup_root = (
            Path(args.backup_dir).expanduser().resolve()
            if args.backup_dir
            else dataset_root.with_name(f"{dataset_root.name}_backup_before_trim")
        )
        source_root = backup_root
        if backup_root.exists():
            raise FileExistsError(
                f"Backup directory '{backup_root}' already exists. "
                "Remove it or choose another --backup-dir."
            )
    else:
        backup_root = None
        source_root = dataset_root
        if output_root == dataset_root:
            raise ValueError("--output-dir must not be the same as --dataset-root. Use --in-place instead.")
        if output_root.exists() and not args.force:
            raise FileExistsError(
                f"Output directory '{output_root}' already exists. "
                "Choose another --output-dir or pass --force to replace it."
            )

    print("Dataset summary before trim")
    print(f"  root: {dataset_root}")
    print(f"  repo id: {repo_id}")
    print(f"  fps: {fps:g}")
    print(f"  total episodes: {total_episodes}")
    print(f"  total frames: {dataset_info['total_frames']}")
    print(f"  output dataset root: {output_root}")
    if backup_root is not None:
        print(f"  backup dataset root: {backup_root}")
    if args.force and output_root.exists():
        print(f"  force: existing output directory will be replaced: {output_root}")

    summarize_trim_plan(trim_plan, selected_episodes)
    reencoded_video_files = estimate_reencoded_video_files(source_meta, trim_plan)
    print(f"  video files requiring re-encode: {reencoded_video_files}")

    if args.dry_run:
        print("\nDry run complete. No files were changed.")
        return

    if args.force and output_root.exists() and not args.in_place:
        shutil.rmtree(output_root)

    moved_to_backup = False
    try:
        if backup_root is not None:
            shutil.move(str(dataset_root), str(backup_root))
            moved_to_backup = True

        trim_dataset_local(
            source_root=source_root,
            output_root=output_root,
            repo_id=repo_id,
            dataset_info=dataset_info,
            trim_plan=trim_plan,
            video_workers=args.video_workers,
        )
        copy_optional_metadata(source_root, output_root)

        deps = require_runtime_dependencies()
        ensure_dataset_stats = deps["ensure_dataset_stats"]
        ensure_dataset_stats(repo_id, output_root, force_recompute=True)

    except Exception:
        if not args.in_place and output_root.exists():
            shutil.rmtree(output_root)
        if moved_to_backup and backup_root is not None and backup_root.exists():
            if dataset_root.exists():
                shutil.rmtree(dataset_root)
            shutil.move(str(backup_root), str(dataset_root))
        raise

    print("\nFinished trimming episode edges.")
    print(f"  output dataset root: {output_root}")
    if backup_root is not None:
        print(f"  original dataset backup kept at: {backup_root}")


if __name__ == "__main__":
    main()
