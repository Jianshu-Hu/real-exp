from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


INFO_PATH = Path("meta/info.json")
DEFAULT_SECTIONS = ("basic", "files", "video")
VALID_SECTIONS = {"basic", "files", "video", "gripper"}


@dataclass
class DatasetIndex:
    root: Path
    info: dict[str, Any]
    episodes: list[dict[str, Any]]
    fps: float
    features: dict[str, Any]
    video_keys: list[str]
    missing_data_files: dict[int, str] = field(default_factory=dict)


@dataclass
class EpisodeSummary:
    episode_index: int
    length_meta: int
    length_rows: int | None = None
    duration_sec: float | None = None
    data_file: str | None = None
    dataset_from_index: int | None = None
    dataset_to_index: int | None = None
    frame_index_min: int | None = None
    frame_index_max: int | None = None
    timestamp_min: float | None = None
    timestamp_max: float | None = None
    checks: dict[str, bool] = field(default_factory=dict)
    videos: dict[str, dict[str, Any]] = field(default_factory=dict)
    gripper: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect episode-level status for a local LeRobot dataset and optionally "
            "write Markdown/JSON/CSV reports."
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
            "Optional comma/whitespace-separated episode indices to inspect, supporting "
            "inclusive ranges like 0,1,4-8,12. If omitted, all episodes are inspected."
        ),
    )
    parser.add_argument(
        "--sections",
        default=",".join(DEFAULT_SECTIONS),
        help=(
            "Comma-separated sections to collect. Available: basic,files,video,gripper. "
            "Default: basic,files,video."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["table", "json", "markdown"],
        default="table",
        help="Terminal output format. Default: table.",
    )
    parser.add_argument(
        "--layout",
        choices=["compact", "blocks"],
        default="compact",
        help="Table layout for terminal output. Default: compact.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for episode_summary.md/json/csv and dataset_overview.json.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=50,
        help="Maximum number of episodes to print to terminal. Default: 50.",
    )
    parser.add_argument(
        "--left-gripper-index",
        type=int,
        default=7,
        help="Left gripper index inside observation.state/action vectors. Default: 7.",
    )
    parser.add_argument(
        "--right-gripper-index",
        type=int,
        default=15,
        help="Right gripper index inside observation.state/action vectors. Default: 15.",
    )
    return parser.parse_args()


def require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pyarrow is required to inspect LeRobot parquet files. "
            "Run this script inside the project's conda/devcontainer environment."
        ) from exc
    return pq


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
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


def parse_sections(text: str) -> set[str]:
    sections = {section.strip() for section in text.replace(" ", "").split(",") if section.strip()}
    if not sections:
        raise ValueError("--sections must contain at least one section.")
    invalid = sorted(sections - VALID_SECTIONS)
    if invalid:
        raise ValueError(f"Invalid --sections entries: {invalid}. Valid sections: {sorted(VALID_SECTIONS)}.")
    return sections


def get_video_keys(info: dict[str, Any]) -> list[str]:
    return [
        feature_name
        for feature_name, feature_spec in info.get("features", {}).items()
        if feature_spec.get("dtype") == "video"
    ]


def get_data_path_for_episode(info: dict[str, Any], episode: dict[str, Any]) -> Path | None:
    data_path_template = info.get("data_path")
    if not data_path_template:
        return None
    if "data/chunk_index" not in episode or "data/file_index" not in episode:
        return None
    return Path(
        data_path_template.format(
            chunk_index=int(episode["data/chunk_index"]),
            file_index=int(episode["data/file_index"]),
        )
    )


def read_parquet_rows(path: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    pq = require_pyarrow()
    schema = pq.read_schema(path)
    available_columns = list(schema.names)
    read_columns = None
    if columns is not None:
        read_columns = [column for column in columns if column in available_columns]
    table = pq.read_table(path, columns=read_columns)
    return table.to_pylist()


def load_episode_metadata(dataset_root: Path) -> list[dict[str, Any]]:
    pq = require_pyarrow()
    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata parquet files found under {dataset_root / 'meta/episodes'}.")

    episodes: list[dict[str, Any]] = []
    for episode_file in episode_files:
        schema = pq.read_schema(episode_file)
        columns = [name for name in schema.names if not name.startswith("stats/")]
        table = pq.read_table(episode_file, columns=columns)
        episodes.extend(table.to_pylist())

    episodes.sort(key=lambda episode: int(episode["episode_index"]))
    return episodes


def load_dataset_index(dataset_root: Path) -> DatasetIndex:
    if not (dataset_root / INFO_PATH).exists():
        raise FileNotFoundError(
            f"'{dataset_root}' is not a LeRobot dataset root. "
            f"Expected to find '{INFO_PATH}' underneath it."
        )

    info = load_json(dataset_root / INFO_PATH)
    episodes = load_episode_metadata(dataset_root)
    fps = float(info["fps"])
    features = info.get("features", {})
    video_keys = get_video_keys(info)
    return DatasetIndex(
        root=dataset_root,
        info=info,
        episodes=episodes,
        fps=fps,
        features=features,
        video_keys=video_keys,
    )


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


def build_rows_by_episode(
    dataset: DatasetIndex,
    selected_episodes: list[int],
    sections: set[str],
) -> dict[int, list[dict[str, Any]]]:
    selected_set = set(selected_episodes)
    episode_by_index = {int(episode["episode_index"]): episode for episode in dataset.episodes}
    file_to_episodes: dict[Path, list[int]] = {}
    for episode_index in selected_episodes:
        data_path = get_data_path_for_episode(dataset.info, episode_by_index[episode_index])
        if data_path is None:
            continue
        file_to_episodes.setdefault(data_path, []).append(episode_index)

    columns = ["episode_index", "frame_index", "timestamp", "index", "task_index"]
    if "gripper" in sections:
        columns.extend(["observation.state", "action"])

    rows_by_episode: dict[int, list[dict[str, Any]]] = {episode_index: [] for episode_index in selected_episodes}
    if file_to_episodes:
        for data_rel_path, episode_indices in sorted(file_to_episodes.items()):
            data_path = dataset.root / data_rel_path
            if not data_path.exists():
                for episode_index in episode_indices:
                    dataset.missing_data_files[episode_index] = str(data_rel_path)
                continue
            rows = read_parquet_rows(data_path, columns=columns)
            episode_set = set(episode_indices)
            for row in rows:
                episode_index = int(row["episode_index"])
                if episode_index in episode_set:
                    row["_source_file"] = str(data_rel_path)
                    rows_by_episode[episode_index].append(row)
        return rows_by_episode

    for data_path in sorted((dataset.root / "data").glob("chunk-*/*.parquet")):
        rows = read_parquet_rows(data_path, columns=columns)
        for row in rows:
            episode_index = int(row["episode_index"])
            if episode_index in selected_set:
                row["_source_file"] = str(data_path.relative_to(dataset.root))
                rows_by_episode[episode_index].append(row)

    return rows_by_episode


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


def safe_min(values: list[float]) -> float | None:
    return min(values) if values else None


def safe_max(values: list[float]) -> float | None:
    return max(values) if values else None


def summarize_gripper_values(
    rows: list[dict[str, Any]],
    feature_name: str,
    left_index: int,
    right_index: int,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    left_values: list[float] = []
    right_values: list[float] = []
    out_of_bounds = 0
    non_binary = 0

    for row in rows:
        values = flatten_numeric(row.get(feature_name))
        if len(values) <= max(left_index, right_index):
            continue
        left = values[left_index]
        right = values[right_index]
        left_values.append(left)
        right_values.append(right)

        for value in (left, right):
            if value < -tolerance or value > 1.0 + tolerance:
                out_of_bounds += 1
            elif abs(value - 0.0) > tolerance and abs(value - 1.0) > tolerance:
                non_binary += 1

    return {
        "left_first": left_values[0] if left_values else None,
        "left_last": left_values[-1] if left_values else None,
        "left_min": safe_min(left_values),
        "left_max": safe_max(left_values),
        "right_first": right_values[0] if right_values else None,
        "right_last": right_values[-1] if right_values else None,
        "right_min": safe_min(right_values),
        "right_max": safe_max(right_values),
        "out_of_bounds_values": out_of_bounds,
        "non_binary_values": non_binary,
    }


def is_contiguous(values: list[int], start: int, length: int) -> bool:
    return values == list(range(start, start + length))


def collect_episode_summary(
    dataset: DatasetIndex,
    episode: dict[str, Any],
    rows: list[dict[str, Any]],
    sections: set[str],
    left_gripper_index: int,
    right_gripper_index: int,
) -> EpisodeSummary:
    episode_index = int(episode["episode_index"])
    length_meta = int(episode["length"])
    summary = EpisodeSummary(
        episode_index=episode_index,
        length_meta=length_meta,
        duration_sec=length_meta / dataset.fps if dataset.fps else None,
    )

    data_rel_path = get_data_path_for_episode(dataset.info, episode)
    if data_rel_path is not None:
        summary.data_file = str(data_rel_path)
    if episode_index in dataset.missing_data_files:
        summary.notes.append(f"missing data file: {dataset.missing_data_files[episode_index]}")
    summary.dataset_from_index = int(episode["dataset_from_index"]) if "dataset_from_index" in episode else None
    summary.dataset_to_index = int(episode["dataset_to_index"]) if "dataset_to_index" in episode else None

    sorted_rows = sorted(rows, key=lambda row: int(row.get("frame_index", 0)))
    summary.length_rows = len(sorted_rows)

    if sorted_rows:
        frame_indices = [int(row["frame_index"]) for row in sorted_rows]
        timestamps = [float(row["timestamp"]) for row in sorted_rows if row.get("timestamp") is not None]
        global_indices = [int(row["index"]) for row in sorted_rows if row.get("index") is not None]

        summary.frame_index_min = min(frame_indices)
        summary.frame_index_max = max(frame_indices)
        summary.timestamp_min = min(timestamps) if timestamps else None
        summary.timestamp_max = max(timestamps) if timestamps else None

        summary.checks["rows_match_length"] = len(sorted_rows) == length_meta
        summary.checks["frame_index_contiguous"] = is_contiguous(frame_indices, 0, len(frame_indices))

        expected_last_timestamp = (length_meta - 1) / dataset.fps if length_meta > 0 else 0.0
        summary.checks["timestamp_starts_zero"] = (
            summary.timestamp_min is not None and abs(summary.timestamp_min) <= 1e-4
        )
        summary.checks["timestamp_end_expected"] = (
            summary.timestamp_max is not None
            and abs(summary.timestamp_max - expected_last_timestamp) <= 1e-3
        )

        if global_indices:
            summary.checks["global_index_contiguous"] = is_contiguous(
                sorted(global_indices),
                min(global_indices),
                len(global_indices),
            )
            if summary.dataset_from_index is not None and summary.dataset_to_index is not None:
                summary.checks["metadata_index_range_match"] = (
                    min(global_indices) == summary.dataset_from_index
                    and max(global_indices) + 1 == summary.dataset_to_index
                )
        else:
            summary.checks["global_index_contiguous"] = False
            summary.checks["metadata_index_range_match"] = False
    else:
        summary.checks["rows_match_length"] = length_meta == 0
        summary.checks["frame_index_contiguous"] = False
        summary.checks["timestamp_starts_zero"] = False
        summary.checks["timestamp_end_expected"] = False
        summary.checks["global_index_contiguous"] = False
        summary.checks["metadata_index_range_match"] = False
        summary.notes.append("no parquet rows found")

    if "video" in sections:
        for video_key in dataset.video_keys:
            short_key = video_key.split(".")[-1]
            video_info = collect_video_summary(dataset, episode, video_key, length_meta)
            summary.videos[short_key] = video_info
        if summary.videos:
            summary.checks["video_metadata_match"] = all(
                bool(video_info.get("ok")) for video_info in summary.videos.values()
            )

    if "gripper" in sections:
        summary.gripper = {
            "observation.state": summarize_gripper_values(
                sorted_rows,
                "observation.state",
                left_gripper_index,
                right_gripper_index,
            ),
            "action": summarize_gripper_values(
                sorted_rows,
                "action",
                left_gripper_index,
                right_gripper_index,
            ),
        }
        if any(
            feature_summary["out_of_bounds_values"] > 0
            for feature_summary in summary.gripper.values()
        ):
            summary.notes.append("gripper out of bounds")
        if any(feature_summary["non_binary_values"] > 0 for feature_summary in summary.gripper.values()):
            summary.notes.append("gripper has non-binary values")

    for check_name, ok in summary.checks.items():
        if not ok:
            summary.notes.append(f"{check_name}=BAD")

    return summary


def collect_video_summary(
    dataset: DatasetIndex,
    episode: dict[str, Any],
    video_key: str,
    length_meta: int,
) -> dict[str, Any]:
    required_keys = [
        f"videos/{video_key}/chunk_index",
        f"videos/{video_key}/file_index",
        f"videos/{video_key}/from_timestamp",
        f"videos/{video_key}/to_timestamp",
    ]
    missing = [key for key in required_keys if key not in episode]
    if missing:
        return {"ok": False, "missing": missing}

    chunk_index = int(episode[f"videos/{video_key}/chunk_index"])
    file_index = int(episode[f"videos/{video_key}/file_index"])
    from_timestamp = float(episode[f"videos/{video_key}/from_timestamp"])
    to_timestamp = float(episode[f"videos/{video_key}/to_timestamp"])
    frame_count = round((to_timestamp - from_timestamp) * dataset.fps)

    video_path_template = dataset.info.get("video_path")
    video_path = None
    exists = None
    if video_path_template:
        video_path = video_path_template.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        exists = (dataset.root / video_path).exists()

    ok = (
        from_timestamp >= -1e-6
        and to_timestamp > from_timestamp
        and frame_count == length_meta
        and (exists is not False)
    )

    return {
        "ok": ok,
        "path": video_path,
        "chunk_index": chunk_index,
        "file_index": file_index,
        "from_timestamp": from_timestamp,
        "to_timestamp": to_timestamp,
        "duration_sec": to_timestamp - from_timestamp,
        "estimated_frames": frame_count,
        "file_exists": exists,
    }


def collect_summaries(
    dataset: DatasetIndex,
    selected_episodes: list[int],
    sections: set[str],
    left_gripper_index: int,
    right_gripper_index: int,
) -> list[EpisodeSummary]:
    rows_by_episode = build_rows_by_episode(dataset, selected_episodes, sections)
    episode_by_index = {int(episode["episode_index"]): episode for episode in dataset.episodes}

    summaries: list[EpisodeSummary] = []
    for episode_index in selected_episodes:
        summaries.append(
            collect_episode_summary(
                dataset=dataset,
                episode=episode_by_index[episode_index],
                rows=rows_by_episode.get(episode_index, []),
                sections=sections,
                left_gripper_index=left_gripper_index,
                right_gripper_index=right_gripper_index,
            )
        )
    return summaries


def yes_no(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "BAD"


def note_text(summary: EpisodeSummary, limit: int = 48) -> str:
    if not summary.notes:
        return "-"
    text = "; ".join(dict.fromkeys(summary.notes))
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def render_compact_table(summaries: list[EpisodeSummary], max_print: int) -> str:
    lines = [
        "ep  len    sec     rows   frame  ts     index  video  notes",
        "--  -----  ------  -----  -----  -----  -----  -----  -----",
    ]
    for summary in summaries[:max_print]:
        checks = summary.checks
        frame_ok = checks.get("frame_index_contiguous") and checks.get("rows_match_length")
        ts_ok = checks.get("timestamp_starts_zero") and checks.get("timestamp_end_expected")
        index_ok = checks.get("global_index_contiguous") and checks.get("metadata_index_range_match")
        video_ok = checks.get("video_metadata_match")
        lines.append(
            f"{summary.episode_index:2d}  "
            f"{summary.length_meta:5d}  "
            f"{(summary.duration_sec or 0.0):6.2f}  "
            f"{(summary.length_rows if summary.length_rows is not None else -1):5d}  "
            f"{yes_no(frame_ok):5s}  "
            f"{yes_no(ts_ok):5s}  "
            f"{yes_no(index_ok):5s}  "
            f"{yes_no(video_ok):5s}  "
            f"{note_text(summary)}"
        )

    if len(summaries) > max_print:
        lines.append(f"... {len(summaries) - max_print} more episode(s) not printed; use --max-print to show more.")

    return "\n".join(lines)


def render_blocks(summaries: list[EpisodeSummary], max_print: int) -> str:
    blocks: list[str] = []
    for summary in summaries[:max_print]:
        lines = [
            f"Episode {summary.episode_index}",
            f"  length: {summary.length_meta} frames / {(summary.duration_sec or 0.0):.2f}s",
            (
                f"  data: {summary.data_file or 'unknown'}, "
                f"index [{summary.dataset_from_index}, {summary.dataset_to_index})"
            ),
            (
                f"  parquet rows: {summary.length_rows}, "
                f"frame_index {summary.frame_index_min}..{summary.frame_index_max}, "
                f"timestamp {format_float(summary.timestamp_min)}..{format_float(summary.timestamp_max)}"
            ),
            f"  checks: {format_checks(summary.checks)}",
        ]

        if summary.videos:
            lines.append("  videos:")
            for camera_key, video_info in summary.videos.items():
                lines.append(
                    "    "
                    f"{camera_key}: {video_info.get('path', 'unknown')}, "
                    f"ts [{format_float(video_info.get('from_timestamp'))}, "
                    f"{format_float(video_info.get('to_timestamp'))}), "
                    f"frames~{video_info.get('estimated_frames')}, "
                    f"exists={video_info.get('file_exists')}, "
                    f"ok={yes_no(video_info.get('ok'))}"
                )

        if summary.gripper:
            lines.append("  gripper:")
            for feature_name, gripper_summary in summary.gripper.items():
                lines.append(
                    "    "
                    f"{feature_name}: "
                    f"L {format_float(gripper_summary.get('left_first'))}->{format_float(gripper_summary.get('left_last'))} "
                    f"[{format_float(gripper_summary.get('left_min'))},{format_float(gripper_summary.get('left_max'))}], "
                    f"R {format_float(gripper_summary.get('right_first'))}->{format_float(gripper_summary.get('right_last'))} "
                    f"[{format_float(gripper_summary.get('right_min'))},{format_float(gripper_summary.get('right_max'))}], "
                    f"out={gripper_summary.get('out_of_bounds_values')}, "
                    f"non_binary={gripper_summary.get('non_binary_values')}"
                )

        if summary.notes:
            lines.append(f"  notes: {'; '.join(dict.fromkeys(summary.notes))}")
        blocks.append("\n".join(lines))

    if len(summaries) > max_print:
        blocks.append(f"... {len(summaries) - max_print} more episode(s) not printed; use --max-print to show more.")

    return "\n\n".join(blocks)


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value_float):
        return str(value)
    return f"{value_float:.3f}"


def format_checks(checks: dict[str, bool]) -> str:
    if not checks:
        return "n/a"
    return ", ".join(f"{name}={yes_no(ok)}" for name, ok in checks.items())


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def build_dataset_overview(dataset: DatasetIndex, selected_episodes: list[int], summaries: list[EpisodeSummary]) -> dict[str, Any]:
    all_episode_lengths = [int(episode["length"]) for episode in dataset.episodes]
    selected_lengths = [summary.length_meta for summary in summaries]
    feature_dims = {
        name: feature.get("shape")
        for name, feature in dataset.features.items()
        if feature.get("dtype") not in {"video", "image"}
    }
    camera_keys = [
        name
        for name, feature in dataset.features.items()
        if feature.get("dtype") in {"video", "image"}
    ]
    return {
        "root": str(dataset.root),
        "fps": dataset.fps,
        "total_episodes_declared": int(dataset.info["total_episodes"]),
        "total_frames_declared": int(dataset.info["total_frames"]),
        "episode_metadata_rows": len(dataset.episodes),
        "sum_episode_lengths": sum(all_episode_lengths),
        "selected_episodes": selected_episodes,
        "selected_episode_count": len(selected_episodes),
        "selected_total_frames": sum(selected_lengths),
        "camera_keys": camera_keys,
        "video_keys": dataset.video_keys,
        "feature_dims": feature_dims,
        "failed_episode_count": sum(1 for summary in summaries if summary.notes),
    }


def render_markdown_report(dataset: DatasetIndex, selected_episodes: list[int], summaries: list[EpisodeSummary]) -> str:
    overview = build_dataset_overview(dataset, selected_episodes, summaries)
    lines = [
        "# LeRobot Episode 状态报告",
        "",
        "## Dataset Overview",
        "",
        f"- root: `{overview['root']}`",
        f"- fps: `{overview['fps']}`",
        f"- total episodes declared: `{overview['total_episodes_declared']}`",
        f"- total frames declared: `{overview['total_frames_declared']}`",
        f"- episode metadata rows: `{overview['episode_metadata_rows']}`",
        f"- sum episode lengths: `{overview['sum_episode_lengths']}`",
        f"- selected episode count: `{overview['selected_episode_count']}`",
        f"- selected total frames: `{overview['selected_total_frames']}`",
        f"- failed episode count: `{overview['failed_episode_count']}`",
        f"- cameras: `{', '.join(overview['camera_keys']) if overview['camera_keys'] else 'none'}`",
        "",
        "## Compact Summary",
        "",
        "```text",
        render_compact_table(summaries, max_print=len(summaries)),
        "```",
        "",
        "## Episode Details",
        "",
        "```text",
        render_blocks(summaries, max_print=len(summaries)),
        "```",
        "",
    ]
    return "\n".join(lines)


def write_reports(output_dir: Path, dataset: DatasetIndex, selected_episodes: list[int], summaries: list[EpisodeSummary]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overview = build_dataset_overview(dataset, selected_episodes, summaries)
    summary_dicts = [asdict(summary) for summary in summaries]

    (output_dir / "dataset_overview.json").write_text(
        json.dumps(to_jsonable(overview), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "episode_summary.json").write_text(
        json.dumps(to_jsonable(summary_dicts), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "episode_summary.md").write_text(
        render_markdown_report(dataset, selected_episodes, summaries),
        encoding="utf-8",
    )

    with (output_dir / "episode_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode_index",
                "length_meta",
                "length_rows",
                "duration_sec",
                "data_file",
                "dataset_from_index",
                "dataset_to_index",
                "frame_index_min",
                "frame_index_max",
                "timestamp_min",
                "timestamp_max",
                "checks",
                "videos",
                "gripper",
                "notes",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            row = asdict(summary)
            row["checks"] = json.dumps(to_jsonable(row["checks"]), ensure_ascii=False)
            row["videos"] = json.dumps(to_jsonable(row["videos"]), ensure_ascii=False)
            row["gripper"] = json.dumps(to_jsonable(row["gripper"]), ensure_ascii=False)
            row["notes"] = "; ".join(row["notes"])
            writer.writerow(row)


def print_dataset_header(dataset: DatasetIndex, selected_episodes: list[int], summaries: list[EpisodeSummary]) -> None:
    overview = build_dataset_overview(dataset, selected_episodes, summaries)
    print("Dataset overview")
    print(f"  root: {overview['root']}")
    print(f"  fps: {overview['fps']:g}")
    print(f"  total episodes declared: {overview['total_episodes_declared']}")
    print(f"  total frames declared: {overview['total_frames_declared']}")
    print(f"  episode metadata rows: {overview['episode_metadata_rows']}")
    print(f"  sum episode lengths: {overview['sum_episode_lengths']}")
    print(f"  selected episodes: {format_episode_indices(selected_episodes)}")
    print(f"  selected total frames: {overview['selected_total_frames']}")
    print(f"  cameras: {', '.join(overview['camera_keys']) if overview['camera_keys'] else 'none'}")
    print(f"  episodes with notes/issues: {overview['failed_episode_count']}")
    print()


def format_episode_indices(indices: list[int]) -> str:
    if not indices:
        return "none"
    if len(indices) <= 20:
        return ", ".join(str(index) for index in indices)
    head = ", ".join(str(index) for index in indices[:10])
    tail = ", ".join(str(index) for index in indices[-5:])
    return f"{head}, ..., {tail}"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    sections = parse_sections(args.sections)
    dataset = load_dataset_index(dataset_root)
    selected_episodes = resolve_selected_episodes(args, int(dataset.info["total_episodes"]))
    summaries = collect_summaries(
        dataset=dataset,
        selected_episodes=selected_episodes,
        sections=sections,
        left_gripper_index=args.left_gripper_index,
        right_gripper_index=args.right_gripper_index,
    )

    if args.format == "json":
        print(json.dumps(to_jsonable([asdict(summary) for summary in summaries]), indent=2, ensure_ascii=False))
    elif args.format == "markdown":
        print(render_markdown_report(dataset, selected_episodes, summaries))
    else:
        print_dataset_header(dataset, selected_episodes, summaries)
        if args.layout == "blocks":
            print(render_blocks(summaries, args.max_print))
        else:
            print(render_compact_table(summaries, args.max_print))

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        write_reports(output_dir, dataset, selected_episodes, summaries)
        print(f"\nReports written to: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
