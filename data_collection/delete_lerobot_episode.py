from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

INFO_PATH = Path("meta/info.json")
ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Delete one or more episodes from a LeRobot dataset recorded by "
            "lerobot_collection.py."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        action="append",
        dest="episode_indices_single",
        help="Episode index to delete. Repeat this flag to delete multiple episodes.",
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        default=None,
        help="Space-separated episode indices to delete.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help=(
            "Optional LeRobot repo id used when loading the dataset. "
            "Defaults to local/<dataset-folder-name>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for the edited dataset. If omitted, a sibling directory "
            "will be created automatically unless --in-place is set."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Replace the original dataset in place. The original dataset is "
            "moved to a backup directory first."
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
        help="Print the planned deletion and exit without modifying anything.",
    )
    return parser.parse_args()


def is_lerobot_dataset_root(root: Path) -> bool:
    return (root / INFO_PATH).exists()


def load_dataset_info(dataset_root: Path) -> dict:
    with (dataset_root / INFO_PATH).open() as f:
        return json.load(f)


def load_action_config(dataset_root: Path) -> dict | None:
    action_config_path = dataset_root / ACTION_CONFIG_PATH
    if not action_config_path.exists():
        return None

    with action_config_path.open() as f:
        return json.load(f)


def collect_episode_indices(args: argparse.Namespace) -> list[int]:
    indices: list[int] = []
    if args.episode_indices_single:
        indices.extend(args.episode_indices_single)
    if args.episode_indices:
        indices.extend(args.episode_indices)

    if not indices:
        raise ValueError("At least one episode index must be provided.")

    if any(index < 0 for index in indices):
        raise ValueError(f"Episode indices must be non-negative, received: {indices}")

    return sorted(set(indices))


def derive_default_output_dir(dataset_root: Path) -> Path:
    candidate = dataset_root.with_name(f"{dataset_root.name}_episode_deleted")
    suffix = 1
    while candidate.exists():
        candidate = dataset_root.with_name(f"{dataset_root.name}_episode_deleted_{suffix}")
        suffix += 1
    return candidate


def resolve_repo_id(dataset_root: Path, repo_id: str | None) -> str:
    if repo_id:
        return repo_id
    return f"local/{dataset_root.name}"


def copy_optional_metadata(source_root: Path, target_root: Path) -> None:
    source_action_config = source_root / ACTION_CONFIG_PATH
    if not source_action_config.exists():
        return

    target_action_config = target_root / ACTION_CONFIG_PATH
    target_action_config.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_action_config, target_action_config)


def load_frame_episode_indices(dataset_root: Path) -> list[int] | None:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return None

    parquet_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        return []

    episode_indices: set[int] = set()
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=["episode_index"])
        episode_indices.update(int(value) for value in table.column("episode_index").to_pylist())

    return sorted(episode_indices)


def format_episode_indices(indices: list[int]) -> str:
    if not indices:
        return "none"

    if len(indices) <= 20:
        return ", ".join(str(index) for index in indices)

    head = ", ".join(str(index) for index in indices[:10])
    tail = ", ".join(str(index) for index in indices[-5:])
    return f"{head}, ..., {tail}"


def print_dataset_summary(dataset_root: Path, label: str) -> None:
    info = load_dataset_info(dataset_root)
    action_config = load_action_config(dataset_root)
    metadata_episode_indices = list(range(int(info["total_episodes"])))
    frame_episode_indices = load_frame_episode_indices(dataset_root)

    feature_names = list(info["features"])
    camera_keys = [
        feature_name
        for feature_name, feature_spec in info["features"].items()
        if feature_name.startswith("observation.images.")
        and feature_spec.get("dtype") in {"video", "image"}
    ]

    print(f"\n{label}")
    print(f"  root: {dataset_root}")
    print(f"  total episodes: {info['total_episodes']}")
    print(f"  total frames: {info['total_frames']}")
    print(f"  total tasks: {info['total_tasks']}")
    print(f"  fps: {info['fps']}")
    print(f"  metadata episode indices: {format_episode_indices(metadata_episode_indices)}")
    if frame_episode_indices is None:
        print("  frame parquet episode indices: unavailable (pyarrow not installed in this environment)")
    else:
        print(f"  frame parquet episode indices: {format_episode_indices(frame_episode_indices)}")
        if frame_episode_indices != metadata_episode_indices:
            print("  warning: metadata episode indices and frame parquet episode indices do not match")
    print(f"  cameras: {', '.join(camera_keys) if camera_keys else 'none'}")
    print(f"  feature count: {len(feature_names)}")

    if action_config is not None:
        print(
            "  action config: "
            f"arm={action_config.get('arm_action_representation', 'unknown')}, "
            f"gripper={action_config.get('gripper_action_representation', 'unknown')}, "
            f"dim={action_config.get('action_dim', 'unknown')}"
        )


def validate_episode_indices(indices: list[int], total_episodes: int) -> None:
    invalid = [index for index in indices if index >= total_episodes]
    if invalid:
        raise ValueError(
            f"Episode indices out of range: {invalid}. "
            f"Dataset contains {total_episodes} episodes indexed 0 to {total_episodes - 1}."
        )

    if len(indices) >= total_episodes:
        raise ValueError("Refusing to delete all episodes from the dataset.")


@dataclass
class LocalDatasetView:
    repo_id: str
    root: Path
    meta: object
    image_transforms: object = None
    delta_timestamps: object = None
    tolerance_s: float = 1e-4


def load_lerobot_dependencies():
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.dataset_tools import (
        _copy_and_reindex_data,
        _copy_and_reindex_episodes_metadata,
        _copy_and_reindex_videos,
    )

    return (
        LeRobotDatasetMetadata,
        _copy_and_reindex_data,
        _copy_and_reindex_episodes_metadata,
        _copy_and_reindex_videos,
    )


def delete_episodes_local(
    source_root: Path,
    output_root: Path,
    repo_id: str,
    episode_indices: list[int],
    dataset_info: dict,
) -> None:
    (
        LeRobotDatasetMetadata,
        copy_and_reindex_data,
        copy_and_reindex_episodes_metadata,
        copy_and_reindex_videos,
    ) = load_lerobot_dependencies()

    source_meta = LeRobotDatasetMetadata(
        repo_id=repo_id,
        root=source_root,
        revision=str(dataset_info["codebase_version"]),
    )
    source_dataset = LocalDatasetView(repo_id=repo_id, root=source_root, meta=source_meta)

    episodes_to_keep = [i for i in range(source_meta.total_episodes) if i not in set(episode_indices)]
    episode_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(episodes_to_keep)}

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

    video_metadata = None
    if source_meta.video_keys:
        video_metadata = copy_and_reindex_videos(source_dataset, new_meta, episode_mapping)

    data_metadata = copy_and_reindex_data(source_dataset, new_meta, episode_mapping)
    copy_and_reindex_episodes_metadata(
        source_dataset,
        new_meta,
        episode_mapping,
        data_metadata,
        video_metadata,
    )


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not is_lerobot_dataset_root(dataset_root):
        raise FileNotFoundError(
            f"'{dataset_root}' is not a LeRobot dataset root. "
            f"Expected to find '{INFO_PATH}' underneath it."
        )

    if args.backup_dir and not args.in_place:
        raise ValueError("--backup-dir can only be used together with --in-place.")
    if args.in_place and args.output_dir:
        raise ValueError("--output-dir cannot be used together with --in-place.")

    episode_indices = collect_episode_indices(args)
    dataset_info = load_dataset_info(dataset_root)
    total_episodes = int(dataset_info["total_episodes"])
    validate_episode_indices(episode_indices, total_episodes)

    remaining_episodes = total_episodes - len(episode_indices)
    repo_id = resolve_repo_id(dataset_root, args.repo_id)

    if args.in_place:
        backup_root = (
            Path(args.backup_dir).expanduser().resolve()
            if args.backup_dir
            else dataset_root.with_name(f"{dataset_root.name}_backup_before_delete")
        )
        output_root = dataset_root
        source_root = backup_root
        if backup_root.exists():
            raise FileExistsError(
                f"Backup directory '{backup_root}' already exists. "
                "Remove it or choose another --backup-dir."
            )
    else:
        backup_root = None
        output_root = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else derive_default_output_dir(dataset_root)
        )
        source_root = dataset_root
        if output_root.exists():
            raise FileExistsError(
                f"Output directory '{output_root}' already exists. "
                "Choose another --output-dir or use --in-place."
            )

    print_dataset_summary(dataset_root, "Dataset summary before deletion")
    print("\nDeletion plan")
    print(f"  repo id: {repo_id}")
    print(f"  delete episodes: {episode_indices}")
    print(f"  remaining episodes after deletion: {remaining_episodes}")
    print(f"  output dataset root: {output_root}")
    if backup_root is not None:
        print(f"  backup dataset root: {backup_root}")

    if args.dry_run:
        print("\nDry run complete. No files were changed.")
        return

    moved_to_backup = False
    try:
        if backup_root is not None:
            shutil.move(str(dataset_root), str(backup_root))
            moved_to_backup = True

        delete_episodes_local(
            source_root=source_root,
            output_root=output_root,
            repo_id=repo_id,
            episode_indices=episode_indices,
            dataset_info=dataset_info,
        )
        copy_optional_metadata(source_root, output_root)

    except Exception:
        if moved_to_backup and backup_root is not None and backup_root.exists():
            if dataset_root.exists():
                shutil.rmtree(dataset_root)
            shutil.move(str(backup_root), str(dataset_root))
        raise

    print(
        "\nFinished deleting episodes."
    )
    print_dataset_summary(output_root, "Dataset summary after deletion")
    if backup_root is not None:
        print(f"Original dataset backup kept at: {backup_root}")


if __name__ == "__main__":
    main()
