from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dataset_stats import ensure_dataset_stats

INFO_PATH = Path("meta/info.json")
TASKS_PATH = Path("meta/tasks.parquet")
EPISODES_DIR = Path("meta/episodes")
CRITICAL_METADATA_FILES = [
    Path("meta/info.json"),
    Path("meta/tasks.parquet"),
    Path("meta/stats.json"),
    Path("meta/real_exp_action_config.json"),
]
DATASET_ALLOW_PATTERNS = [
    "README.md",
    ".gitattributes",
    "data/**/*.parquet",
    "meta/**/*.json",
    "meta/**/*.parquet",
    "videos/**/*.mp4",
]
DATASET_ALLOW_PATTERNS_NO_VIDEOS = [
    pattern for pattern in DATASET_ALLOW_PATTERNS if not pattern.startswith("videos/")
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push a local LeRobot dataset to the Hugging Face Hub."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the local LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face dataset repo id, for example <user>/<dataset-name>.",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Hub branch to push to. Defaults to 'main'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the Hub dataset as a private repo.",
    )
    parser.add_argument(
        "--license",
        default="apache-2.0",
        help="Dataset license written to the dataset card.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=None,
        help="Optional dataset tag. Repeat to add multiple tags.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Upload metadata/parquet only and skip the videos directory.",
    )
    parser.add_argument(
        "--upload-large-folder",
        action="store_true",
        help="Use the large-folder uploader for bigger datasets.",
    )
    parser.add_argument(
        "--no-tag-version",
        action="store_true",
        help="Do not create the LeRobot codebase-version tag on the Hub repo.",
    )
    parser.add_argument(
        "--delete-remote-backups",
        action="store_true",
        help="Delete existing *.bak files from the target Hub dataset repo after pushing.",
    )
    parser.add_argument(
        "--skip-explicit-metadata-upload",
        action="store_true",
        help=(
            "Do not explicitly upload meta/info.json, meta/tasks.parquet, meta/stats.json, "
            "and meta/real_exp_action_config.json after the normal LeRobot push."
        ),
    )
    return parser.parse_args()


def delete_remote_backup_files(repo_id: str, branch: str | None) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=branch)
    backup_files = [path for path in files if path.endswith(".bak")]
    if not backup_files:
        return []

    api.delete_files(
        repo_id=repo_id,
        delete_patterns=backup_files,
        repo_type="dataset",
        revision=branch,
        commit_message="Remove backup parquet files",
    )
    return backup_files


def read_actual_dataset_counts(dataset_root: Path) -> tuple[int, int] | None:
    episode_files = sorted((dataset_root / EPISODES_DIR).glob("**/*.parquet"))
    if not episode_files:
        return None

    import pandas as pd

    max_episode_index = -1
    total_frames = 0
    for episode_file in episode_files:
        episodes_df = pd.read_parquet(episode_file, columns=["episode_index", "length"])
        if episodes_df.empty:
            continue
        max_episode_index = max(max_episode_index, int(episodes_df["episode_index"].max()))
        total_frames += int(episodes_df["length"].sum())

    if max_episode_index < 0:
        return None
    return max_episode_index + 1, total_frames


def repair_info_if_needed(dataset_root: Path) -> bool:
    info_path = dataset_root / INFO_PATH
    actual_counts = read_actual_dataset_counts(dataset_root)
    if actual_counts is None:
        return False

    actual_episodes, actual_frames = actual_counts
    with info_path.open() as f:
        info = json.load(f)

    declared_episodes = int(info.get("total_episodes", 0))
    declared_frames = int(info.get("total_frames", 0))
    if declared_episodes == actual_episodes and declared_frames == actual_frames:
        return False

    if actual_episodes < declared_episodes or actual_frames < declared_frames:
        raise ValueError(
            "meta/info.json declares more data than episode metadata. "
            f"Declared episodes/frames: {declared_episodes}/{declared_frames}; "
            f"actual: {actual_episodes}/{actual_frames}."
        )

    info["total_episodes"] = actual_episodes
    info["total_frames"] = actual_frames
    info["splits"] = {"train": f"0:{actual_episodes}"}
    with info_path.open("w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print(
        "Updated meta/info.json from episode metadata: "
        f"episodes {declared_episodes} -> {actual_episodes}, "
        f"frames {declared_frames} -> {actual_frames}"
    )
    return True


def repair_tasks_if_needed(dataset_root: Path) -> bool:
    tasks_path = dataset_root / TASKS_PATH
    if not tasks_path.exists():
        return False

    import pandas as pd

    tasks = pd.read_parquet(tasks_path)
    if "task_index" not in tasks.columns:
        raise ValueError(f"{tasks_path} is missing task_index column.")

    task_rows_by_index = {
        int(task_index): str(task)
        for task, task_index in zip(tasks.index.tolist(), tasks["task_index"].tolist(), strict=True)
    }
    episode_tasks_by_index: dict[int, str] = {}
    for episode_file in sorted((dataset_root / EPISODES_DIR).glob("**/*.parquet")):
        episodes_df = pd.read_parquet(episode_file)
        if "tasks" not in episodes_df.columns:
            continue
        for task_list in episodes_df["tasks"].tolist():
            if task_list is None:
                continue
            for task in task_list:
                task_text = str(task)
                if task_text in tasks.index:
                    task_index = int(tasks.loc[task_text, "task_index"])
                else:
                    next_index = max([*task_rows_by_index.keys(), *episode_tasks_by_index.keys()], default=-1) + 1
                    task_index = next_index
                episode_tasks_by_index[task_index] = task_text

    data_task_indices: set[int] = set()
    for data_file in sorted((dataset_root / "data").glob("**/*.parquet")):
        data_df = pd.read_parquet(data_file, columns=["task_index"])
        data_task_indices.update(int(value) for value in data_df["task_index"].dropna().unique().tolist())

    missing_task_indices = sorted(data_task_indices - set(task_rows_by_index))
    if not missing_task_indices:
        return False

    repaired = dict(task_rows_by_index)
    for task_index in missing_task_indices:
        task_text = episode_tasks_by_index.get(task_index)
        if task_text is None:
            raise ValueError(
                f"Data files reference task_index {task_index}, but no matching task text was found "
                "in episode metadata."
            )
        repaired[task_index] = task_text

    repaired_tasks = pd.DataFrame(
        {"task_index": sorted(repaired)},
        index=pd.Index([repaired[index] for index in sorted(repaired)], name="task"),
    )
    repaired_tasks.to_parquet(tasks_path)
    print(
        "Updated meta/tasks.parquet from episode/data metadata: added task indices "
        f"{', '.join(str(index) for index in missing_task_indices)}"
    )
    return True


def repair_local_metadata(dataset_root: Path) -> None:
    repair_info_if_needed(dataset_root)
    repair_tasks_if_needed(dataset_root)


def upload_critical_metadata_files(repo_id: str, dataset_root: Path, branch: str | None) -> list[Path]:
    api = HfApi()
    uploaded_files = []
    for rel_path in CRITICAL_METADATA_FILES:
        local_path = dataset_root / rel_path
        if not local_path.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=rel_path.as_posix(),
            repo_id=repo_id,
            repo_type="dataset",
            revision=branch,
            commit_message=f"Update dataset metadata {rel_path.as_posix()}",
        )
        uploaded_files.append(rel_path)
    return uploaded_files


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not (dataset_root / INFO_PATH).exists():
        raise FileNotFoundError(
            f"{dataset_root} does not look like a LeRobot dataset root; missing {(dataset_root / INFO_PATH)}."
        )

    repair_local_metadata(dataset_root)
    dataset = ensure_dataset_stats(args.repo_id, dataset_root)
    print(f"Loaded dataset from {dataset.root}")
    print(f"Pushing to https://huggingface.co/datasets/{args.repo_id}")
    dataset.push_to_hub(
        branch=args.branch,
        tags=args.tags,
        license=args.license,
        tag_version=not args.no_tag_version,
        push_videos=not args.skip_videos,
        private=args.private,
        allow_patterns=DATASET_ALLOW_PATTERNS_NO_VIDEOS if args.skip_videos else DATASET_ALLOW_PATTERNS,
        upload_large_folder=args.upload_large_folder,
    )
    if not args.skip_explicit_metadata_upload:
        uploaded_files = upload_critical_metadata_files(args.repo_id, dataset_root, args.branch)
        if uploaded_files:
            print("Explicitly uploaded metadata files:")
            for path in uploaded_files:
                print(f"  {path.as_posix()}")
    if args.delete_remote_backups:
        deleted_files = delete_remote_backup_files(args.repo_id, args.branch)
        if deleted_files:
            print("Deleted remote backup files:")
            for path in deleted_files:
                print(f"  {path}")
        else:
            print("No remote backup files found.")
    print("Push complete.")


if __name__ == "__main__":
    main()
