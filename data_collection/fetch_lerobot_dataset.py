from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

INFO_PATH = Path("meta/info.json")
EPISODES_DIR = Path("meta/episodes")
TASKS_PATH = Path("meta/tasks.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a LeRobot dataset from the Hugging Face Hub into a local directory."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id, for example <user>/<dataset-name>.",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help=(
            "Local directory for the downloaded dataset. "
            "Defaults to data/<repo-name> under this repository."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help=(
            "Hub branch, tag, or commit to fetch. "
            "Defaults to 'main' so the local dataset matches the remote branch by default."
        ),
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Download metadata/parquet only and skip videos.",
    )
    parser.add_argument(
        "--force-cache-sync",
        action="store_true",
        default=True,
        help="Refresh metadata/cache bookkeeping when loading an existing local copy.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help=(
            "Delete the target local dataset directory before fetching. "
            "Enabled by default so fetch replaces the local copy with the current remote contents."
        ),
    )
    parser.add_argument(
        "--repair-info",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After fetching, update derived local metadata if downloaded episode/data files are newer "
            "than the dataset-level metadata. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Keep the existing local dataset directory instead of replacing it before fetch.",
    )
    parser.add_argument(
        "--no-force-cache-sync",
        dest="force_cache_sync",
        action="store_false",
        help="Do not force a refresh when loading an existing local copy.",
    )
    return parser.parse_args()


def default_local_dir(repo_id: str) -> Path:
    repo_name = repo_id.split("/")[-1]
    return REPO_ROOT / "data" / repo_name


def remove_existing_dataset(local_dir: Path) -> None:
    if not local_dir.exists():
        return
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Cannot clean non-directory path: {local_dir}")

    resolved_repo_root = REPO_ROOT.resolve()
    resolved_data_root = (REPO_ROOT / "data").resolve()
    resolved_local_dir = local_dir.resolve()
    if resolved_local_dir in (resolved_repo_root, resolved_data_root, Path("/")):
        raise ValueError(f"Refusing to clean unsafe dataset path: {resolved_local_dir}")
    if resolved_data_root not in resolved_local_dir.parents and resolved_repo_root not in resolved_local_dir.parents:
        raise ValueError(
            f"Refusing to clean path outside this repository. Remove it manually if intended: {resolved_local_dir}"
        )

    print(f"Removing existing local dataset: {resolved_local_dir}")
    shutil.rmtree(resolved_local_dir)


def read_actual_dataset_counts(local_dir: Path) -> tuple[int, int] | None:
    episode_files = sorted((local_dir / EPISODES_DIR).glob("**/*.parquet"))
    if not episode_files:
        return None

    import pandas as pd

    max_episode_index = -1
    total_frames = 0
    for episode_file in episode_files:
        episodes_df = pd.read_parquet(
            episode_file,
            columns=["episode_index", "length"],
        )
        if episodes_df.empty:
            continue
        max_episode_index = max(max_episode_index, int(episodes_df["episode_index"].max()))
        total_frames += int(episodes_df["length"].sum())

    if max_episode_index < 0:
        return None
    return max_episode_index + 1, total_frames


def repair_info_if_needed(local_dir: Path) -> bool:
    info_path = local_dir / INFO_PATH
    if not info_path.exists():
        return False

    actual_counts = read_actual_dataset_counts(local_dir)
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
        print(
            "WARNING: meta/info.json declares more data than episode metadata. "
            f"Declared episodes/frames: {declared_episodes}/{declared_frames}; "
            f"actual: {actual_episodes}/{actual_frames}. Not repairing automatically."
        )
        return False

    info["total_episodes"] = actual_episodes
    info["total_frames"] = actual_frames
    info["splits"] = {"train": f"0:{actual_episodes}"}
    with info_path.open("w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print(
        "Repaired stale meta/info.json: "
        f"episodes {declared_episodes} -> {actual_episodes}, "
        f"frames {declared_frames} -> {actual_frames}"
    )
    return True


def repair_tasks_if_needed(local_dir: Path) -> bool:
    tasks_path = local_dir / TASKS_PATH
    if not tasks_path.exists():
        return False

    import pandas as pd

    tasks = pd.read_parquet(tasks_path)
    if "task_index" not in tasks.columns:
        print(f"WARNING: {tasks_path} is missing task_index column. Not repairing automatically.")
        return False

    task_rows_by_index = {
        int(task_index): str(task)
        for task, task_index in zip(tasks.index.tolist(), tasks["task_index"].tolist(), strict=True)
    }
    episode_files = sorted((local_dir / EPISODES_DIR).glob("**/*.parquet"))
    episode_tasks_by_index: dict[int, str] = {}
    for episode_file in episode_files:
        try:
            episodes_df = pd.read_parquet(episode_file, columns=["tasks"])
        except (KeyError, ValueError):
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
    for data_file in sorted((local_dir / "data").glob("**/*.parquet")):
        try:
            data_df = pd.read_parquet(data_file, columns=["task_index"])
        except (KeyError, ValueError):
            continue
        data_task_indices.update(int(value) for value in data_df["task_index"].dropna().unique().tolist())

    missing_task_indices = sorted(data_task_indices - set(task_rows_by_index))
    if not missing_task_indices:
        return False

    repaired = dict(task_rows_by_index)
    for task_index in missing_task_indices:
        task_text = episode_tasks_by_index.get(task_index)
        if task_text is None:
            print(
                "WARNING: data files reference task_index "
                f"{task_index}, but no matching task text was found in episode metadata. "
                "Not repairing tasks automatically."
            )
            return False
        repaired[task_index] = task_text

    repaired_tasks = pd.DataFrame(
        {"task_index": sorted(repaired)},
        index=pd.Index([repaired[index] for index in sorted(repaired)], name="task"),
    )
    repaired_tasks.to_parquet(tasks_path)
    print(
        "Repaired stale meta/tasks.parquet: added task indices "
        f"{', '.join(str(index) for index in missing_task_indices)}"
    )
    return True


def repair_metadata_if_needed(local_dir: Path) -> bool:
    repaired_info = repair_info_if_needed(local_dir)
    repaired_tasks = repair_tasks_if_needed(local_dir)
    return repaired_info or repaired_tasks


def main() -> None:
    args = parse_args()
    local_dir = (
        Path(args.local_dir).expanduser().resolve()
        if args.local_dir is not None
        else default_local_dir(args.repo_id).resolve()
    )
    if args.clean:
        remove_existing_dataset(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=local_dir,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        download_videos=not args.skip_videos,
    )
    if args.repair_info and repair_metadata_if_needed(local_dir):
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=local_dir,
            revision=args.revision,
            force_cache_sync=False,
            download_videos=not args.skip_videos,
        )
    print(f"Fetched dataset {args.repo_id}")
    print(f"Local root: {dataset.root}")
    print(f"Episodes: {dataset.num_episodes}")
    print(f"Frames: {dataset.num_frames}")


if __name__ == "__main__":
    main()
