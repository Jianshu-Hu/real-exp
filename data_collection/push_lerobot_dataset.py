from __future__ import annotations

import argparse
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
        default=None,
        help="Optional Hub branch to push to.",
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


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not (dataset_root / INFO_PATH).exists():
        raise FileNotFoundError(
            f"{dataset_root} does not look like a LeRobot dataset root; missing {(dataset_root / INFO_PATH)}."
        )

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
