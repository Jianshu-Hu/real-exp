from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

INFO_PATH = Path("meta/info.json")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not (dataset_root / INFO_PATH).exists():
        raise FileNotFoundError(
            f"{dataset_root} does not look like a LeRobot dataset root; missing {(dataset_root / INFO_PATH)}."
        )

    dataset = LeRobotDataset(repo_id=args.repo_id, root=dataset_root)
    print(f"Loaded dataset from {dataset.root}")
    print(f"Pushing to https://huggingface.co/datasets/{args.repo_id}")
    dataset.push_to_hub(
        branch=args.branch,
        tags=args.tags,
        license=args.license,
        tag_version=not args.no_tag_version,
        push_videos=not args.skip_videos,
        private=args.private,
        upload_large_folder=args.upload_large_folder,
    )
    print("Push complete.")


if __name__ == "__main__":
    main()
