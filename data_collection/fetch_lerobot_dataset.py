from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if str(LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


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
        default=None,
        help="Optional Hub branch, tag, or commit to fetch.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Download metadata/parquet only and skip videos.",
    )
    parser.add_argument(
        "--force-cache-sync",
        action="store_true",
        help="Refresh metadata/cache bookkeeping when loading an existing local copy.",
    )
    return parser.parse_args()


def default_local_dir(repo_id: str) -> Path:
    repo_name = repo_id.split("/")[-1]
    return REPO_ROOT / "data" / repo_name


def main() -> None:
    args = parse_args()
    local_dir = (
        Path(args.local_dir).expanduser().resolve()
        if args.local_dir is not None
        else default_local_dir(args.repo_id).resolve()
    )
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=local_dir,
        revision=args.revision,
        force_cache_sync=args.force_cache_sync,
        download_videos=not args.skip_videos,
    )
    print(f"Fetched dataset {args.repo_id}")
    print(f"Local root: {dataset.root}")
    print(f"Episodes: {dataset.num_episodes}")
    print(f"Frames: {dataset.num_frames}")


if __name__ == "__main__":
    main()
