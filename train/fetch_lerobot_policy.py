from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE = REPO_ROOT / ".hf-cache"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "fetched_policies"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a LeRobot policy from the Hugging Face Hub into a local directory."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face model repo id, for example <user>/<policy-name>.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help=(
            "Local directory for the downloaded policy. "
            "Defaults to outputs/fetched_policies/<repo-name>."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hub branch, tag, or commit to fetch.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to the cached login token.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local Hub cache files and avoid network access.",
    )
    return parser.parse_args()


def ensure_runtime_env() -> None:
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_CACHE))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)


def default_local_dir(repo_id: str) -> Path:
    repo_name = repo_id.split("/")[-1]
    return DEFAULT_OUTPUT_ROOT / repo_name


def main() -> None:
    args = parse_args()
    ensure_runtime_env()

    local_dir = (
        args.local_dir.expanduser().resolve()
        if args.local_dir is not None
        else default_local_dir(args.repo_id).resolve()
    )
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    resolved_dir = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        token=args.token,
        local_dir=local_dir,
        local_files_only=args.local_files_only,
    )
    print(f"Fetched policy {args.repo_id}")
    print(f"Local path: {resolved_dir}")


if __name__ == "__main__":
    main()
