from __future__ import annotations

import argparse
import os
import shutil
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
        default="main",
        help=(
            "Hub branch, tag, or commit to fetch. "
            "Defaults to 'main' so the local policy matches the remote branch by default."
        ),
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
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help=(
            "Delete the target local policy directory before fetching. "
            "Enabled by default so fetch replaces the local copy with the current remote contents."
        ),
    )
    parser.add_argument(
        "--no-clean",
        dest="clean",
        action="store_false",
        help="Keep the existing local policy directory instead of replacing it before fetch.",
    )
    return parser.parse_args()


def ensure_runtime_env() -> None:
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_CACHE))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)


def default_local_dir(repo_id: str) -> Path:
    repo_name = repo_id.split("/")[-1]
    return DEFAULT_OUTPUT_ROOT / repo_name


def remove_existing_policy(local_dir: Path) -> None:
    if not local_dir.exists():
        return
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Cannot clean non-directory path: {local_dir}")

    resolved_repo_root = REPO_ROOT.resolve()
    resolved_outputs_root = (REPO_ROOT / "outputs").resolve()
    resolved_local_dir = local_dir.resolve()
    if resolved_local_dir in (resolved_repo_root, resolved_outputs_root, Path("/")):
        raise ValueError(f"Refusing to clean unsafe policy path: {resolved_local_dir}")
    if resolved_outputs_root not in resolved_local_dir.parents and resolved_repo_root not in resolved_local_dir.parents:
        raise ValueError(
            f"Refusing to clean path outside this repository. Remove it manually if intended: {resolved_local_dir}"
        )

    print(f"Removing existing local policy: {resolved_local_dir}")
    shutil.rmtree(resolved_local_dir)


def main() -> None:
    args = parse_args()
    ensure_runtime_env()

    local_dir = (
        args.local_dir.expanduser().resolve()
        if args.local_dir is not None
        else default_local_dir(args.repo_id).resolve()
    )
    if args.clean:
        remove_existing_policy(local_dir)
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
