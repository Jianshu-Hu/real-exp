from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE = REPO_ROOT / ".hf-cache"
PRETRAINED_MODEL_DIRNAME = "pretrained_model"
CONFIG_FILENAME = "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push a saved LeRobot policy directory to the Hugging Face Hub."
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        required=True,
        help="Path to a saved policy directory or to a checkpoint directory containing pretrained_model/.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face model repo id, for example <user>/<policy-name>.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional Hub branch to push to.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the Hub model as a private repo.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to the cached login token.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload LeRobot policy",
        help="Commit message used for the Hub upload.",
    )
    return parser.parse_args()


def ensure_runtime_env() -> None:
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_CACHE))
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)


def resolve_policy_dir(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if (candidate / CONFIG_FILENAME).exists():
        return candidate

    nested = candidate / PRETRAINED_MODEL_DIRNAME
    if (nested / CONFIG_FILENAME).exists():
        return nested

    raise FileNotFoundError(
        f"Could not find {CONFIG_FILENAME} in {candidate} or {nested}. "
        "Pass a saved policy directory or a checkpoint directory containing pretrained_model/."
    )


def main() -> None:
    args = parse_args()
    ensure_runtime_env()

    policy_dir = resolve_policy_dir(args.policy_path)
    api = HfApi(token=args.token)
    repo_id = api.create_repo(
        repo_id=args.repo_id,
        private=args.private,
        exist_ok=True,
        repo_type="model",
    ).repo_id

    if args.branch:
        api.create_branch(
            repo_id=repo_id,
            branch=args.branch,
            repo_type="model",
            exist_ok=True,
        )

    print(f"Resolved local policy directory: {policy_dir}")
    print(f"Pushing to https://huggingface.co/{repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=policy_dir,
        repo_type="model",
        revision=args.branch,
        commit_message=args.commit_message,
    )
    print("Push complete.")


if __name__ == "__main__":
    main()
