from __future__ import annotations

import argparse
import os
from pathlib import Path

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy_config
from lerobot.scripts.lerobot_train import train


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "pick_and_place_test"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_HF_CACHE = REPO_ROOT / ".hf-cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LeRobot imitation-learning policy on a local dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the local LeRobot dataset root.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default="local/pick_and_place_test",
        help="Dataset repo id recorded in LeRobot metadata.",
    )
    parser.add_argument(
        "--policy-type",
        choices=("act", "diffusion"),
        default="act",
        help="Imitation-learning policy family to train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where checkpoints and logs will be written.",
    )
    parser.add_argument("--steps", type=int, default=50_000, help="Number of optimizer steps.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--save-freq", type=int, default=5_000, help="Checkpoint save frequency.")
    parser.add_argument("--log-freq", type=int, default=100, help="Logging frequency in steps.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override, for example cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default="real-exp",
        help="Weights & Biases project name if wandb is enabled.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Enable push-to-hub for the trained policy.",
    )
    parser.add_argument(
        "--policy-repo-id",
        default=None,
        help="Hub repo id for the policy. Required only when --push-to-hub is set.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output dir containing a prior training run.",
    )

    # ACT-specific knobs.
    parser.add_argument("--act-chunk-size", type=int, default=100, help="ACT action chunk size.")
    parser.add_argument(
        "--act-kl-weight", type=float, default=10.0, help="ACT KL loss weight."
    )

    # Diffusion-specific knobs.
    parser.add_argument("--diffusion-horizon", type=int, default=16, help="Diffusion horizon.")
    parser.add_argument(
        "--diffusion-n-obs-steps",
        type=int,
        default=2,
        help="Number of observation steps for diffusion policy.",
    )
    parser.add_argument(
        "--diffusion-n-action-steps",
        type=int,
        default=8,
        help="Number of action steps executed per diffusion rollout.",
    )

    return parser.parse_args()


def ensure_runtime_env() -> None:
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_CACHE))
    hf_datasets_cache = Path(
        os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets")
    )
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)


def build_policy_config(args: argparse.Namespace):
    common_kwargs = {
        "device": args.device,
        "push_to_hub": args.push_to_hub,
        "repo_id": args.policy_repo_id,
    }

    if args.policy_type == "act":
        return make_policy_config(
            "act",
            chunk_size=args.act_chunk_size,
            n_action_steps=args.act_chunk_size,
            kl_weight=args.act_kl_weight,
            **common_kwargs,
        )

    if args.policy_type == "diffusion":
        return make_policy_config(
            "diffusion",
            horizon=args.diffusion_horizon,
            n_obs_steps=args.diffusion_n_obs_steps,
            n_action_steps=args.diffusion_n_action_steps,
            **common_kwargs,
        )

    raise ValueError(f"Unsupported policy type: {args.policy_type}")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (DEFAULT_OUTPUT_ROOT / f"{args.dataset_root.name}_{args.policy_type}").resolve()


def main() -> None:
    args = parse_args()
    ensure_runtime_env()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError("--policy-repo-id is required when --push-to-hub is set.")

    output_dir = resolve_output_dir(args)

    dataset_cfg = DatasetConfig(
        repo_id=args.dataset_repo_id,
        root=str(dataset_root),
    )
    policy_cfg = build_policy_config(args)
    wandb_cfg = WandBConfig(
        enable=not args.disable_wandb,
        project=args.wandb_project,
        mode="disabled" if args.disable_wandb else None,
    )

    cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=f"{args.dataset_root.name}_{args.policy_type}",
        resume=args.resume,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        steps=args.steps,
        eval_freq=0,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        wandb=wandb_cfg,
    )

    train(cfg)


if __name__ == "__main__":
    main()
