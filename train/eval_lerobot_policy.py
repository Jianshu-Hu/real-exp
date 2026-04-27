from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE = REPO_ROOT / ".hf-cache"
DEFAULT_HF_DATASETS_CACHE = DEFAULT_HF_CACHE / "datasets"
os.environ["HF_HOME"] = str(DEFAULT_HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(DEFAULT_HF_DATASETS_CACHE)

DATA_COLLECTION_DIR = REPO_ROOT / "data_collection"
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))

from accelerate import Accelerator

from lerobot.configs.default import DatasetConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from image_preprocessing import ResizePadConfig, infer_square_resize_pad_size_from_policy_features
from train_lerobot_policy import (
    DEFAULT_DATASET_ROOT,
    apply_dataset_image_transform,
    build_dataloader,
    ensure_runtime_env,
    evaluate_validation_loss,
    require_absolute_joint_action_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved LeRobot checkpoints on a held-out validation split using deployment-style "
            "policy inference."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Root directory containing training runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Specific run directory to evaluate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--policy-type",
        choices=("act", "diffusion", "all"),
        default="all",
        help="Policy type to evaluate when discovering runs or validating explicit run dirs.",
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("latest", "all"),
        default="latest",
        help="Evaluate only the latest checkpoint in each run, or all checkpoints.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override dataset root. Defaults to the checkpoint train_config dataset root.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        default=None,
        help="Override dataset repo id. Defaults to the checkpoint train_config dataset repo id.",
    )
    parser.add_argument(
        "--val-episodes",
        type=int,
        nargs="*",
        default=None,
        help="Explicit validation episode ids. Defaults to the held-out complement of train_config.dataset.episodes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Validation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Validation dataloader worker count.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on validation batches per checkpoint.",
    )
    parser.add_argument(
        "--policy-device",
        default="cpu",
        help="Device override passed into the policy config, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--diffusion-noise-scheduler-type",
        choices=("DDPM", "DDIM"),
        default="DDIM",
        help="Inference-time diffusion scheduler override. Ignored for ACT checkpoints.",
    )
    parser.add_argument(
        "--diffusion-num-inference-steps",
        type=int,
        default=10,
        help="Inference-time reverse diffusion step override. Ignored for ACT checkpoints.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the evaluation summary JSON.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def policy_type_matches(policy_type: str, requested_policy_type: str) -> bool:
    return requested_policy_type == "all" or policy_type == requested_policy_type


def discover_run_dirs(outputs_root: Path, requested_policy_type: str) -> list[Path]:
    run_dirs: list[Path] = []
    for candidate in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        checkpoint_root = candidate / "checkpoints"
        if not checkpoint_root.exists():
            continue
        configs = sorted(checkpoint_root.glob("*/pretrained_model/config.json"))
        if not configs:
            continue
        config = load_json(configs[-1])
        policy_type = str(config.get("type", ""))
        if policy_type_matches(policy_type, requested_policy_type):
            run_dirs.append(candidate)
    return run_dirs


def resolve_checkpoint_dirs(run_dir: Path, selection: str) -> list[Path]:
    checkpoint_dirs: list[Path] = []
    for path in (run_dir / "checkpoints").iterdir():
        if not path.is_dir():
            continue
        if not path.name.isdigit():
            continue
        pretrained_dir = path / "pretrained_model"
        if pretrained_dir.is_dir():
            checkpoint_dirs.append(pretrained_dir)
    checkpoint_dirs.sort(key=lambda path: int(path.parent.name))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found under {run_dir / 'checkpoints'}")
    if selection == "latest":
        return [checkpoint_dirs[-1]]
    return checkpoint_dirs


def infer_validation_episodes(
    train_config: dict[str, Any],
    dataset_root: Path,
    explicit_val_episodes: list[int] | None,
) -> list[int]:
    if explicit_val_episodes is not None:
        return sorted(explicit_val_episodes)

    dataset_info = load_json(dataset_root / "meta" / "info.json")
    total_episodes = int(dataset_info["total_episodes"])
    train_episodes = train_config.get("dataset", {}).get("episodes")
    if not isinstance(train_episodes, list):
        raise ValueError(
            "train_config.json does not contain dataset.episodes, and --val-episodes was not provided."
        )

    held_out = sorted(set(range(total_episodes)) - set(int(episode) for episode in train_episodes))
    if not held_out:
        raise ValueError(
            "No held-out validation episodes were inferred from train_config.json. "
            "Pass --val-episodes explicitly."
        )
    return held_out


def make_dataset_cfg(repo_id: str, root: Path, episodes: list[int]) -> DatasetConfig:
    return DatasetConfig(repo_id=repo_id, root=str(root), episodes=episodes)


def build_resize_pad_config(policy) -> ResizePadConfig:
    image_size = infer_square_resize_pad_size_from_policy_features(policy.config.image_features)
    if image_size is None:
        return ResizePadConfig(enabled=False, size=224, fill=0.0)
    return ResizePadConfig(enabled=True, size=image_size, fill=0.0)


def build_policy_cli_overrides(policy_type: str, args: argparse.Namespace) -> list[str]:
    cli_overrides = [f"--device={args.policy_device}"]
    if policy_type == "diffusion":
        cli_overrides.extend(
            [
                f"--noise_scheduler_type={args.diffusion_noise_scheduler_type}",
                f"--num_inference_steps={args.diffusion_num_inference_steps}",
            ]
        )
    return cli_overrides


def evaluate_checkpoint(
    checkpoint_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    train_config_path = checkpoint_dir / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(f"Missing train_config.json at {train_config_path}")

    train_config = load_json(train_config_path)
    dataset_root = (args.dataset_root or Path(train_config["dataset"]["root"]) or DEFAULT_DATASET_ROOT).resolve()
    dataset_repo_id = args.dataset_repo_id or str(train_config["dataset"]["repo_id"])
    val_episodes = infer_validation_episodes(train_config, dataset_root, args.val_episodes)

    require_absolute_joint_action_dataset(dataset_root)

    policy_config = load_json(checkpoint_dir / "config.json")
    policy_type = str(policy_config["type"])
    if not policy_type_matches(policy_type, args.policy_type):
        raise ValueError(
            f"Checkpoint {checkpoint_dir} is policy type {policy_type}, "
            f"but --policy-type={args.policy_type} was requested."
        )

    cli_overrides = build_policy_cli_overrides(policy_type, args)
    policy_class = get_policy_class(policy_type)
    policy = policy_class.from_pretrained(checkpoint_dir, cli_overrides=cli_overrides)

    resize_pad_config = build_resize_pad_config(policy)
    dataset_cfg = make_dataset_cfg(dataset_repo_id, dataset_root, val_episodes)
    build_cfg = SimpleNamespace(
        dataset=dataset_cfg,
        policy=policy.config,
        num_workers=args.num_workers,
        tolerance_s=float(train_config.get("tolerance_s", 1e-4)),
    )
    val_dataset = make_dataset(build_cfg)
    apply_dataset_image_transform(val_dataset, resize_pad_config)

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=checkpoint_dir,
        preprocessor_overrides={"device_processor": {"device": args.policy_device}},
        postprocessor_overrides={"device_processor": {"device": args.policy_device}},
    )

    accelerator = Accelerator(cpu=args.policy_device == "cpu")
    val_dataloader = build_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        shuffle=False,
    )
    policy = accelerator.prepare(policy)
    val_dataloader = accelerator.prepare_data_loader(val_dataloader)

    val_loss = evaluate_validation_loss(
        policy=policy,
        dataloader=val_dataloader,
        preprocessor=preprocessor,
        accelerator=accelerator,
        max_batches=args.max_batches,
    )

    return {
        "run_dir": str(checkpoint_dir.parent.parent.parent),
        "checkpoint_step": int(checkpoint_dir.parent.name),
        "checkpoint_dir": str(checkpoint_dir),
        "policy_type": policy_type,
        "dataset_root": str(dataset_root),
        "dataset_repo_id": dataset_repo_id,
        "val_episodes": val_episodes,
        "policy_device": args.policy_device,
        "noise_scheduler_type": args.diffusion_noise_scheduler_type if policy_type == "diffusion" else None,
        "num_inference_steps": args.diffusion_num_inference_steps if policy_type == "diffusion" else None,
        "val_loss": float(val_loss),
    }


def main() -> None:
    args = parse_args()
    ensure_runtime_env()

    run_dirs = (
        [path.resolve() for path in args.run_dir]
        if args.run_dir
        else discover_run_dirs(args.outputs_root, args.policy_type)
    )
    if not run_dirs:
        raise FileNotFoundError(
            f"No {args.policy_type} run directories found under {args.outputs_root}"
        )

    results: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        checkpoint_dirs = resolve_checkpoint_dirs(run_dir, args.checkpoint_selection)
        for checkpoint_dir in checkpoint_dirs:
            result = evaluate_checkpoint(checkpoint_dir, args)
            results.append(result)
            message = (
                f"{result['run_dir']} "
                f"type={result['policy_type']} "
                f"step={result['checkpoint_step']} "
                f"val_loss={result['val_loss']:.6f}"
            )
            if result["policy_type"] == "diffusion":
                message += (
                    f" scheduler={result['noise_scheduler_type']} "
                    f"steps={result['num_inference_steps']}"
                )
            print(message)

    results.sort(key=lambda item: (Path(item["run_dir"]).name, item["policy_type"], item["checkpoint_step"]))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
