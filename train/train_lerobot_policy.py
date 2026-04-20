from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_COLLECTION_DIR = REPO_ROOT / "data_collection"
if str(DATA_COLLECTION_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_COLLECTION_DIR))

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from lerobot.configs.default import DatasetConfig, WandBConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_train import update_policy
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import init_logging

from dataset_stats import ensure_dataset_stats
from image_preprocessing import (
    ResizePadConfig,
    apply_resize_pad_to_feature_specs,
    make_resize_pad_transform,
)

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "pick_and_place_test"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs"
DEFAULT_HF_CACHE = REPO_ROOT / ".hf-cache"


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader worker count.")
    parser.add_argument("--save-freq", type=int, default=2_500, help="Checkpoint save frequency.")
    parser.add_argument("--log-freq", type=int, default=100, help="Logging frequency in steps.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed.")
    parser.add_argument(
        "--image-resize-pad-size",
        type=int,
        default=224,
        help="Resize camera images aspect-preservingly and pad to this square size. Set <=0 to disable.",
    )
    parser.add_argument(
        "--image-resize-pad-fill",
        type=float,
        default=0.0,
        help="Constant fill value used for padded image regions.",
    )
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

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Fraction of episodes reserved for validation.",
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=None,
        help="Validation frequency in training steps. Defaults to --save-freq. Set to 0 to disable validation.",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=None,
        help="Validation batch size. Defaults to --batch-size.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional cap on validation batches per evaluation pass.",
    )

    # ACT-specific knobs.
    parser.add_argument("--act-chunk-size", type=int, default=32, help="ACT action chunk size.")
    parser.add_argument("--act-kl-weight", type=float, default=10.0, help="ACT KL loss weight.")

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
    hf_datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets"))
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


def resolve_resize_pad_config(args: argparse.Namespace) -> ResizePadConfig:
    return ResizePadConfig(
        enabled=args.image_resize_pad_size > 0,
        size=max(1, args.image_resize_pad_size),
        fill=args.image_resize_pad_fill,
    )


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (DEFAULT_OUTPUT_ROOT / f"{args.dataset_root.name}_{args.policy_type}").resolve()


def resolve_episode_split(args: argparse.Namespace, total_episodes: int) -> tuple[list[int], list[int]]:
    all_episodes = list(range(total_episodes))

    if args.val_ratio <= 0:
        return all_episodes, []

    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be in (0, 1) when provided.")

    val_count = max(1, int(math.ceil(total_episodes * args.val_ratio)))
    if val_count >= total_episodes:
        raise ValueError("Validation split would consume all episodes.")

    shuffled_episodes = all_episodes.copy()
    random.Random(args.seed).shuffle(shuffled_episodes)
    val_episode_set = set(shuffled_episodes[:val_count])
    train_episodes = [episode for episode in all_episodes if episode not in val_episode_set]
    val_episodes = [episode for episode in all_episodes if episode in val_episode_set]
    return train_episodes, val_episodes


def make_local_dataset_cfg(
    repo_id: str,
    root: Path,
    episodes: list[int],
) -> DatasetConfig:
    return DatasetConfig(repo_id=repo_id, root=str(root), episodes=episodes)


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
):
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def apply_dataset_image_transform(dataset, resize_pad_config: ResizePadConfig) -> None:
    apply_resize_pad_to_feature_specs(dataset.meta.info["features"], resize_pad_config)
    transform = make_resize_pad_transform(resize_pad_config)
    if transform is not None:
        dataset.set_image_transforms(transform)


def evaluate_validation_loss(
    policy: PreTrainedPolicy,
    dataloader,
    preprocessor,
    accelerator: Accelerator,
    max_batches: int | None,
) -> float:
    was_training = policy.training
    policy.train()
    losses: list[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = preprocessor(batch)
            with accelerator.autocast():
                loss, _ = policy.forward(batch)
            reduced_loss = accelerator.gather_for_metrics(loss.detach().reshape(1))
            losses.extend(float(value.item()) for value in reduced_loss)

    if not was_training:
        policy.eval()
    if not losses:
        raise ValueError("Validation dataloader produced no batches.")
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    ensure_runtime_env()
    val_freq = args.save_freq if args.val_freq is None else args.val_freq

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    ensure_dataset_stats(args.dataset_repo_id, dataset_root)

    if args.push_to_hub and not args.policy_repo_id:
        raise ValueError("--policy-repo-id is required when --push-to-hub is set.")

    dataset_info_path = dataset_root / "meta" / "info.json"
    if not dataset_info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {dataset_info_path}")

    dataset_info = json.loads(dataset_info_path.read_text())
    total_episodes = int(dataset_info["total_episodes"])
    train_episodes, val_episodes = resolve_episode_split(args, total_episodes)
    resize_pad_config = resolve_resize_pad_config(args)
    apply_resize_pad_to_feature_specs(dataset_info["features"], resize_pad_config)

    if not train_episodes:
        raise ValueError("Training split is empty.")
    if val_freq > 0 and not val_episodes:
        raise ValueError("Validation is enabled but validation split is empty.")

    output_dir = resolve_output_dir(args)
    policy_cfg = build_policy_config(args)
    wandb_cfg = WandBConfig(
        enable=not args.disable_wandb,
        project=args.wandb_project,
        mode="disabled" if args.disable_wandb else None,
    )
    train_dataset_cfg = make_local_dataset_cfg(
        args.dataset_repo_id,
        dataset_root,
        train_episodes,
    )

    cfg = TrainPipelineConfig(
        dataset=train_dataset_cfg,
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
    cfg.validate()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    force_cpu = cfg.policy.device == "cpu"
    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
        cpu=force_cpu,
    )
    init_logging(accelerator=accelerator)

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    is_main_process = accelerator.is_main_process
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        print(f"Train episodes: {train_episodes}")
        print(f"Validation episodes: {val_episodes}")
        print(f"Validation frequency: {val_freq}")

    train_dataset = make_dataset(cfg)
    apply_dataset_image_transform(train_dataset, resize_pad_config)
    val_dataset = None
    if val_episodes:
        val_cfg = TrainPipelineConfig(
            dataset=make_local_dataset_cfg(
                args.dataset_repo_id,
                dataset_root,
                val_episodes,
            ),
            policy=policy_cfg,
            output_dir=output_dir,
            job_name=cfg.job_name,
            resume=False,
            seed=args.seed,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            steps=args.steps,
            eval_freq=0,
            log_freq=args.log_freq,
            save_freq=args.save_freq,
            wandb=wandb_cfg,
        )
        val_dataset = make_dataset(val_cfg)
        apply_dataset_image_transform(val_dataset, resize_pad_config)

    policy = make_policy(cfg=cfg.policy, ds_meta=train_dataset.meta, rename_map=cfg.rename_map)

    processor_kwargs: dict[str, Any] = {}
    postprocessor_kwargs: dict[str, Any] = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = train_dataset.meta.stats

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        shuffle=True,
    )
    val_batch_size = args.val_batch_size or args.batch_size
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=val_batch_size,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            shuffle=False,
        )

    policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, train_dataloader, lr_scheduler
    )
    if val_dataloader is not None:
        val_dataloader = accelerator.prepare_data_loader(val_dataloader)

    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size,
        train_dataset.num_frames,
        train_dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    train_iter = iter(train_dataloader)
    policy.train()
    loop_start_time = time.perf_counter()
    start_step = step

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )

        step += 1
        train_tracker.step()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_val_step = val_freq > 0 and step % val_freq == 0 and val_dataloader is not None

        if is_log_step:
            elapsed_s = time.perf_counter() - loop_start_time
            completed_steps = max(1, step - start_step)
            seconds_per_step = elapsed_s / completed_steps
            remaining_steps = max(0, cfg.steps - step)
            eta_s = seconds_per_step * remaining_steps
            print(
                f"{train_tracker} elapsed:{format_duration(elapsed_s)} eta:{format_duration(eta_s)}"
            )
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_log_dict["elapsed_s"] = elapsed_s
                wandb_log_dict["eta_s"] = eta_s
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)
            accelerator.wait_for_everyone()

        if is_val_step:
            val_loss = evaluate_validation_loss(
                policy=policy,
                dataloader=val_dataloader,
                preprocessor=preprocessor,
                accelerator=accelerator,
                max_batches=args.max_val_batches,
            )
            if is_main_process:
                print(f"validation step={step} loss={val_loss:.6f}")
                if wandb_logger:
                    wandb_logger.log_dict({"val_loss": val_loss}, step=step, mode="eval")


if __name__ == "__main__":
    main()
