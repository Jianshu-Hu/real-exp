from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

from image_preprocessing import ResizePadSquare, infer_square_resize_pad_size_from_policy_features
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


ARM_INDICES = np.array([*range(7), *range(8, 15)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline audit of a LeRobot checkpoint against recorded observations and actions."
    )
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", default="local/audit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--startup-frames",
        type=int,
        default=15,
        help="Audit this many initial frames of every episode (15 frames = one second at 15 Hz).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Also audit every Nth frame across each episode. Set 0 to audit startup frames only.",
    )
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--max-lag-frames", type=int, default=12)
    parser.add_argument(
        "--diffusion-noise-seed",
        type=int,
        default=0,
        help="Use the same deterministic DDIM noise seed as deployment. Set a negative value for random noise.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def load_dataset_arrays(root: Path) -> tuple[dict[int, dict[str, np.ndarray]], int]:
    episodes: dict[int, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"state": [], "action": [], "frame": []}
    )
    total_frames = 0
    for parquet_path in sorted((root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(parquet_path, columns=["episode_index", "frame_index", "observation.state", "action"])
        for episode, frame, state, action in zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            table["observation.state"].to_pylist(),
            table["action"].to_pylist(),
            strict=True,
        ):
            episodes[int(episode)]["state"].append(np.asarray(state, dtype=np.float64))
            episodes[int(episode)]["action"].append(np.asarray(action, dtype=np.float64))
            episodes[int(episode)]["frame"].append(np.asarray(frame, dtype=np.int64))
            total_frames += 1

    finalized: dict[int, dict[str, np.ndarray]] = {}
    for episode, values in episodes.items():
        order = np.argsort(np.asarray(values["frame"]))
        finalized[episode] = {
            key: np.asarray(value)[order] for key, value in values.items()
        }
    return finalized, total_frames


def audit_recorded_actions(root: Path, fps: float, max_lag_frames: int) -> dict[str, Any]:
    episodes, total_frames = load_dataset_arrays(root)
    action_minus_state: list[float] = []
    action_step: list[float] = []
    action_velocity: list[float] = []
    action_acceleration: list[float] = []
    lag_errors: dict[int, list[float]] = defaultdict(list)

    for values in episodes.values():
        state = values["state"][:, ARM_INDICES]
        action = values["action"][:, ARM_INDICES]
        action_minus_state.extend(np.abs(action - state).reshape(-1).tolist())
        if len(action) > 1:
            first_difference = np.abs(np.diff(action, axis=0))
            action_step.extend(first_difference.reshape(-1).tolist())
            action_velocity.extend((first_difference * fps).reshape(-1).tolist())
        if len(action) > 2:
            second_difference = np.abs(np.diff(action, n=2, axis=0))
            action_acceleration.extend((second_difference * fps * fps).reshape(-1).tolist())

        for lag in range(max_lag_frames + 1):
            if len(state) <= lag:
                continue
            lag_errors[lag].extend(np.abs(action[: len(state) - lag] - state[lag:]).reshape(-1).tolist())

    lag_summary = {str(lag): quantiles(errors) for lag, errors in lag_errors.items()}
    best_lag = min(lag_errors, key=lambda lag: float(np.mean(lag_errors[lag])))
    return {
        "episodes": len(episodes),
        "frames": total_frames,
        "arm_action_minus_state_rad": quantiles(action_minus_state),
        "arm_action_step_rad": quantiles(action_step),
        "arm_action_velocity_rad_per_sec": quantiles(action_velocity),
        "arm_action_acceleration_rad_per_sec2": quantiles(action_acceleration),
        "state_action_alignment": {
            "best_nonnegative_lag_frames": int(best_lag),
            "per_lag_abs_error_rad": lag_summary,
        },
    }


def selected_indices(dataset: LeRobotDataset, startup_frames: int, stride: int, max_samples: int) -> list[int]:
    raw = dataset.hf_dataset.select_columns(["episode_index", "frame_index"])
    selected: list[int] = []
    for index, (episode, frame) in enumerate(zip(raw["episode_index"], raw["frame_index"], strict=True)):
        del episode  # Keeping the explicit unpack makes the selection contract clear.
        if int(frame) < startup_frames or (stride > 0 and int(frame) % stride == 0):
            selected.append(index)
        if len(selected) >= max_samples:
            break
    return selected


def make_raw_prediction(policy, policy_type: str, processed: dict[str, torch.Tensor], image_keys: list[str], seed: int):
    if policy_type == "act":
        return policy.predict_action_chunk(processed)

    inference_batch = {OBS_STATE: processed[OBS_STATE]}
    if image_keys:
        inference_batch[OBS_IMAGES] = torch.stack([processed[key] for key in image_keys], dim=-4)

    noise = None
    if seed >= 0:
        generator = torch.Generator(device=inference_batch[OBS_STATE].device)
        generator.manual_seed(seed)
        noise = torch.randn(
            size=(
                inference_batch[OBS_STATE].shape[0],
                policy.config.horizon,
                policy.config.action_feature.shape[0],
            ),
            device=inference_batch[OBS_STATE].device,
            dtype=inference_batch[OBS_STATE].dtype,
            generator=generator,
        )
    return policy.diffusion.generate_actions(inference_batch, noise=noise)


def unnormalize_actions(postprocessor, normalized_actions: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [postprocessor(normalized_actions[:, step, :]) for step in range(normalized_actions.shape[1])],
        dim=1,
    )


def audit_policy(args: argparse.Namespace, fps: float) -> dict[str, Any]:
    config = json.loads((args.policy_path / "config.json").read_text())
    policy_type = str(config["type"])
    if policy_type not in {"act", "diffusion"}:
        raise ValueError(f"Only ACT and diffusion are supported. Got '{policy_type}'.")

    policy_class = get_policy_class(policy_type)
    policy = policy_class.from_pretrained(args.policy_path).to(args.device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    resize_size = infer_square_resize_pad_size_from_policy_features(policy.config.image_features)
    image_transform = ResizePadSquare(resize_size) if resize_size is not None else None
    dataset_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root,
        delta_timestamps=resolve_delta_timestamps(policy.config, dataset_meta),
        image_transforms=image_transform,
    )
    image_keys = list(policy.config.image_features)
    indices = selected_indices(dataset, args.startup_frames, args.stride, args.max_samples)

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    first_predictions: dict[int, tuple[int, np.ndarray]] = {}
    policy.eval()
    with torch.inference_mode():
        for number, index in enumerate(indices, start=1):
            item = dataset[index]
            episode = as_int(item["episode_index"])
            frame = as_int(item["frame_index"])
            group = "startup" if frame < args.startup_frames else "all_stride_samples"

            observation = {OBS_STATE: item[OBS_STATE]}
            observation.update({key: item[key] for key in image_keys})
            processed = preprocessor(observation)
            if policy_type == "diffusion":
                # Dataset samples are (history, feature); the deployed server receives
                # (batch, history, feature), so recreate its explicit batch dimension.
                processed = {
                    key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value
                    for key, value in processed.items()
                }
            predicted_normalized = make_raw_prediction(
                policy, policy_type, processed, image_keys, args.diffusion_noise_seed
            )
            predicted = unnormalize_actions(postprocessor, predicted_normalized).squeeze(0).cpu().numpy()

            target_offset = 0 if policy_type == "act" else policy.config.n_obs_steps - 1
            target = item[ACTION][target_offset].cpu().numpy()
            state = item[OBS_STATE][-1].cpu().numpy() if item[OBS_STATE].ndim == 2 else item[OBS_STATE].cpu().numpy()
            predicted_arm = predicted[:, ARM_INDICES]
            target_arm = target[ARM_INDICES]
            state_arm = state[ARM_INDICES]

            grouped[group]["first_prediction_minus_target_rad"].extend(
                np.abs(predicted_arm[0] - target_arm).tolist()
            )
            grouped[group]["first_prediction_minus_state_rad"].extend(
                np.abs(predicted_arm[0] - state_arm).tolist()
            )
            if len(predicted_arm) > 1:
                grouped[group]["predicted_chunk_step_rad"].extend(
                    np.abs(np.diff(predicted_arm, axis=0)).reshape(-1).tolist()
                )
                grouped[group]["predicted_chunk_velocity_rad_per_sec"].extend(
                    (np.abs(np.diff(predicted_arm, axis=0)) * fps).reshape(-1).tolist()
                )

            previous = first_predictions.get(episode)
            if previous is not None and frame > previous[0]:
                grouped[group]["replanned_first_action_step_rad"].extend(
                    np.abs(predicted_arm[0] - previous[1]).tolist()
                )
            first_predictions[episode] = (frame, predicted_arm[0])

            if number % 100 == 0:
                print(f"policy audit: {number}/{len(indices)} samples")

    return {
        "policy_path": str(args.policy_path),
        "policy_type": policy_type,
        "device": args.device,
        "samples_requested": len(indices),
        "startup_frames": args.startup_frames,
        "stride": args.stride,
        "diffusion_noise_seed": args.diffusion_noise_seed if policy_type == "diffusion" else None,
        "metrics_rad": {
            group: {name: quantiles(values) for name, values in metrics.items()}
            for group, metrics in grouped.items()
        },
    }


def main() -> None:
    args = parse_args()
    args.policy_path = args.policy_path.resolve()
    args.dataset_root = args.dataset_root.resolve()
    info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    fps = float(info["fps"])

    print("Auditing recorded action/state alignment...")
    report: dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "fps": fps,
        "dataset_metrics": audit_recorded_actions(args.dataset_root, fps, args.max_lag_frames),
    }
    print("Replaying checkpoint on recorded observations...")
    report["policy_metrics"] = audit_policy(args, fps)

    output = args.output or (args.policy_path / "offline_audit.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Wrote report to {output}")


if __name__ == "__main__":
    main()
