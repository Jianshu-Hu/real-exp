from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "pick_and_place_test"
DEFAULT_HF_CACHE = REPO_ROOT / ".hf-cache"
ACTION_CONFIG_REL_PATH = Path("meta/real_exp_action_config.json")
INFO_REL_PATH = Path("meta/info.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect or serve a trained LeRobot policy for deployment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a trained policy checkpoint and the expected dataset/action contract.",
    )
    inspect_parser.add_argument(
        "--policy-path",
        type=Path,
        required=True,
        help="Path to a trained LeRobot checkpoint or saved policy directory.",
    )
    inspect_parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the local dataset root used for training.",
    )

    serve_parser = subparsers.add_parser(
        "server",
        help="Start the LeRobot async inference server on the machine that runs policy inference.",
    )
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server bind host.")
    serve_parser.add_argument("--port", type=int, default=8080, help="Server bind port.")
    serve_parser.add_argument("--fps", type=int, default=15, help="Expected control frequency.")
    serve_parser.add_argument(
        "--inference-latency",
        type=float,
        default=None,
        help="Target inference latency in seconds. Defaults to 1/fps.",
    )
    serve_parser.add_argument(
        "--obs-queue-timeout",
        type=float,
        default=2.0,
        help="Observation queue timeout in seconds.",
    )

    return parser.parse_args()


def ensure_runtime_env() -> None:
    hf_home = Path(os.environ.get("HF_HOME", DEFAULT_HF_CACHE))
    hf_datasets_cache = Path(os.environ.get("HF_DATASETS_CACHE", hf_home / "datasets"))
    hf_home.mkdir(parents=True, exist_ok=True)
    hf_datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_datasets_cache)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def infer_policy_type(config: dict[str, Any]) -> str:
    policy_type = config.get("type")
    if not isinstance(policy_type, str) or not policy_type:
        raise ValueError(f"Could not infer policy type from config.json: {config}")
    return policy_type


def infer_actions_per_chunk(policy_type: str, config: dict[str, Any]) -> int:
    if policy_type == "act":
        return int(config.get("n_action_steps", config.get("chunk_size", 1)))
    if policy_type == "diffusion":
        return int(config.get("n_action_steps", 1))
    if policy_type == "vqbet":
        return int(config.get("n_action_pred_token", 1))
    return 1


def describe_action_layout(action_dim: int) -> str:
    if action_dim == 16:
        return "[Left Arm(7), Left Gripper(1), Right Arm(7), Right Gripper(1)]"
    if action_dim == 14:
        return "[Left Arm(7), Right Arm(7)]"
    return f"Unsupported or custom action layout with dim={action_dim}"


def inspect_policy(policy_path: Path, dataset_root: Path) -> None:
    from lerobot.policies.factory import get_policy_class

    policy_path = policy_path.resolve()
    dataset_root = dataset_root.resolve()

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    config_path = policy_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing policy config: {config_path}")

    dataset_info_path = dataset_root / INFO_REL_PATH
    action_config_path = dataset_root / ACTION_CONFIG_REL_PATH
    if not dataset_info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {dataset_info_path}")

    policy_cfg = load_json(config_path)
    dataset_info = load_json(dataset_info_path)
    action_cfg = load_json(action_config_path) if action_config_path.exists() else None

    policy_type = infer_policy_type(policy_cfg)
    actions_per_chunk = infer_actions_per_chunk(policy_type, policy_cfg)
    image_keys = [
        key
        for key, spec in dataset_info["features"].items()
        if key.startswith("observation.images.") and spec.get("dtype") in {"image", "video"}
    ]
    state_dim = int(dataset_info["features"]["observation.state"]["shape"][0])
    action_dim = int(dataset_info["features"]["action"]["shape"][0])

    policy_class = get_policy_class(policy_type)
    policy = policy_class.from_pretrained(policy_path)

    print("Deployment inspection")
    print("---------------------")
    print(f"policy_path: {policy_path}")
    print(f"policy_type: {policy_type}")
    print(f"recommended_actions_per_chunk: {actions_per_chunk}")
    print(f"dataset_root: {dataset_root}")
    print(f"dataset_fps: {dataset_info['fps']}")
    print(f"dataset_total_episodes: {dataset_info['total_episodes']}")
    print(f"dataset_state_dim: {state_dim}")
    print(f"dataset_action_dim: {action_dim}")
    print(f"dataset_action_layout: {describe_action_layout(action_dim)}")
    print(f"dataset_image_keys: {', '.join(image_keys)}")
    if action_cfg is not None:
        print(
            "dataset_action_representation: "
            f"arm={action_cfg.get('arm_action_representation')}, "
            f"gripper={action_cfg.get('gripper_action_representation')}"
        )

    print()
    print("Policy expectations")
    print("-------------------")
    print(f"policy_device_default: {policy.config.device}")
    print(f"policy_input_image_keys: {', '.join(policy.config.image_features.keys())}")
    print(
        "policy_uses_state: "
        f"{'yes' if policy.config.robot_state_feature is not None else 'no'}"
    )
    print(
        "policy_action_feature_shape: "
        f"{None if policy.config.action_feature is None else policy.config.action_feature.shape}"
    )

    print()
    print("Executor contract")
    print("-----------------")
    print("The robot-side executor should:")
    print("1. Read observation.state and camera frames at the control FPS.")
    print("2. Package observations with keys matching the dataset/policy contract.")
    print("3. Send observations to the policy server and receive action chunks.")
    print("4. Interpret action outputs using the dataset action representation.")
    print("5. Apply safety checks, filtering, interpolation, and watchdog logic locally.")


def run_server(args: argparse.Namespace) -> None:
    from lerobot.async_inference.configs import PolicyServerConfig
    from lerobot.async_inference.policy_server import serve

    inference_latency = args.inference_latency
    if inference_latency is None:
        inference_latency = 1.0 / args.fps

    cfg = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    serve(cfg)


def main() -> None:
    args = parse_args()
    ensure_runtime_env()

    if args.command == "inspect":
        inspect_policy(args.policy_path, args.dataset_root)
        return

    if args.command == "server":
        run_server(args)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
