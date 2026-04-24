from __future__ import annotations

import argparse
import json
import logging
import os
import pickle  # nosec
import time
import traceback
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty
from typing import Any

import torch

from image_preprocessing import ResizePadSquare, infer_square_resize_pad_size_from_policy_features

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


def resize_pad_robot_observation_image(
    image: torch.Tensor,
    resize_dims: tuple[int, int, int],
    image_preprocess: ResizePadSquare | None,
) -> torch.Tensor:
    assert image.ndim == 3, f"Image must be (H, W, C)! Received {image.shape}"
    image = image.permute(2, 0, 1)

    if image_preprocess is not None:
        return image_preprocess(image)

    dims = (resize_dims[1], resize_dims[2])
    image_batched = image.unsqueeze(0)
    resized = torch.nn.functional.interpolate(image_batched, size=dims, mode="bilinear", align_corners=False)
    return resized.squeeze(0)


def raw_observation_to_observation_with_resize_pad(
    raw_observation: dict[str, Any],
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, Any],
    image_preprocess: ResizePadSquare | None,
) -> dict[str, Any]:
    from lerobot.async_inference.helpers import (
        extract_images_from_raw_observation,
        extract_state_from_raw_observation,
        is_image_key,
        make_lerobot_observation,
        prepare_image,
    )

    lerobot_obs = make_lerobot_observation(raw_observation, lerobot_features)
    image_keys = list(filter(is_image_key, lerobot_obs))

    observation: dict[str, Any] = {  # state is expected as (B, state_dim)
        "observation.state": extract_state_from_raw_observation(lerobot_obs)
    }

    for image_key in image_keys:
        raw_image = extract_images_from_raw_observation(lerobot_obs, image_key)
        resized_image = resize_pad_robot_observation_image(
            torch.as_tensor(raw_image),
            policy_image_features[image_key].shape,
            image_preprocess,
        )
        observation[image_key] = prepare_image(resized_image).unsqueeze(0)

    if "task" in raw_observation:
        observation["task"] = raw_observation["task"]

    return observation


def make_deployment_policy_server():
    from lerobot.async_inference.constants import SUPPORTED_POLICIES
    from lerobot.async_inference.helpers import Observation, RemotePolicyConfig
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.transport import services_pb2

    class DeploymentPolicyServer(PolicyServer):
        def SendPolicyInstructions(self, request, context):  # noqa: N802
            if not self.running:
                self.logger.warning("Server is not running. Ignoring policy instructions.")
                return services_pb2.Empty()

            client_id = context.peer()
            policy_specs = pickle.loads(request.data)  # nosec

            if not isinstance(policy_specs, RemotePolicyConfig):
                raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

            if policy_specs.policy_type not in SUPPORTED_POLICIES:
                raise ValueError(
                    f"Policy type {policy_specs.policy_type} not supported. "
                    f"Supported policies: {SUPPORTED_POLICIES}"
                )

            self.logger.info(
                f"Receiving policy instructions from {client_id} | "
                f"Policy type: {policy_specs.policy_type} | "
                f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
                f"Actions per chunk: {policy_specs.actions_per_chunk} | "
                f"Device: {policy_specs.device}"
            )

            self.device = policy_specs.device
            self.policy_type = policy_specs.policy_type
            self.lerobot_features = policy_specs.lerobot_features
            self.actions_per_chunk = policy_specs.actions_per_chunk

            policy_class = get_policy_class(self.policy_type)

            start = time.perf_counter()
            self.policy = policy_class.from_pretrained(
                policy_specs.pretrained_name_or_path,
                cli_overrides=[f"--device={self.device}"],
            )
            self.policy.to(self.device)
            max_actions_per_chunk = infer_actions_per_chunk(self.policy_type, asdict(self.policy.config))
            if self.actions_per_chunk > max_actions_per_chunk:
                raise ValueError(
                    f"actions_per_chunk ({self.actions_per_chunk}) cannot exceed the policy maximum "
                    f"chunk size ({max_actions_per_chunk})."
                )

            device_override = {"device": self.device}
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=policy_specs.pretrained_name_or_path,
                preprocessor_overrides={
                    "device_processor": device_override,
                    "rename_observations_processor": {"rename_map": policy_specs.rename_map},
                },
                postprocessor_overrides={"device_processor": device_override},
            )
            image_size = infer_square_resize_pad_size_from_policy_features(self.policy_image_features)
            self.image_preprocess = ResizePadSquare(size=image_size) if image_size is not None else None

            end = time.perf_counter()
            self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")
            if self.image_preprocess is not None:
                self.logger.info(
                    "Using aspect-preserving resize + constant padding during inference "
                    f"for square image inputs of size {self.image_preprocess.size}"
                )
            else:
                self.logger.info("Using upstream direct resize during inference image preparation")

            return services_pb2.Empty()

        def _predict_action_chunk(self, observation_t) -> list[Any]:
            start_prepare = time.perf_counter()
            observation: Observation = raw_observation_to_observation_with_resize_pad(
                observation_t.get_observation(),
                self.lerobot_features,
                self.policy_image_features,
                self.image_preprocess,
            )
            prepare_time = time.perf_counter() - start_prepare

            start_preprocess = time.perf_counter()
            observation = self.preprocessor(observation)
            self.last_processed_obs = observation_t
            preprocessing_time = time.perf_counter() - start_preprocess

            start_inference = time.perf_counter()
            action_tensor = self._get_action_chunk(observation)
            inference_time = time.perf_counter() - start_inference
            self.logger.info(
                f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
            )

            start_postprocess = time.perf_counter()
            _, chunk_size, _ = action_tensor.shape
            processed_actions = []
            for i in range(chunk_size):
                single_action = action_tensor[:, i, :]
                processed_action = self.postprocessor(single_action)
                processed_actions.append(processed_action)

            action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
            self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")
            action_tensor = action_tensor.detach().cpu()

            action_chunk = self._time_action_chunk(
                observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
            )
            postprocess_stops = time.perf_counter()
            postprocessing_time = postprocess_stops - start_postprocess

            self.logger.info(
                f"Observation {observation_t.get_timestep()} | "
                f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
            )
            self.logger.debug(
                f"Observation {observation_t.get_timestep()} | "
                f"Prepare time: {1000 * prepare_time:.2f}ms | "
                f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
                f"Inference time: {1000 * inference_time:.2f}ms | "
                f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
                f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
            )

            return action_chunk

        def GetActions(self, request, context):  # noqa: N802
            try:
                getactions_starts = time.perf_counter()
                obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
                self.logger.info(
                    f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
                )

                with self._predicted_timesteps_lock:
                    self._predicted_timesteps.add(obs.get_timestep())

                start_time = time.perf_counter()
                action_chunk = self._predict_action_chunk(obs)
                inference_time = time.perf_counter() - start_time

                start_time = time.perf_counter()
                actions_bytes = pickle.dumps(action_chunk)  # nosec
                serialize_time = time.perf_counter() - start_time

                actions = services_pb2.Actions(data=actions_bytes)

                self.logger.info(
                    f"Action chunk #{obs.get_timestep()} generated | "
                    f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
                )

                self.logger.debug(
                    f"Action chunk #{obs.get_timestep()} generated | "
                    f"Inference time: {inference_time:.2f}s |"
                    f"Serialize time: {serialize_time:.2f}s |"
                    f"Total time: {inference_time + serialize_time:.2f}s"
                )

                time.sleep(
                    max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
                )

                return actions

            except Empty:
                return services_pb2.Empty()
            except Exception as exc:
                self.logger.error(f"Error in StreamActions: {exc}")
                self.logger.error(traceback.format_exc())
                return services_pb2.Empty()

    return DeploymentPolicyServer


def serve_deployment_policy_server(cfg) -> None:
    import grpc

    from lerobot.transport import services_pb2_grpc

    DeploymentPolicyServer = make_deployment_policy_server()

    logging.info(pformat(asdict(cfg)))

    policy_server = DeploymentPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")


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
    max_actions_per_chunk = infer_actions_per_chunk(policy_type, policy_cfg)
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
    print(f"max_actions_per_chunk: {max_actions_per_chunk}")
    print(f"dataset_root: {dataset_root}")
    print(f"dataset_fps: {dataset_info['fps']}")
    print(f"dataset_total_episodes: {dataset_info['total_episodes']}")
    print(f"dataset_state_dim: {state_dim}")
    print(f"dataset_action_dim: {action_dim}")
    print(f"dataset_action_layout: {describe_action_layout(action_dim)}")
    print(f"dataset_image_keys: {', '.join(image_keys)}")
    if action_cfg is not None:
        arm_action_representation = action_cfg.get("arm_action_representation")
        print(
            "dataset_action_representation: "
            f"arm={arm_action_representation}, "
            f"gripper={action_cfg.get('gripper_action_representation')}"
        )
        if arm_action_representation != "absolute_joint_position":
            print(
                "warning: current deployment expects arm=absolute_joint_position; "
                "old delta_joint_position policies should be retrained on a new dataset."
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
    serve_deployment_policy_server(cfg)


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
