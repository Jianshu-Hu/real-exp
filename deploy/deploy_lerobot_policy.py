"""Inspect trained policies or serve them for remote robot-side execution.

Usage:
    python deploy/deploy_lerobot_policy.py inspect --policy-path outputs/my_policy
    python deploy/deploy_lerobot_policy.py server --host 0.0.0.0 --port 8080 \
        --policy-path outputs/my_policy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle  # nosec
import sys
import time
import traceback
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent import futures
from collections import deque
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty
from typing import Any

import torch
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.image_preprocessing import ResizePadSquare, infer_square_resize_pad_size_from_policy_features
from utils.trajectory_metadata import (
    describe_trajectory_layout,
    require_dataset_trajectory_config,
    validate_action_trajectory_contract,
    validate_trajectory_config,
)
from utils.deployment_metadata import deployment_camera_names, load_checkpoint_deployment_contract
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.transport.utils import receive_bytes_in_chunks

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
        default=None,
        help="Optional source dataset for an additional checkpoint-vs-dataset consistency check.",
    )

    serve_parser = subparsers.add_parser(
        "server",
        help="Start the LeRobot async inference server on the machine that runs policy inference.",
    )
    serve_parser.add_argument("--host", default="0.0.0.0", help="Server bind host.")
    serve_parser.add_argument("--port", type=int, default=8080, help="Server bind port.")
    serve_parser.add_argument("--metadata-port", type=int, default=8081, help="Read-only deployment metadata HTTP port.")
    serve_parser.add_argument("--policy-path", type=Path, required=True, help="Checkpoint directory owned by this server.")
    serve_parser.add_argument("--fps", type=float, default=None, help="Expected control frequency; defaults to checkpoint metadata.")
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
    serve_parser.add_argument(
        "--camera-cache-address",
        default="tcp://127.0.0.1:5557",
        help="Loopback ZMQ camera-bundle cache published by the deployment bridge.",
    )
    serve_parser.add_argument(
        "--max-observation-age",
        type=float,
        default=0.25,
        help="Reject policy requests whose referenced camera bundle is older than this many seconds.",
    )
    serve_parser.add_argument(
        "--max-camera-skew",
        type=float,
        default=0.067,
        help="Reject camera bundles whose inter-camera timestamp skew exceeds this value.",
    )
    serve_parser.add_argument(
        "--diffusion-noise-scheduler-type",
        choices=("DDPM", "DDIM"),
        default="DDIM",
        help="Override the diffusion scheduler when loading a diffusion policy on the server.",
    )
    serve_parser.add_argument(
        "--diffusion-num-inference-steps",
        type=int,
        default=10,
        help="Override the diffusion denoising steps when loading a diffusion policy on the server.",
    )
    serve_parser.add_argument(
        "--diffusion-fixed-noise-seed",
        type=int,
        default=0,
        help=(
            "Use a fixed random seed for diffusion inference noise. Defaults to 0 for deterministic "
            "deployment. Set to a different integer to test another deterministic seed."
        ),
    )
    serve_parser.add_argument(
        "--disable-diffusion-fixed-noise",
        action="store_true",
        help="Disable fixed diffusion inference noise and use stochastic sampling.",
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


def policy_feature_dim(
    policy_config: dict[str, Any], section: str, feature_key: str, *, source: str
) -> int:
    try:
        shape = policy_config[section][feature_key]["shape"]
    except KeyError as exc:
        raise ValueError(
            f"{source} is missing {section}.{feature_key}.shape."
        ) from exc
    if not isinstance(shape, (list, tuple)) or len(shape) != 1:
        raise ValueError(
            f"{source} {section}.{feature_key}.shape must contain one dimension, got {shape!r}."
        )
    try:
        dimension = int(shape[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} {section}.{feature_key}.shape has an invalid dimension: {shape!r}."
        ) from exc
    if dimension <= 0:
        raise ValueError(
            f"{source} {section}.{feature_key}.shape must be positive, got {shape!r}."
        )
    return dimension


def validate_checkpoint_trajectory_contract(
    policy_config: dict[str, Any],
    trajectory_config: dict[str, Any],
    *,
    source: str,
    live_state_dim: int | None = None,
) -> dict[str, Any]:
    """Verify a checkpoint's declared robot layout matches its model features."""
    policy_state_dim = policy_feature_dim(
        policy_config, "input_features", "observation.state", source=source
    )
    policy_action_dim = policy_feature_dim(
        policy_config, "output_features", "action", source=source
    )
    trajectory_config = validate_trajectory_config(
        trajectory_config,
        policy_state_dim,
        policy_action_dim,
        source=f"{source} trajectory metadata",
    )
    if live_state_dim is not None and live_state_dim != policy_state_dim:
        raise ValueError(
            "Deployment observation state dimension does not match the checkpoint trajectory "
            f"contract: live={live_state_dim}, checkpoint={policy_state_dim}."
        )
    return trajectory_config


def trajectory_contract_mismatches(
    expected: dict[str, Any], actual: dict[str, Any]
) -> list[str]:
    fields = (
        "arm_mode",
        "arms",
        "end_effector",
        "include_gripper",
        "include_hand",
        "robot_state_dim",
        "action_dim",
    )
    return [
        f"{field}: dataset={expected[field]!r}, checkpoint={actual[field]!r}"
        for field in fields
        if expected[field] != actual[field]
    ]


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
    rename_map: dict[str, str] | None = None,
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
        policy_image_key = (rename_map or {}).get(image_key, image_key)
        if policy_image_key not in policy_image_features:
            # A live bridge may expose more cameras than a policy consumes. Keep the
            # observation contract permissive and omit those images before preprocessing.
            continue
        resized_image = resize_pad_robot_observation_image(
            torch.as_tensor(raw_image),
            policy_image_features[policy_image_key].shape,
            image_preprocess,
        )
        observation[image_key] = prepare_image(resized_image).unsqueeze(0)

    if "task" in raw_observation:
        observation["task"] = raw_observation["task"]

    return observation


class CameraBundleCache:
    """Bounded server-local cache of synchronized camera bundles from the ROS bridge."""

    def __init__(self, address: str, max_entries: int = 8) -> None:
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self.address = address
        self.max_entries = max_entries
        self._bundles: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, max_entries)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.connect(address)
        self._thread = threading.Thread(target=self._receive_loop, daemon=True, name="camera-bundle-cache")

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._shutdown.set()
        self._thread.join(timeout=1.0)
        self._socket.close(0)
        self._context.term()

    def _receive_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._shutdown.is_set():
            if self._socket not in dict(poller.poll(100)):
                continue
            packet = self._socket.recv_pyobj()
            if not isinstance(packet, dict):
                continue
            sequence = packet.get("camera_bundle_sequence")
            cameras = packet.get("cameras")
            if sequence is None or not isinstance(cameras, dict):
                continue
            with self._lock:
                self._bundles[int(sequence)] = packet
                for stale_sequence in sorted(self._bundles)[:-self.max_entries]:
                    self._bundles.pop(stale_sequence, None)

    def get(self, sequence: int) -> dict[str, Any] | None:
        with self._lock:
            return self._bundles.get(int(sequence))


def resolve_server_local_observation(
    timed_observation: Any,
    policy_image_features: dict[str, Any],
    lerobot_features: dict[str, dict[str, Any]],
    camera_bundle_cache: CameraBundleCache | None,
    max_observation_age_s: float,
    max_camera_skew_s: float,
    rename_map: dict[str, str] | None = None,
) -> Any:
    """Resolve a robot-side metadata packet to the exact server-local RGB bundle.

    This is deliberately independent of ``PolicyServer`` so the cache protocol can be tested
    without loading a checkpoint or starting gRPC. The function either returns a complete
    observation or raises a descriptive error before preprocessing can see incomplete images.
    """
    raw = timed_observation.get_observation()
    expected_names = [
        key.split("observation.images.", 1)[-1]
        for key, feature in lerobot_features.items()
        if key.startswith("observation.images.")
        and feature.get("dtype") in {"image", "video"}
        and (rename_map or {}).get(key, key) in policy_image_features
    ]
    if not expected_names:
        return timed_observation
    request_debug = getattr(timed_observation, "deployment_debug", {}) or {}
    sequence = request_debug.get("camera_bundle_sequence")
    if sequence is None:
        raise RuntimeError(
            "Policy expects camera images, but the observation has no camera_bundle_sequence "
            "reference for the server-local cache."
        )
    if camera_bundle_cache is None:
        raise RuntimeError("A camera bundle was referenced but the server cache is disabled.")

    bundle = camera_bundle_cache.get(int(sequence))
    if bundle is None:
        raise RuntimeError(f"Camera bundle #{sequence} is unavailable or expired from the cache.")
    cameras = bundle.get("cameras")
    cached_sequence = bundle.get("camera_bundle_sequence")
    if cached_sequence is None or int(cached_sequence) != int(sequence):
        raise RuntimeError(
            f"Camera cache returned bundle #{cached_sequence} for requested sequence #{sequence}."
        )
    if not isinstance(cameras, dict):
        raise RuntimeError(f"Camera bundle #{sequence} has no camera dictionary.")
    missing = [name for name in expected_names if name not in cameras]
    if missing:
        raise RuntimeError(
            f"Camera bundle #{sequence} is missing policy cameras: {', '.join(missing)}."
        )

    camera_sync = bundle.get("camera_sync") or {}
    if not camera_sync.get("bundle_ready", False):
        raise RuntimeError(f"Camera bundle #{sequence} is not marked ready.")
    camera_stamps: dict[str, float] = {}
    for name in expected_names:
        camera = cameras[name]
        if not isinstance(camera, dict) or "rgb" not in camera:
            raise RuntimeError(f"Camera bundle #{sequence} has no RGB payload for '{name}'.")
        image = camera["rgb"]
        expected_shape = (lerobot_features.get(f"observation.images.{name}") or {}).get("shape")
        actual_shape = tuple(getattr(image, "shape", ()))
        declared_shape = tuple(camera.get("shape", ()))
        if expected_shape and declared_shape and tuple(expected_shape) != declared_shape:
            raise RuntimeError(
                f"Camera '{name}' shape {declared_shape} in bundle #{sequence} does not match "
                f"the live feature contract {tuple(expected_shape)}."
            )
        if declared_shape and actual_shape and declared_shape != actual_shape:
            raise RuntimeError(
                f"Camera '{name}' declares shape {declared_shape} but carries RGB shape {actual_shape}."
            )
        stamp_s = camera.get("stamp_s")
        if stamp_s is None:
            raise RuntimeError(f"Camera bundle #{sequence} has no timestamp for '{name}'.")
        camera_stamps[name] = float(stamp_s)

    computed_skew_s = max(camera_stamps.values()) - min(camera_stamps.values())
    declared_skew_s = camera_sync.get("max_skew_s")
    if declared_skew_s is not None and float(declared_skew_s) > max_camera_skew_s:
        raise RuntimeError(
            f"Camera bundle #{sequence} skew {float(declared_skew_s):.3f}s exceeds "
            f"{max_camera_skew_s:.3f}s."
        )
    if computed_skew_s > max_camera_skew_s:
        raise RuntimeError(
            f"Camera bundle #{sequence} computed skew {computed_skew_s:.3f}s exceeds "
            f"{max_camera_skew_s:.3f}s."
        )

    reference_s = camera_sync.get("reference_stamp_s")
    if reference_s is None:
        reference_s = bundle.get("bridge_publish_s")
    if reference_s is None:
        raise RuntimeError(f"Camera bundle #{sequence} has no freshness timestamp.")
    age_s = time.time() - float(reference_s)
    if age_s < -1.0 or age_s > max_observation_age_s:
        raise RuntimeError(
            f"Camera bundle #{sequence} age {age_s:.3f}s is outside the allowed "
            f"window ({max_observation_age_s:.3f}s)."
        )

    state_stamp_s = request_debug.get("robot_state_stamp_s")
    if state_stamp_s is None:
        state_stamp_s = bundle.get("robot_state_stamp_s")
    if state_stamp_s is None:
        raise RuntimeError(f"Observation referencing camera bundle #{sequence} has no robot-state timestamp.")
    state_age_s = time.time() - float(state_stamp_s)
    if state_age_s < -1.0 or state_age_s > max_observation_age_s:
        raise RuntimeError(
            f"Observation robot-state age {state_age_s:.3f}s is outside the allowed "
            f"window ({max_observation_age_s:.3f}s)."
        )
    state_camera_skew_s = abs(float(state_stamp_s) - float(reference_s))
    if state_camera_skew_s > max_observation_age_s:
        raise RuntimeError(
            f"Observation state/camera skew {state_camera_skew_s:.3f}s exceeds "
            f"{max_observation_age_s:.3f}s for bundle #{sequence}."
        )

    raw.update({name: cameras[name]["rgb"] for name in expected_names})
    timed_observation.deployment_debug = {
        **request_debug,
        "camera_bundle_age_s_at_server": age_s,
        "robot_state_age_s_at_server": state_age_s,
        "camera_computed_skew_s_at_server": computed_skew_s,
        "state_camera_skew_s_at_server": state_camera_skew_s,
        "camera_sync": camera_sync,
    }
    return timed_observation


def make_deployment_policy_server(
    diffusion_cli_overrides: list[str] | None = None,
    diffusion_fixed_noise_seed: int | None = None,
    camera_bundle_cache: CameraBundleCache | None = None,
    max_observation_age_s: float = 0.25,
    max_camera_skew_s: float = 0.067,
    checkpoint_path: Path | None = None,
    deployment_contract: dict[str, Any] | None = None,
):
    from lerobot.async_inference.constants import SUPPORTED_POLICIES
    from lerobot.async_inference.helpers import Observation, RemotePolicyConfig
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.transport import services_pb2

    class DeploymentPolicyServer(PolicyServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.checkpoint_path = checkpoint_path
            self.deployment_contract = deployment_contract

        def _resolve_server_local_observation(self, timed_observation):
            return resolve_server_local_observation(
                timed_observation,
                self.policy_image_features,
                self.lerobot_features,
                camera_bundle_cache,
                max_observation_age_s,
                max_camera_skew_s,
                self.rename_map,
            )

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
            if self.deployment_contract is not None and policy_specs.policy_type != self.deployment_contract["policy_type"]:
                raise ValueError(
                    "Deployment executor policy type disagrees with checkpoint metadata: "
                    f"client={policy_specs.policy_type!r}, server={self.deployment_contract['policy_type']!r}"
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
            self.rename_map = policy_specs.rename_map
            if self.deployment_contract is None:
                raise RuntimeError("Policy server has no configured deployment contract.")
            self.actions_per_chunk = int(policy_specs.actions_per_chunk)
            max_contract_chunk = int(self.deployment_contract["max_actions_per_chunk"])
            if self.actions_per_chunk <= 0 or self.actions_per_chunk > max_contract_chunk:
                raise ValueError(
                    "Deployment executor actions_per_chunk exceeds the checkpoint contract: "
                    f"client={self.actions_per_chunk}, maximum={max_contract_chunk}"
                )

            if self.checkpoint_path is None:
                raise RuntimeError("Policy server has no configured checkpoint deployment contract.")
            pretrained_path = self.checkpoint_path
            if not pretrained_path.exists():
                raise FileNotFoundError(
                    f"Policy checkpoint does not exist on the policy server: {pretrained_path}"
                )
            if not pretrained_path.is_dir():
                raise NotADirectoryError(
                    f"Policy checkpoint must be a directory on the policy server: {pretrained_path}"
                )
            required_files = ("config.json", "model.safetensors")
            missing_files = [name for name in required_files if not (pretrained_path / name).is_file()]
            if missing_files:
                raise FileNotFoundError(
                    f"Policy checkpoint {pretrained_path} is missing required files: "
                    + ", ".join(missing_files)
                )
            policy_state_shape = tuple(
                policy_specs.lerobot_features.get("observation.state", {}).get("shape", ())
            )
            if len(policy_state_shape) != 1:
                raise ValueError(
                    "Deployment observations must provide a one-dimensional observation.state feature; "
                    f"got {policy_state_shape!r}."
                )
            checkpoint_trajectory = validate_checkpoint_trajectory_contract(
                load_json(pretrained_path / "config.json"),
                self.deployment_contract["trajectory_config"],
                source=str(pretrained_path),
                live_state_dim=int(policy_state_shape[0]),
            )
            validate_action_trajectory_contract(
                self.deployment_contract["action_config"],
                checkpoint_trajectory,
                source=str(pretrained_path / "meta"),
            )
            client_trajectory = getattr(policy_specs, "trajectory_config", None)
            if not isinstance(client_trajectory, dict):
                raise ValueError(
                    "Deployment executor did not provide its dataset trajectory contract. "
                    "Update the robot-side executor before loading this checkpoint."
                )
            client_trajectory = validate_trajectory_config(
                client_trajectory,
                int(policy_state_shape[0]),
                checkpoint_trajectory["action_dim"],
                source="deployment executor trajectory metadata",
            )
            contract_mismatches = trajectory_contract_mismatches(
                client_trajectory, checkpoint_trajectory
            )
            if contract_mismatches:
                raise ValueError(
                    "Deployment executor and checkpoint trajectory contracts disagree: "
                    + "; ".join(contract_mismatches)
                )
            expected_cameras = set(deployment_camera_names(self.deployment_contract))
            actual_cameras = {
                key.removeprefix("observation.images.")
                for key in policy_specs.lerobot_features
                if key.startswith("observation.images.")
            }
            if expected_cameras != actual_cameras:
                raise ValueError(
                    "Deployment executor camera features disagree with checkpoint metadata: "
                    f"expected={sorted(expected_cameras)}, actual={sorted(actual_cameras)}"
                )
            self.logger.info(
                "Policy checkpoint preflight passed: "
                f"path={pretrained_path}, model_bytes={(pretrained_path / 'model.safetensors').stat().st_size}"
            )

            policy_class = get_policy_class(self.policy_type)
            cli_overrides = [f"--device={self.device}"]
            if self.policy_type == "diffusion" and diffusion_cli_overrides:
                cli_overrides.extend(diffusion_cli_overrides)
                self.logger.info(f"Applying diffusion CLI overrides: {diffusion_cli_overrides}")

            start = time.perf_counter()
            try:
                self.policy = policy_class.from_pretrained(
                    str(pretrained_path),
                    cli_overrides=cli_overrides,
                )
            except Exception as exc:
                self.logger.error(
                    "Policy checkpoint loading failed for "
                    f"{pretrained_path} on device {self.device}: {exc}"
                )
                self.logger.error(traceback.format_exc())
                raise RuntimeError(
                    f"Could not load policy checkpoint {pretrained_path} on device {self.device}: {exc}"
                ) from exc
            self.policy.to(self.device)
            self.policy.eval()
            max_actions_per_chunk = infer_actions_per_chunk(self.policy_type, asdict(self.policy.config))
            if self.actions_per_chunk > max_actions_per_chunk:
                raise ValueError(
                    f"actions_per_chunk ({self.actions_per_chunk}) cannot exceed the policy maximum "
                    f"chunk size ({max_actions_per_chunk})."
                )

            device_override = {"device": self.device}
            self.preprocessor, self.postprocessor = make_pre_post_processors(
                self.policy.config,
                pretrained_path=str(pretrained_path),
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

            if self.policy_type == "diffusion":
                self._diffusion_history = {
                    OBS_STATE: deque(maxlen=self.policy.config.n_obs_steps),
                }
                if self.policy.config.image_features:
                    self._diffusion_history[OBS_IMAGES] = deque(maxlen=self.policy.config.n_obs_steps)
                if self.policy.config.env_state_feature:
                    self._diffusion_history[OBS_ENV_STATE] = deque(maxlen=self.policy.config.n_obs_steps)
            else:
                self._diffusion_history = None

            return services_pb2.Empty()

        def _prepare_policy_observation(self, observation_t) -> Observation:
            observation: Observation = raw_observation_to_observation_with_resize_pad(
                observation_t.get_observation(),
                self.lerobot_features,
                self.policy_image_features,
                self.image_preprocess,
                self.rename_map,
            )
            observation = self.preprocessor(observation)
            self.last_processed_obs = observation_t
            return observation

        def _update_diffusion_history(self, observation: dict[str, torch.Tensor]) -> None:
            if self.policy_type != "diffusion" or self._diffusion_history is None:
                return

            history_input = {OBS_STATE: observation[OBS_STATE]}
            if self.policy.config.image_features:
                history_input[OBS_IMAGES] = torch.stack(
                    [observation[key] for key in self.policy.config.image_features],
                    dim=-4,
                )
            if self.policy.config.env_state_feature and OBS_ENV_STATE in observation:
                history_input[OBS_ENV_STATE] = observation[OBS_ENV_STATE]

            self._diffusion_history = populate_queues(self._diffusion_history, history_input)

        def _build_diffusion_history_batch(self) -> dict[str, torch.Tensor]:
            if self._diffusion_history is None:
                raise RuntimeError("Diffusion history is not initialized.")

            history_batch: dict[str, torch.Tensor] = {}
            for key, queue in self._diffusion_history.items():
                if not queue:
                    raise RuntimeError(
                        f"Diffusion deployment could not build history for '{key}'. "
                        "No observations have been buffered yet."
                    )
                history_batch[key] = torch.stack(list(queue), dim=1).clone()

            return history_batch

        def _make_fixed_diffusion_noise(self, history_batch: dict[str, torch.Tensor]) -> torch.Tensor:
            if diffusion_fixed_noise_seed is None:
                raise RuntimeError("Fixed diffusion noise requested without a seed.")

            state = history_batch[OBS_STATE]
            generator = torch.Generator(device=state.device)
            generator.manual_seed(diffusion_fixed_noise_seed)
            return torch.randn(
                size=(
                    state.shape[0],
                    self.policy.config.horizon,
                    self.policy.config.action_feature.shape[0],
                ),
                dtype=state.dtype,
                device=state.device,
                generator=generator,
            )

        def _predict_action_chunk(self, observation_t) -> list[Any]:
            observation = self._prepare_policy_observation(observation_t)
            action_tensor = self._get_action_chunk(observation, observation_t)

            _, chunk_size, _ = action_tensor.shape
            processed_actions = []
            for i in range(chunk_size):
                single_action = action_tensor[:, i, :]
                processed_action = self.postprocessor(single_action)
                processed_actions.append(processed_action)

            action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
            action_tensor = action_tensor.detach().cpu()

            action_anchor_timestep = getattr(observation_t, "action_timestep", observation_t.get_timestep())
            action_chunk = self._time_action_chunk(
                observation_t.get_timestamp(), list(action_tensor), action_anchor_timestep
            )

            return action_chunk

        def _get_action_chunk(self, observation: dict[str, torch.Tensor], observation_t=None) -> torch.Tensor:
            if self.policy_type == "diffusion":
                history_batch = getattr(observation_t, "diffusion_history_batch", None)
                if history_batch is None:
                    history_batch = self._build_diffusion_history_batch()
                noise = (
                    self._make_fixed_diffusion_noise(history_batch)
                    if diffusion_fixed_noise_seed is not None
                    else None
                )
                with torch.inference_mode():
                    chunk = self.policy.diffusion.generate_actions(history_batch, noise=noise)
                if chunk.ndim != 3:
                    chunk = chunk.unsqueeze(0)
                return chunk[:, : self.actions_per_chunk, :]

            with torch.inference_mode():
                return super()._get_action_chunk(observation)

        def SendObservations(self, request_iterator, context):  # noqa: N802
            received_bytes = receive_bytes_in_chunks(
                request_iterator, None, self.shutdown_event, self.logger
            )
            timed_observation = pickle.loads(received_bytes)  # nosec
            try:
                timed_observation = self._resolve_server_local_observation(timed_observation)
            except Exception as exc:
                self.logger.error(f"Rejecting deployment observation: {exc}")
                timed_observation.rejected_reason = str(exc)
                if timed_observation.must_go:
                    self._enqueue_observation(timed_observation)
                return services_pb2.Empty()

            if self.policy_type == "diffusion" and self.preprocessor is not None:
                try:
                    observation = self._prepare_policy_observation(timed_observation)
                    self._update_diffusion_history(observation)
                except Exception as exc:
                    self.logger.error(f"Error updating diffusion history from observation stream: {exc}")
                    self.logger.error(traceback.format_exc())

                # For diffusion, every observation updates history, but only `must_go` observations
                # should trigger a fresh chunk request.
                if timed_observation.must_go:
                    try:
                        timed_observation.diffusion_history_batch = self._build_diffusion_history_batch()
                    except Exception as exc:
                        self.logger.error(f"Error snapshotting diffusion history: {exc}")
                        self.logger.error(traceback.format_exc())
                        return services_pb2.Empty()
                    self._enqueue_observation(timed_observation)
                return services_pb2.Empty()

            self._enqueue_observation(timed_observation)

            return services_pb2.Empty()

        def GetActions(self, request, context):  # noqa: N802
            try:
                getactions_starts = time.perf_counter()
                obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
                rejected_reason = getattr(obs, "rejected_reason", None)
                if rejected_reason:
                    self.logger.error(f"Action request rejected: {rejected_reason}")
                    return services_pb2.Empty()

                with self._predicted_timesteps_lock:
                    self._predicted_timesteps.add(obs.get_timestep())

                action_chunk = self._predict_action_chunk(obs)

                actions_bytes = pickle.dumps(action_chunk)  # nosec

                actions = services_pb2.Actions(data=actions_bytes)

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


def serve_deployment_policy_server(
    cfg,
    diffusion_cli_overrides: list[str] | None = None,
    diffusion_fixed_noise_seed: int | None = None,
    camera_cache_address: str = "tcp://127.0.0.1:5557",
    max_observation_age_s: float = 0.25,
    max_camera_skew_s: float = 0.067,
    checkpoint_path: Path | None = None,
    deployment_contract: dict[str, Any] | None = None,
    metadata_port: int = 8081,
) -> None:
    import grpc

    from lerobot.transport import services_pb2_grpc

    camera_bundle_cache = CameraBundleCache(camera_cache_address)
    camera_bundle_cache.start()
    DeploymentPolicyServer = make_deployment_policy_server(
        diffusion_cli_overrides,
        diffusion_fixed_noise_seed=diffusion_fixed_noise_seed,
        camera_bundle_cache=camera_bundle_cache,
        max_observation_age_s=max_observation_age_s,
        max_camera_skew_s=max_camera_skew_s,
        checkpoint_path=checkpoint_path,
        deployment_contract=deployment_contract,
    )

    logging.info(pformat(asdict(cfg)))

    policy_server = DeploymentPolicyServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    class MetadataHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/deployment-metadata":
                self.send_error(404)
                return
            payload = json.dumps(deployment_contract, sort_keys=True).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    metadata_server = ThreadingHTTPServer((cfg.host, metadata_port), MetadataHandler)
    metadata_thread = threading.Thread(target=metadata_server.serve_forever, daemon=True)
    metadata_thread.start()

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    try:
        server.wait_for_termination()
    finally:
        metadata_server.shutdown()
        metadata_server.server_close()
        camera_bundle_cache.close()
        policy_server.logger.info("Server terminated")


def inspect_policy(policy_path: Path, dataset_root: Path | None = None) -> None:
    from lerobot.policies.factory import get_policy_class

    policy_path = policy_path.resolve()
    dataset_root = dataset_root.resolve() if dataset_root is not None else None

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path not found: {policy_path}")
    if dataset_root is not None and not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    config_path = policy_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing policy config: {config_path}")

    policy_cfg = load_json(config_path)
    checkpoint_contract = load_checkpoint_deployment_contract(policy_path)
    dataset_info = {
        "fps": checkpoint_contract["fps"],
        "features": checkpoint_contract["features"],
        "total_episodes": "unknown",
    }
    action_cfg = checkpoint_contract["action_config"]
    trajectory_cfg = checkpoint_contract["trajectory_config"]
    if dataset_root is not None:
        dataset_info_path = dataset_root / INFO_REL_PATH
        action_config_path = dataset_root / ACTION_CONFIG_REL_PATH
        if not dataset_info_path.exists() or not action_config_path.exists():
            raise FileNotFoundError(f"Dataset metadata is incomplete under {dataset_root / 'meta'}")
        source_info = load_json(dataset_info_path)
        source_action = load_json(action_config_path)
        source_trajectory = require_dataset_trajectory_config(dataset_root)
        validate_action_trajectory_contract(source_action, source_trajectory, source=str(dataset_root / "meta"))
        if trajectory_contract_mismatches(source_trajectory, trajectory_cfg):
            raise ValueError("Dataset and checkpoint trajectory contracts disagree.")
        dataset_info = source_info

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
    print(f"dataset_root: {dataset_root if dataset_root is not None else '<embedded checkpoint metadata>'}")
    print(f"dataset_fps: {dataset_info['fps']}")
    print(f"dataset_total_episodes: {dataset_info['total_episodes']}")
    print(f"dataset_state_dim: {state_dim}")
    print(f"dataset_action_dim: {action_dim}")
    print(f"dataset_action_layout: {describe_trajectory_layout(trajectory_cfg)}")
    print(
        "dataset_trajectory_setting: "
        f"{trajectory_cfg['end_effector']}/{trajectory_cfg['arm_mode']}"
    )
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

    checkpoint_path = args.policy_path.expanduser().resolve()
    deployment_contract = load_checkpoint_deployment_contract(checkpoint_path)
    metadata_fps = float(deployment_contract["fps"])
    if args.fps is not None and not np.isclose(float(args.fps), metadata_fps):
        raise ValueError(
            f"--fps={args.fps:g} does not match checkpoint metadata fps={metadata_fps:g}."
        )
    fps = metadata_fps

    inference_latency = args.inference_latency
    if inference_latency is None:
        inference_latency = 1.0 / fps

    cfg = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=fps,
        inference_latency=inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    diffusion_cli_overrides = [
        f"--noise_scheduler_type={args.diffusion_noise_scheduler_type}",
        f"--num_inference_steps={args.diffusion_num_inference_steps}",
    ]
    fixed_noise_seed = None if args.disable_diffusion_fixed_noise else args.diffusion_fixed_noise_seed
    serve_deployment_policy_server(
        cfg,
        diffusion_cli_overrides=diffusion_cli_overrides,
        diffusion_fixed_noise_seed=fixed_noise_seed,
        camera_cache_address=args.camera_cache_address,
        max_observation_age_s=args.max_observation_age,
        max_camera_skew_s=args.max_camera_skew,
        checkpoint_path=checkpoint_path,
        deployment_contract=deployment_contract,
        metadata_port=args.metadata_port,
    )


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
