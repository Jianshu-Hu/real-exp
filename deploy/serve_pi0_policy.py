"""Serve the RMBench Franka Pi0 checkpoint over the OpenPI websocket API."""

from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.pi0_deployment import (  # noqa: E402
    PI0_TRAIN_CONFIG,
    load_pi0_deployment_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--metadata-port", type=int, default=8081)
    parser.add_argument("--camera-cache-address", default="tcp://127.0.0.1:5557")
    parser.add_argument("--max-observation-age", type=float, default=0.25)
    parser.add_argument("--max-camera-skew", type=float, default=0.067)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument(
        "--history-overflow", choices=("error", "hold", "slide", "grow"), default="grow"
    )
    return parser.parse_args()


class CameraBundleCache:
    """Small loopback cache of synchronized RGB bundles published by the ROS bridge."""

    def __init__(self, address: str, max_entries: int = 16) -> None:
        import zmq

        self._zmq = zmq
        self._bundles: OrderedDict[int, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, max_entries)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.connect(address)
        self._max_entries = max_entries
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="pi0-camera-cache"
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._socket.close(0)
        self._context.term()

    def _run(self) -> None:
        poller = self._zmq.Poller()
        poller.register(self._socket, self._zmq.POLLIN)
        while not self._stop.is_set():
            if self._socket not in dict(poller.poll(100)):
                continue
            packet = self._socket.recv_pyobj()
            sequence = (
                packet.get("camera_bundle_sequence")
                if isinstance(packet, dict)
                else None
            )
            if sequence is None:
                continue
            with self._lock:
                self._bundles[int(sequence)] = packet
                self._bundles.move_to_end(int(sequence))
                while len(self._bundles) > self._max_entries:
                    self._bundles.popitem(last=False)

    def get(self, sequence: int) -> dict | None:
        with self._lock:
            return self._bundles.get(int(sequence))


class ValidatedPi0Policy:
    """Validate the wire contract and expose only finite absolute targets."""

    def __init__(
        self,
        policy: object,
        camera_cache: CameraBundleCache,
        max_observation_age: float,
        max_camera_skew: float,
        contract: dict,
    ) -> None:
        self._policy = policy
        self._camera_cache = camera_cache
        self._max_observation_age = max_observation_age
        self._max_camera_skew = max_camera_skew
        self.metadata = dict(contract)
        self.metadata["action_output_representation"] = "absolute_target"
        self.metadata["history_reset"] = "on_server_start_or_new_client_connection"

    def reset(self) -> None:
        self._policy.reset_history()

    def infer(self, request: dict) -> dict:
        sequence = request.get("camera_bundle_sequence")
        if sequence is None:
            raise ValueError("Pi0 request has no camera_bundle_sequence.")
        bundle = self._camera_cache.get(int(sequence))
        if bundle is None:
            raise RuntimeError(f"Camera bundle #{sequence} is unavailable or expired.")
        sync = bundle.get("camera_sync") or {}
        if not sync.get("bundle_ready", False):
            raise RuntimeError(f"Camera bundle #{sequence} is not synchronized.")
        skew = float(sync.get("max_skew_s", float("inf")))
        if skew > self._max_camera_skew:
            raise RuntimeError(
                f"Camera bundle #{sequence} skew {skew:.3f}s exceeds {self._max_camera_skew:.3f}s."
            )
        reference_stamp = sync.get("reference_stamp_s")
        if reference_stamp is None:
            raise RuntimeError(f"Camera bundle #{sequence} has no freshness timestamp.")
        age = time.time() - float(reference_stamp)
        if age < -1.0 or age > self._max_observation_age:
            raise RuntimeError(
                f"Camera bundle #{sequence} age {age:.3f}s exceeds {self._max_observation_age:.3f}s."
            )
        state_stamp = request.get("robot_state_stamp_s")
        if state_stamp is None:
            raise RuntimeError(
                f"Request for camera bundle #{sequence} has no robot-state timestamp."
            )
        state_age = time.time() - float(state_stamp)
        if state_age < -1.0 or state_age > self._max_observation_age:
            raise RuntimeError(
                f"Robot state age {state_age:.3f}s exceeds {self._max_observation_age:.3f}s."
            )
        state_camera_skew = abs(float(state_stamp) - float(reference_stamp))
        if state_camera_skew > self._max_observation_age:
            raise RuntimeError(
                f"Robot state/camera skew {state_camera_skew:.3f}s exceeds "
                f"{self._max_observation_age:.3f}s."
            )
        cameras = bundle.get("cameras") or {}
        camera_names = list(self.metadata["camera_names"])
        missing = sorted(set(camera_names) - set(cameras))
        if missing:
            raise RuntimeError(
                f"Camera bundle #{sequence} is missing {', '.join(missing)}."
            )
        for name in camera_names:
            expected_shape = tuple(
                self.metadata["features"][f"observation.images.{name}"]["shape"]
            )
            actual_shape = tuple(np.asarray(cameras[name].get("rgb")).shape)
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"Camera {name} in bundle #{sequence} has shape {actual_shape}; "
                    f"expected {expected_shape}."
                )
        observation = {
            "state": request.get("state"),
            "images": {
                self.metadata["camera_key_map"][name]: np.transpose(
                    np.asarray(cameras[name]["rgb"]), (2, 0, 1)
                )
                for name in camera_names
            },
            "prompt": request.get("prompt") or self.metadata["prompt"],
        }
        state = np.asarray(observation.get("state"), dtype=np.float32)
        state_dim = int(self.metadata["trajectory_config"]["robot_state_dim"])
        if state.shape != (state_dim,) or not np.isfinite(state).all():
            raise ValueError(
                f"Pi0 state must be finite with shape ({state_dim},), got {state.shape}."
            )
        images = observation.get("images")
        if not isinstance(images, dict):
            raise ValueError("Pi0 observation must contain an images mapping.")
        expected = set(self.metadata["camera_key_map"].values())
        if set(images) != expected:
            raise ValueError(
                f"Pi0 expects image keys {sorted(expected)}; got {sorted(images)}."
            )
        for name in sorted(expected):
            image = np.asarray(images[name])
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"{name} must be CHW RGB with shape (3,H,W), got {image.shape}."
                )
        result = self._policy.infer(observation)
        actions = np.asarray(result.get("actions"))
        action_shape = (
            int(self.metadata["max_actions_per_chunk"]),
            int(self.metadata["trajectory_config"]["action_dim"]),
        )
        if actions.shape != action_shape or not np.isfinite(actions).all():
            raise RuntimeError(
                f"Pi0 returned an invalid action chunk: {actions.shape}."
            )
        # create_trained_policy's output transform has already restored the
        # chunk-origin joint delta to absolute joint targets. Do not add state.
        return {**result, "actions": actions}


class MetadataHandler(BaseHTTPRequestHandler):
    contract: dict = {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/deployment-metadata":
            self.send_error(404)
            return
        payload = json.dumps(self.contract).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/deployment-metadata":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        logging.debug(format, *args)


class Pi0JsonWebsocketServer:
    def __init__(self, policy: ValidatedPi0Policy, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self._active_client = False

    async def _handler(self, websocket: object) -> None:
        import websockets

        if self._active_client:
            await websocket.close(
                code=1013, reason="Pi0 history already has an active client"
            )
            return
        self._active_client = True
        self.policy.reset()
        logging.info(
            "Pi0 executor connected from %s; history reset", websocket.remote_address
        )
        try:
            await websocket.send(json.dumps(self.policy.metadata))
            async for message in websocket:
                try:
                    request = json.loads(message)
                    result = await asyncio.to_thread(self.policy.infer, request)
                    await websocket.send(
                        json.dumps({"actions": result["actions"].tolist()})
                    )
                except Exception as exc:
                    logging.error("Pi0 inference request failed: %s", exc)
                    logging.debug(traceback.format_exc())
                    await websocket.send(json.dumps({"error": str(exc)}))
        except websockets.ConnectionClosed:
            pass
        finally:
            self._active_client = False
            logging.info("Pi0 executor disconnected")

    async def run(self) -> None:
        import websockets.asyncio.server

        async with websockets.asyncio.server.serve(
            self._handler, self.host, self.port, compression=None, max_size=None
        ) as server:
            await server.serve_forever()


def create_checkpoint_policy(
    checkpoint: Path,
    contract: dict,
    *,
    default_prompt: str,
    history_overflow: str,
) -> object:
    """Build the OpenPI model and native Franka transforms for a resolved contract."""
    from openpi import transforms
    from openpi.models import history
    from openpi.policies import policy_config
    from openpi.training import config

    base = config.get_config(PI0_TRAIN_CONFIG)
    if contract["train_config"] == PI0_TRAIN_CONFIG:
        train_config = base
    else:
        history_path = checkpoint / "assets" / "history_config.json"
        if not history_path.is_file():
            raise FileNotFoundError(
                f"Auto-detected Pi0 profile requires {history_path} to reconstruct the model."
            )
        saved_history = json.loads(history_path.read_text())
        history_fields = {
            field.name for field in dataclasses.fields(history.HistoryEncoderConfig)
        }
        history_config = history.HistoryEncoderConfig(
            **{
                key: value
                for key, value in saved_history.items()
                if key in history_fields
            }
        )
        action_dim = int(contract["trajectory_config"]["action_dim"])
        if history_config.action_target_dim != action_dim:
            raise ValueError(
                f"Checkpoint history action_target_dim={history_config.action_target_dim} "
                f"does not match detected action_dim={action_dim}."
            )
        model = dataclasses.replace(base.model, history=history_config)
        mask_parts: list[int] = []
        for _ in contract["trajectory_config"]["arms"]:
            mask_parts.extend((7, -1))
        data = config.LeRobotAlohaDataConfig(
            repo_id=contract["checkpoint_asset_id"],
            assets=config.AssetsConfig(asset_id=contract["checkpoint_asset_id"]),
            base_config=base.data.base_config,
            use_delta_joint_actions=True,
            delta_joint_mask=transforms.make_bool_mask(*mask_parts),
            action_output_dim=action_dim,
            default_prompt=default_prompt,
            adapt_to_pi=False,
        )
        train_config = dataclasses.replace(
            base,
            name=f"pi0_auto_{contract['checkpoint_profile']}",
            model=model,
            data=data,
            policy_metadata={
                "robot": f"franka_{contract['trajectory_config']['arm_mode']}",
                "action_output_dim": action_dim,
                "action_representation": "absolute_joint_targets_and_absolute_gripper",
                "training_action_representation": "joint_delta_from_chunk_origin",
                "dataset_fps": contract["fps"],
            },
        )
    return policy_config.create_trained_policy(
        train_config,
        checkpoint,
        default_prompt=default_prompt,
        asset_id=contract["checkpoint_asset_id"],
        history_overflow=history_overflow,
    )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not 1 <= args.port <= 65535 or not 1 <= args.metadata_port <= 65535:
        raise ValueError("Server ports must be in [1, 65535].")
    if args.port == args.metadata_port:
        raise ValueError("Websocket and metadata ports must differ.")
    if args.max_observation_age <= 0 or args.max_camera_skew <= 0:
        raise ValueError("Freshness and camera-skew limits must be positive.")
    contract = load_pi0_deployment_contract(checkpoint)
    default_prompt = args.default_prompt or str(contract["prompt"])

    logging.info("Loading Pi0 checkpoint %s", checkpoint)
    trained_policy = create_checkpoint_policy(
        checkpoint,
        contract,
        default_prompt=default_prompt,
        history_overflow=args.history_overflow,
    )
    camera_cache = CameraBundleCache(args.camera_cache_address)
    camera_cache.start()
    policy = ValidatedPi0Policy(
        trained_policy,
        camera_cache,
        args.max_observation_age,
        args.max_camera_skew,
        contract,
    )
    policy.reset()
    MetadataHandler.contract = policy.metadata
    metadata_server = ThreadingHTTPServer(
        (args.host, args.metadata_port), MetadataHandler
    )
    metadata_thread = threading.Thread(
        target=metadata_server.serve_forever, daemon=True
    )
    metadata_thread.start()
    logging.info(
        "Pi0 checkpoint loaded; serving websocket :%d, metadata HTTP :%d",
        args.port,
        args.metadata_port,
    )
    try:
        asyncio.run(Pi0JsonWebsocketServer(policy, args.host, args.port).run())
    finally:
        metadata_server.shutdown()
        metadata_server.server_close()
        camera_cache.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
