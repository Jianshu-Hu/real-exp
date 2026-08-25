"""Checkpoint-local deployment contracts and read-only remote discovery."""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.trajectory_metadata import (
    validate_action_trajectory_contract,
    validate_trajectory_config,
)

ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
TRAJECTORY_CONFIG_PATH = Path("meta/real_exp_trajectory_config.json")
DATASET_INFO_PATH = Path("meta/info.json")
DEPLOYMENT_CONFIG_PATH = Path("meta/real_exp_deployment_config.json")
DEPLOYMENT_CONFIG_SCHEMA_VERSION = 1


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint deployment metadata: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata must be a JSON object: {path}")
    return value


def infer_actions_per_chunk(policy_type: str, policy_config: dict[str, Any]) -> int:
    if policy_type == "act":
        return int(policy_config.get("n_action_steps", policy_config.get("chunk_size", 1)))
    if policy_type == "diffusion":
        return int(policy_config.get("n_action_steps", 1))
    if policy_type == "vqbet":
        return int(policy_config.get("n_action_pred_token", 1))
    return 1


def deployment_features(dataset_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features = dataset_info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Dataset info must contain a features mapping.")
    selected = {
        key: value
        for key, value in features.items()
        if key in {"observation.state", "action"}
        or key.startswith("observation.images.")
    }
    if "observation.state" not in selected or "action" not in selected:
        raise ValueError("Dataset info must declare observation.state and action features.")
    if not any(key.startswith("observation.images.") for key in selected):
        raise ValueError("Deployment requires at least one image/video observation feature.")
    return selected


def build_deployment_contract(
    policy_config: dict[str, Any],
    dataset_info: dict[str, Any],
    action_config: dict[str, Any],
    trajectory_config: dict[str, Any],
) -> dict[str, Any]:
    policy_type = policy_config.get("type")
    if not isinstance(policy_type, str) or not policy_type:
        raise ValueError("Policy config has no valid type.")
    features = deployment_features(dataset_info)
    try:
        state_dim = int(features["observation.state"]["shape"][0])
        action_dim = int(features["action"]["shape"][0])
        fps = float(dataset_info["fps"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Dataset deployment feature dimensions/FPS are invalid.") from exc
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Dataset deployment FPS must be positive and finite, got {fps!r}.")
    policy_input = policy_config.get("input_features", {})
    policy_output = policy_config.get("output_features", {})
    try:
        policy_state_dim = int(policy_input["observation.state"]["shape"][0])
        policy_action_dim = int(policy_output["action"]["shape"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Policy config must declare one-dimensional state and action features.") from exc
    if policy_state_dim != state_dim or policy_action_dim != action_dim:
        raise ValueError(
            "Checkpoint model state/action dimensions disagree with embedded dataset metadata: "
            f"model={policy_state_dim}/{policy_action_dim}, metadata={state_dim}/{action_dim}"
        )
    policy_cameras = {
        key.removeprefix("observation.images.")
        for key in policy_input
        if key.startswith("observation.images.")
    }
    metadata_cameras = {
        key.removeprefix("observation.images.")
        for key in features
        if key.startswith("observation.images.")
    }
    if policy_cameras != metadata_cameras:
        raise ValueError(
            "Checkpoint model camera features disagree with embedded dataset metadata: "
            f"model={sorted(policy_cameras)}, metadata={sorted(metadata_cameras)}"
        )
    trajectory_config = validate_trajectory_config(
        trajectory_config,
        state_dim,
        action_dim,
        source="checkpoint trajectory metadata",
    )
    validate_action_trajectory_contract(
        action_config, trajectory_config, source="checkpoint metadata"
    )
    return {
        "schema_version": DEPLOYMENT_CONFIG_SCHEMA_VERSION,
        "policy_type": policy_type,
        "actions_per_chunk": infer_actions_per_chunk(policy_type, policy_config),
        "max_actions_per_chunk": infer_actions_per_chunk(policy_type, policy_config),
        "n_obs_steps": int(policy_config.get("n_obs_steps", 1)),
        "fps": fps,
        "camera_names": sorted([
            key.removeprefix("observation.images.")
            for key in features
            if key.startswith("observation.images.")
        ]),
        "features": features,
        "trajectory_config": trajectory_config,
        "action_config": action_config,
    }


def validate_deployment_contract(
    contract: dict[str, Any], *, source: str = "deployment contract"
) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise ValueError(f"{source} must be a JSON object.")
    if contract.get("schema_version") != DEPLOYMENT_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source} schema_version must be {DEPLOYMENT_CONFIG_SCHEMA_VERSION}, "
            f"got {contract.get('schema_version')!r}."
        )
    policy_type = contract.get("policy_type")
    if not isinstance(policy_type, str) or not policy_type:
        raise ValueError(f"{source} has no valid policy_type.")
    try:
        actions_per_chunk = int(contract["actions_per_chunk"])
        max_actions_per_chunk = int(contract.get("max_actions_per_chunk", actions_per_chunk))
        fps = float(contract["fps"])
        n_obs_steps = int(contract.get("n_obs_steps", 1))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} has invalid actions_per_chunk or fps.") from exc
    if actions_per_chunk <= 0:
        raise ValueError(f"{source} actions_per_chunk must be positive.")
    if max_actions_per_chunk <= 0 or actions_per_chunk > max_actions_per_chunk:
        raise ValueError(f"{source} has invalid action chunk limits.")
    if n_obs_steps <= 0:
        raise ValueError(f"{source} n_obs_steps must be positive.")
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{source} fps must be positive and finite.")
    features = contract.get("features")
    if not isinstance(features, dict):
        raise ValueError(f"{source} has no features mapping.")
    try:
        state_dim = int(features["observation.state"]["shape"][0])
        action_dim = int(features["action"]["shape"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} state/action feature dimensions are invalid.") from exc
    trajectory = validate_trajectory_config(
        contract.get("trajectory_config"),
        state_dim,
        action_dim,
        source=f"{source} trajectory metadata",
    )
    action = contract.get("action_config")
    if not isinstance(action, dict):
        raise ValueError(f"{source} has no action_config mapping.")
    validate_action_trajectory_contract(action, trajectory, source=source)
    cameras = deployment_camera_names(contract)
    if not cameras:
        raise ValueError(f"{source} has no image/video camera features.")
    normalized = dict(contract)
    normalized.update(
        {
            "actions_per_chunk": actions_per_chunk,
            "max_actions_per_chunk": max_actions_per_chunk,
            "n_obs_steps": n_obs_steps,
            "fps": fps,
            "trajectory_config": trajectory,
        }
    )
    return normalized


def write_checkpoint_deployment_metadata(
    policy_dir: Path,
    dataset_info: dict[str, Any],
    action_config: dict[str, Any],
    trajectory_config: dict[str, Any],
) -> dict[str, Any]:
    """Write all metadata needed to deploy without retaining the source dataset."""
    policy_dir = policy_dir.expanduser().resolve()
    meta_dir = policy_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    values = {
        DATASET_INFO_PATH: dataset_info,
        ACTION_CONFIG_PATH: action_config,
        TRAJECTORY_CONFIG_PATH: trajectory_config,
    }
    for relative_path, value in values.items():
        (policy_dir / relative_path).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
    policy_config = _load_object(policy_dir / "config.json")
    contract = build_deployment_contract(
        policy_config, dataset_info, action_config, trajectory_config
    )
    (policy_dir / DEPLOYMENT_CONFIG_PATH).write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    return contract


def load_checkpoint_deployment_contract(policy_dir: Path) -> dict[str, Any]:
    """Load a self-contained checkpoint and reject internally inconsistent metadata."""
    policy_dir = policy_dir.expanduser().resolve()
    policy_config = _load_object(policy_dir / "config.json")
    dataset_info = _load_object(policy_dir / DATASET_INFO_PATH)
    action_config = _load_object(policy_dir / ACTION_CONFIG_PATH)
    trajectory_config = _load_object(policy_dir / TRAJECTORY_CONFIG_PATH)
    expected = build_deployment_contract(
        policy_config, dataset_info, action_config, trajectory_config
    )
    stored = validate_deployment_contract(
        _load_object(policy_dir / DEPLOYMENT_CONFIG_PATH), source=str(policy_dir)
    )
    if stored != expected:
        raise ValueError(
            f"Checkpoint deployment manifest disagrees with its config/meta files: {policy_dir}"
        )
    return stored


def fetch_deployment_contract(url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"Deployment metadata server returned HTTP {response.status}.")
        value = json.loads(response.read().decode("utf-8"))
    return validate_deployment_contract(value, source=url)


def metadata_url_from_server_address(server_address: str, metadata_address: str | None = None) -> str:
    if metadata_address:
        if metadata_address.startswith("http://") or metadata_address.startswith("https://"):
            return metadata_address.rstrip("/") + "/deployment-metadata"
        host, _, port = metadata_address.rpartition(":")
        if not host or not port:
            raise ValueError(f"Metadata address must be HOST:PORT or URL, got {metadata_address!r}.")
        return f"http://{host}:{port}/deployment-metadata"
    host = server_address.rsplit(":", 1)[0]
    return f"http://{host}:8081/deployment-metadata"


def deployment_camera_names(contract: dict[str, Any]) -> list[str]:
    feature_cameras = [
        key.removeprefix("observation.images.")
        for key, feature in contract["features"].items()
        if key.startswith("observation.images.")
        and feature.get("dtype") in {"image", "video"}
    ]
    declared = contract.get("camera_names", feature_cameras)
    if not isinstance(declared, list) or len(declared) != len(set(declared)):
        raise ValueError("Deployment contract camera_names must be a unique list.")
    if set(declared) != set(feature_cameras):
        raise ValueError("Deployment contract camera_names disagree with its features.")
    return list(declared)


def deployment_lines(contract: dict[str, Any]) -> list[str]:
    contract = validate_deployment_contract(contract)
    trajectory = contract["trajectory_config"]
    return [
        trajectory["arm_mode"],
        trajectory["end_effector"],
        f"{contract['fps']:g}",
        str(trajectory["robot_state_dim"]),
        str(trajectory["action_dim"]),
        trajectory["state_action_mode"],
        ",".join(deployment_camera_names(contract)),
        contract["policy_type"],
        str(contract["actions_per_chunk"]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--url")
    parser.add_argument("--deployment-lines", action="store_true")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    contract = (
        load_checkpoint_deployment_contract(args.checkpoint)
        if args.checkpoint is not None
        else fetch_deployment_contract(args.url, args.timeout)
    )
    if args.deployment_lines:
        print("\n".join(deployment_lines(contract)))
    else:
        print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
