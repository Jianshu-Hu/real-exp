"""Resolve deployable Franka Pi0 checkpoint contracts.

OpenPI Orbax checkpoints do not retain LeRobot's deployment manifest. The
normalization asset retained with the checkpoint identifies the native Franka
layout used by the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PI0_PROTOCOL_VERSION = 1
PI0_TRAIN_CONFIG = "pi0_base_franka_left_memory_260915_anchor_adaln_h50_30k_v1"
PI0_PRESS_BUTTON_TRAIN_CONFIG = (
    "pi0_base_franka_press_button_260917_anchor_adaln_bz32_h50_30k"
)
PI0_PROMPT = (
    "There are four mats, one block, and a button on the table. "
    "One block is on one of the mats. First, put the block to the center, "
    "then press the button. Then, put the block back in its original position."
)
PI0_FPS = 15.0
PI0_HORIZON = 50
PI0_DEFAULT_ACTIONS_PER_CHUNK = 8

PI0_CHECKPOINT_PROFILES: dict[str, dict[str, Any]] = {
    "memory_260915-franka-left-2view-v1": {
        "train_config": PI0_TRAIN_CONFIG,
        "arm_mode": "left",
        "camera_names": ["cam_front", "cam_left"],
        "prompt": PI0_PROMPT,
    },
    "memory_260915-franka-left-front-v1": {
        "train_config": (
            "pi0_base_franka_left_memory_260915_front_anchor_adaln_h50_30k_v1"
        ),
        "arm_mode": "left",
        "camera_names": ["cam_front"],
        "prompt": PI0_PROMPT,
    },
    "press_button-memory-260917-franka-3view-v1": {
        "train_config": PI0_PRESS_BUTTON_TRAIN_CONFIG,
        "arm_mode": "duo",
        "camera_names": ["cam_front", "cam_left", "cam_right"],
        "prompt": "Press the button.",
    },
}


def _franka_contract(
    *,
    arm_mode: str,
    asset_id: str,
    train_config: str,
    prompt: str,
    camera_names: list[str] | None = None,
) -> dict[str, Any]:
    if arm_mode not in {"left", "right", "duo"}:
        raise ValueError(f"Unsupported Franka Pi0 arm mode {arm_mode!r}.")
    arms = ["left", "right"] if arm_mode == "duo" else [arm_mode]
    if camera_names is None:
        camera_names = [
            "cam_front",
            *(["cam_left"] if "left" in arms else []),
            *(["cam_right"] if "right" in arms else []),
        ]
    camera_names = list(camera_names)
    allowed_cameras = {"cam_front", *(f"cam_{side}" for side in arms)}
    if (
        not camera_names
        or len(camera_names) != len(set(camera_names))
        or "cam_front" not in camera_names
        or set(camera_names) - allowed_cameras
    ):
        raise ValueError(
            f"Invalid cameras {camera_names!r} for Franka arm mode {arm_mode!r}."
        )
    camera_key_map = {"cam_front": "cam_high"}
    if "cam_left" in camera_names:
        camera_key_map["cam_left"] = "cam_left_wrist"
    if "cam_right" in camera_names:
        camera_key_map["cam_right"] = "cam_right_wrist"
    state_dim = 8 * len(arms)
    trajectory = {
        "schema_version": 2,
        "end_effector": "gripper",
        "arm_mode": arm_mode,
        "arms": arms,
        "include_gripper": True,
        "include_hand": False,
        "robot_state_dim": state_dim,
        "action_dim": state_dim,
        "state_action_mode": "joint",
        "state_representation": "joint",
        "action_representation": "delta_joint_position",
        "delta_alignment": "chunk_anchor",
    }
    action = {
        "schema_version": 2,
        "action_dim": state_dim,
        "arm_mode": arm_mode,
        "include_gripper": True,
        "include_hand": False,
        "include_right_arm": arm_mode == "duo",
        "state_action_mode": "joint",
        "state_representation": "joint",
        "action_representation": "delta_joint_position",
        "arm_action_representation": "delta_joint_position",
        "gripper_action_representation": "absolute_width",
        "delta_alignment": "chunk_anchor",
        "transport_action_representation": "absolute_target",
    }
    features: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": [state_dim]},
        "action": {"dtype": "float32", "shape": [state_dim]},
    }
    for name in camera_names:
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": [480, 640, 3] if name == "cam_front" else [240, 424, 3],
        }
    return {
        "schema_version": 1,
        "protocol": "real-exp-pi0-websocket",
        "protocol_version": PI0_PROTOCOL_VERSION,
        "policy_type": "pi0",
        "train_config": train_config,
        "checkpoint_asset_id": asset_id,
        "prompt": prompt,
        "actions_per_chunk": PI0_DEFAULT_ACTIONS_PER_CHUNK,
        "max_actions_per_chunk": PI0_HORIZON,
        "n_obs_steps": 1,
        "fps": PI0_FPS,
        "camera_names": camera_names,
        "camera_key_map": camera_key_map,
        "features": features,
        "trajectory_config": trajectory,
        "action_config": action,
    }


def pi0_deployment_contract() -> dict[str, Any]:
    """Return the legacy left-arm/two-view Franka Pi0 contract."""
    return _franka_contract(
        arm_mode="left",
        asset_id="memory_260915-franka-left-2view-v1",
        train_config=PI0_TRAIN_CONFIG,
        prompt=PI0_PROMPT,
    )


def _norm_asset_paths(checkpoint: Path) -> list[Path]:
    return sorted(
        path
        for path in (checkpoint / "assets").glob("*/norm_stats.json")
        if path.is_file()
    )


def _active_norm_dimensions(norm_stats_path: Path, key: str) -> int:
    try:
        stats = json.loads(norm_stats_path.read_text())["norm_stats"][key]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Pi0 normalization stats: {norm_stats_path}") from exc
    vectors = [stats.get(name) for name in ("mean", "std", "q01", "q99")]
    vectors = [vector for vector in vectors if isinstance(vector, list)]
    if not vectors:
        raise ValueError(f"Invalid Pi0 normalization stats: {norm_stats_path}")
    width = max(len(vector) for vector in vectors)
    active = [
        index
        for index in range(width)
        if any(
            index < len(vector) and abs(float(vector[index])) > 1e-12
            for vector in vectors
        )
    ]
    return active[-1] + 1 if active else 0


def load_pi0_deployment_contract(checkpoint: Path | str) -> dict[str, Any]:
    """Derive a supported native-Franka contract from an Orbax checkpoint."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    if (
        not checkpoint.is_dir()
        or not (checkpoint / "_CHECKPOINT_METADATA").is_file()
        or not (checkpoint / "params").is_dir()
    ):
        raise FileNotFoundError(
            f"{checkpoint} is not a complete OpenPI Orbax checkpoint."
        )
    assets = _norm_asset_paths(checkpoint)
    if len(assets) != 1:
        raise ValueError(
            f"{checkpoint} must contain exactly one assets/*/norm_stats.json file."
        )
    norm_stats_path = assets[0]
    asset_id = norm_stats_path.parent.name
    state_dim = _active_norm_dimensions(norm_stats_path, "state")
    action_dim = _active_norm_dimensions(norm_stats_path, "actions")
    if state_dim != action_dim or state_dim not in {8, 16}:
        raise ValueError(
            f"Unsupported Pi0 state/action layout {state_dim}/{action_dim} in {norm_stats_path}."
        )
    profile = PI0_CHECKPOINT_PROFILES.get(asset_id)
    if profile is not None:
        arm_mode = str(profile["arm_mode"])
        expected_dim = 16 if arm_mode == "duo" else 8
        if state_dim != expected_dim:
            raise ValueError(
                f"Pi0 asset {asset_id!r} is registered as {arm_mode} ({expected_dim}-D), "
                f"but its normalization stats use {state_dim} dimensions."
            )
        train_config = str(profile["train_config"])
        camera_names = list(profile["camera_names"])
        prompt = str(profile["prompt"])
    else:
        asset_lower = asset_id.lower()
        if state_dim == 16:
            arm_mode = "duo"
        elif "franka-right" in asset_lower or "franka_right" in asset_lower:
            arm_mode = "right"
        elif "franka-left" in asset_lower or "franka_left" in asset_lower:
            arm_mode = "left"
        else:
            raise ValueError(
                f"Cannot determine the single-arm side from Pi0 asset {asset_id!r}; "
                "use a norm-stats asset name containing franka-left or franka-right."
            )
        train_config = "auto"
        camera_names = None
        prompt = PI0_PROMPT
    contract = _franka_contract(
        arm_mode=arm_mode,
        asset_id=asset_id,
        train_config=train_config,
        prompt=prompt,
        camera_names=camera_names,
    )
    contract["checkpoint_profile"] = (
        f"franka_{arm_mode}_{len(contract['camera_names'])}view"
    )
    return contract


def deployment_lines(contract: dict[str, Any]) -> list[str]:
    trajectory = contract["trajectory_config"]
    return [
        str(trajectory["arm_mode"]),
        str(trajectory["end_effector"]),
        f"{float(contract['fps']):g}",
        str(trajectory["robot_state_dim"]),
        str(trajectory["action_dim"]),
        str(trajectory["state_action_mode"]),
        ",".join(contract["camera_names"]),
        str(contract["policy_type"]),
        str(contract["actions_per_chunk"]),
    ]


def is_pi0_checkpoint(path: Any) -> bool:
    try:
        load_pi0_deployment_contract(path)
    except (FileNotFoundError, ValueError, TypeError):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--deployment-lines", action="store_true")
    args = parser.parse_args()
    contract = (
        load_pi0_deployment_contract(args.checkpoint)
        if args.checkpoint
        else pi0_deployment_contract()
    )
    if args.deployment_lines:
        print("\n".join(deployment_lines(contract)))
    else:
        print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
