"""Metadata describing the hardware setting used for a LeRobot trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRAJECTORY_CONFIG_PATH = Path("meta/real_exp_trajectory_config.json")
TRAJECTORY_CONFIG_SCHEMA_VERSION = 1
END_EFFECTOR_MODES = {"arm", "gripper", "hand"}
ARM_MODES = {"duo", "left", "right"}


def normalize_arm_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "single":
        normalized = "left"
    if normalized not in ARM_MODES:
        raise ValueError(f"Unsupported arm mode {value!r}; expected duo, left, or right.")
    return normalized


def trajectory_config_from_packet(packet: dict[str, Any]) -> dict[str, Any]:
    include_hand = bool(packet.get("include_hand", False))
    include_gripper = bool(packet.get("include_gripper", False))
    if include_hand and include_gripper:
        raise ValueError("A trajectory cannot include both a Wuji hand and a Franka gripper.")

    arm_mode = packet.get("arm_mode")
    if arm_mode is None:
        arm_mode = "duo" if bool(packet.get("include_right_arm", True)) else "left"
    arm_mode = normalize_arm_mode(str(arm_mode))
    if arm_mode == "duo" and not bool(packet.get("include_right_arm", True)):
        raise ValueError("The bridge packet declares arm_mode=duo but has no right arm.")
    if arm_mode != "duo" and bool(packet.get("include_right_arm", False)):
        raise ValueError(f"The bridge packet declares arm_mode={arm_mode} but includes a right arm.")

    end_effector = "hand" if include_hand else "gripper" if include_gripper else "arm"
    return {
        "schema_version": TRAJECTORY_CONFIG_SCHEMA_VERSION,
        "end_effector": end_effector,
        "arm_mode": arm_mode,
        "arms": ["left", "right"] if arm_mode == "duo" else [arm_mode],
        "include_gripper": include_gripper,
        "include_hand": include_hand,
        "robot_state_dim": int(packet["robot_state_dim"]),
        "action_dim": int(packet["action_dim"]),
    }


def legacy_trajectory_config(action_config: dict[str, Any], state_dim: int, action_dim: int) -> dict[str, Any]:
    include_hand = bool(action_config.get("include_hand", False))
    include_gripper = bool(action_config.get("include_gripper", False))
    recorded_arm_mode = action_config.get("arm_mode")
    arm_mode = (
        normalize_arm_mode(str(recorded_arm_mode))
        if recorded_arm_mode is not None
        else "duo" if bool(action_config.get("include_right_arm", True)) else "left"
    )
    return {
        "schema_version": TRAJECTORY_CONFIG_SCHEMA_VERSION,
        "end_effector": "hand" if include_hand else "gripper" if include_gripper else "arm",
        "arm_mode": arm_mode,
        "arms": ["left", "right"] if arm_mode == "duo" else [arm_mode],
        "include_gripper": include_gripper,
        "include_hand": include_hand,
        "robot_state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "legacy_inferred": True,
    }


def load_trajectory_config(dataset_root: Path, action_config: dict[str, Any], state_dim: int, action_dim: int) -> dict[str, Any]:
    path = dataset_root / TRAJECTORY_CONFIG_PATH
    if path.exists():
        config = json.loads(path.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Trajectory metadata must be a JSON object: {path}")
        return config
    return legacy_trajectory_config(action_config, state_dim, action_dim)


def validate_setting(config: dict[str, Any], end_effector: str, arm_mode: str) -> None:
    expected_end_effector = str(end_effector).strip().lower()
    if expected_end_effector not in END_EFFECTOR_MODES:
        raise ValueError(f"Unsupported end-effector mode {end_effector!r}; expected arm, gripper, or hand.")
    expected_arm_mode = normalize_arm_mode(arm_mode)
    recorded_end_effector = str(config.get("end_effector", "")).strip().lower()
    recorded_arm_mode = normalize_arm_mode(str(config.get("arm_mode", "")))
    mismatches: list[str] = []
    if recorded_end_effector != expected_end_effector:
        mismatches.append(f"end-effector is '{expected_end_effector}', trajectory requires '{recorded_end_effector}'")
    if recorded_arm_mode != expected_arm_mode:
        mismatches.append(f"arm setting is '{expected_arm_mode}', trajectory requires '{recorded_arm_mode}'")
    if mismatches:
        raise ValueError("Replay setting does not match trajectory metadata: " + "; ".join(mismatches))


def write_trajectory_config(dataset_root: Path, config: dict[str, Any]) -> None:
    path = dataset_root / TRAJECTORY_CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def load_dataset_trajectory_config(dataset_root: Path) -> dict[str, Any]:
    dataset_root = dataset_root.expanduser()
    trajectory_path = dataset_root / TRAJECTORY_CONFIG_PATH
    if trajectory_path.exists():
        config = json.loads(trajectory_path.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Trajectory metadata must be a JSON object: {trajectory_path}")
        return config

    action_path = dataset_root / "meta/real_exp_action_config.json"
    info_path = dataset_root / "meta/info.json"
    if not action_path.exists() or not info_path.exists():
        raise FileNotFoundError(
            f"Trajectory metadata is missing at {trajectory_path}, and the legacy action/info metadata "
            "needed to infer it is incomplete."
        )
    action_config = json.loads(action_path.read_text())
    info = json.loads(info_path.read_text())
    features = info.get("features", {})
    try:
        state_dim = int(features["observation.state"]["shape"][0])
        action_dim = int(features["action"]["shape"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not infer trajectory dimensions from {info_path}") from exc
    return legacy_trajectory_config(action_config, state_dim, action_dim)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect real-exp trajectory hardware metadata.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--field", choices=["end_effector", "arm_mode"], default=None)
    args = parser.parse_args()
    config = load_dataset_trajectory_config(args.dataset_root)
    if args.field is not None:
        print(config[args.field])
    else:
        print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
