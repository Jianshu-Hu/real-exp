"""Metadata describing the hardware setting used for a LeRobot trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

TRAJECTORY_CONFIG_PATH = Path("meta/real_exp_trajectory_config.json")
TRAJECTORY_CONFIG_SCHEMA_VERSION = 2
END_EFFECTOR_MODES = {"arm", "gripper", "hand"}
ARM_MODES = {"duo", "left", "right"}
STATE_ACTION_MODES = {"joint", "end_effector"}
ARM_JOINT_DIM = 7
EE_STATE_DIM = 9
EE_ACTION_DIM = 6
HAND_JOINT_DIM = 20


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
    state_action_mode = str(packet.get("state_action_mode", "joint")).strip().lower()
    if state_action_mode in {"ee", "pose", "end_effector_pose", "end-effector"}:
        state_action_mode = "end_effector"
    if state_action_mode not in STATE_ACTION_MODES:
        raise ValueError(
            f"Unsupported state_action_mode {state_action_mode!r}; expected joint or end_effector."
        )
    state_arm_dim = ARM_JOINT_DIM if state_action_mode == "joint" else EE_STATE_DIM
    action_arm_dim = ARM_JOINT_DIM if state_action_mode == "joint" else EE_ACTION_DIM
    state_block_dim = state_arm_dim
    action_block_dim = action_arm_dim
    if include_gripper:
        state_block_dim += 1
        action_block_dim += 1
    elif include_hand:
        state_block_dim += HAND_JOINT_DIM
        action_block_dim += HAND_JOINT_DIM
    arm_count = 2 if arm_mode == "duo" else 1
    expected_state_dim = state_block_dim * arm_count
    expected_action_dim = action_block_dim * arm_count
    if int(packet["robot_state_dim"]) != expected_state_dim or int(packet["action_dim"]) != expected_action_dim:
        raise ValueError(
            f"Bridge packet state/action dimensions {packet['robot_state_dim']}/{packet['action_dim']} "
            f"do not match {state_action_mode} trajectory layout "
            f"({expected_state_dim}/{expected_action_dim})."
        )
    return {
        "schema_version": TRAJECTORY_CONFIG_SCHEMA_VERSION,
        "end_effector": end_effector,
        "arm_mode": arm_mode,
        "arms": ["left", "right"] if arm_mode == "duo" else [arm_mode],
        "include_gripper": include_gripper,
        "include_hand": include_hand,
        "robot_state_dim": int(packet["robot_state_dim"]),
        "action_dim": int(packet["action_dim"]),
        "state_action_mode": state_action_mode,
        "state_representation": "joint" if state_action_mode == "joint" else "end_effector_position_rotation_6d",
        "action_representation": "delta_joint_position" if state_action_mode == "joint" else "delta_end_effector_position_rotation_vector",
        "delta_alignment": "one_step",
    }


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


def validate_trajectory_config(
    config: dict[str, Any],
    state_dim: int,
    action_dim: int,
    *,
    source: str = "trajectory metadata",
) -> dict[str, Any]:
    """Validate and normalize the metadata-driven robot vector contract."""
    schema_version = int(config.get("schema_version", -1))
    if schema_version != TRAJECTORY_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"{source} has unsupported schema_version={config.get('schema_version')!r}; "
            f"expected {TRAJECTORY_CONFIG_SCHEMA_VERSION}."
        )
    arm_mode = normalize_arm_mode(str(config.get("arm_mode", "")))
    expected_arms = ["left", "right"] if arm_mode == "duo" else [arm_mode]
    arms = config.get("arms")
    if arms != expected_arms:
        raise ValueError(f"{source} arms={arms!r} does not match arm_mode={arm_mode!r}.")

    end_effector = str(config.get("end_effector", "")).strip().lower()
    if end_effector not in END_EFFECTOR_MODES:
        raise ValueError(
            f"{source} has unsupported end_effector={end_effector!r}; "
            "expected arm, gripper, or hand."
        )
    include_gripper = bool(config.get("include_gripper", False))
    include_hand = bool(config.get("include_hand", False))
    if include_gripper != (end_effector == "gripper") or include_hand != (end_effector == "hand"):
        raise ValueError(
            f"{source} end-effector flags are inconsistent: end_effector={end_effector!r}, "
            f"include_gripper={include_gripper}, include_hand={include_hand}."
        )

    state_action_mode = str(config.get("state_action_mode", "joint")).strip().lower()
    if state_action_mode in {"ee", "pose", "end_effector_pose", "end-effector"}:
        state_action_mode = "end_effector"
    if state_action_mode not in STATE_ACTION_MODES:
        raise ValueError(
            f"{source} has unsupported state_action_mode={state_action_mode!r}; "
            "expected joint or end_effector."
        )
    expected_state_representation = (
        "joint" if state_action_mode == "joint" else "end_effector_position_rotation_6d"
    )
    expected_action_representation = (
        "delta_joint_position" if state_action_mode == "joint"
        else "delta_end_effector_position_rotation_vector"
    )
    if config.get("state_representation", expected_state_representation) != expected_state_representation:
        raise ValueError(f"{source} state_representation is inconsistent with state_action_mode.")
    if config.get("action_representation", expected_action_representation) != expected_action_representation:
        raise ValueError(f"{source} action_representation is inconsistent with state_action_mode.")
    state_arm_dim = ARM_JOINT_DIM if state_action_mode == "joint" else EE_STATE_DIM
    action_arm_dim = ARM_JOINT_DIM if state_action_mode == "joint" else EE_ACTION_DIM
    state_block_dim = state_arm_dim
    action_block_dim = action_arm_dim
    if end_effector == "gripper":
        state_block_dim += 1
        action_block_dim += 1
    elif end_effector == "hand":
        state_block_dim += HAND_JOINT_DIM
        action_block_dim += HAND_JOINT_DIM
    expected_state_dim = state_block_dim * len(expected_arms)
    expected_action_dim = action_block_dim * len(expected_arms)
    recorded_state_dim = int(config.get("robot_state_dim", -1))
    recorded_action_dim = int(config.get("action_dim", -1))
    if (recorded_state_dim, recorded_action_dim) != (int(state_dim), int(action_dim)):
        raise ValueError(
            f"{source} records state/action dimensions "
            f"{recorded_state_dim}/{recorded_action_dim}, but the feature contract is "
            f"{state_dim}/{action_dim}."
        )
    if int(state_dim) != expected_state_dim or int(action_dim) != expected_action_dim:
        raise ValueError(
            f"{source} describes {len(expected_arms)} {end_effector} arm block(s), which require "
            f"{expected_state_dim}/{expected_action_dim} values, but state/action dimensions are "
            f"{state_dim}/{action_dim}."
        )
    return {
        **config,
        "arm_mode": arm_mode,
        "arms": expected_arms,
        "end_effector": end_effector,
        "include_gripper": include_gripper,
        "include_hand": include_hand,
        "robot_state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "state_action_mode": state_action_mode,
        "schema_version": schema_version,
        "state_representation": expected_state_representation,
        "action_representation": expected_action_representation,
    }


def validate_action_trajectory_contract(
    action_config: dict[str, Any],
    trajectory_config: dict[str, Any],
    *,
    source: str = "dataset metadata",
) -> None:
    """Require action semantics metadata to agree with the vector layout."""
    state_action_mode = trajectory_config.get("state_action_mode", "joint")
    expected_state_representation = (
        "joint" if state_action_mode == "joint" else "end_effector_position_rotation_6d"
    )
    expected_action_representation = (
        "delta_joint_position" if state_action_mode == "joint"
        else "delta_end_effector_position_rotation_vector"
    )
    expected = {
        "action_dim": int(trajectory_config["action_dim"]),
        "arm_mode": trajectory_config["arm_mode"],
        "include_gripper": bool(trajectory_config["include_gripper"]),
        "include_hand": bool(trajectory_config["include_hand"]),
        "include_right_arm": trajectory_config["arm_mode"] == "duo",
        "state_action_mode": state_action_mode,
        "state_representation": expected_state_representation,
        "action_representation": expected_action_representation,
    }
    mismatches = [
        f"{key}: action metadata={action_config.get(key, value)!r}, trajectory metadata={value!r}"
        for key, value in expected.items()
        if action_config.get(key, value) != value
    ]
    if mismatches:
        raise ValueError(f"{source} contracts disagree: " + "; ".join(mismatches))

    if action_config.get("delta_alignment") == "chunk_anchor":
        if action_config.get("transport_action_representation") != "absolute_target":
            raise ValueError(
                f"{source} chunk-anchored actions must declare "
                "transport_action_representation='absolute_target'."
            )

    arm_representation = str(action_config.get("arm_action_representation", "")).strip().lower()
    state_action_mode = str(trajectory_config.get("state_action_mode", "joint")).strip().lower()
    expected_arm_representation = (
        "delta_joint_position" if state_action_mode == "joint"
        else "delta_end_effector_position_rotation_vector"
    )
    if arm_representation != expected_arm_representation:
        raise ValueError(
            f"{source} requires arm_action_representation={expected_arm_representation!r}; "
            f"got {arm_representation!r}."
        )
    end_effector = trajectory_config["end_effector"]
    if end_effector == "gripper":
        gripper_representation = str(
            action_config.get("gripper_action_representation", "")
        ).strip().lower()
        if gripper_representation != "absolute_width":
            raise ValueError(
                f"{source} requires gripper_action_representation='absolute_width'; "
                f"got {gripper_representation!r}."
            )
    elif end_effector == "hand":
        hand_representation = str(
            action_config.get("hand_action_representation", "")
        ).strip().lower()
        if hand_representation != "absolute_joint_position":
            raise ValueError(
                f"{source} requires hand_action_representation='absolute_joint_position'; "
                f"got {hand_representation!r}."
            )


def require_dataset_trajectory_config(dataset_root: Path) -> dict[str, Any]:
    """Load the current trajectory contract without legacy inference."""
    dataset_root = dataset_root.expanduser().resolve()
    trajectory_path = dataset_root / TRAJECTORY_CONFIG_PATH
    info_path = dataset_root / "meta/info.json"
    if not trajectory_path.exists():
        raise FileNotFoundError(
            f"Missing trajectory metadata: {trajectory_path}. Current training and deployment "
            "require an explicit real_exp_trajectory_config.json."
        )
    if not info_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {info_path}")
    config = json.loads(trajectory_path.read_text())
    info = json.loads(info_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Trajectory metadata must be a JSON object: {trajectory_path}")
    try:
        state_dim = int(info["features"]["observation.state"]["shape"][0])
        action_dim = int(info["features"]["action"]["shape"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not read state/action dimensions from {info_path}") from exc
    return validate_trajectory_config(
        config, state_dim, action_dim, source=str(trajectory_path)
    )


def describe_trajectory_layout(config: dict[str, Any]) -> str:
    parts: list[str] = []
    end_effector = str(config["end_effector"])
    if config.get("state_action_mode", "joint") == "joint":
        state_label, action_label = "Joint state(7)", "Joint delta(7)"
    else:
        state_label, action_label = "EE state(9)", "EE delta(6)"
    for side in config["arms"]:
        parts.append(f"{side.title()} {state_label}/{action_label}")
        if end_effector == "gripper":
            parts.append(f"{side.title()} Gripper(1)")
        elif end_effector == "hand":
            parts.append(f"{side.title()} Hand(20)")
    return "[" + ", ".join(parts) + "]"


def split_trajectory_vector(
    values: Any, config: dict[str, Any], *, kind: str = "action"
) -> dict[str, Any]:
    """Split one state/action vector according to the explicit trajectory metadata."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    if kind not in {"state", "action"}:
        raise ValueError(f"Unsupported trajectory vector kind {kind!r}.")
    expected_dim = int(config[f"{kind}_dim"] if f"{kind}_dim" in config else config["robot_state_dim" if kind == "state" else "action_dim"])
    if array.ndim != 1 or array.shape[0] != expected_dim:
        raise ValueError(
            f"Expected a one-dimensional {expected_dim}-value trajectory vector, got {array.shape}."
        )
    result: dict[str, Any] = {
        "left_arm": None,
        "left_ee_pose": None,
        "left_delta_ee_pose": None,
        "left_gripper": None,
        "left_hand": None,
        "right_arm": None,
        "right_ee_pose": None,
        "right_delta_ee_pose": None,
        "right_gripper": None,
        "right_hand": None,
    }
    offset = 0
    state_action_mode = str(config.get("state_action_mode", "joint")).strip().lower()
    for side in config["arms"]:
        if state_action_mode == "joint":
            result[f"{side}_arm"] = array[offset : offset + ARM_JOINT_DIM]
            offset += ARM_JOINT_DIM
        else:
            arm_dim = EE_STATE_DIM if kind == "state" else EE_ACTION_DIM
            pose = array[offset : offset + arm_dim]
            result[f"{side}_arm"] = pose
            if kind == "state":
                result[f"{side}_ee_pose"] = pose
            else:
                result[f"{side}_delta_ee_pose"] = pose
            offset += arm_dim
        if config["end_effector"] == "gripper":
            result[f"{side}_gripper"] = float(array[offset])
            offset += 1
        elif config["end_effector"] == "hand":
            result[f"{side}_hand"] = array[offset : offset + HAND_JOINT_DIM]
            offset += HAND_JOINT_DIM
    if offset != expected_dim:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Trajectory layout consumed {offset} of {expected_dim} values.")
    return result


def absolute_transport_action_dim(config: dict[str, Any]) -> int:
    """Return the executor transport width for one absolute target vector."""
    arm_dim = ARM_JOINT_DIM if config["state_action_mode"] == "joint" else EE_STATE_DIM
    extra_dim = (
        1 if config["end_effector"] == "gripper"
        else HAND_JOINT_DIM if config["end_effector"] == "hand"
        else 0
    )
    return (arm_dim + extra_dim) * len(config["arms"])


def split_absolute_transport_action(values: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Split the absolute target vector sent from the policy server to an executor."""
    transport_config = dict(config)
    transport_config["robot_state_dim"] = absolute_transport_action_dim(config)
    return split_trajectory_vector(values, transport_config, kind="state")


def validate_live_packet(config: dict[str, Any], packet: dict[str, Any]) -> None:
    """Require the live bridge to match the dataset/checkpoint hardware contract."""
    packet_config = trajectory_config_from_packet(packet)
    fields = (
        "arm_mode",
        "arms",
        "end_effector",
        "include_gripper",
        "include_hand",
        "robot_state_dim",
        "action_dim",
        "state_action_mode",
        "state_representation",
        "action_representation",
    )
    mismatches = [
        f"{field}: dataset={config.get(field)!r}, live={packet_config.get(field)!r}"
        for field in fields
        if config.get(field) != packet_config.get(field)
    ]
    if mismatches:
        raise ValueError(
            "Live deployment bridge does not match trajectory metadata: " + "; ".join(mismatches)
        )
    if config.get("state_action_mode", "joint") == "end_effector":
        import numpy as np

        expected_dims = {
            "ee_pose": EE_STATE_DIM * len(config["arms"]),
            "target_ee_pose": EE_STATE_DIM * len(config["arms"]),
            "delta_ee_pose": EE_ACTION_DIM * len(config["arms"]),
        }
        for key, expected_dim in expected_dims.items():
            values = np.asarray(packet.get(key, []), dtype=float)
            if values.shape != (expected_dim,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Live deployment bridge {key} has shape {values.shape}; "
                    f"expected a finite {expected_dim}-value vector."
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect real-exp trajectory hardware metadata.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--field", choices=["end_effector", "arm_mode"], default=None)
    parser.add_argument(
        "--deployment-lines",
        action="store_true",
        help=(
            "Print the validated explicit deployment contract as stable newline-delimited fields: "
            "arm mode, end effector, fps, state dim, action dim, and comma-separated cameras."
        ),
    )
    args = parser.parse_args()
    if args.deployment_lines:
        config = require_dataset_trajectory_config(args.dataset_root)
        dataset_root = args.dataset_root.expanduser().resolve()
        info = json.loads((dataset_root / "meta/info.json").read_text())
        action_path = dataset_root / "meta/real_exp_action_config.json"
        if not action_path.is_file():
            raise FileNotFoundError(f"Missing dataset action metadata: {action_path}")
        validate_action_trajectory_contract(
            json.loads(action_path.read_text()), config, source=str(dataset_root / "meta")
        )
        cameras = [
            key.removeprefix("observation.images.")
            for key, feature in info["features"].items()
            if key.startswith("observation.images.")
            and feature.get("dtype") in {"image", "video"}
        ]
        if not cameras:
            raise ValueError("Deployment dataset has no image/video observation features.")
        for value in (
            config["arm_mode"],
            config["end_effector"],
            info["fps"],
            config["robot_state_dim"],
            config["action_dim"],
            ",".join(cameras),
        ):
            print(value)
    elif args.field is not None:
        config = require_dataset_trajectory_config(args.dataset_root)
        print(config[args.field])
    else:
        config = require_dataset_trajectory_config(args.dataset_root)
        print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
