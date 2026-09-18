"""Fixed deployment contract for the RMBench Franka Pi0 checkpoint.

The Orbax checkpoint does not contain the LeRobot ``config.json``/``meta``
manifest used by the older deployment path.  These values come from
``libs/RMBench/policy/pi0/README_REAL.md`` and
``openpi.shared.franka_memory`` and deliberately live outside the checkpoint
so an extracted checkpoint remains byte-for-byte identical to training output.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


PI0_PROTOCOL_VERSION = 1
PI0_TRAIN_CONFIG = "pi0_base_franka_left_memory_260915_anchor_adaln_h50_30k_v1"
PI0_PROMPT = (
    "There are four mats, one block, and a button on the table. "
    "One block is on one of the mats. First, put the block to the center, "
    "then press the button. Then, put the block back in its original position."
)
PI0_FPS = 15.0
PI0_HORIZON = 50
PI0_DEFAULT_ACTIONS_PER_CHUNK = 8


def pi0_deployment_contract() -> dict[str, Any]:
    """Return the robot/camera contract expected by ``pi0_15000``."""
    trajectory = {
        "schema_version": 2,
        "end_effector": "gripper",
        "arm_mode": "left",
        "arms": ["left"],
        "include_gripper": True,
        "include_hand": False,
        "robot_state_dim": 8,
        "action_dim": 8,
        "state_action_mode": "joint",
        "state_representation": "joint",
        # Training represents arm actions as chunk-origin deltas. OpenPI's
        # output transform has already decoded them before network transport.
        "action_representation": "delta_joint_position",
        "delta_alignment": "chunk_anchor",
    }
    action = {
        "schema_version": 2,
        "action_dim": 8,
        "arm_mode": "left",
        "include_gripper": True,
        "include_hand": False,
        "include_right_arm": False,
        "state_action_mode": "joint",
        "state_representation": "joint",
        "action_representation": "delta_joint_position",
        "arm_action_representation": "delta_joint_position",
        "gripper_action_representation": "absolute_width",
        "delta_alignment": "chunk_anchor",
        "transport_action_representation": "absolute_target",
    }
    return {
        "schema_version": 1,
        "protocol": "real-exp-pi0-websocket",
        "protocol_version": PI0_PROTOCOL_VERSION,
        "policy_type": "pi0",
        "train_config": PI0_TRAIN_CONFIG,
        "prompt": PI0_PROMPT,
        "actions_per_chunk": PI0_DEFAULT_ACTIONS_PER_CHUNK,
        "max_actions_per_chunk": PI0_HORIZON,
        "n_obs_steps": 1,
        "fps": PI0_FPS,
        "camera_names": ["cam_front", "cam_left"],
        "camera_key_map": {
            "cam_front": "cam_high",
            "cam_left": "cam_left_wrist",
        },
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
            },
            "observation.images.cam_front": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
            "observation.images.cam_left": {
                "dtype": "video",
                "shape": [240, 424, 3],
            },
            "action": {
                "dtype": "float32",
                "shape": [8],
            },
        },
        "trajectory_config": trajectory,
        "action_config": action,
    }


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
    """Recognize a complete OpenPI Orbax step directory."""
    return (
        path.is_dir()
        and (path / "_CHECKPOINT_METADATA").is_file()
        and (path / "params").is_dir()
        and (path / "assets" / "memory_260915-franka-left-2view-v1" / "norm_stats.json").is_file()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-lines", action="store_true")
    args = parser.parse_args()
    contract = pi0_deployment_contract()
    if args.deployment_lines:
        print("\n".join(deployment_lines(contract)))
    else:
        print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
