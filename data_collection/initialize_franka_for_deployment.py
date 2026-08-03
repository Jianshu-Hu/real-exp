from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
from pathlib import Path
from typing import Any

import numpy as np

import reset_pylibfranka as reset


EXECUTE_CONFIRMATION = "INITIALIZE_FRANKA_FOR_DEPLOYMENT"

# FR3 joint limits from franka_description/robots/fr3/joint_limits.yaml.
FR3_JOINT_LOWER_RAD = np.array(
    [-2.9007, -1.8361, -2.9007, -3.0770, -2.8763, 0.4398, -3.0508],
    dtype=float,
)
FR3_JOINT_UPPER_RAD = np.array(
    [2.9007, 1.8361, 2.9007, -0.1169, 2.8763, 4.6216, 3.0508],
    dtype=float,
)
JOINT_LIMIT_MARGIN_RAD = 0.05
FRANKA_HAND_MAX_WIDTH_M = 0.08
GRIPPER_MEASUREMENT_TOLERANCE_M = 0.002


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize both Franka arms for policy deployment. Select either a dataset "
            "episode start or a postprocessed absolute action supplied by a policy."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--episode",
        nargs="?",
        const="random",
        metavar="INDEX",
        help=(
            "Use frame 0 from a dataset episode. Omit INDEX to randomly select an "
            "episode containing the requested --frame-index."
        ),
    )
    source.add_argument(
        "--policy",
        nargs="+",
        metavar="TARGET",
        help=(
            "Use a postprocessed absolute policy action. Supply 14/16 numbers, a JSON "
            "array, or a JSON file containing action/policy_action/initial_state/target."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="LeRobot dataset root. Required with --episode.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Episode-local frame used with --episode (default: 0).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional reproducible seed for random episode selection.",
    )
    parser.add_argument(
        "--policy-gripper-representation",
        choices=("binary_open_close", "absolute_width"),
        default="binary_open_close",
        help=(
            "Interpretation of dimensions 7 and 15 in a 16-D policy action "
            "(default: binary_open_close)."
        ),
    )
    parser.add_argument("--ip-left", default="172.16.0.3", help="Left Franka IP address.")
    parser.add_argument("--ip-right", default="172.16.0.2", help="Right Franka IP address.")
    parser.add_argument(
        "--skip-grippers",
        action="store_true",
        help="Initialize only the arms and leave both grippers unchanged.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move the robots. Without this flag, only validate and print the target.",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help=f"Required with --execute. Must equal {EXECUTE_CONFIRMATION!r}.",
    )
    return parser.parse_args()


def state_fingerprint(values: np.ndarray) -> str:
    canonical = ",".join(f"{float(value):.9f}" for value in np.asarray(values, dtype=float))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:16]


def dataset_parquet_files(dataset_root: Path) -> tuple[Path, list[Path]]:
    root = dataset_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet data files found under {root / 'data'}")
    return root, paths


def episodes_with_frame(dataset_root: Path, frame_index: int) -> list[int]:
    if frame_index < 0:
        raise ValueError("--frame-index must be non-negative.")
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyarrow is required for --episode mode. Run this command in the environment "
            "that can read the LeRobot dataset."
        ) from exc

    _, paths = dataset_parquet_files(dataset_root)
    episodes: set[int] = set()
    for path in paths:
        table = pq.read_table(path, columns=["episode_index", "frame_index"])
        for episode, frame in zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            strict=True,
        ):
            if int(frame) == frame_index:
                episodes.add(int(episode))
    return sorted(episodes)


def select_episode(args: argparse.Namespace) -> int:
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required with --episode.")

    available = episodes_with_frame(args.dataset_root, args.frame_index)
    if not available:
        raise ValueError(
            f"No episode in {args.dataset_root.expanduser().resolve()} contains "
            f"frame_index={args.frame_index}."
        )

    if args.episode == "random":
        selected = random.Random(args.random_seed).choice(available)
        print(f"Randomly selected episode {selected} from {len(available)} available episode(s).")
        return selected

    try:
        selected = int(args.episode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--episode must be an integer when supplied, got {args.episode!r}.") from exc
    if selected not in available:
        raise ValueError(
            f"Episode {selected} does not contain frame_index={args.frame_index}. "
            f"Available episodes: {available}"
        )
    return selected


def extract_vector_from_json(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("policy_action", "action", "initial_state", "target"):
            if key in value:
                candidate = value[key]
                if isinstance(candidate, dict):
                    for nested_key in ("values", "action", "target"):
                        if nested_key in candidate:
                            return candidate[nested_key]
                return candidate
    raise ValueError(
        "Policy JSON must be an array or contain one of: policy_action, action, "
        "initial_state, target."
    )


def load_policy_vector(tokens: list[str]) -> np.ndarray:
    if len(tokens) > 1:
        raw_values: Any = tokens
    else:
        token = tokens[0].strip()
        candidate_path = Path(token).expanduser()
        if candidate_path.is_file():
            text = candidate_path.read_text().strip()
            try:
                raw_values = extract_vector_from_json(json.loads(text))
            except json.JSONDecodeError:
                records = [line for line in text.splitlines() if line.strip()]
                if not records:
                    raise ValueError(f"Policy target file is empty: {candidate_path}")
                raw_values = extract_vector_from_json(json.loads(records[0]))
        elif token.startswith("["):
            raw_values = extract_vector_from_json(json.loads(token))
        else:
            raw_values = [part.strip() for part in token.split(",") if part.strip()]

    try:
        vector = np.asarray([float(value) for value in raw_values], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("--policy target must contain only numeric values.") from exc
    if vector.ndim != 1 or vector.size not in (14, 16):
        raise ValueError(
            f"--policy requires a 14-D arm action or 16-D arm+gripper action; got shape {vector.shape}."
        )
    return vector


def normalize_gripper_width(value: float, *, source: str) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{source} gripper target is not finite: {value}")
    if value < -GRIPPER_MEASUREMENT_TOLERANCE_M or value > (
        FRANKA_HAND_MAX_WIDTH_M + GRIPPER_MEASUREMENT_TOLERANCE_M
    ):
        raise ValueError(
            f"{source} gripper width {value:.6f} m is outside the Franka Hand range "
            f"[0, {FRANKA_HAND_MAX_WIDTH_M:.3f}] m."
        )
    clipped = float(np.clip(value, 0.0, FRANKA_HAND_MAX_WIDTH_M))
    if clipped != value:
        print(f"{source} gripper width clipped from {value:.6f} m to {clipped:.6f} m.")
    return clipped


def policy_gripper_width(value: float, representation: str, *, source: str) -> float:
    if representation == "binary_open_close":
        if not np.isfinite(value) or value < -0.05 or value > 1.05:
            raise ValueError(f"{source} binary gripper action must be near [0, 1], got {value}.")
        return FRANKA_HAND_MAX_WIDTH_M if value >= 0.5 else 0.0
    return normalize_gripper_width(value, source=source)


def validate_arm_target(name: str, target_q: np.ndarray) -> np.ndarray:
    target_q = np.asarray(target_q, dtype=float)
    if target_q.shape != (7,):
        raise ValueError(f"{name} arm target must have shape (7,), got {target_q.shape}.")
    if not np.all(np.isfinite(target_q)):
        raise ValueError(f"{name} arm target contains NaN or infinity: {target_q}")

    safe_lower = FR3_JOINT_LOWER_RAD + JOINT_LIMIT_MARGIN_RAD
    safe_upper = FR3_JOINT_UPPER_RAD - JOINT_LIMIT_MARGIN_RAD
    invalid = np.flatnonzero((target_q < safe_lower) | (target_q > safe_upper))
    if invalid.size:
        details = ", ".join(
            (
                f"J{index + 1}={target_q[index]:.6f} "
                f"(safe range {safe_lower[index]:.6f}..{safe_upper[index]:.6f})"
            )
            for index in invalid
        )
        raise ValueError(f"{name} arm target violates FR3 joint-limit safety margins: {details}")
    return target_q


def load_episode_target(args: argparse.Namespace) -> dict[str, object]:
    episode = select_episode(args)
    state = reset.load_dataset_reset_state(args.dataset_root, episode, args.frame_index)
    split = reset.split_dual_arm_state(state)
    root = args.dataset_root.expanduser().resolve()
    return {
        "source": f"dataset={root}, episode={episode}, frame_index={args.frame_index}",
        "fingerprint_values": np.asarray(state, dtype=float),
        "left_arm": np.asarray(split["left_arm"], dtype=float),
        "left_gripper": normalize_gripper_width(
            float(split["left_gripper"]), source="Left dataset"
        ),
        "right_arm": np.asarray(split["right_arm"], dtype=float),
        "right_gripper": normalize_gripper_width(
            float(split["right_gripper"]), source="Right dataset"
        ),
    }


def load_policy_target(args: argparse.Namespace) -> dict[str, object]:
    vector = load_policy_vector(args.policy)
    if vector.size == 16:
        left_arm = vector[0:7]
        left_gripper = policy_gripper_width(
            float(vector[7]),
            args.policy_gripper_representation,
            source="Left policy",
        )
        right_arm = vector[8:15]
        right_gripper = policy_gripper_width(
            float(vector[15]),
            args.policy_gripper_representation,
            source="Right policy",
        )
    else:
        left_arm = vector[0:7]
        left_gripper = None
        right_arm = vector[7:14]
        right_gripper = None

    return {
        "source": (
            f"postprocessed policy absolute action ({vector.size}D, "
            f"gripper={args.policy_gripper_representation})"
        ),
        "fingerprint_values": vector,
        "left_arm": np.asarray(left_arm, dtype=float),
        "left_gripper": left_gripper,
        "right_arm": np.asarray(right_arm, dtype=float),
        "right_gripper": right_gripper,
    }


def validate_target(target: dict[str, object]) -> None:
    target["left_arm"] = validate_arm_target(
        "Left", np.asarray(target["left_arm"], dtype=float)
    )
    target["right_arm"] = validate_arm_target(
        "Right", np.asarray(target["right_arm"], dtype=float)
    )


def print_target(args: argparse.Namespace, target: dict[str, object]) -> None:
    print("Franka deployment initialization preview")
    print("----------------------------------------")
    print(f"Target source: {target['source']}")
    print(f"Left arm IP: {args.ip_left}")
    print(f"Right arm IP: {args.ip_right}")
    reset.print_array("Left arm target qpos", np.asarray(target["left_arm"], dtype=float))
    reset.print_array("Right arm target qpos", np.asarray(target["right_arm"], dtype=float))
    if target["left_gripper"] is None or target["right_gripper"] is None:
        print("Gripper targets: not supplied; grippers will be left unchanged.")
    else:
        reset.print_array(
            "Left gripper target width",
            np.asarray([target["left_gripper"]], dtype=float),
        )
        reset.print_array(
            "Right gripper target width",
            np.asarray([target["right_gripper"]], dtype=float),
        )
    print(
        "Motion limits: max joint velocity "
        f"{reset.RESET_MAX_JOINT_VELOCITIES_RAD_PER_S.tolist()} rad/s"
    )
    print(
        "Motion limits: max joint acceleration "
        f"{reset.RESET_MAX_JOINT_ACCELERATIONS_RAD_PER_S2.tolist()} rad/s^2"
    )
    print(
        f"FR3 target validation margin: {JOINT_LIMIT_MARGIN_RAD:.3f} rad inside each joint limit"
    )
    print(
        "Target SHA256 fingerprint: "
        f"{state_fingerprint(np.asarray(target['fingerprint_values'], dtype=float))}"
    )


def run_initialization(args: argparse.Namespace, target: dict[str, object]) -> None:
    abort_event = mp.Event()
    arm_processes = [
        mp.Process(
            target=reset.arm_worker,
            args=(
                args.ip_left,
                np.asarray(target["left_arm"], dtype=float),
                "Left Arm",
                abort_event,
            ),
            name="left_arm_deployment_initialization",
        ),
        mp.Process(
            target=reset.arm_worker,
            args=(
                args.ip_right,
                np.asarray(target["right_arm"], dtype=float),
                "Right Arm",
                abort_event,
            ),
            name="right_arm_deployment_initialization",
        ),
    ]
    for process in arm_processes:
        process.start()
    for process in arm_processes:
        process.join()

    failed = [process.name for process in arm_processes if process.exitcode not in (0, None)]
    if failed:
        abort_event.set()
        raise SystemExit(f"Franka initialization workers failed: {failed}")
    if abort_event.is_set():
        raise SystemExit("Franka arm initialization aborted due to an error.")

    move_grippers = (
        not args.skip_grippers
        and target["left_gripper"] is not None
        and target["right_gripper"] is not None
    )
    if move_grippers:
        reset.move_gripper(args.ip_left, float(target["left_gripper"]), "Left", abort_event)
        reset.move_gripper(args.ip_right, float(target["right_gripper"]), "Right", abort_event)
    if abort_event.is_set():
        raise SystemExit("Franka initialization completed with gripper errors.")
    print("Franka deployment initialization completed.")


def main() -> None:
    args = parse_args()
    target = (
        load_episode_target(args)
        if hasattr(args, "episode") and args.episode is not None
        else load_policy_target(args)
    )
    validate_target(target)
    print_target(args, target)

    if not args.execute:
        print()
        print("Preview only: no robot connection was opened and no motion was commanded.")
        print(
            "After checking the source, target values, and physical workspace, rerun with "
            f"--execute --confirm {EXECUTE_CONFIRMATION}."
        )
        return
    if args.confirm != EXECUTE_CONFIRMATION:
        raise SystemExit(
            f"Refusing to move: --execute requires --confirm {EXECUTE_CONFIRMATION}."
        )

    print()
    print("EXECUTION ENABLED: both Franka arms may move now.")
    run_initialization(args, target)


if __name__ == "__main__":
    main()
