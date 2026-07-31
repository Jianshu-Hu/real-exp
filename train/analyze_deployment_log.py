from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-real-exp"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_JOINT_SPIKE_THRESHOLD = 0.12
DEFAULT_ARM_NORM_SPIKE_THRESHOLD = 0.25
DEFAULT_TARGET_STATE_THRESHOLD = 0.25
DEFAULT_TRACKING_LAG_STEPS = 1
DEFAULT_TOP_K_SPIKES = 20

LEFT_JOINT_NAMES = [f"left_j{idx}" for idx in range(7)]
RIGHT_JOINT_NAMES = [f"right_j{idx}" for idx in range(7)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Franka deployment logs and generate plots, CSVs, and summaries."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        action="append",
        required=True,
        help="Deployment log directory containing metadata.json and samples.jsonl. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Defaults to <log-dir>/analysis for one log, or "
            "<first-log-parent>/analysis_compare_<timestamp> for multiple logs."
        ),
    )
    parser.add_argument("--compare", action="store_true", help="Generate multi-run comparison outputs.")
    parser.add_argument(
        "--joint-spike-threshold",
        type=float,
        default=DEFAULT_JOINT_SPIKE_THRESHOLD,
        help="Per-joint target jump spike threshold in rad/frame.",
    )
    parser.add_argument(
        "--arm-norm-spike-threshold",
        type=float,
        default=DEFAULT_ARM_NORM_SPIKE_THRESHOLD,
        help="Per-arm target jump norm spike threshold in rad/frame.",
    )
    parser.add_argument(
        "--target-state-threshold",
        type=float,
        default=DEFAULT_TARGET_STATE_THRESHOLD,
        help="Per-arm target-minus-state max-absolute threshold in rad.",
    )
    parser.add_argument(
        "--tracking-lag-steps",
        type=int,
        default=DEFAULT_TRACKING_LAG_STEPS,
        help="Number of action records to lag for rough robot-state tracking error.",
    )
    parser.add_argument(
        "--top-k-spikes",
        type=int,
        default=DEFAULT_TOP_K_SPIKES,
        help="Number of largest spike rows to write and visualize.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Write only CSV/JSON/text outputs.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def get_nested(value: dict[str, Any] | None, path: str, default: Any = None) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)
    return default if current is None else current


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def finite_array(values: Iterable[Any]) -> np.ndarray:
    finite: list[float] = []
    for value in values:
        number = as_float(value)
        if number is not None:
            finite.append(number)
    return np.asarray(finite, dtype=float)


def stats(values: Iterable[Any]) -> dict[str, float | int | None]:
    array = finite_array(values)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
            "rms": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p95": float(np.percentile(array, 95)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def vector_or_none(value: Any) -> np.ndarray | None:
    if not isinstance(value, list):
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size not in {14, 16}:
        return None
    return array


def fixed_vector_or_none(value: Any, length: int) -> np.ndarray | None:
    if not isinstance(value, list):
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size != length:
        return None
    return array


def split_vector(value: Any) -> dict[str, np.ndarray | float | None]:
    array = vector_or_none(value)
    if array is None:
        return {
            "left_arm": None,
            "right_arm": None,
            "left_gripper": None,
            "right_gripper": None,
        }
    if array.size == 16:
        return {
            "left_arm": array[0:7],
            "left_gripper": float(array[7]),
            "right_arm": array[8:15],
            "right_gripper": float(array[15]),
        }
    return {
        "left_arm": array[0:7],
        "left_gripper": None,
        "right_arm": array[7:14],
        "right_gripper": None,
    }


def array_norm(value: np.ndarray | None) -> float | None:
    if value is None:
        return None
    return float(np.linalg.norm(value))


def array_abs_max(value: np.ndarray | None) -> float | None:
    if value is None or value.size == 0:
        return None
    return float(np.max(np.abs(value)))


def subtract(left: np.ndarray | None, right: np.ndarray | None) -> np.ndarray | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    return left - right


def flatten_arm(prefix: str, values: np.ndarray | None, row: dict[str, Any], names: list[str]) -> None:
    if values is None:
        for name in names:
            row[f"{prefix}_{name}"] = None
        return
    for name, value in zip(names, values, strict=True):
        row[f"{prefix}_{name}"] = float(value)


def flatten_metric(prefix: str, values: np.ndarray | None, row: dict[str, Any], names: list[str]) -> None:
    flatten_arm(prefix, values, row, names)
    row[f"{prefix}_norm"] = array_norm(values)
    row[f"{prefix}_abs_max"] = array_abs_max(values)


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def last_chunk_before(action_elapsed: float | None, chunk_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if action_elapsed is None:
        return None
    latest: dict[str, Any] | None = None
    for row in chunk_rows:
        elapsed = as_float(row.get("elapsed_s"))
        if elapsed is not None and elapsed <= action_elapsed:
            latest = row
        if elapsed is not None and elapsed > action_elapsed:
            break
    return latest


def build_chunk_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if record.get("event") != "action_chunk_received":
            continue
        row: dict[str, Any] = {
            "row_index": index,
            "elapsed_s": as_float(record.get("elapsed_s")),
            "wall_time": as_float(record.get("wall_time")),
            "incoming_count": record.get("incoming_count"),
            "filtered_count": record.get("filtered_count"),
            "inflight_latency_s": as_float(record.get("inflight_latency_s")),
            "overlap_count": record.get("overlap_count"),
            "new_action_count": record.get("new_action_count"),
            "dropped_action_count": record.get("dropped_action_count"),
            "latest_executed_timestep": record.get("latest_executed_timestep"),
            "cleared_inflight_observation_timestep": record.get(
                "cleared_inflight_observation_timestep"
            ),
        }
        for source_name in ["overlap_raw_delta", "overlap_blended_delta", "queue_after_update_delta"]:
            for arm in ["left", "right"]:
                for stat_name in ["norm", "abs_max"]:
                    for reducer in ["mean", "median", "max", "min"]:
                        path = f"aggregate.{source_name}.{arm}_{stat_name}.{reducer}"
                        row[f"{source_name}_{arm}_{stat_name}_{reducer}"] = get_nested(record, path)
        rows.append(row)
    rows.sort(key=lambda item: item.get("elapsed_s") if item.get("elapsed_s") is not None else -1)
    return rows


def build_observation_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_must_go_elapsed: float | None = None
    for index, record in enumerate(records):
        if record.get("event") != "observation_sent":
            continue
        elapsed = as_float(record.get("elapsed_s"))
        must_go = bool(record.get("must_go"))
        camera_ages = [
            value
            for value in (record.get("executor_wall_minus_camera_stamp_s") or {}).values()
            if as_float(value) is not None
        ]
        history_quality = record.get("diffusion_history_quality") or {}
        must_go_interval = None
        if must_go and previous_must_go_elapsed is not None and elapsed is not None:
            must_go_interval = elapsed - previous_must_go_elapsed
        if must_go and elapsed is not None:
            previous_must_go_elapsed = elapsed
        row = {
            "row_index": index,
            "elapsed_s": elapsed,
            "wall_time": as_float(record.get("wall_time")),
            "observation_timestep": record.get("observation_timestep"),
            "action_anchor_timestep": record.get("action_anchor_timestep"),
            "must_go": must_go,
            "must_go_interval_s": must_go_interval,
            "queue_size": get_nested(record, "queue.size"),
            "queue_first_timestep": get_nested(record, "queue.first_timestep"),
            "queue_last_timestep": get_nested(record, "queue.last_timestep"),
            "inflight_observation_timestep": record.get("inflight_observation_timestep"),
            "latest_executed_timestep": record.get("latest_executed_timestep"),
            "dropped_zmq_packets": record.get("dropped_zmq_packets"),
            "bridge_to_executor_s": as_float(record.get("bridge_to_executor_s")),
            "camera_age_mean_s": float(np.mean(camera_ages)) if camera_ages else None,
            "camera_age_max_s": float(np.max(camera_ages)) if camera_ages else None,
            "history_ready": history_quality.get("ready"),
            "history_len": history_quality.get("history_len"),
            "required_history_len": history_quality.get("required_history_len"),
            "history_repeated_camera_count": len(history_quality.get("repeated_cameras") or []),
            "history_max_camera_age_s": history_quality.get("max_camera_age_s"),
        }
        rows.append(row)
    return rows


def build_action_rows(
    records: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    tracking_lag_steps: int,
) -> list[dict[str, Any]]:
    action_records = [
        (index, record)
        for index, record in enumerate(records)
        if record.get("event") in {"action_executed", "action_predicted"}
    ]
    rows: list[dict[str, Any]] = []
    split_cache: list[dict[str, dict[str, np.ndarray | float | None]]] = []

    for action_index, (record_index, record) in enumerate(action_records):
        robot_split = split_vector(record.get("robot_state"))
        predicted_split = split_vector(record.get("predicted_action"))
        split_cache.append({"robot": robot_split, "predicted": predicted_split})

        previous = split_cache[action_index - 1] if action_index > 0 else None
        left_target_delta = subtract(
            predicted_split["left_arm"], previous["predicted"]["left_arm"] if previous else None
        )
        right_target_delta = subtract(
            predicted_split["right_arm"], previous["predicted"]["right_arm"] if previous else None
        )
        left_state_delta = subtract(
            robot_split["left_arm"], previous["robot"]["left_arm"] if previous else None
        )
        right_state_delta = subtract(
            robot_split["right_arm"], previous["robot"]["right_arm"] if previous else None
        )

        logged_left_tms = fixed_vector_or_none(record.get("left_target_minus_state"), 7)
        logged_right_tms = fixed_vector_or_none(record.get("right_target_minus_state"), 7)
        if logged_left_tms is not None:
            left_target_minus_state = logged_left_tms
        else:
            left_target_minus_state = subtract(
                predicted_split["left_arm"], robot_split["left_arm"]
            )
        if logged_right_tms is not None:
            right_target_minus_state = logged_right_tms
        else:
            right_target_minus_state = subtract(
                predicted_split["right_arm"], robot_split["right_arm"]
            )

        chunk = last_chunk_before(as_float(record.get("elapsed_s")), chunk_rows)
        row: dict[str, Any] = {
            "record_index": record_index,
            "action_index": action_index,
            "event": record.get("event"),
            "elapsed_s": as_float(record.get("elapsed_s")),
            "wall_time": as_float(record.get("wall_time")),
            "action_timestep": record.get("action_timestep"),
            "action_timestamp": as_float(record.get("action_timestamp")),
            "latest_executed_timestep": record.get("latest_executed_timestep"),
            "execute_wall_dt": as_float(record.get("execute_wall_dt")),
            "queue_before_size": get_nested(record, "queue_before_pop.size"),
            "queue_after_size": get_nested(record, "queue_after_pop.size"),
            "last_chunk_overlap_count": chunk.get("overlap_count") if chunk else None,
            "last_chunk_new_action_count": chunk.get("new_action_count") if chunk else None,
            "last_chunk_inflight_latency_s": chunk.get("inflight_latency_s") if chunk else None,
            "left_target_delta_norm": array_norm(left_target_delta),
            "right_target_delta_norm": array_norm(right_target_delta),
            "left_target_delta_abs_max": array_abs_max(left_target_delta),
            "right_target_delta_abs_max": array_abs_max(right_target_delta),
            "left_state_delta_norm": array_norm(left_state_delta),
            "right_state_delta_norm": array_norm(right_state_delta),
            "left_state_delta_abs_max": array_abs_max(left_state_delta),
            "right_state_delta_abs_max": array_abs_max(right_state_delta),
            "left_target_minus_state_norm": array_norm(left_target_minus_state),
            "right_target_minus_state_norm": array_norm(right_target_minus_state),
            "left_target_minus_state_abs_max": array_abs_max(left_target_minus_state),
            "right_target_minus_state_abs_max": array_abs_max(right_target_minus_state),
            "left_gripper_state": robot_split["left_gripper"],
            "right_gripper_state": robot_split["right_gripper"],
            "left_gripper_action": predicted_split["left_gripper"],
            "right_gripper_action": predicted_split["right_gripper"],
        }
        flatten_arm("left_state", robot_split["left_arm"], row, LEFT_JOINT_NAMES)
        flatten_arm("right_state", robot_split["right_arm"], row, RIGHT_JOINT_NAMES)
        flatten_arm("left_action", predicted_split["left_arm"], row, LEFT_JOINT_NAMES)
        flatten_arm("right_action", predicted_split["right_arm"], row, RIGHT_JOINT_NAMES)
        flatten_metric("left_target_delta", left_target_delta, row, LEFT_JOINT_NAMES)
        flatten_metric("right_target_delta", right_target_delta, row, RIGHT_JOINT_NAMES)
        flatten_metric("left_state_delta", left_state_delta, row, LEFT_JOINT_NAMES)
        flatten_metric("right_state_delta", right_state_delta, row, RIGHT_JOINT_NAMES)
        flatten_metric("left_target_minus_state", left_target_minus_state, row, LEFT_JOINT_NAMES)
        flatten_metric("right_target_minus_state", right_target_minus_state, row, RIGHT_JOINT_NAMES)
        rows.append(row)

    if tracking_lag_steps > 0:
        for index, row in enumerate(rows):
            lagged_index = index + tracking_lag_steps
            if lagged_index >= len(split_cache):
                left_tracking = None
                right_tracking = None
            else:
                future_robot = split_cache[lagged_index]["robot"]
                current_predicted = split_cache[index]["predicted"]
                left_tracking = subtract(
                    future_robot["left_arm"], current_predicted["left_arm"]
                )
                right_tracking = subtract(
                    future_robot["right_arm"], current_predicted["right_arm"]
                )
            row["tracking_lag_steps"] = tracking_lag_steps
            flatten_metric("left_tracking_error", left_tracking, row, LEFT_JOINT_NAMES)
            flatten_metric("right_tracking_error", right_tracking, row, RIGHT_JOINT_NAMES)
    else:
        for row in rows:
            row["tracking_lag_steps"] = tracking_lag_steps

    return rows


def build_spike_rows(
    action_rows: list[dict[str, Any]],
    joint_threshold: float,
    arm_norm_threshold: float,
    target_state_threshold: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], int]:
    spikes: list[dict[str, Any]] = []
    arm_specs = [
        ("left", LEFT_JOINT_NAMES),
        ("right", RIGHT_JOINT_NAMES),
    ]
    for row in action_rows:
        for arm, names in arm_specs:
            norm_value = as_float(row.get(f"{arm}_target_delta_norm"))
            if norm_value is not None and norm_value > arm_norm_threshold:
                spikes.append(spike_row(row, arm, "target_delta_norm", None, norm_value))

            target_state_value = as_float(row.get(f"{arm}_target_minus_state_abs_max"))
            if target_state_value is not None and target_state_value > target_state_threshold:
                spikes.append(
                    spike_row(row, arm, "target_minus_state_abs_max", None, target_state_value)
                )

            for name in names:
                value = as_float(row.get(f"{arm}_target_delta_{name}"))
                if value is not None and abs(value) > joint_threshold:
                    spikes.append(
                        spike_row(row, arm, "target_delta_joint", name, value, abs(value))
                    )

    spikes.sort(key=lambda item: abs(float(item["ranking_value"])), reverse=True)
    return spikes[: max(top_k, 0)], len(spikes)


def spike_row(
    action_row: dict[str, Any],
    arm: str,
    metric: str,
    joint: str | None,
    value: float,
    ranking_value: float | None = None,
) -> dict[str, Any]:
    return {
        "action_index": action_row.get("action_index"),
        "elapsed_s": action_row.get("elapsed_s"),
        "action_timestep": action_row.get("action_timestep"),
        "arm": arm,
        "joint": joint,
        "metric": metric,
        "value": value,
        "ranking_value": abs(value) if ranking_value is None else ranking_value,
        "last_chunk_overlap_count": action_row.get("last_chunk_overlap_count"),
        "last_chunk_new_action_count": action_row.get("last_chunk_new_action_count"),
        "last_chunk_inflight_latency_s": action_row.get("last_chunk_inflight_latency_s"),
        f"{arm}_target_delta_norm": action_row.get(f"{arm}_target_delta_norm"),
        f"{arm}_target_delta_abs_max": action_row.get(f"{arm}_target_delta_abs_max"),
        f"{arm}_target_minus_state_abs_max": action_row.get(
            f"{arm}_target_minus_state_abs_max"
        ),
    }


def build_summary(
    log_dir: Path,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    spike_count: int,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    elapsed_values = [row.get("elapsed_s") for row in action_rows + observation_rows + chunk_rows]
    elapsed_array = finite_array(elapsed_values)
    duration = float(np.max(elapsed_array) - np.min(elapsed_array)) if elapsed_array.size else 0.0
    event_counts: dict[str, int] = {}
    for record in records:
        event = str(record.get("event"))
        event_counts[event] = event_counts.get(event, 0) + 1

    summary = {
        "run_name": log_dir.name,
        "log_dir": str(log_dir),
        "metadata": metadata,
        "thresholds": thresholds,
        "duration_s": duration,
        "event_counts": event_counts,
        "actions": {
            "count": len(action_rows),
            "left_target_delta_norm": stats(row.get("left_target_delta_norm") for row in action_rows),
            "right_target_delta_norm": stats(
                row.get("right_target_delta_norm") for row in action_rows
            ),
            "left_target_delta_abs_max": stats(
                row.get("left_target_delta_abs_max") for row in action_rows
            ),
            "right_target_delta_abs_max": stats(
                row.get("right_target_delta_abs_max") for row in action_rows
            ),
            "left_state_delta_norm": stats(row.get("left_state_delta_norm") for row in action_rows),
            "right_state_delta_norm": stats(
                row.get("right_state_delta_norm") for row in action_rows
            ),
            "left_target_minus_state_abs_max": stats(
                row.get("left_target_minus_state_abs_max") for row in action_rows
            ),
            "right_target_minus_state_abs_max": stats(
                row.get("right_target_minus_state_abs_max") for row in action_rows
            ),
            "left_tracking_error_norm": stats(
                row.get("left_tracking_error_norm") for row in action_rows
            ),
            "right_tracking_error_norm": stats(
                row.get("right_tracking_error_norm") for row in action_rows
            ),
            "execute_wall_dt": stats(row.get("execute_wall_dt") for row in action_rows),
        },
        "chunks": {
            "count": len(chunk_rows),
            "inflight_latency_s": stats(row.get("inflight_latency_s") for row in chunk_rows),
            "overlap_count": stats(row.get("overlap_count") for row in chunk_rows),
            "new_action_count": stats(row.get("new_action_count") for row in chunk_rows),
            "overlap_raw_delta_right_norm_mean": stats(
                row.get("overlap_raw_delta_right_norm_mean") for row in chunk_rows
            ),
            "overlap_blended_delta_right_norm_mean": stats(
                row.get("overlap_blended_delta_right_norm_mean") for row in chunk_rows
            ),
        },
        "observations": {
            "count": len(observation_rows),
            "must_go_count": sum(1 for row in observation_rows if row.get("must_go")),
            "must_go_interval_s": stats(row.get("must_go_interval_s") for row in observation_rows),
            "queue_size": stats(row.get("queue_size") for row in observation_rows),
            "dropped_zmq_packets": stats(
                row.get("dropped_zmq_packets") for row in observation_rows
            ),
            "camera_age_max_s": stats(row.get("camera_age_max_s") for row in observation_rows),
        },
        "spikes": {
            "count": spike_count,
        },
    }
    return summary


def write_summary_text(path: Path, summary: dict[str, Any]) -> None:
    def stat_line(label: str, item: dict[str, Any]) -> str:
        return (
            f"{label}: count={item.get('count')} "
            f"median={fmt(item.get('median'))} max={fmt(item.get('max'))} "
            f"p95={fmt(item.get('p95'))}"
        )

    lines = [
        "Deployment Log Analysis",
        "=======================",
        f"run_name: {summary['run_name']}",
        f"log_dir: {summary['log_dir']}",
        f"duration_s: {fmt(summary['duration_s'])}",
        f"policy_type: {get_nested(summary, 'metadata.policy_type')}",
        f"actions_per_chunk: {get_nested(summary, 'metadata.actions_per_chunk')}",
        f"chunk_size_threshold: {get_nested(summary, 'metadata.chunk_size_threshold')}",
        f"aggregate_ratio_old: {get_nested(summary, 'metadata.aggregate_ratio_old')}",
        f"event_counts: {summary['event_counts']}",
        "",
        "Action Metrics",
        "--------------",
        stat_line(
            "left_target_delta_norm",
            get_nested(summary, "actions.left_target_delta_norm", {}),
        ),
        stat_line(
            "right_target_delta_norm",
            get_nested(summary, "actions.right_target_delta_norm", {}),
        ),
        stat_line(
            "left_target_minus_state_abs_max",
            get_nested(summary, "actions.left_target_minus_state_abs_max", {}),
        ),
        stat_line(
            "right_target_minus_state_abs_max",
            get_nested(summary, "actions.right_target_minus_state_abs_max", {}),
        ),
        stat_line(
            "left_tracking_error_norm",
            get_nested(summary, "actions.left_tracking_error_norm", {}),
        ),
        stat_line(
            "right_tracking_error_norm",
            get_nested(summary, "actions.right_tracking_error_norm", {}),
        ),
        "",
        "Chunk And Timing Metrics",
        "------------------------",
        stat_line(
            "inflight_latency_s",
            get_nested(summary, "chunks.inflight_latency_s", {}),
        ),
        stat_line("overlap_count", get_nested(summary, "chunks.overlap_count", {})),
        stat_line("new_action_count", get_nested(summary, "chunks.new_action_count", {})),
        stat_line(
            "must_go_interval_s",
            get_nested(summary, "observations.must_go_interval_s", {}),
        ),
        stat_line("queue_size", get_nested(summary, "observations.queue_size", {})),
        "",
        "Spikes",
        "------",
        f"spike_count: {get_nested(summary, 'spikes.count')}",
        f"thresholds: {summary['thresholds']}",
        "",
    ]
    path.write_text("\n".join(lines))


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.6g}"


def list_values(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = as_float(row.get("elapsed_s"))
        y = as_float(row.get(key))
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def add_spike_lines(ax: Any, spike_rows: list[dict[str, Any]], arm: str | None = None) -> None:
    added = False
    for spike in spike_rows:
        if arm is not None and spike.get("arm") != arm:
            continue
        elapsed = as_float(spike.get("elapsed_s"))
        if elapsed is None:
            continue
        ax.axvline(elapsed, color="crimson", alpha=0.12, linewidth=0.8)
        added = True
    if added:
        ax.plot([], [], color="crimson", alpha=0.4, label="spike")


def plot_overview(
    path: Path,
    action_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    spike_rows: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

    for key, label in [
        ("left_target_delta_norm", "left target delta"),
        ("right_target_delta_norm", "right target delta"),
    ]:
        x, y = list_values(action_rows, key)
        axes[0].plot(x, y, label=label)
    add_spike_lines(axes[0], spike_rows)
    axes[0].set_ylabel("rad/frame")
    axes[0].set_title("Predicted action target jump")
    axes[0].legend(loc="upper right")

    for key, label in [
        ("left_target_minus_state_norm", "left target-state"),
        ("right_target_minus_state_norm", "right target-state"),
    ]:
        x, y = list_values(action_rows, key)
        axes[1].plot(x, y, label=label)
    add_spike_lines(axes[1], spike_rows)
    axes[1].set_ylabel("rad")
    axes[1].set_title("Predicted target minus current robot state")
    axes[1].legend(loc="upper right")

    for key, label in [
        ("left_state_delta_norm", "left state delta"),
        ("right_state_delta_norm", "right state delta"),
    ]:
        x, y = list_values(action_rows, key)
        axes[2].plot(x, y, label=label)
    axes[2].set_ylabel("rad/frame")
    axes[2].set_title("Actual robot state motion")
    axes[2].legend(loc="upper right")

    x, y = list_values(chunk_rows, "inflight_latency_s")
    axes[3].plot(x, y, marker=".", label="inflight latency")
    axes[3].set_ylabel("s")
    axes[3].set_title("Policy server latency")
    axes[3].legend(loc="upper right")

    x, y = list_values(observation_rows, "queue_size")
    axes[4].plot(x, y, label="queue size")
    must_go_times = [
        row["elapsed_s"]
        for row in observation_rows
        if row.get("must_go") and as_float(row.get("elapsed_s")) is not None
    ]
    for elapsed in must_go_times:
        axes[4].axvline(elapsed, color="tab:orange", alpha=0.15, linewidth=0.8)
    axes[4].plot([], [], color="tab:orange", alpha=0.5, label="must_go")
    axes[4].set_ylabel("actions")
    axes[4].set_title("Action queue and must_go observations")
    axes[4].set_xlabel("elapsed_s")
    axes[4].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_arm_state_vs_action(
    path: Path,
    action_rows: list[dict[str, Any]],
    spike_rows: list[dict[str, Any]],
    arm: str,
    names: list[str],
) -> None:
    fig, axes = plt.subplots(7, 1, figsize=(16, 18), sharex=True)
    for axis, joint_name in zip(axes, names, strict=True):
        x, state = list_values(action_rows, f"{arm}_state_{joint_name}")
        action_x, action = list_values(action_rows, f"{arm}_action_{joint_name}")
        axis.plot(x, state, label="robot_state", linewidth=1.4)
        if action.size:
            axis.plot(action_x, action, label="predicted_action", linewidth=1.1)
        add_spike_lines(axis, spike_rows, arm=arm)
        axis.set_ylabel(joint_name)
        axis.grid(True, alpha=0.25)
    axes[0].set_title(f"{arm} arm robot state vs predicted action")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("elapsed_s")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def metric_matrix(rows: list[dict[str, Any]], prefix: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    values: list[list[float]] = []
    for row in rows:
        elapsed = as_float(row.get("elapsed_s"))
        if elapsed is None:
            continue
        current: list[float] = []
        ok = True
        for name in names:
            value = as_float(row.get(f"{prefix}_{name}"))
            if value is None:
                ok = False
                break
            current.append(value)
        if ok:
            xs.append(elapsed)
            values.append(current)
    if not values:
        return np.asarray([], dtype=float), np.zeros((0, len(names)), dtype=float)
    return np.asarray(xs, dtype=float), np.asarray(values, dtype=float)


def plot_arm_errors(
    path: Path,
    action_rows: list[dict[str, Any]],
    arm: str,
    names: list[str],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
    heatmap_specs = [
        (f"{arm}_target_minus_state", "target minus state per joint"),
        (f"{arm}_target_delta", "predicted action delta per joint"),
    ]
    for axis, (prefix, title) in zip(axes[:2], heatmap_specs, strict=True):
        xs, matrix = metric_matrix(action_rows, prefix, names)
        if matrix.size:
            extent = [float(xs[0]), float(xs[-1]), 0, len(names)]
            image = axis.imshow(
                matrix.T,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=extent,
                cmap="coolwarm",
            )
            axis.set_yticks(np.arange(len(names)) + 0.5)
            axis.set_yticklabels(names)
            fig.colorbar(image, ax=axis, fraction=0.015, pad=0.01)
        axis.set_title(f"{arm} {title}")
        axis.set_ylabel("joint")

    for key, label in [
        (f"{arm}_tracking_error_norm", "tracking error norm"),
        (f"{arm}_target_minus_state_norm", "target-state norm"),
    ]:
        x, y = list_values(action_rows, key)
        axes[2].plot(x, y, label=label)
    axes[2].set_title(f"{arm} tracking and target-state error")
    axes[2].set_xlabel("elapsed_s")
    axes[2].set_ylabel("rad")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gripper(path: Path, action_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    for axis, side in zip(axes, ["left", "right"], strict=True):
        for key, label in [
            (f"{side}_gripper_state", "state"),
            (f"{side}_gripper_action", "predicted_action"),
        ]:
            x, y = list_values(action_rows, key)
            if y.size:
                axis.plot(x, y, label=label)
        axis.set_title(f"{side} gripper")
        axis.set_ylabel("value")
        axis.legend(loc="upper right")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("elapsed_s")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_chunk_queue_latency(
    path: Path,
    chunk_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(16, 13), sharex=True)

    x, y = list_values(observation_rows, "queue_size")
    axes[0].plot(x, y, label="queue size")
    axes[0].set_title("Queue size")
    axes[0].set_ylabel("actions")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    for key, label in [("overlap_count", "overlap"), ("new_action_count", "new")]:
        x, y = list_values(chunk_rows, key)
        axes[1].plot(x, y, marker=".", label=label)
    axes[1].set_title("Chunk overlap and new action counts")
    axes[1].set_ylabel("actions")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)

    for key, label in [
        ("overlap_raw_delta_left_norm_mean", "left raw"),
        ("overlap_blended_delta_left_norm_mean", "left blended"),
        ("overlap_raw_delta_right_norm_mean", "right raw"),
        ("overlap_blended_delta_right_norm_mean", "right blended"),
    ]:
        x, y = list_values(chunk_rows, key)
        axes[2].plot(x, y, marker=".", label=label)
    axes[2].set_title("Overlap raw vs blended action delta")
    axes[2].set_ylabel("rad")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.25)

    x, y = list_values(chunk_rows, "inflight_latency_s")
    axes[3].plot(x, y, marker=".", label="inflight latency")
    axes[3].set_title("Inference latency")
    axes[3].set_xlabel("elapsed_s")
    axes[3].set_ylabel("s")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spike_windows(
    path: Path,
    action_rows: list[dict[str, Any]],
    spike_rows: list[dict[str, Any]],
    top_k: int,
) -> None:
    rows_to_plot = spike_rows[: max(top_k, 0)]
    if not rows_to_plot:
        fig, axis = plt.subplots(1, 1, figsize=(10, 4))
        axis.text(0.5, 0.5, "No spikes detected", ha="center", va="center")
        axis.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    fig, axes = plt.subplots(len(rows_to_plot), 1, figsize=(16, 3.2 * len(rows_to_plot)), sharex=False)
    if len(rows_to_plot) == 1:
        axes = [axes]

    for axis, spike in zip(axes, rows_to_plot, strict=True):
        action_index = int(spike.get("action_index") or 0)
        start = max(0, action_index - 5)
        end = min(len(action_rows), action_index + 6)
        window = action_rows[start:end]
        arm = str(spike.get("arm"))
        x, target_delta = list_values(window, f"{arm}_target_delta_norm")
        target_state_x, target_state = list_values(window, f"{arm}_target_minus_state_norm")
        if target_delta.size:
            axis.plot(x, target_delta, marker=".", label=f"{arm} target delta norm")
        if target_state.size:
            axis.plot(
                target_state_x,
                target_state,
                marker=".",
                label=f"{arm} target-state norm",
            )
        elapsed = as_float(spike.get("elapsed_s"))
        if elapsed is not None:
            axis.axvline(elapsed, color="crimson", linewidth=1.2, alpha=0.7)
        axis.set_title(
            f"spike action_index={spike.get('action_index')} "
            f"timestep={spike.get('action_timestep')} {arm} "
            f"{spike.get('metric')} {spike.get('joint') or ''} value={fmt(spike.get('value'))}"
        )
        axis.set_ylabel("rad")
        axis.legend(loc="upper right")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("elapsed_s")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_plots(
    output_dir: Path,
    action_rows: list[dict[str, Any]],
    chunk_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    spike_rows: list[dict[str, Any]],
    top_k_spikes: int,
) -> None:
    if action_rows or chunk_rows or observation_rows:
        plot_overview(
            output_dir / "overview.png",
            action_rows,
            chunk_rows,
            observation_rows,
            spike_rows,
        )
    if action_rows:
        plot_arm_state_vs_action(
            output_dir / "left_arm_state_vs_action.png",
            action_rows,
            spike_rows,
            "left",
            LEFT_JOINT_NAMES,
        )
        plot_arm_state_vs_action(
            output_dir / "right_arm_state_vs_action.png",
            action_rows,
            spike_rows,
            "right",
            RIGHT_JOINT_NAMES,
        )
        plot_arm_errors(output_dir / "left_arm_errors.png", action_rows, "left", LEFT_JOINT_NAMES)
        plot_arm_errors(output_dir / "right_arm_errors.png", action_rows, "right", RIGHT_JOINT_NAMES)
        plot_gripper(output_dir / "gripper.png", action_rows)
        plot_spike_windows(output_dir / "spike_windows.png", action_rows, spike_rows, top_k_spikes)
    if chunk_rows or observation_rows:
        plot_chunk_queue_latency(output_dir / "chunk_queue_latency.png", chunk_rows, observation_rows)


def analyze_one_log(
    log_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    log_dir = log_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_json(log_dir / "metadata.json")
    records = load_jsonl(log_dir / "samples.jsonl")
    chunk_rows = build_chunk_rows(records)
    observation_rows = build_observation_rows(records)
    action_rows = build_action_rows(records, chunk_rows, args.tracking_lag_steps)
    spike_rows, spike_count = build_spike_rows(
        action_rows,
        args.joint_spike_threshold,
        args.arm_norm_spike_threshold,
        args.target_state_threshold,
        args.top_k_spikes,
    )
    thresholds = {
        "joint_spike_threshold": args.joint_spike_threshold,
        "arm_norm_spike_threshold": args.arm_norm_spike_threshold,
        "target_state_threshold": args.target_state_threshold,
        "tracking_lag_steps": args.tracking_lag_steps,
        "top_k_spikes": args.top_k_spikes,
    }
    summary = build_summary(
        log_dir,
        metadata,
        records,
        action_rows,
        chunk_rows,
        observation_rows,
        spike_count,
        thresholds,
    )

    (output_dir / "summary.json").write_text(json.dumps(json_safe(summary), indent=2) + "\n")
    write_summary_text(output_dir / "summary.txt", summary)
    csv_write(output_dir / "action_timeseries.csv", action_rows)
    csv_write(output_dir / "chunk_timeseries.csv", chunk_rows)
    csv_write(output_dir / "observation_timeseries.csv", observation_rows)
    csv_write(output_dir / "spikes.csv", spike_rows)
    if not args.no_plots:
        write_plots(
            output_dir,
            action_rows,
            chunk_rows,
            observation_rows,
            spike_rows,
            args.top_k_spikes,
        )
    return summary


def compare_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": summary.get("run_name"),
        "actions_per_chunk": get_nested(summary, "metadata.actions_per_chunk"),
        "chunk_size_threshold": get_nested(summary, "metadata.chunk_size_threshold"),
        "aggregate_ratio_old": get_nested(summary, "metadata.aggregate_ratio_old"),
        "duration_s": summary.get("duration_s"),
        "median_left_target_delta_norm": get_nested(
            summary, "actions.left_target_delta_norm.median"
        ),
        "max_left_target_delta_norm": get_nested(summary, "actions.left_target_delta_norm.max"),
        "median_right_target_delta_norm": get_nested(
            summary, "actions.right_target_delta_norm.median"
        ),
        "max_right_target_delta_norm": get_nested(summary, "actions.right_target_delta_norm.max"),
        "median_left_target_minus_state_abs": get_nested(
            summary, "actions.left_target_minus_state_abs_max.median"
        ),
        "max_left_target_minus_state_abs": get_nested(
            summary, "actions.left_target_minus_state_abs_max.max"
        ),
        "median_right_target_minus_state_abs": get_nested(
            summary, "actions.right_target_minus_state_abs_max.median"
        ),
        "max_right_target_minus_state_abs": get_nested(
            summary, "actions.right_target_minus_state_abs_max.max"
        ),
        "median_inflight_latency": get_nested(summary, "chunks.inflight_latency_s.median"),
        "max_inflight_latency": get_nested(summary, "chunks.inflight_latency_s.max"),
        "mean_overlap_count": get_nested(summary, "chunks.overlap_count.mean"),
        "mean_new_action_count": get_nested(summary, "chunks.new_action_count.mean"),
        "spike_count": get_nested(summary, "spikes.count"),
    }


def write_compare_outputs(output_dir: Path, summaries: list[dict[str, Any]], no_plots: bool) -> None:
    rows = [compare_row(summary) for summary in summaries]
    csv_write(output_dir / "compare_summary.csv", rows)
    (output_dir / "compare_summary.json").write_text(json.dumps(json_safe(rows), indent=2) + "\n")
    if no_plots or not rows:
        return

    labels = [str(row.get("run_name")) for row in rows]
    metrics = [
        ("max_right_target_delta_norm", "max right target delta norm"),
        ("median_right_target_delta_norm", "median right target delta norm"),
        ("max_right_target_minus_state_abs", "max right target-state abs"),
        ("median_inflight_latency", "median inflight latency"),
        ("spike_count", "spike count"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 4 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    for axis, (key, title) in zip(axes, metrics, strict=True):
        values = [as_float(row.get(key)) or 0.0 for row in rows]
        axis.bar(np.arange(len(labels)), values)
        axis.set_title(title)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "compare_overview.png", dpi=150)
    plt.close(fig)


def default_output_dir(args: argparse.Namespace) -> Path:
    log_dirs = [path.expanduser().resolve() for path in args.log_dir]
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    if len(log_dirs) == 1 and not args.compare:
        return log_dirs[0] / "analysis"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dirs[0].parent / f"analysis_compare_{timestamp}"


def main() -> None:
    args = parse_args()
    if args.tracking_lag_steps < 0:
        raise ValueError("--tracking-lag-steps must be non-negative")
    if args.top_k_spikes < 0:
        raise ValueError("--top-k-spikes must be non-negative")

    log_dirs = [path.expanduser().resolve() for path in args.log_dir]
    output_root = default_output_dir(args)
    output_root.mkdir(parents=True, exist_ok=True)

    multi_run = len(log_dirs) > 1 or args.compare
    summaries: list[dict[str, Any]] = []
    for log_dir in log_dirs:
        if multi_run:
            run_output_dir = output_root / log_dir.name
        else:
            run_output_dir = output_root
        summary = analyze_one_log(log_dir, run_output_dir, args)
        summaries.append(summary)
        print(f"Wrote analysis for {log_dir.name}: {run_output_dir}")

    if multi_run:
        write_compare_outputs(output_root, summaries, args.no_plots)
        print(f"Wrote comparison analysis: {output_root}")


if __name__ == "__main__":
    main()
