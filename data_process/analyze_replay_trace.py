from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

try:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - depends on the active analysis environment
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:  # pragma: no cover - optional fallback plotting backend
    Image = None
    ImageDraw = None
    ImageFont = None

ACTION_CONFIG_PATH = Path("meta/real_exp_action_config.json")
REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S = np.array([0.35, 0.35, 0.35, 0.35, 0.50, 0.50, 0.50], dtype=float)
REPLAY_MAX_JOINT_ACCELERATIONS_RAD_PER_S2 = np.array([0.80, 0.80, 0.80, 0.80, 1.20, 1.20, 1.20], dtype=float)
JOINT_LABELS = [f"J{index}" for index in range(1, 8)]
PLOT_COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (23, 190, 207),
    (0, 0, 0),
]
DEFAULT_PLOT_DPI = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate static dataset audits and optional replay trace analysis reports."
    )
    parser.add_argument("--dataset-root", required=True, type=Path, help="LeRobot dataset root.")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to analyze.")
    parser.add_argument("--trace", type=Path, default=None, help="Optional controller-matched replay trace CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output report directory.")
    parser.add_argument("--lag-max", type=int, default=10, help="Maximum lag in frames for action/state lag sweep.")
    parser.add_argument(
        "--tracking-error-threshold",
        type=float,
        default=0.05,
        help="Joint tracking error threshold in rad for replay trace summaries.",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=DEFAULT_PLOT_DPI,
        help="PNG plot DPI when matplotlib is available.",
    )
    parser.add_argument("--write-svg", action="store_true", help="Also write SVG copies of matplotlib figures.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_episode(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    info = load_json(dataset_root / "meta" / "info.json")
    action_config_path = dataset_root / ACTION_CONFIG_PATH
    action_config = load_json(action_config_path) if action_config_path.exists() else {}
    rows: list[tuple[int, float, list[float], list[float]]] = []
    available: set[int] = set()
    for parquet_file in sorted((dataset_root / "data").glob("chunk-*/*.parquet")):
        table = pq.read_table(
            parquet_file,
            columns=["episode_index", "frame_index", "timestamp", "observation.state", "action"],
        )
        data = table.to_pydict()
        for row_episode, frame_index, timestamp, state, action in zip(
            data["episode_index"],
            data["frame_index"],
            data["timestamp"],
            data["observation.state"],
            data["action"],
            strict=True,
        ):
            available.add(int(row_episode))
            if int(row_episode) != episode_index:
                continue
            rows.append((int(frame_index), float(timestamp), state, action))
    if not rows:
        raise ValueError(f"Episode {episode_index} not found. Available episodes: {sorted(available)}")
    rows.sort(key=lambda item: item[0])
    return {
        "info": info,
        "action_config": action_config,
        "frame_indices": np.asarray([row[0] for row in rows], dtype=int),
        "timestamps": np.asarray([row[1] for row in rows], dtype=float),
        "states": np.asarray([row[2] for row in rows], dtype=float),
        "actions": np.asarray([row[3] for row in rows], dtype=float),
    }


def joint_slices() -> dict[str, slice]:
    return {
        "left": slice(0, 7),
        "right": slice(8, 15),
    }


def gripper_indices() -> dict[str, int]:
    return {"left": 7, "right": 15}


def stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"mean": None, "max": None, "p95": None}
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "max": None, "p95": None}
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def per_joint_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.abs(np.asarray(values, dtype=float))
    result: dict[str, list[Any]] = {"mean": [], "max": [], "p95": [], "valid_count": []}
    for joint_index in range(values.shape[1]):
        finite_values = values[:, joint_index]
        finite_values = finite_values[np.isfinite(finite_values)]
        result["valid_count"].append(int(finite_values.size))
        if finite_values.size == 0:
            result["mean"].append(None)
            result["max"].append(None)
            result["p95"].append(None)
            continue
        result["mean"].append(round(float(np.mean(finite_values)), 8))
        result["max"].append(round(float(np.max(finite_values)), 8))
        result["p95"].append(round(float(np.percentile(finite_values, 95)), 8))
    return result


def max_abs_error_per_frame(error: np.ndarray) -> np.ndarray:
    abs_error = np.abs(np.asarray(error, dtype=float))
    per_frame = np.full(abs_error.shape[0], np.nan, dtype=float)
    for row_index, row in enumerate(abs_error):
        finite_values = row[np.isfinite(row)]
        if finite_values.size:
            per_frame[row_index] = float(np.max(finite_values))
    return per_frame


def trace_lag_sweep(target: np.ndarray, actual: np.ndarray, lag_max: int) -> dict[str, Any]:
    rows: list[dict[str, float | int | None]] = []
    max_lag = min(lag_max, max(len(target) - 1, 0))
    for lag in range(max_lag + 1):
        if lag == 0:
            diff = target - actual
        else:
            diff = target[:-lag] - actual[lag:]
        per_frame = max_abs_error_per_frame(diff)
        finite = per_frame[np.isfinite(per_frame)]
        if finite.size == 0:
            rows.append(
                {
                    "lag_frames": lag,
                    "lag_error_mean": None,
                    "lag_error_p95": None,
                    "lag_error_max": None,
                    "valid_frames": 0,
                }
            )
            continue
        rows.append(
            {
                "lag_frames": lag,
                "lag_error_mean": float(np.mean(finite)),
                "lag_error_p95": float(np.percentile(finite, 95)),
                "lag_error_max": float(np.max(finite)),
                "valid_frames": int(finite.size),
            }
        )

    valid_rows = [row for row in rows if row["lag_error_mean"] is not None]
    best = min(valid_rows, key=lambda row: float(row["lag_error_mean"])) if valid_rows else None
    return {"rows": rows, "best": best}


def scalar_counts(values: np.ndarray) -> dict[str, int]:
    finite_values = values[np.isfinite(values)]
    return {str(float(value)): int(np.sum(finite_values == value)) for value in sorted(set(finite_values.tolist()))}


def scalar_range(values: np.ndarray) -> dict[str, Any]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {"min": None, "max": None, "valid_count": 0}
    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "valid_count": int(finite_values.size),
    }


def lag_sweep(actions: np.ndarray, states: np.ndarray, lag_max: int) -> dict[str, Any]:
    lag_rows: list[dict[str, float | int]] = []
    for lag in range(lag_max + 1):
        if lag == 0:
            diff = actions - states
        else:
            diff = actions[:-lag] - states[lag:]
        per_frame = np.max(np.abs(diff), axis=1)
        lag_rows.append(
            {
                "lag_frames": lag,
                "lag_error_mean": float(np.mean(per_frame)),
                "lag_error_p95": float(np.percentile(per_frame, 95)),
                "lag_error_max": float(np.max(per_frame)),
            }
        )
    best = min(lag_rows, key=lambda row: float(row["lag_error_mean"]))
    return {"rows": lag_rows, "best": best}


def velocity_acceleration_metrics(values: np.ndarray, fps: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    velocity = np.diff(values, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps if len(velocity) > 1 else np.empty((0, values.shape[1]))
    velocity_over = np.abs(velocity) > REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S
    acceleration_over = np.abs(acceleration) > REPLAY_MAX_JOINT_ACCELERATIONS_RAD_PER_S2
    return {
        "velocity": velocity,
        "acceleration": acceleration,
        "velocity_max_per_joint": np.max(np.abs(velocity), axis=0).round(8).tolist() if len(velocity) else [0.0] * 7,
        "acceleration_max_per_joint": (
            np.max(np.abs(acceleration), axis=0).round(8).tolist() if len(acceleration) else [0.0] * 7
        ),
        "velocity_over_limit_frames": int(np.sum(np.any(velocity_over, axis=1))) if len(velocity_over) else 0,
        "acceleration_over_limit_frames": (
            int(np.sum(np.any(acceleration_over, axis=1))) if len(acceleration_over) else 0
        ),
        "velocity_over_limit_per_joint": np.sum(velocity_over, axis=0).astype(int).tolist() if len(velocity_over) else [0] * 7,
        "acceleration_over_limit_per_joint": (
            np.sum(acceleration_over, axis=0).astype(int).tolist() if len(acceleration_over) else [0] * 7
        ),
    }


def timestamp_metrics(frame_indices: np.ndarray, timestamps: np.ndarray, fps: float) -> dict[str, Any]:
    expected_indices = np.arange(frame_indices[0], frame_indices[0] + len(frame_indices))
    unique_count = len(set(frame_indices.tolist()))
    missing_count = int(max(0, expected_indices.size - unique_count))
    timestamp_deltas = np.diff(timestamps)
    expected_dt = 1.0 / fps
    relative_expected_timestamps = timestamps[0] + (frame_indices - frame_indices[0]) / fps
    return {
        "first": float(timestamps[0]),
        "last": float(timestamps[-1]),
        "actual_duration_s": float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
        "expected_duration_s": float((len(timestamps) - 1) / fps),
        "max_abs_error_vs_frame_index": float(np.max(np.abs(timestamps - relative_expected_timestamps))),
        "frame_indices_continuous": bool(np.all(frame_indices == expected_indices)),
        "missing_frame_count": missing_count,
        "duplicate_frame_count": int(len(frame_indices) - unique_count),
        "timestamps_strictly_increasing": bool(np.all(timestamp_deltas > 0.0)) if len(timestamp_deltas) else True,
        "timestamp_dt_mean_s": float(np.mean(timestamp_deltas)) if len(timestamp_deltas) else None,
        "timestamp_dt_max_abs_error_s": (
            float(np.max(np.abs(timestamp_deltas - expected_dt))) if len(timestamp_deltas) else None
        ),
    }


def compute_static_metrics(episode: dict[str, Any], lag_max: int) -> dict[str, Any]:
    states = episode["states"]
    actions = episode["actions"]
    frame_indices = episode["frame_indices"]
    timestamps = episode["timestamps"]
    fps = float(episode["info"]["fps"])
    metrics: dict[str, Any] = {
        "dataset": {
            "fps": fps,
            "frames": int(len(frame_indices)),
            "duration_s": float(len(frame_indices) / fps),
            "state_dim": int(states.shape[1]),
            "action_dim": int(actions.shape[1]),
            "action_config": episode["action_config"],
        },
        "timestamp": timestamp_metrics(frame_indices, timestamps, fps),
        "arms": {},
        "grippers": {},
    }

    for arm, sl in joint_slices().items():
        arm_state = states[:, sl]
        arm_action = actions[:, sl]
        action_current = arm_action - arm_state
        action_next = arm_action[:-1] - arm_state[1:]
        lag_result = lag_sweep(arm_action, arm_state, lag_max)
        lag_result["best_lag_s"] = float(lag_result["best"]["lag_frames"] / fps)
        arm_metrics = {
            "action_minus_current_state": per_joint_stats(action_current),
            "action_minus_next_state": per_joint_stats(action_next),
            "lag_sweep": lag_result,
            "state_dynamics": strip_arrays(velocity_acceleration_metrics(arm_state, fps)),
            "action_dynamics": strip_arrays(velocity_acceleration_metrics(arm_action, fps)),
        }
        metrics["arms"][arm] = arm_metrics

    for arm, index in gripper_indices().items():
        state_values = states[:, index]
        action_values = actions[:, index]
        transitions = int(np.sum(np.abs(np.diff(action_values)) > 1e-6))
        counts = {str(float(value)): int(np.sum(action_values == value)) for value in sorted(set(action_values.tolist()))}
        metrics["grippers"][arm] = {
            "state_min": float(np.min(state_values)),
            "state_max": float(np.max(state_values)),
            "action_counts": counts,
            "action_transitions": transitions,
        }

    return metrics


def strip_arrays(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"velocity", "acceleration"}}


def make_output_dirs(output: Path) -> Path:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    return figures


def fallback_plot_available() -> bool:
    return Image is not None and ImageDraw is not None and ImageFont is not None


def downsample_series(x_values: np.ndarray, y_values: np.ndarray, max_points: int = 900) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite_mask]
    y_values = y_values[finite_mask]
    if len(x_values) <= max_points:
        return x_values, y_values
    indices = np.linspace(0, len(x_values) - 1, max_points).astype(int)
    return x_values[indices], y_values[indices]


def pil_font(size: int = 14) -> Any:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def value_range(series: list[np.ndarray]) -> tuple[float, float]:
    finite_values = [np.asarray(values, dtype=float)[np.isfinite(values)] for values in series]
    finite_values = [values for values in finite_values if values.size]
    if not finite_values:
        return -1.0, 1.0
    merged = np.concatenate(finite_values)
    min_value = float(np.min(merged))
    max_value = float(np.max(merged))
    if abs(max_value - min_value) < 1e-9:
        pad = max(abs(max_value) * 0.1, 1.0)
        return min_value - pad, max_value + pad
    pad = (max_value - min_value) * 0.08
    return min_value - pad, max_value + pad


def draw_line_plot(
    path: Path,
    title: str,
    x_values: np.ndarray,
    series: list[tuple[str, np.ndarray, tuple[int, int, int], bool]],
    *,
    y_label: str,
    width: int = 1200,
    height: int = 520,
) -> None:
    if not fallback_plot_available():
        return

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = pil_font(18)
    label_font = pil_font(13)
    small_font = pil_font(11)
    left, top, right, bottom = 80, 55, width - 30, height - 70
    draw.text((left, 18), title, fill=(20, 20, 20), font=title_font)
    draw.rectangle((left, top, right, bottom), outline=(190, 190, 190))
    for fraction in (0.25, 0.5, 0.75):
        y = int(top + (bottom - top) * fraction)
        draw.line((left, y, right, y), fill=(235, 235, 235))

    finite_x = np.asarray(x_values, dtype=float)
    finite_x = finite_x[np.isfinite(finite_x)]
    if finite_x.size == 0:
        image.save(path)
        return
    x_min = float(np.min(finite_x))
    x_max = float(np.max(finite_x))
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0

    y_min, y_max = value_range([values for _, values, _, _ in series])
    draw.text((15, (top + bottom) // 2), y_label, fill=(60, 60, 60), font=small_font)
    draw.text((left, bottom + 12), f"x: {x_min:g}..{x_max:g}", fill=(70, 70, 70), font=small_font)
    draw.text((left, top - 18), f"y: {y_min:.3g}..{y_max:.3g}", fill=(70, 70, 70), font=small_font)

    def point_to_pixel(x: float, y: float) -> tuple[int, int]:
        px = left + int((x - x_min) / (x_max - x_min) * (right - left))
        py = bottom - int((y - y_min) / (y_max - y_min) * (bottom - top))
        return px, py

    for index, (label, values, color, dashed) in enumerate(series):
        xs, ys = downsample_series(x_values, np.asarray(values, dtype=float))
        if len(xs) < 2:
            continue
        points = [point_to_pixel(float(x), float(y)) for x, y in zip(xs, ys, strict=True)]
        if dashed:
            for segment_index in range(0, len(points) - 1, 2):
                draw.line((points[segment_index], points[segment_index + 1]), fill=color, width=2)
        else:
            draw.line(points, fill=color, width=2)
        legend_x = left + 8 + (index % 4) * 210
        legend_y = bottom + 30 + (index // 4) * 16
        draw.line((legend_x, legend_y + 7, legend_x + 20, legend_y + 7), fill=color, width=2)
        draw.text((legend_x + 26, legend_y), label, fill=(40, 40, 40), font=small_font)

    image.save(path)


def save_static_figures_fallback(episode: dict[str, Any], metrics: dict[str, Any], output: Path) -> dict[str, str]:
    if not fallback_plot_available():
        return {}
    figures_dir = make_output_dirs(output)
    states = episode["states"]
    actions = episode["actions"]
    frame_indices = episode["frame_indices"]
    fps = float(episode["info"]["fps"])
    paths: dict[str, str] = {}

    state_action_panels = Image.new("RGB", (1200, 1760), "white")
    panel_paths: list[Path] = []
    for arm, sl in joint_slices().items():
        panel_path = figures_dir / f"_tmp_{arm}_state_action.png"
        series: list[tuple[str, np.ndarray, tuple[int, int, int], bool]] = []
        for joint in range(7):
            series.append((f"state {JOINT_LABELS[joint]}", states[:, sl][:, joint], PLOT_COLORS[joint], False))
            series.append((f"action {JOINT_LABELS[joint]}", actions[:, sl][:, joint], PLOT_COLORS[joint], True))
        draw_line_plot(panel_path, f"{arm.title()} Arm: observation.state vs action", frame_indices, series, y_label="rad", height=520)
        panel_paths.append(panel_path)

    for arm, index in gripper_indices().items():
        panel_path = figures_dir / f"_tmp_{arm}_gripper_static.png"
        draw_line_plot(
            panel_path,
            f"{arm.title()} Gripper: state width vs binary action",
            frame_indices,
            [
                ("state width", states[:, index], PLOT_COLORS[0], False),
                ("binary action", actions[:, index], PLOT_COLORS[3], False),
            ],
            y_label="value",
            height=360,
        )
        panel_paths.append(panel_path)

    y_offset = 0
    for panel_path in panel_paths:
        panel = Image.open(panel_path)
        state_action_panels.paste(panel, (0, y_offset))
        y_offset += panel.height
        panel_path.unlink(missing_ok=True)
    path = figures_dir / "joint_state_vs_action.png"
    state_action_panels.crop((0, 0, 1200, y_offset)).save(path)
    paths["joint_state_vs_action"] = str(path.relative_to(output))

    lag_series = []
    lag_frames = np.asarray(
        [row["lag_frames"] for row in metrics["arms"]["left"]["lag_sweep"]["rows"]],
        dtype=float,
    )
    for color_index, arm in enumerate(("left", "right")):
        rows = metrics["arms"][arm]["lag_sweep"]["rows"]
        lag_mean = np.asarray([row["lag_error_mean"] for row in rows], dtype=float)
        lag_p95 = np.asarray([row["lag_error_p95"] for row in rows], dtype=float)
        lag_series.append((f"{arm} mean", lag_mean, PLOT_COLORS[color_index], False))
        lag_series.append((f"{arm} p95", lag_p95, PLOT_COLORS[color_index + 3], True))
    path = figures_dir / "lag_sweep.png"
    draw_line_plot(path, "Action-State Lag Sweep", lag_frames, lag_series, y_label="rad", height=460)
    paths["lag_sweep"] = str(path.relative_to(output))

    dynamics_panels = Image.new("RGB", (1200, 1040), "white")
    panel_paths = []
    for arm, sl in joint_slices().items():
        arm_action = actions[:, sl]
        dyn = velocity_acceleration_metrics(arm_action, fps)
        velocity = dyn["velocity"]
        acceleration = dyn["acceleration"]
        velocity_series = [(JOINT_LABELS[joint], velocity[:, joint], PLOT_COLORS[joint], False) for joint in range(7)]
        accel_series = [(JOINT_LABELS[joint], acceleration[:, joint], PLOT_COLORS[joint], False) for joint in range(7)]
        if len(velocity):
            velocity_series.extend(
                [
                    ("+max limit", np.full(len(velocity), float(np.max(REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S))), (180, 0, 0), True),
                    ("-max limit", np.full(len(velocity), float(-np.max(REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S))), (180, 0, 0), True),
                ]
            )
        if len(acceleration):
            accel_series.extend(
                [
                    ("+max limit", np.full(len(acceleration), float(np.max(REPLAY_MAX_JOINT_ACCELERATIONS_RAD_PER_S2))), (180, 0, 0), True),
                    ("-max limit", np.full(len(acceleration), float(-np.max(REPLAY_MAX_JOINT_ACCELERATIONS_RAD_PER_S2))), (180, 0, 0), True),
                ]
            )
        panel_path = figures_dir / f"_tmp_{arm}_velocity.png"
        draw_line_plot(panel_path, f"{arm.title()} Action Velocity", frame_indices[1:], velocity_series, y_label="rad/s", height=260)
        panel_paths.append(panel_path)
        panel_path = figures_dir / f"_tmp_{arm}_accel.png"
        draw_line_plot(panel_path, f"{arm.title()} Action Acceleration", frame_indices[2:], accel_series, y_label="rad/s^2", height=260)
        panel_paths.append(panel_path)
    y_offset = 0
    for panel_path in panel_paths:
        panel = Image.open(panel_path)
        dynamics_panels.paste(panel, (0, y_offset))
        y_offset += panel.height
        panel_path.unlink(missing_ok=True)
    path = figures_dir / "velocity_acceleration_limits.png"
    dynamics_panels.crop((0, 0, 1200, y_offset)).save(path)
    paths["velocity_acceleration_limits"] = str(path.relative_to(output))
    return paths


def save_figure(fig: Any, path: Path, output: Path, dpi: int, write_svg: bool) -> str:
    fig.savefig(path, dpi=dpi)
    if write_svg:
        fig.savefig(path.with_suffix(".svg"))
    return str(path.relative_to(output))


def save_static_figures(
    episode: dict[str, Any],
    metrics: dict[str, Any],
    output: Path,
    lag_max: int,
    dpi: int,
    write_svg: bool,
) -> dict[str, str]:
    if plt is None:
        return save_static_figures_fallback(episode, metrics, output)
    figures_dir = make_output_dirs(output)
    states = episode["states"]
    actions = episode["actions"]
    frame_indices = episode["frame_indices"]
    fps = float(episode["info"]["fps"])
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for axis, (arm, sl) in zip(axes[0:2], joint_slices().items(), strict=True):
        for joint in range(7):
            axis.plot(frame_indices, states[:, sl][:, joint], linewidth=0.8, alpha=0.7, label=f"state {JOINT_LABELS[joint]}")
            axis.plot(frame_indices, actions[:, sl][:, joint], linewidth=0.8, linestyle="--", alpha=0.7, label=f"action {JOINT_LABELS[joint]}")
        axis.set_title(f"{arm.title()} Arm: observation.state vs action")
        axis.set_ylabel("rad")
    for axis, (arm, index) in zip(axes[2:4], gripper_indices().items(), strict=True):
        axis.plot(frame_indices, states[:, index], label="state width", linewidth=1.0)
        axis.step(frame_indices, actions[:, index], label="binary action", linewidth=1.0, where="post")
        axis.set_title(f"{arm.title()} Gripper")
        axis.set_ylabel("value")
        axis.legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("frame_index")
    for axis in axes[:2]:
        axis.legend(loc="upper right", ncol=4, fontsize=6)
    fig.tight_layout()
    path = figures_dir / "joint_state_vs_action.png"
    paths["joint_state_vs_action"] = save_figure(fig, path, output, dpi, write_svg)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for axis, arm in zip(axes, ("left", "right"), strict=True):
        rows = metrics["arms"][arm]["lag_sweep"]["rows"]
        lag_frames = [row["lag_frames"] for row in rows]
        mean_errors = [row["lag_error_mean"] for row in rows]
        p95_errors = [row["lag_error_p95"] for row in rows]
        axis.plot(lag_frames, mean_errors, marker="o", label="mean max joint error")
        axis.plot(lag_frames, p95_errors, marker="o", label="p95 max joint error")
        axis.set_title(f"{arm.title()} Action-State Lag Sweep")
        axis.set_xlabel("lag frames")
        axis.set_ylabel("rad")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    path = figures_dir / "lag_sweep.png"
    paths["lag_sweep"] = save_figure(fig, path, output, dpi, write_svg)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    for col, (arm, sl) in enumerate(joint_slices().items()):
        arm_action = actions[:, sl]
        dyn = velocity_acceleration_metrics(arm_action, fps)
        velocity = dyn["velocity"]
        acceleration = dyn["acceleration"]
        for joint in range(7):
            axes[0, col].plot(frame_indices[1:], velocity[:, joint], linewidth=0.8, label=JOINT_LABELS[joint])
            if len(acceleration):
                axes[1, col].plot(frame_indices[2:], acceleration[:, joint], linewidth=0.8, label=JOINT_LABELS[joint])
        axes[0, col].set_title(f"{arm.title()} Action Velocity")
        axes[1, col].set_title(f"{arm.title()} Action Acceleration")
        axes[0, col].set_ylabel("rad/s")
        axes[1, col].set_ylabel("rad/s^2")
        axes[1, col].set_xlabel("frame_index")
        for limit in REPLAY_MAX_JOINT_VELOCITIES_RAD_PER_S:
            axes[0, col].axhline(limit, color="red", alpha=0.08)
            axes[0, col].axhline(-limit, color="red", alpha=0.08)
        for limit in REPLAY_MAX_JOINT_ACCELERATIONS_RAD_PER_S2:
            axes[1, col].axhline(limit, color="red", alpha=0.08)
            axes[1, col].axhline(-limit, color="red", alpha=0.08)
        axes[0, col].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = figures_dir / "velocity_acceleration_limits.png"
    paths["velocity_acceleration_limits"] = save_figure(fig, path, output, dpi, write_svg)
    plt.close(fig)
    return paths


def load_trace(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def trace_column_array(rows: list[dict[str, str]], prefix: str) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        row_values: list[float] = []
        for joint in range(1, 8):
            raw = row.get(f"{prefix}_{joint}", "")
            row_values.append(float(raw) if raw not in {"", None} else np.nan)
        values.append(row_values)
    return np.asarray(values, dtype=float)


def scalar_trace_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        raw = row.get(key, "")
        values.append(float(raw) if raw not in {"", None} else np.nan)
    return np.asarray(values, dtype=float)


def compute_trace_metrics(rows: list[dict[str, str]], threshold: float, lag_max: int) -> dict[str, Any]:
    if not rows:
        return {"available": False, "warnings": ["Trace file is empty."]}
    metrics: dict[str, Any] = {
        "available": True,
        "frames": len(rows),
        "aborted": any(row.get("abort_requested") == "True" for row in rows),
        "duration_s": stats(scalar_trace_column(rows, "time_s"))["max"],
        "controller_ready_frames": sum(row.get("controller_ready") == "True" for row in rows),
        "warnings": [],
        "arms": {},
        "grippers": {},
    }
    for arm in ("left", "right"):
        error = trace_column_array(rows, f"{arm}_error_q")
        target = trace_column_array(rows, f"{arm}_target_q")
        actual = trace_column_array(rows, f"{arm}_actual_q")
        per_frame = max_abs_error_per_frame(error)
        finite_rows = np.isfinite(per_frame)
        if int(np.sum(finite_rows)) == 0:
            metrics["warnings"].append(f"No valid {arm} actual joint samples were present in the trace.")
        metrics["arms"][arm] = {
            "valid_actual_frames": int(np.sum(finite_rows)),
            "missing_actual_frames": int(len(rows) - np.sum(finite_rows)),
            "tracking_error": finite_stats(per_frame),
            "per_joint": per_joint_stats(error),
            "frames_over_threshold": int(np.sum(per_frame > threshold)),
            "lag_sweep": trace_lag_sweep(target, actual, lag_max),
        }

    for arm in ("left", "right"):
        target = scalar_trace_column(rows, f"{arm}_gripper_target")
        actual = scalar_trace_column(rows, f"{arm}_gripper_actual")
        target_finite = target[np.isfinite(target)]
        metrics["grippers"][arm] = {
            "target_counts": scalar_counts(target),
            "target_transitions": int(np.sum(np.abs(np.diff(target_finite)) > 1e-6)) if len(target_finite) > 1 else 0,
            "actual_range": scalar_range(actual),
        }
    return metrics


def save_trace_figures(rows: list[dict[str, str]], output: Path, dpi: int, write_svg: bool) -> dict[str, str]:
    if not rows:
        return {}
    if plt is None:
        return save_trace_figures_fallback(rows, output)
    figures_dir = make_output_dirs(output)
    frame_indices = np.asarray([int(row["frame_index"]) for row in rows], dtype=int)
    paths: dict[str, str] = {}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for axis, arm in zip(axes, ("left", "right"), strict=True):
        error = trace_column_array(rows, f"{arm}_error_q")
        for joint in range(7):
            axis.plot(frame_indices, error[:, joint], linewidth=0.8, label=JOINT_LABELS[joint])
        axis.set_title(f"{arm.title()} Replay Tracking Error")
        axis.set_ylabel("rad")
        axis.grid(True, alpha=0.3)
        axis.legend(ncol=4, fontsize=7)
    axes[-1].set_xlabel("frame_index")
    fig.tight_layout()
    path = figures_dir / "tracking_error.png"
    paths["tracking_error"] = save_figure(fig, path, output, dpi, write_svg)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for axis, arm in zip(axes, ("left", "right"), strict=True):
        target = scalar_trace_column(rows, f"{arm}_gripper_target")
        actual = scalar_trace_column(rows, f"{arm}_gripper_actual")
        axis.step(frame_indices, target, where="post", label="target")
        axis.plot(frame_indices, actual, label="actual")
        axis.set_title(f"{arm.title()} Gripper Replay")
        axis.set_ylabel("value")
        axis.legend(fontsize=8)
    axes[-1].set_xlabel("frame_index")
    fig.tight_layout()
    path = figures_dir / "gripper.png"
    paths["gripper"] = save_figure(fig, path, output, dpi, write_svg)
    plt.close(fig)
    return paths


def save_trace_figures_fallback(rows: list[dict[str, str]], output: Path) -> dict[str, str]:
    if not fallback_plot_available() or not rows:
        return {}
    figures_dir = make_output_dirs(output)
    frame_indices = np.asarray([int(row["frame_index"]) for row in rows], dtype=int)
    paths: dict[str, str] = {}

    tracking_panels = Image.new("RGB", (1200, 1040), "white")
    panel_paths: list[Path] = []
    for arm in ("left", "right"):
        error = trace_column_array(rows, f"{arm}_error_q")
        series = [(JOINT_LABELS[joint], error[:, joint], PLOT_COLORS[joint], False) for joint in range(7)]
        panel_path = figures_dir / f"_tmp_{arm}_tracking.png"
        draw_line_plot(panel_path, f"{arm.title()} Replay Tracking Error", frame_indices, series, y_label="rad")
        panel_paths.append(panel_path)
    y_offset = 0
    for panel_path in panel_paths:
        panel = Image.open(panel_path)
        tracking_panels.paste(panel, (0, y_offset))
        y_offset += panel.height
        panel_path.unlink(missing_ok=True)
    path = figures_dir / "tracking_error.png"
    tracking_panels.crop((0, 0, 1200, y_offset)).save(path)
    paths["tracking_error"] = str(path.relative_to(output))

    gripper_panels = Image.new("RGB", (1200, 720), "white")
    panel_paths = []
    for arm in ("left", "right"):
        panel_path = figures_dir / f"_tmp_{arm}_gripper_trace.png"
        draw_line_plot(
            panel_path,
            f"{arm.title()} Gripper Replay",
            frame_indices,
            [
                ("target", scalar_trace_column(rows, f"{arm}_gripper_target"), PLOT_COLORS[0], False),
                ("actual", scalar_trace_column(rows, f"{arm}_gripper_actual"), PLOT_COLORS[3], False),
            ],
            y_label="value",
            height=360,
        )
        panel_paths.append(panel_path)
    y_offset = 0
    for panel_path in panel_paths:
        panel = Image.open(panel_path)
        gripper_panels.paste(panel, (0, y_offset))
        y_offset += panel.height
        panel_path.unlink(missing_ok=True)
    path = figures_dir / "gripper.png"
    gripper_panels.crop((0, 0, 1200, y_offset)).save(path)
    paths["gripper"] = str(path.relative_to(output))
    return paths


def write_metrics_csv(path: Path, static_metrics: dict[str, Any], trace_metrics: dict[str, Any] | None) -> None:
    rows: list[dict[str, Any]] = []
    for arm in ("left", "right"):
        arm_metrics = static_metrics["arms"][arm]
        rows.append(
            {
                "section": "static",
                "arm": arm,
                "metric": "best_lag_frames",
                "value": arm_metrics["lag_sweep"]["best"]["lag_frames"],
            }
        )
        rows.append(
            {
                "section": "static",
                "arm": arm,
                "metric": "best_lag_error_mean",
                "value": arm_metrics["lag_sweep"]["best"]["lag_error_mean"],
            }
        )
        rows.append(
            {
                "section": "static",
                "arm": arm,
                "metric": "action_velocity_over_limit_frames",
                "value": arm_metrics["action_dynamics"]["velocity_over_limit_frames"],
            }
        )
        rows.append(
            {
                "section": "static",
                "arm": arm,
                "metric": "action_acceleration_over_limit_frames",
                "value": arm_metrics["action_dynamics"]["acceleration_over_limit_frames"],
            }
        )
        if trace_metrics and trace_metrics.get("available"):
            rows.append(
                {
                    "section": "trace",
                    "arm": arm,
                    "metric": "tracking_error_max",
                    "value": trace_metrics["arms"][arm]["tracking_error"]["max"],
                }
            )
            rows.append(
                {
                    "section": "trace",
                    "arm": arm,
                    "metric": "frames_over_tracking_threshold",
                    "value": trace_metrics["arms"][arm]["frames_over_threshold"],
                }
            )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "arm", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def html_summary_list(title: str, items: dict[str, Any]) -> str:
    rows = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td><pre>{html.escape(json.dumps(value, indent=2, sort_keys=True))}</pre></td></tr>"
        for key, value in items.items()
    )
    return f"<h2>{html.escape(title)}</h2><table>{rows}</table>"


def write_html_report(
    path: Path,
    static_metrics: dict[str, Any],
    trace_metrics: dict[str, Any] | None,
    figures: dict[str, str],
    trace_path: Path | None,
) -> None:
    title = "Replay Trace Analysis" if trace_path else "Static Dataset Audit"
    if figures:
        image_tags = "\n".join(
            f'<section><h3>{html.escape(name.replace("_", " ").title())}</h3><img src="{html.escape(rel_path)}" alt="{html.escape(name)}"></section>'
            for name, rel_path in figures.items()
        )
    else:
        image_tags = "<p>No figures were generated. Install matplotlib or pillow in the active environment to enable plots.</p>"
    trace_section = ""
    if trace_metrics is not None:
        trace_section = html_summary_list("Replay Trace Metrics", trace_metrics)

    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; color: #222; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem; vertical-align: top; text-align: left; }}
    pre {{ white-space: pre-wrap; margin: 0; }}
    img {{ max-width: 100%; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {html_summary_list("Dataset Summary", static_metrics["dataset"])}
  {html_summary_list("Timestamp Summary", static_metrics["timestamp"])}
  {html_summary_list("Arm Static Metrics", static_metrics["arms"])}
  {html_summary_list("Gripper Metrics", static_metrics["grippers"])}
  {trace_section}
  <h2>Figures</h2>
  {image_tags}
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episode = load_episode(args.dataset_root, args.episode)
    static_metrics = compute_static_metrics(episode, args.lag_max)
    figures = save_static_figures(episode, static_metrics, args.output, args.lag_max, args.plot_dpi, args.write_svg)
    warnings = []
    if plt is None:
        if fallback_plot_available():
            warnings.append("matplotlib is not installed; generated simpler fallback PNG figures with pillow.")
        else:
            warnings.append("matplotlib and pillow are not installed in the active environment; skipped figure generation.")

    trace_metrics: dict[str, Any] | None = None
    if args.trace is not None:
        trace_rows = load_trace(args.trace)
        trace_metrics = compute_trace_metrics(trace_rows, args.tracking_error_threshold, args.lag_max)
        figures.update(save_trace_figures(trace_rows, args.output, args.plot_dpi, args.write_svg))

    summary = {
        "dataset_root": str(args.dataset_root),
        "episode": args.episode,
        "trace": None if args.trace is None else str(args.trace),
        "plot_backend": "matplotlib" if plt is not None else ("pillow" if fallback_plot_available() else "none"),
        "plot_dpi": args.plot_dpi,
        "figures": figures,
        "warnings": warnings,
        "static": static_metrics,
        "trace_metrics": trace_metrics,
    }
    write_json(args.output / "summary.json", summary)
    write_metrics_csv(args.output / "metrics.csv", static_metrics, trace_metrics)
    write_html_report(args.output / "index.html", static_metrics, trace_metrics, figures, args.trace)
    print(f"Wrote report to {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
