from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np


FRANKA_HAND_MAX_WIDTH_M = 0.08


@dataclass(frozen=True)
class EpisodeStartTarget:
    episode_index: int
    frame_index: int
    dataset_state: np.ndarray
    action_target: np.ndarray


def select_episode_index(
    available_episodes: list[int],
    selector: str,
    *,
    random_seed: int | None,
) -> int:
    if not available_episodes:
        raise ValueError("No dataset episodes contain the requested frame.")
    if selector == "random":
        return random.Random(random_seed).choice(sorted(available_episodes))
    try:
        selected = int(selector)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"--episode-start must be an integer when supplied, got {selector!r}."
        ) from exc
    if selected not in available_episodes:
        raise ValueError(
            f"Episode {selected} does not contain the requested frame. "
            f"Available episodes: {sorted(available_episodes)}"
        )
    return selected


def state_to_action_target(
    state: np.ndarray,
    *,
    action_dim: int,
    gripper_action_representation: str,
) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    if state.ndim != 1 or state.size not in (14, 16):
        raise ValueError(
            f"Episode start state must be a 14D or 16D vector, got shape {state.shape}."
        )
    if not np.all(np.isfinite(state)):
        raise ValueError("Episode start state contains NaN or infinity.")

    if action_dim == 14:
        if state.size == 14:
            return state.copy()
        return np.concatenate((state[0:7], state[8:15]))

    if action_dim != 16 or state.size != 16:
        raise ValueError(
            f"Cannot convert {state.size}D state to {action_dim}D action target."
        )

    target = state.copy()
    representation = gripper_action_representation.strip().lower()
    if representation == "binary_open_close":
        threshold_m = FRANKA_HAND_MAX_WIDTH_M / 2.0
        target[7] = 1.0 if state[7] >= threshold_m else 0.0
        target[15] = 1.0 if state[15] >= threshold_m else 0.0
    elif representation == "absolute_width":
        target[7] = float(np.clip(state[7] / FRANKA_HAND_MAX_WIDTH_M, 0.0, 1.0))
        target[15] = float(np.clip(state[15] / FRANKA_HAND_MAX_WIDTH_M, 0.0, 1.0))
    else:
        raise ValueError(
            "Unsupported gripper action representation for episode start: "
            f"{gripper_action_representation!r}."
        )
    return target


def load_episode_start_target(
    dataset_root: Path,
    selector: str,
    *,
    frame_index: int,
    random_seed: int | None,
    action_dim: int,
    gripper_action_representation: str,
) -> EpisodeStartTarget:
    if frame_index < 0:
        raise ValueError("--episode-start-frame-index must be non-negative.")

    root = dataset_root.expanduser().resolve()
    paths = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(
            f"--episode-start requires dataset parquet files under {root / 'data'}. "
            "Dataset metadata alone is not sufficient."
        )

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "--episode-start requires pyarrow in the executor environment."
        ) from exc

    states_by_episode: dict[int, np.ndarray] = {}
    for path in paths:
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "observation.state"],
        )
        for episode, frame, state in zip(
            table["episode_index"].to_pylist(),
            table["frame_index"].to_pylist(),
            table["observation.state"].to_pylist(),
            strict=True,
        ):
            if int(frame) != frame_index:
                continue
            states_by_episode.setdefault(int(episode), np.asarray(state, dtype=float))

    selected = select_episode_index(
        list(states_by_episode),
        selector,
        random_seed=random_seed,
    )
    state = states_by_episode[selected]
    return EpisodeStartTarget(
        episode_index=selected,
        frame_index=frame_index,
        dataset_state=state,
        action_target=state_to_action_target(
            state,
            action_dim=action_dim,
            gripper_action_representation=gripper_action_representation,
        ),
    )
