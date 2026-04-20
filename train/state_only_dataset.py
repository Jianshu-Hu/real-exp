from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from lerobot.datasets.utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
    hf_transform_to_torch,
    load_nested_dataset,
    load_info,
    load_stats,
    load_subtasks,
    load_tasks,
)


def normalize_feature_shapes(features: dict[str, dict]) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    for key, feature in features.items():
        shape = feature.get("shape")
        normalized[key] = {
            **feature,
            "shape": tuple(shape) if shape is not None else shape,
        }
    return normalized


def load_episode_rows(root: Path) -> list[dict]:
    episode_paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not episode_paths:
        raise FileNotFoundError(f"Missing episode metadata under {root / 'meta' / 'episodes'}")

    episodes_df = pd.concat([pd.read_parquet(path) for path in episode_paths], ignore_index=True)
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)
    return episodes_df.to_dict(orient="records")


class StateOnlyLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.root = Path(root) if root is not None else None
        self.episodes = episodes
        if self.root is None:
            raise ValueError("StateOnlyLeRobotDataset requires an explicit local dataset root.")

        info = load_info(self.root)
        tasks = load_tasks(self.root)
        subtasks = load_subtasks(self.root)
        stats = load_stats(self.root)
        episode_rows = load_episode_rows(self.root)
        self._episode_rows = episode_rows
        self._episode_index_lookup = {
            int(ep["episode_index"]): ep for ep in self._episode_rows
        }
        self.meta = SimpleNamespace(
            repo_id=repo_id,
            root=self.root,
            revision=revision,
            info=info,
            episodes=episode_rows,
            tasks=tasks,
            subtasks=subtasks,
            stats=stats,
            fps=info["fps"],
            features=info["features"],
            total_episodes=info["total_episodes"],
        )
        self.features = normalize_feature_shapes(self.meta.features)
        self._absolute_to_relative_idx: dict[int, int] | None = None

        self.delta_indices: dict[str, list[int]] | None = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, self.meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, self.meta.fps)

        hf_features = get_hf_features_from_features(self.features)
        self.hf_dataset = load_nested_dataset(
            self.root / "data",
            features=hf_features,
            episodes=self.episodes,
        )
        self.hf_dataset.set_transform(hf_transform_to_torch)
        self._build_index_mapping()

    def _build_index_mapping(self) -> None:
        self._absolute_to_relative_idx = None
        if self.episodes is not None:
            self._absolute_to_relative_idx = {
                abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
                for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
            }

    @property
    def num_frames(self) -> int:
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        ep = self._episode_index_lookup[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key, q_idx in query_indices.items():
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, value in query_result.items():
                item[key] = value

        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks.iloc[task_idx].name

        if "subtask_index" in self.features and self.meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self.meta.subtasks.iloc[subtask_idx].name

        return item
