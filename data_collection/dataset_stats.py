from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

LOGGER = logging.getLogger(__name__)

IMAGENET_CAMERA_STATS = {
    "mean": np.array([[[0.485]], [[0.456]], [[0.406]]], dtype=np.float32),
    "std": np.array([[[0.229]], [[0.224]], [[0.225]]], dtype=np.float32),
    "min": np.array([[[0.0]], [[0.0]], [[0.0]]], dtype=np.float32),
    "max": np.array([[[1.0]], [[1.0]], [[1.0]]], dtype=np.float32),
}


def ensure_dataset_stats(
    repo_id: str,
    dataset_root: Path,
    *,
    force_recompute: bool = False,
    add_imagenet_camera_stats: bool = True,
) -> LeRobotDataset:
    """Ensure a local LeRobot dataset has training-ready stats metadata.

    LeRobot 0.5.2 training expects ``meta/stats.json`` to exist. Numeric stats
    are recomputed from parquet data when missing or explicitly requested. Camera
    entries are populated with ImageNet mean/std so the default
    ``DatasetConfig.use_imagenet_stats=True`` path can safely overwrite/use them.
    """
    dataset_root = dataset_root.expanduser().resolve()
    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)

    stats_missing = dataset.meta.stats is None
    if stats_missing or force_recompute:
        LOGGER.warning("Generating dataset stats at %s", dataset_root / "meta" / "stats.json")
        recompute_stats(dataset, skip_image_video=True)

    if dataset.meta.stats is None:
        dataset.meta.stats = {}

    stats_updated = stats_missing or force_recompute
    if add_imagenet_camera_stats:
        for key in dataset.meta.camera_keys:
            camera_stats = dataset.meta.stats.setdefault(key, {})
            for stat_name, stat_value in IMAGENET_CAMERA_STATS.items():
                if stat_name not in camera_stats:
                    camera_stats[stat_name] = stat_value
                    stats_updated = True
            if "count" not in camera_stats:
                camera_stats["count"] = np.array([dataset.num_frames], dtype=np.int64)
                stats_updated = True

    if stats_updated:
        write_stats(dataset.meta.stats, dataset_root)

    return dataset
