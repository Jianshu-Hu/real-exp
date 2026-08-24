from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import pytest

from train.train_lerobot_policy import format_run_timestamp, resolve_output_dir
from train.eval_lerobot_policy import discover_run_dirs


def training_args(
    *,
    dataset_root: Path,
    policy_type: str = "act",
    output_dir: Path | None = None,
    resume: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_root=dataset_root,
        policy_type=policy_type,
        output_dir=output_dir,
        resume=resume,
    )


def test_format_run_timestamp_is_sortable_and_filesystem_safe() -> None:
    now = dt.datetime(2026, 8, 19, 14, 5, 7)

    assert format_run_timestamp(now) == "2026-08-19_14-05-07"


def test_default_output_dir_adds_timestamped_run_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("train.train_lerobot_policy.DEFAULT_OUTPUT_ROOT", tmp_path / "outputs")
    args = training_args(dataset_root=Path("data/test-traj-gen-pick-and-place"))
    now = dt.datetime(2026, 8, 19, 14, 5, 7)

    output_dir = resolve_output_dir(args, now=now)

    assert output_dir == (
        tmp_path
        / "outputs"
        / "test-traj-gen-pick-and-place_act"
        / "2026-08-19_14-05-07"
    ).resolve()


def test_explicit_output_dir_is_parent_for_new_run(tmp_path) -> None:
    output_parent = tmp_path / "custom-runs"
    args = training_args(
        dataset_root=Path("data/test-traj-gen-pick-and-place"),
        output_dir=output_parent,
    )

    output_dir = resolve_output_dir(args, now=dt.datetime(2026, 8, 19, 14, 5, 7))

    assert output_dir == output_parent.resolve() / "2026-08-19_14-05-07"


def test_resume_uses_exact_explicit_run_directory_without_new_timestamp(tmp_path) -> None:
    existing_run = tmp_path / "custom-runs" / "2026-08-18_20-51-53"
    args = training_args(
        dataset_root=Path("data/test-traj-gen-pick-and-place"),
        output_dir=existing_run,
        resume=True,
    )

    assert resolve_output_dir(args, now=dt.datetime(2026, 8, 19, 14, 5, 7)) == existing_run.resolve()


def test_resume_requires_exact_output_directory() -> None:
    args = training_args(
        dataset_root=Path("data/test-traj-gen-pick-and-place"),
        resume=True,
    )

    with pytest.raises(ValueError, match="--resume requires --output-dir"):
        resolve_output_dir(args)


def test_eval_discovers_timestamped_and_legacy_run_directories(tmp_path) -> None:
    timestamped_config = (
        tmp_path
        / "dataset_act"
        / "2026-08-19_14-05-07"
        / "checkpoints"
        / "005000"
        / "pretrained_model"
        / "config.json"
    )
    legacy_config = (
        tmp_path
        / "legacy_act"
        / "checkpoints"
        / "005000"
        / "pretrained_model"
        / "config.json"
    )
    for config_path in (timestamped_config, legacy_config):
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps({"type": "act"}))

    runs = discover_run_dirs(tmp_path, "act")

    assert runs == [timestamped_config.parents[3], legacy_config.parents[3]]
