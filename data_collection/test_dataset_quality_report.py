from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import dataset_quality_report as quality


class DatasetQualityReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset_root = self.root / "dataset"
        self._create_dataset(self.dataset_root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_dataset(self, root: Path) -> None:
        (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
        (root / "data" / "chunk-000").mkdir(parents=True)
        info = {
            "codebase_version": "v3.0",
            "robot_type": "fr3_duo",
            "total_episodes": 1,
            "total_frames": 8,
            "total_tasks": 1,
            "fps": 15,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [16]},
                "action": {"dtype": "float32", "shape": [16]},
            },
        }
        (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
        action_config = {
            "arm_action_representation": "absolute_joint_position",
            "gripper_action_representation": "binary_open_close",
            "action_dim": 16,
        }
        (root / "meta" / "real_exp_action_config.json").write_text(
            json.dumps(action_config), encoding="utf-8"
        )

        base_arm = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.5, 0.0], dtype=np.float32)
        state = np.zeros((8, 16), dtype=np.float32)
        action = np.zeros((8, 16), dtype=np.float32)
        state[:, 0:7] = base_arm
        state[:, 8:15] = base_arm
        action[:, 0:7] = base_arm
        action[:, 8:15] = base_arm
        state[:, 7] = 0.08
        state[:, 15] = 0.08
        action[:, 7] = 1.0
        action[:, 15] = 1.0
        action[3, 14] = 3.04
        action[4, 14] = 0.0

        data_table = pa.table(
            {
                "episode_index": pa.array([0] * 8, type=pa.int64()),
                "frame_index": pa.array(list(range(8)), type=pa.int64()),
                "timestamp": pa.array(np.arange(8) / 15.0, type=pa.float64()),
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 16)),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 16)),
            }
        )
        pq.write_table(data_table, root / "data" / "chunk-000" / "file-000.parquet")
        episode_table = pa.table(
            {
                "episode_index": pa.array([0], type=pa.int64()),
                "length": pa.array([8], type=pa.int64()),
            }
        )
        pq.write_table(
            episode_table,
            root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        )

    def test_scan_detects_motion_quality_findings(self) -> None:
        report = quality.scan_dataset(
            self.dataset_root,
            quality.Thresholds(),
            check_video_frames=False,
        )
        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("action_position_safety_margin", codes)
        self.assertIn("action_acceleration_safety_limit", codes)
        self.assertEqual(report["summary"]["status"], "UNSAFE")

        plan = quality.build_repair_plan(report)
        self.assertEqual(plan["episodes"][0]["recommended_action"], "constrained_action_repair")

    def test_reports_must_stay_outside_dataset(self) -> None:
        report = quality.scan_dataset(
            self.dataset_root,
            quality.Thresholds(),
            check_video_frames=False,
        )
        with self.assertRaises(ValueError):
            quality.write_report(report, self.dataset_root / "quality_report")

        external = self.root / "reports" / "run"
        quality.write_report(report, external)
        self.assertTrue((external / "summary.md").is_file())
        self.assertTrue((external / "report.json").is_file())
        self.assertTrue((external / "events.csv").is_file())
        self.assertTrue((external / "repair_plan.json").is_file())

    def test_constrained_action_repair_respects_limits(self) -> None:
        thresholds = quality.Thresholds()
        initial = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.5, 0.0], dtype=np.float64)
        raw = np.repeat(initial[None, :], 60, axis=0)
        raw[2:, 0] = 1.0
        raw[20:, 0] = -1.0
        raw[40:, 0] = 0.5
        safe = quality.rate_limit_arm_targets(raw, initial, 15.0, thresholds, np)

        lower = np.asarray(quality.FR3_JOINT_LOWER_RAD) + thresholds.position_margin_rad
        upper = np.asarray(quality.FR3_JOINT_UPPER_RAD) - thresholds.position_margin_rad
        max_velocity = np.asarray(quality.FR3_MAX_VELOCITY_RAD_S) * thresholds.safety_factor
        max_acceleration = (
            np.asarray(quality.FR3_MAX_ACCELERATION_RAD_S2) * thresholds.safety_factor
        )
        points = np.vstack([initial, safe])
        velocity = np.diff(points, axis=0) * 15.0
        acceleration = np.diff(np.vstack([np.zeros((1, 7)), velocity]), axis=0) * 15.0

        self.assertTrue(np.all(safe >= lower - 1e-10))
        self.assertTrue(np.all(safe <= upper + 1e-10))
        self.assertTrue(np.all(np.abs(velocity) <= max_velocity + 1e-10))
        self.assertTrue(np.all(np.abs(acceleration) <= max_acceleration + 1e-10))

    def test_action_repair_rewrites_only_derived_dataset(self) -> None:
        derived = self.root / "derived"
        shutil.copytree(self.dataset_root, derived)
        source_table_before = pq.read_table(
            self.dataset_root / "data" / "chunk-000" / "file-000.parquet"
        )
        source_actions_before = source_table_before["action"].to_pylist()

        corrections = quality.rewrite_selected_actions(
            derived,
            {0},
            quality.Thresholds(),
            max_correction_rad=0.01,
            allow_large_repair=True,
        )

        source_actions_after = pq.read_table(
            self.dataset_root / "data" / "chunk-000" / "file-000.parquet"
        )["action"].to_pylist()
        derived_actions = pq.read_table(
            derived / "data" / "chunk-000" / "file-000.parquet"
        )["action"].to_pylist()
        self.assertEqual(source_actions_before, source_actions_after)
        self.assertNotEqual(source_actions_before, derived_actions)
        self.assertGreater(corrections[0], 0.0)

    def test_apply_keeps_reports_outside_derived_dataset(self) -> None:
        report = quality.scan_dataset(
            self.dataset_root,
            quality.Thresholds(),
            check_video_frames=False,
        )
        report_dir = self.root / "reports" / "scan"
        quality.write_report(report, report_dir)
        plan_path = report_dir / "repair_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan["episodes"][0]["selected_action"] = "keep"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        derived = self.root / "derived_keep"

        quality.apply_plan(
            SimpleNamespace(
                plan=str(plan_path),
                output_dir=str(derived),
                repo_id=None,
                video_workers=None,
                max_action_correction_rad=0.05,
                allow_large_action_repair=False,
                allow_unresolved=False,
            )
        )

        self.assertTrue((derived / "meta" / "info.json").is_file())
        self.assertFalse(any(derived.rglob("report.json")))
        self.assertFalse(any(derived.rglob("repair_plan.json")))
        self.assertTrue(any(report_dir.glob("post_repair_*/report.json")))


if __name__ == "__main__":
    unittest.main()
