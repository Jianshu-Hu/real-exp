from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from deployment_start_target import (
    load_episode_start_target,
    select_episode_index,
    state_to_action_target,
)


LEFT = np.array([0.1, -0.2, 0.3, -1.4, 0.5, 1.6, -0.7])
RIGHT = np.array([-0.1, 0.2, -0.3, -1.5, -0.5, 1.7, 0.7])


class DeploymentStartTargetTest(unittest.TestCase):
    def test_specific_and_seeded_random_episode_selection(self) -> None:
        self.assertEqual(
            select_episode_index([2, 4, 8], "4", random_seed=None),
            4,
        )
        first = select_episode_index([2, 4, 8], "random", random_seed=7)
        second = select_episode_index([8, 2, 4], "random", random_seed=7)
        self.assertEqual(first, second)

    def test_binary_grippers_are_derived_from_state_widths(self) -> None:
        state = np.concatenate((LEFT, [0.08], RIGHT, [0.0]))
        target = state_to_action_target(
            state,
            action_dim=16,
            gripper_action_representation="binary_open_close",
        )
        np.testing.assert_allclose(target[0:7], LEFT)
        np.testing.assert_allclose(target[8:15], RIGHT)
        self.assertEqual(target[7], 1.0)
        self.assertEqual(target[15], 0.0)

    def test_absolute_width_grippers_are_normalized_for_bridge(self) -> None:
        state = np.concatenate((LEFT, [0.04], RIGHT, [0.08]))
        target = state_to_action_target(
            state,
            action_dim=16,
            gripper_action_representation="absolute_width",
        )
        self.assertEqual(target[7], 0.5)
        self.assertEqual(target[15], 1.0)

    def test_sixteen_dimensional_state_can_feed_fourteen_dimensional_action(self) -> None:
        state = np.concatenate((LEFT, [0.08], RIGHT, [0.08]))
        target = state_to_action_target(
            state,
            action_dim=14,
            gripper_action_representation="binary_open_close",
        )
        np.testing.assert_allclose(target, np.concatenate((LEFT, RIGHT)))

    def test_loads_requested_episode_frame_from_dataset_parquet(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ModuleNotFoundError:
            self.skipTest("pyarrow is not installed")

        episode_2 = np.concatenate((LEFT, [0.08], RIGHT, [0.0]))
        episode_5 = np.concatenate((LEFT + 0.1, [0.0], RIGHT - 0.1, [0.08]))
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            data_dir = dataset_root / "data" / "chunk-000"
            data_dir.mkdir(parents=True)
            table = pa.table(
                {
                    "episode_index": [2, 2, 5, 5],
                    "frame_index": [0, 1, 0, 1],
                    "observation.state": [
                        episode_2.tolist(),
                        episode_2.tolist(),
                        episode_5.tolist(),
                        episode_5.tolist(),
                    ],
                }
            )
            pq.write_table(table, data_dir / "file-000.parquet")

            target = load_episode_start_target(
                dataset_root,
                "5",
                frame_index=0,
                random_seed=None,
                action_dim=16,
                gripper_action_representation="binary_open_close",
            )

        self.assertEqual(target.episode_index, 5)
        self.assertEqual(target.frame_index, 0)
        np.testing.assert_allclose(target.dataset_state, episode_5)
        self.assertEqual(target.action_target[7], 0.0)
        self.assertEqual(target.action_target[15], 1.0)


if __name__ == "__main__":
    unittest.main()
