from __future__ import annotations

import argparse
import unittest

import numpy as np

from policy_action_limiter import (
    FR3_JOINT_UPPER_RAD,
    PolicyActionLimiter,
    PolicyStartAligner,
    add_policy_action_limit_arguments,
)


LEFT_HOME = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0])
RIGHT_HOME = np.array([0.0, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0])


def robot_vector(left: np.ndarray, right: np.ndarray, *, with_grippers: bool) -> np.ndarray:
    if with_grippers:
        return np.concatenate((left, [0.08], right, [0.08]))
    return np.concatenate((left, right))


class PolicyActionLimitArgumentTest(unittest.TestCase):
    @staticmethod
    def parse_limit(*arguments: str) -> bool:
        parser = argparse.ArgumentParser()
        add_policy_action_limit_arguments(parser)
        return bool(parser.parse_args(list(arguments)).limit)

    def test_limiter_is_enabled_by_default(self) -> None:
        self.assertTrue(self.parse_limit())

    def test_no_limit_disables_limiter(self) -> None:
        self.assertFalse(self.parse_limit("--no-limit"))

    def test_limit_remains_a_compatible_explicit_enable(self) -> None:
        self.assertTrue(self.parse_limit("--limit"))


class PolicyActionLimiterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.limiter = PolicyActionLimiter(fps=15)

    def test_small_safe_action_stays_one_frame(self) -> None:
        state = robot_vector(LEFT_HOME, RIGHT_HOME, with_grippers=True)
        action = robot_vector(
            LEFT_HOME + 0.01,
            RIGHT_HOME - 0.01,
            with_grippers=True,
        )

        plan = self.limiter.plan(state, action)

        self.assertEqual(len(plan.steps), 1)
        self.assertFalse(plan.was_stretched)
        np.testing.assert_allclose(plan.steps[-1].left_joint_target, LEFT_HOME + 0.01)
        np.testing.assert_allclose(plan.steps[-1].right_joint_target, RIGHT_HOME - 0.01)

    def test_large_action_is_stretched_inside_limits(self) -> None:
        state = robot_vector(LEFT_HOME, RIGHT_HOME, with_grippers=False)
        action = robot_vector(
            LEFT_HOME + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            RIGHT_HOME - np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            with_grippers=False,
        )

        plan = self.limiter.plan(state, action)

        self.assertGreater(len(plan.steps), 1)
        self.assertTrue(plan.was_stretched)
        self.assertLessEqual(plan.max_velocity_ratio, 1.0 + 1e-12)
        self.assertLessEqual(plan.max_acceleration_ratio, 1.0 + 1e-12)
        np.testing.assert_allclose(plan.steps[-1].left_joint_target, action[0:7])
        np.testing.assert_allclose(plan.steps[-1].right_joint_target, action[7:14])

    def test_non_finite_action_is_rejected(self) -> None:
        state = robot_vector(LEFT_HOME, RIGHT_HOME, with_grippers=False)
        action = state.copy()
        action[3] = np.nan

        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            self.limiter.plan(state, action)

    def test_policy_target_near_hard_limit_is_rejected(self) -> None:
        state = robot_vector(LEFT_HOME, RIGHT_HOME, with_grippers=False)
        action = state.copy()
        action[0] = FR3_JOINT_UPPER_RAD[0] - 0.01

        with self.assertRaisesRegex(ValueError, "outside the allowed FR3 range"):
            self.limiter.plan(state, action)


class PolicyStartAlignerTest(unittest.TestCase):
    def test_alignment_is_conservative_and_requires_fresh_replan(self) -> None:
        state = robot_vector(LEFT_HOME, RIGHT_HOME, with_grippers=True)
        action = robot_vector(
            LEFT_HOME + np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            RIGHT_HOME - np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            with_grippers=True,
        )
        regular_plan = PolicyActionLimiter(fps=15).plan(state, action)
        aligner = PolicyStartAligner(fps=15, settle_frames=2)

        alignment_plan = aligner.start(state, action)

        self.assertGreater(len(alignment_plan.steps), len(regular_plan.steps))
        self.assertFalse(aligner.allows_observations)
        trajectory_updates = [
            aligner.update(state, now_monotonic=float(index))
            for index in range(len(alignment_plan.steps))
        ]
        self.assertEqual(
            sum(update.send_grippers for update in trajectory_updates),
            1,
        )
        self.assertTrue(trajectory_updates[-1].send_grippers)

        settle_start = float(len(alignment_plan.steps) - 1)
        first_settle = aligner.update(action, now_monotonic=settle_start + 0.01)
        second_settle = aligner.update(action, now_monotonic=settle_start + 0.02)
        self.assertFalse(first_settle.alignment_finished)
        self.assertTrue(second_settle.alignment_finished)
        self.assertEqual(aligner.phase, PolicyStartAligner.WAITING_REPLAN)
        self.assertTrue(aligner.allows_observations)
        self.assertTrue(aligner.blocks_normal_execution)

        hold = aligner.update(action, now_monotonic=settle_start + 0.03)
        self.assertFalse(hold.send_grippers)
        aligner.mark_replan_received()
        self.assertEqual(aligner.phase, PolicyStartAligner.COMPLETE)
        self.assertFalse(aligner.blocks_normal_execution)

    def test_alignment_times_out_when_robot_does_not_reach_target(self) -> None:
        state = robot_vector(LEFT_HOME, RIGHT_HOME, with_grippers=False)
        action = state.copy()
        action[0] += 0.1
        aligner = PolicyStartAligner(
            fps=15,
            settle_frames=2,
            settle_timeout_s=0.1,
        )
        plan = aligner.start(state, action)
        for index in range(len(plan.steps)):
            aligner.update(state, now_monotonic=float(index))

        with self.assertRaisesRegex(TimeoutError, "did not settle"):
            aligner.update(
                state,
                now_monotonic=float(len(plan.steps)) + 0.2,
            )

    def test_low_stiffness_joint_seven_uses_larger_settle_tolerance(self) -> None:
        state = robot_vector(LEFT_HOME, RIGHT_HOME, with_grippers=False)
        action = state.copy()
        action[6] += 0.2
        aligner = PolicyStartAligner(fps=15, settle_frames=1)
        plan = aligner.start(state, action)
        for index in range(len(plan.steps)):
            aligner.update(state, now_monotonic=float(index))

        compliant_state = action.copy()
        compliant_state[6] += 0.058
        update = aligner.update(
            compliant_state,
            now_monotonic=float(len(plan.steps)) + 0.01,
        )

        self.assertTrue(update.alignment_finished)
        self.assertEqual(aligner.phase, PolicyStartAligner.WAITING_REPLAN)

    def test_same_residual_is_rejected_on_high_stiffness_joint(self) -> None:
        state = robot_vector(LEFT_HOME, RIGHT_HOME, with_grippers=False)
        action = state.copy()
        action[0] += 0.2
        aligner = PolicyStartAligner(
            fps=15,
            settle_frames=1,
            settle_timeout_s=0.1,
        )
        plan = aligner.start(state, action)
        for index in range(len(plan.steps)):
            aligner.update(state, now_monotonic=float(index))

        residual_state = action.copy()
        residual_state[0] += 0.058
        with self.assertRaisesRegex(TimeoutError, "did not settle"):
            aligner.update(
                residual_state,
                now_monotonic=float(len(plan.steps)) + 0.2,
            )


if __name__ == "__main__":
    unittest.main()
