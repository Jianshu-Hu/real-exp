from __future__ import annotations

import sys
from pathlib import Path

import pytest


SIMTOOLREAL_ROOT = Path(__file__).resolve().parents[1]
if str(SIMTOOLREAL_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMTOOLREAL_ROOT))

from transport import make_joint_state, make_joint_target, make_object_pose, make_policy_action, make_policy_observation, validate_packet


def test_joint_packet_round_trip() -> None:
    packet = make_joint_state([0.1, -0.2], ["a", "b"], velocities=[1.0, 2.0], timestamp_ns=123)
    assert validate_packet(packet) == packet


def test_object_pose_round_trip() -> None:
    packet = make_object_pose(
        [[1, 0, 0, 0.1], [0, 1, 0, -0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
        timestamp_ns=456,
    )
    assert validate_packet(packet) == packet
    assert packet["pose"][3:12:4] == [0.1, -0.2, 0.3]


def test_rejects_invalid_pose() -> None:
    with pytest.raises(ValueError, match="16 values"):
        make_object_pose([1, 2, 3])


def test_joint_target_round_trip() -> None:
    packet = make_joint_target([0.1] * 27, timestamp_ns=789)
    assert validate_packet(packet) == packet


def test_policy_rpc_packets_round_trip() -> None:
    observation = make_policy_observation([0.0] * 134, timestamp_ns=10)
    action = make_policy_action([0.0] * 27, timestamp_ns=11)
    assert validate_packet(observation) == observation
    assert validate_packet(action) == action
