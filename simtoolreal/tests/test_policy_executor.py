from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = ROOT / "simtoolreal"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from policy_executor import PolicyExecutor, PolicyRpcClient, _state


def test_executor_builds_and_splits_simtoolreal_target() -> None:
    upstream = ROOT / "libs/SimToolReal-Franka-Wuji2"
    urdf = upstream / "assets/urdf/franka_wuji_right/fr3v2_wuji_hand2_right.urdf"
    executor = PolicyExecutor(
        robot_urdf=urdf,
        upstream_root=upstream,
        config=None,
        checkpoint=None,
        mock_policy=True,
    )
    pose = np.eye(4)
    pose[:3, 3] = (0.1, -0.2, 0.65)
    action, target = executor.infer(
        np.zeros(27),
        np.zeros(27),
        pose,
        pose,
        np.ones(3),
        np.eye(4),
    )
    assert action.shape == (27,)
    assert target.full.shape == (27,)
    assert target.arm.shape == (7,)
    assert target.hand.shape == (20,)


def test_executor_accepts_right_bridge_state_packet() -> None:
    q, qd, stamp = _state({
        "arm_mode": "right",
        "include_hand": True,
        "joint_state": list(np.arange(27, dtype=float)),
        "robot_state_stamp_s": 123.0,
    })
    assert q.shape == qd.shape == (27,)
    assert stamp == 123_000_000_000


def test_executor_rejects_non_right_bridge_state() -> None:
    import pytest
    with pytest.raises(ValueError, match="right-arm"):
        _state({"arm_mode": "left", "include_hand": True, "joint_state": [0.0] * 27})


def test_policy_rpc_client_round_trip() -> None:
    import threading
    import zmq

    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.bind("inproc://simtoolreal-policy-test")

    def serve_once() -> None:
        request = server.recv_json()
        assert request["kind"] == "policy_observation"
        server.send_json({"protocol": 1, "kind": "policy_action", "timestamp_ns": 2,
                          "source": "test", "action": [0.0] * 27})

    thread = threading.Thread(target=serve_once)
    thread.start()
    client = PolicyRpcClient(context, "inproc://simtoolreal-policy-test")
    np.testing.assert_allclose(client.act(np.zeros(134)), 0.0)
    client.close()
    thread.join(timeout=1.0)
    server.close(0)
    context.term()
