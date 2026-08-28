from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import zmq

import sys

ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = ROOT / "simtoolreal"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from policy_executor import PolicyRpcClient


def test_mock_policy_rpc_serves_134_to_27() -> None:
    context = zmq.Context()
    endpoint = "inproc://simtoolreal-rpc-integration"
    ready = threading.Event()

    def run_server() -> None:
        socket = context.socket(zmq.REP)
        socket.bind(endpoint)
        from rl_policy import MockPolicy
        from transport import make_policy_action, validate_packet
        ready.set()
        try:
            request = validate_packet(socket.recv_json())
            assert request["kind"] == "policy_observation"
            assert len(request["observation"]) == 134
            socket.send_json(make_policy_action(MockPolicy(27).act(np.zeros(134)).tolist()))
        finally:
            socket.close(0)

    thread = threading.Thread(target=run_server)
    thread.start()
    assert ready.wait(1.0)
    client = PolicyRpcClient(context, endpoint)
    action = client.act(np.zeros(134, dtype=np.float32))
    client.close()
    thread.join(timeout=1.0)
    context.term()
    assert action.shape == (27,)
