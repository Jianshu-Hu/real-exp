from collections import deque
import copy
import json
import threading
import time
from types import SimpleNamespace

import pytest
import zmq

from softhand_grasp import client, geometry, protocol, server


def server_result(request):
    # Already-calculated poses; no camera points or calibration cross the wire.
    return {"side": "right", "pose_frame": "right_base", "live_capture": True,
            "capture_started_unix_s": time.time(), "completed_unix_s": time.time(),
            "plan": {"description": request["description"],
                     "grasp_pose_xyzrpy": [0.4, 0.2, 0.4, 0, 0, 0],
                     "lift_pose_xyzrpy": [0.4, 0.2, 0.5, 0, 0, 0]}}


@pytest.fixture
def reply():
    request = protocol.request_message("red cup")
    response = server.handle_request(request, server_result, deque(maxlen=256))
    return request, response


@pytest.mark.parametrize("change,pattern", [
    ({"request_id": "wrong"}, "request_id"),
    ({"side": "left"}, "right-arm"),
    ({"pose_frame": "camera"}, "right-arm"),
    ({"capture_started_unix_s": 1}, "stale"),
    ({"completed_unix_s": float("nan")}, "finite"),
    ({"live_capture": "true"}, "live"),
    ({"live_capture": False}, "replay"),
])
def test_client_rejects_wrong_or_stale_response(reply, change, pattern):
    request, response = reply
    response.update(change)
    with pytest.raises(ValueError, match=pattern):
        protocol.validate_response(response, request, execute=True)


def test_pose_validation_and_no_client_recalculation(reply, monkeypatch):
    request, response = reply
    plan = protocol.validate_response(response, request, execute=True)
    poses = []

    def forbidden(*_):
        pytest.fail("Client recalculated a server pose")

    monkeypatch.setattr(geometry, "target_poses", forbidden)
    client.run_sequence(client.load_config(), SimpleNamespace(move_to=poses.append), lambda: plan)
    assert poses[1:] == [plan["grasp_pose_xyzrpy"], plan["lift_pose_xyzrpy"]]
    bad = copy.deepcopy(response)
    bad["plan"]["lift_pose_xyzrpy"][0] = 100
    with pytest.raises(ValueError, match="workspace"):
        protocol.validate_response(bad, request)
    bad["plan"]["lift_pose_xyzrpy"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        protocol.validate_response(bad, request)


def test_server_error_and_duplicate_do_not_call_inference_again():
    seen, calls = deque(maxlen=256), []

    def broken(request):
        calls.append(request)
        raise RuntimeError("camera disconnected")

    request = protocol.request_message("cup")
    response = server.handle_request(request, broken, seen)
    assert not response["ok"] and "camera disconnected" in response["error"]
    with pytest.raises(RuntimeError, match="camera disconnected"):
        protocol.validate_response(response, request)
    assert "Duplicate" in server.handle_request(request, broken, seen)["error"]
    assert len(calls) == 1


def test_real_tcp_request_reply_and_malformed_request_recovery():
    context = zmq.Context()
    ready, address, errors = threading.Event(), [], []

    def serve():
        try:
            with context.socket(zmq.REP) as socket:
                socket.setsockopt(zmq.RCVTIMEO, 3000)
                port = socket.bind_to_random_port("tcp://127.0.0.1")
                address.append(f"tcp://127.0.0.1:{port}")
                ready.set()
                seen = deque(maxlen=256)
                server.serve_once(socket, server_result, seen)
                server.serve_once(socket, server_result, seen)
        except BaseException as exc:
            errors.append(exc)
            ready.set()

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        assert ready.wait(3) and address
        with context.socket(zmq.REQ) as bad:
            bad.setsockopt(zmq.LINGER, 0)
            bad.setsockopt(zmq.RCVTIMEO, 3000)
            bad.connect(address[0])
            bad.send(b"this is not JSON")
            assert bad.recv_json()["ok"] is False
        args = SimpleNamespace(object="red cup", instance_index=None, execute=False,
                               server_address=address[0], request_timeout_s=3, max_plan_age_s=30)
        result = client.request_plan(args)
        assert result["plan"]["description"] == "red cup"
        assert result["plan"]["grasp_pose_xyzrpy"] == [0.4, 0.2, 0.4, 0, 0, 0]
    finally:
        thread.join(4)
        context.destroy(linger=0)
    assert not thread.is_alive() and not errors


def test_timeout_aborts_before_approach_or_pressure():
    with zmq.Context() as context, context.socket(zmq.REP) as silent:
        port = silent.bind_to_random_port("tcp://127.0.0.1")
        args = SimpleNamespace(object="cup", instance_index=None, execute=True,
                               server_address=f"tcp://127.0.0.1:{port}",
                               request_timeout_s=0.05, max_plan_age_s=30)
        moves = []
        with pytest.raises(TimeoutError, match="No grasp response"):
            client.run_sequence(client.load_config(), SimpleNamespace(move_to=moves.append),
                                lambda: client.request_plan(args)["plan"], execute=True)
        assert len(moves) == 1


def test_server_owns_capture_and_pose_calculation(tmp_path, monkeypatch):
    commands = []
    args = SimpleNamespace(runs_dir=tmp_path, capture_dir=None, camera_python="camera-python",
                           sam_python="sam-python", checkpoint=tmp_path / "sam3.pt", device="cuda")
    request = protocol.request_message("red cup", 1)

    def fake_run(command, **kwargs):
        commands.append(command)
        output = command[command.index("--output-dir") + 1]
        assert "PYTHONPATH" in kwargs["env"]
        if "softhand_grasp.perception" in command:
            from pathlib import Path
            (Path(output) / "plan.json").write_text(json.dumps(server_result(request)["plan"]))

    monkeypatch.setattr(server.subprocess, "run", fake_run)
    result = server.infer_request(request, args, geometry.load_config())
    assert commands[0][:3] == ["camera-python", "-m", "grasp.realsense_capture"]
    assert commands[1][:3] == ["sam-python", "-m", "softhand_grasp.perception"]
    assert commands[1][-2:] == ["--instance-index", "1"]
    assert result["live_capture"] is True and result["pose_frame"] == "right_base"
    # Replay mode is explicitly marked and does not start camera capture.
    args.capture_dir = tmp_path / "saved_capture"
    commands.clear()
    result = server.infer_request(request, args, geometry.load_config())
    assert len(commands) == 1 and not result["live_capture"]
