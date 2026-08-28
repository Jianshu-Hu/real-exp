from __future__ import annotations

import math
import socket
import time

import pytest

from data_collection.meta_quest.protocol import ProtocolError, parse_pose_line
from data_collection.meta_quest.power import QuestKeepAwake
from data_collection.meta_quest.receiver import PoseReceiver
from data_collection.meta_quest.visualizer import rotate_vector_xyzw
from data_collection.meta_quest.calibration import PoseCalibrator, RigidPose


def test_parse_upstream_wrist_pose() -> None:
    pose = parse_pose_line(
        "Right wrist:, 0.2502, 1.0635, 0.2540, 0.194, -0.116, 0.094, -0.970"
    )

    assert pose is not None
    assert pose.side == "right"
    assert pose.source == "wrist"
    assert pose.position == (0.2502, 1.0635, 0.2540)
    assert math.isclose(sum(value * value for value in pose.quaternion_xyzw), 1.0)
    assert pose.tracked is True
    assert pose.layout == "wrist_pose_v1"


def test_parse_legacy_tracked_first_controller_pose() -> None:
    pose = parse_pose_line(
        "Left controller:, 1, 0.1, 0.2, 0.3, 0, 0, 0, 1, "
        "0, 0, 1, 0, 1, 0, 1, 0, 0, 0.7"
    )

    assert pose is not None
    assert pose.side == "left"
    assert pose.source == "controller"
    assert pose.position == (0.1, 0.2, 0.3)
    assert pose.quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)
    assert pose.tracked is True
    assert pose.grasp == 0.7
    assert pose.layout == "controller_tracked_first_v1"


def test_parse_position_first_controller_pose_with_debug_metadata() -> None:
    pose = parse_pose_line(
        "Right controller | f = 42 | t = 123456789:, "
        "0.1, 0.2, 0.3, 0, 0, 0, 1, "
        "0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4"
    )

    assert pose is not None
    assert pose.position == (0.1, 0.2, 0.3)
    assert pose.tracked is True
    assert pose.grasp == 0.4
    assert pose.frame_id == 42
    assert pose.device_timestamp_ns == 123456789
    assert pose.layout == "controller_position_first_v1"


def test_parse_installed_apk_compact_controller_pose() -> None:
    pose = parse_pose_line(
        "Right controller:, 0.1392, 0.8692, -0.2335, "
        "-0.040, 0.851, -0.132, 0.508, 0.75, 1, 0"
    )

    assert pose is not None
    assert pose.position == (0.1392, 0.8692, -0.2335)
    assert pose.tracked is True
    assert pose.grasp == 0.75
    assert pose.clutch is True
    assert pose.record is False
    assert pose.layout == "controller_compact_controls_v1"


def test_ignore_landmarks_and_reject_malformed_pose() -> None:
    assert parse_pose_line("Right landmarks:, 0, 0, 0") is None
    assert parse_pose_line("Fist: Closed") is None
    with pytest.raises(ProtocolError, match="expected 10, 17, or 18"):
        parse_pose_line("Right controller:, 1, 2, 3")
    with pytest.raises(ProtocolError, match="clutch/record flags"):
        parse_pose_line("Right controller:, 1, 2, 3, 0, 0, 0, 1, 0.5, 2, 0")
    with pytest.raises(ProtocolError, match="zero length"):
        parse_pose_line("Left wrist:, 0, 0, 0, 0, 0, 0, 0")


def test_receiver_accepts_concurrent_stream_records() -> None:
    with PoseReceiver(port=0) as receiver:
        with (
            socket.create_connection(("127.0.0.1", receiver.port)) as right,
            socket.create_connection(("127.0.0.1", receiver.port)) as left,
        ):
            right.sendall(b"Right wrist:, 1, 2,")
            right.sendall(b" 3, 0, 0, 0, 1\nRight controller:, 1, 2, 3\n")
            left.sendall(b"Left wrist:, 4, 5, 6, 0, 0, 0, 1\n")

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            snapshot = receiver.snapshot()
            if snapshot.pose_records == 2 and snapshot.rejected_pose_records == 1:
                break
            time.sleep(0.01)

        snapshot = receiver.snapshot()
        assert snapshot.accepted_connections == 2
        assert snapshot.pose_records == 2
        assert snapshot.rejected_pose_records == 1
        assert snapshot.poses[("wrist", "right")].sample.position == (1.0, 2.0, 3.0)
        assert snapshot.poses[("wrist", "left")].sample.position == (4.0, 5.0, 6.0)


def test_receiver_measures_each_controller_frequency() -> None:
    receiver = PoseReceiver(frequency_window_s=1.0)
    line = "Right controller:, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0"
    receiver._record_line(line)
    time.sleep(0.02)
    receiver._record_line(line)

    hz = receiver.snapshot().receive_hz[("controller", "right")]
    assert 20.0 < hz < 100.0


def test_visualizer_rotates_controller_axes_from_xyzw_quaternion() -> None:
    half_sqrt_two = math.sqrt(0.5)
    rotated = rotate_vector_xyzw(
        (0.0, 0.0, half_sqrt_two, half_sqrt_two),
        (1.0, 0.0, 0.0),
    )

    assert rotated == pytest.approx((0.0, 1.0, 0.0))


def test_keep_awake_is_acquired_only_for_an_active_stream(monkeypatch) -> None:
    actions: list[str] = []
    guard = QuestKeepAwake()
    monkeypatch.setattr(guard, "_broadcast", lambda action: actions.append(action) or True)

    assert guard.update(False) is False
    assert actions == []
    assert guard.update(True) is True
    assert len(actions) == 1
    assert actions[0].endswith("prox_close")
    assert guard.update(True) is True
    assert len(actions) == 1
    assert guard.update(False) is False
    assert actions[-1].endswith("automation_disable")


def test_pose_calibration_maps_anchor_to_predefined_target() -> None:
    source_anchor = RigidPose((1.0, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    target_anchor = RigidPose((0.4, -0.2, 0.7), (0.0, 0.0, 0.0, 1.0))
    calibrator = PoseCalibrator(target_anchor)

    assert calibrator.update(source_anchor, False) is None
    mapped_anchor = calibrator.update(source_anchor, True)
    assert mapped_anchor is not None
    assert mapped_anchor.position == pytest.approx(target_anchor.position)
    assert mapped_anchor.quaternion_xyzw == pytest.approx(target_anchor.quaternion_xyzw)
    assert calibrator.calibrated is True

    moved_source = RigidPose((1.2, 2.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    moved_target = calibrator.update(moved_source, False)
    assert moved_target is not None
    assert moved_target.position == pytest.approx((0.6, -0.2, 0.7))


def test_pose_calibration_button_only_captures_on_rising_edge() -> None:
    target = RigidPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    calibrator = PoseCalibrator(target)
    first = RigidPose((1.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    second = RigidPose((2.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    calibrator.update(first, True)
    held = calibrator.update(second, True)
    assert held is not None
    assert held.position == pytest.approx((1.0, 0.0, 0.0))


def test_pose_calibration_rejects_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="zero length"):
        RigidPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
