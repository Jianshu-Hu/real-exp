from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from .teleop import (
    HAND2_JOINT_NIDS,
    LANDMARK_NAMES,
    diagnostic_comm_summary,
    hand_joint_positions,
    read_hand_tuning,
    require_clean_diagnostics,
    select_device,
    skeleton_keypoints,
    validate_diagnostics,
)


def _skeleton_frame(*, confidence: float = 1.0):
    joints = []
    for index, name in enumerate(LANDMARK_NAMES):
        joints.append(
            SimpleNamespace(
                name=name,
                confidence=confidence,
                pose=SimpleNamespace(position=[0.01 * index, 0.02 * index, 0.0]),
            )
        )
    return SimpleNamespace(joints=joints)


def test_skeleton_keypoints_validates_order_shape_and_confidence() -> None:
    keypoints = skeleton_keypoints(_skeleton_frame(), min_confidence=0.25)
    assert keypoints.shape == (21, 3)
    assert keypoints.dtype == np.float32

    low_confidence = _skeleton_frame()
    low_confidence.joints[4].confidence = 0.1
    with pytest.raises(ValueError, match="thumb_tip"):
        skeleton_keypoints(low_confidence, min_confidence=0.25)

    wrong_order = _skeleton_frame()
    wrong_order.joints[0], wrong_order.joints[1] = wrong_order.joints[1], wrong_order.joints[0]
    with pytest.raises(ValueError, match="landmark order"):
        skeleton_keypoints(wrong_order, min_confidence=0.25)


def test_hand_joint_positions_returns_firmware_order() -> None:
    joints = [
        SimpleNamespace(nid=nid, position=float(index))
        for index, nid in reversed(list(enumerate(HAND2_JOINT_NIDS)))
    ]
    positions = hand_joint_positions(SimpleNamespace(joints=joints))
    np.testing.assert_allclose(positions, np.arange(20, dtype=np.float64))

    with pytest.raises(ValueError, match="missing joint NIDs"):
        hand_joint_positions(SimpleNamespace(joints=joints[:-1]))


def test_validate_diagnostics_rejects_faults_and_missing_joints() -> None:
    healthy = [
        SimpleNamespace(nid=nid, error_code_current=0) for nid in HAND2_JOINT_NIDS
    ]
    validate_diagnostics(SimpleNamespace(joints=healthy))

    healthy[7].error_code_current = 0x1234
    with pytest.raises(RuntimeError, match="0x1234"):
        validate_diagnostics(SimpleNamespace(joints=healthy))

    warnings = validate_diagnostics(
        SimpleNamespace(joints=healthy),
        lambda code: {"name": "SyntheticWarning", "severity": "Warning"},
    )
    assert warnings == [(HAND2_JOINT_NIDS[7], 0x1234, "SyntheticWarning")]
    with pytest.raises(RuntimeError, match="missing joint NIDs"):
        validate_diagnostics(SimpleNamespace(joints=healthy[:-1]))


def test_diagnostic_warnings_block_enable_without_explicit_override() -> None:
    warnings = {(7, 0x0003, "Enc1BitRate")}
    with pytest.raises(RuntimeError, match="refusing to enable"):
        require_clean_diagnostics(warnings, allow_warnings=False)
    require_clean_diagnostics(warnings, allow_warnings=True)
    require_clean_diagnostics(set(), allow_warnings=False)


def test_diagnostic_comm_summary_separates_host_and_internal_bus_quality() -> None:
    comm = SimpleNamespace(
        e2e_received=999,
        e2e_lost=1,
        e2e_reordered=2,
        e2e_duplicates=3,
        sdk_dropped=4,
        rpc_retries=5,
        rpc_total=100,
        rpc_timeouts=6,
        comm_get_failures=7,
    )
    joints = [
        SimpleNamespace(
            nid=nid,
            comm_response_rate_pct=99 if nid == HAND2_JOINT_NIDS[0] else 100,
            comm_timeout_total=2 if nid == HAND2_JOINT_NIDS[0] else 0,
            error_code_current=0x0006 if nid == HAND2_JOINT_NIDS[0] else 0,
        )
        for nid in HAND2_JOINT_NIDS
    ]
    overall, details = diagnostic_comm_summary(SimpleNamespace(comm=comm, joints=joints))
    assert "loss=0.1000%" in overall
    assert "rpc_timeouts=6" in overall
    assert details == ["NID 1: response=99% timeouts=2 error=0x0006"]


def test_read_hand_tuning_reads_all_effort_and_mit_values() -> None:
    hand = SimpleNamespace(
        effort_limit=lambda: SimpleNamespace(get=lambda: [1.5] * 20),
        mit_params=lambda: SimpleNamespace(
            get=lambda: [SimpleNamespace(kp=3.0, kd=0.05)] * 20
        ),
    )
    efforts, kps, kds = read_hand_tuning(hand)
    assert efforts == [1.5] * 20
    assert kps == [3.0] * 20
    assert kds == [0.05] * 20

    hand.effort_limit = lambda: SimpleNamespace(get=lambda: [1.5] * 19)
    with pytest.raises(RuntimeError, match="20 hand effort limits"):
        read_hand_tuning(hand)


def test_select_device_requires_unambiguous_match() -> None:
    glove_type = object()
    devices = [
        SimpleNamespace(sn="WG1", address="192.168.1.101:50001", device_type=glove_type),
        SimpleNamespace(sn="WG2", address="192.168.1.100:50001", device_type=glove_type),
    ]
    assert select_device(devices, glove_type, "WG1", "Wuji Glove") is devices[0]
    assert select_device(devices, glove_type, "192.168.1.101", "Wuji Glove") is devices[0]
    with pytest.raises(RuntimeError, match="Multiple"):
        select_device(devices, glove_type, None, "Wuji Glove")
    with pytest.raises(RuntimeError, match="No Wuji Glove"):
        select_device(devices, glove_type, "missing", "Wuji Glove")
