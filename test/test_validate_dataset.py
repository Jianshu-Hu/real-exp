from __future__ import annotations

from fractions import Fraction

import av
import numpy as np

from data_collection.validate_dataset import (
    check_joint_safety_constraints,
    check_physical_video_frames,
    check_state_action_semantics,
)
from utils.limit import (
    FR3_SAFE_POSITION_LOWER_RAD,
    FR3_SAFE_POSITION_UPPER_RAD,
)


def dual_arm_vector(left: np.ndarray, right: np.ndarray) -> list[float]:
    return [*left, 0.5, *right, 0.5]


def test_dataset_warning_reports_violation_frame_indices() -> None:
    midpoint = 0.5 * (FR3_SAFE_POSITION_LOWER_RAD + FR3_SAFE_POSITION_UPPER_RAD)
    states = [dual_arm_vector(midpoint, midpoint) for _ in range(4)]
    actions = [dual_arm_vector(midpoint, midpoint) for _ in range(4)]
    states[2] = dual_arm_vector(
        np.array(
            [FR3_SAFE_POSITION_UPPER_RAD[0] + 0.01, *midpoint[1:]],
        ),
        midpoint,
    )
    rows = [
        {
            "frame_index": frame_index + 10,
            "timestamp": frame_index * 0.1,
            "observation.state": state,
            "action": action,
        }
        for frame_index, (state, action) in enumerate(zip(states, actions))
    ]

    issues, warnings, metrics = check_joint_safety_constraints(
        rows,
        "absolute_joint_position",
    )

    assert issues == []
    assert len(warnings) == 2
    assert warnings[0].startswith("left measured-state validity violations:")
    assert "position=1 frames=[12]" in warnings[0]
    assert any(
        warning.startswith("left sampled state motion warnings:")
        for warning in warnings
    )
    assert metrics["state_violation_steps"] == 1
    assert metrics["state_motion_warning_steps"] == 2
    assert metrics["action_violation_steps"] == 0
    assert metrics["action_waypoint_slew_steps"] == 0


def test_dataset_treats_absolute_action_derivatives_as_waypoint_slew() -> None:
    midpoint = 0.5 * (FR3_SAFE_POSITION_LOWER_RAD + FR3_SAFE_POSITION_UPPER_RAD)
    states = [dual_arm_vector(midpoint, midpoint) for _ in range(4)]
    actions = [dual_arm_vector(midpoint, midpoint) for _ in range(4)]
    unsafe_action = midpoint.copy()
    unsafe_action[0] += 0.3
    actions[1] = dual_arm_vector(unsafe_action, midpoint)
    rows = [
        {
            "frame_index": frame_index,
            "timestamp": frame_index * 0.1,
            "observation.state": state,
            "action": action,
        }
        for frame_index, (state, action) in enumerate(zip(states, actions))
    ]

    issues, warnings, metrics = check_joint_safety_constraints(
        rows,
        "absolute_joint_position",
    )

    assert issues == []
    assert warnings == []
    assert metrics["state_violation_steps"] == 0
    assert metrics["action_violation_steps"] == 0
    assert metrics["action_waypoint_slew_steps"] == 3


def test_dataset_still_checks_accepted_action_position_envelope() -> None:
    midpoint = 0.5 * (FR3_SAFE_POSITION_LOWER_RAD + FR3_SAFE_POSITION_UPPER_RAD)
    states = [dual_arm_vector(midpoint, midpoint) for _ in range(3)]
    actions = [dual_arm_vector(midpoint, midpoint) for _ in range(3)]
    unsafe_action = midpoint.copy()
    unsafe_action[0] = FR3_SAFE_POSITION_UPPER_RAD[0] + 0.01
    actions[1] = dual_arm_vector(unsafe_action, midpoint)
    rows = [
        {
            "frame_index": frame_index,
            "timestamp": frame_index * 0.1,
            "observation.state": state,
            "action": action,
        }
        for frame_index, (state, action) in enumerate(zip(states, actions))
    ]

    issues, warnings, metrics = check_joint_safety_constraints(
        rows,
        "absolute_joint_position",
    )

    assert issues == []
    assert len(warnings) == 1
    assert warnings[0].startswith("left accepted action-target validity violations:")
    assert "position=1 frames=[1]" in warnings[0]
    assert metrics["state_violation_steps"] == 0
    assert metrics["action_violation_steps"] == 1
    assert metrics["action_waypoint_slew_steps"] == 2


def test_single_right_arm_hand_layout_ignores_hand_as_grippers() -> None:
    midpoint = 0.5 * (FR3_SAFE_POSITION_LOWER_RAD + FR3_SAFE_POSITION_UPPER_RAD)
    hand = np.linspace(-1.5, 1.5, 20)
    vector = [*midpoint, *hand]
    rows = [
        {
            "frame_index": frame_index,
            "timestamp": frame_index * 0.1,
            "observation.state": vector,
            "action": vector,
        }
        for frame_index in range(3)
    ]
    trajectory_config = {
        "end_effector": "hand",
        "arm_mode": "right",
        "arms": ["right"],
    }

    semantic_issues, semantic_metrics = check_state_action_semantics(
        episode_index=0,
        rows=rows,
        arm_action_representation="absolute_joint_position",
        delta_action_tolerance=1e-4,
        action_outlier_threshold=0.3,
        gripper_min=0.0,
        gripper_max=1.0,
        gripper_tolerance=1e-5,
        trajectory_config=trajectory_config,
    )
    safety_issues, _, safety_metrics = check_joint_safety_constraints(
        rows,
        "absolute_joint_position",
        trajectory_config,
    )

    assert semantic_issues == []
    assert semantic_metrics["gripper_checked"] is False
    assert semantic_metrics["gripper_outlier_frames"] == []
    assert semantic_metrics["max_left_arm_delta"] == 0.0
    assert semantic_metrics["max_right_arm_delta"] == max(abs(midpoint))
    assert safety_issues == []
    assert safety_metrics["state_violation_steps"] == 0
    assert safety_metrics["action_violation_steps"] == 0


def write_test_video(path, *, frame_count: int, fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 16
        stream.height = 12
        stream.pix_fmt = "yuv420p"
        for frame_index in range(frame_count):
            image = np.full((12, 16, 3), frame_index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = frame_index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def video_fixture_metadata(frame_count: int, fps: int = 10):
    video_key = "observation.images.test"
    info = {
        "fps": fps,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {video_key: {"dtype": "video", "shape": [3, 12, 16]}},
    }
    episodes = [
        {
            "episode_index": 0,
            "length": frame_count,
            f"videos/{video_key}/chunk_index": 0,
            f"videos/{video_key}/file_index": 0,
        }
    ]
    return video_key, info, episodes


def test_physical_video_quality_fully_decodes_valid_stream(tmp_path) -> None:
    video_key, info, episodes = video_fixture_metadata(frame_count=5)
    video_path = tmp_path / info["video_path"].format(
        video_key=video_key,
        chunk_index=0,
        file_index=0,
    )
    write_test_video(video_path, frame_count=5)

    assert check_physical_video_frames(tmp_path, info, episodes, [video_key]) == []


def test_physical_video_quality_rejects_truncated_stream(tmp_path) -> None:
    video_key, info, episodes = video_fixture_metadata(frame_count=5)
    video_path = tmp_path / info["video_path"].format(
        video_key=video_key,
        chunk_index=0,
        file_index=0,
    )
    write_test_video(video_path, frame_count=5)
    video_bytes = video_path.read_bytes()
    video_path.write_bytes(video_bytes[: len(video_bytes) // 2])

    issues = check_physical_video_frames(tmp_path, info, episodes, [video_key])

    assert any(
        "full decode failed" in issue
        or "physical frames" in issue
        or "invalid ffprobe output" in issue
        for issue in issues
    )
