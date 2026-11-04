"""Synchronization and offline selection checks for continuous video capture."""

import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.prepare_wrist_recording import select_windows
from calibration.record_wrist_calibration import StateHistory, interpolate_state, select_cameras


@pytest.mark.parametrize('output', ['/left', '/right', '/both'])
def test_rejects_unset_run_path_before_loading_hardware_modules(output):
    script = Path(__file__).resolve().parents[1] / 'record_wrist_calibration.py'
    result = subprocess.run([sys.executable, str(script), '--side', 'left', '--output', output], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert 'unset $calib_run' in result.stderr
    assert 'Waiting for fresh' not in result.stdout


def state(stamp, xyz=(0., 0., 0.), angle=0.):
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler('z', angle, degrees=True).as_matrix()
    pose[:3, 3] = xyz
    return {'stamp': stamp, 'received': stamp, 'B_T_E': pose.tolist(), 'F_T_E': np.eye(4).tolist()}


def test_interpolates_image_time_and_shortest_rotation_across_wrap():
    before = state(100., (0, 0, 0), 170)
    after = state(100.08, (.08, .16, .24), -170)
    result = interpolate_state(before, after, 100.02)
    np.testing.assert_allclose(np.asarray(result['B_T_E'])[:3, 3], [.02, .04, .06], atol=1e-10)
    expected = Rotation.from_euler('z', 175, degrees=True).as_matrix()
    np.testing.assert_allclose(np.asarray(result['B_T_E'])[:3, :3], expected, atol=1e-10)
    assert result['bracket_gap_s'] == pytest.approx(.08)


@pytest.mark.parametrize('image_stamp,end_stamp', [(99.99, 100.08), (100.09, 100.08), (100.04, 100.5)])
def test_never_extrapolates_or_bridges_state_dropout(image_stamp, end_stamp):
    with pytest.raises(ValueError, match='bracketing'):
        interpolate_state(state(100.), state(end_stamp), image_stamp)


def test_rejects_changed_tool_frame():
    first, last = state(100), state(100.04)
    last['F_T_E'][0][3] = .01
    with pytest.raises(ValueError, match='EE definition changed'):
        interpolate_state(first, last, 100.02)


def test_waits_for_measured_state_after_the_image():
    history = StateHistory()
    assert history.add(state(100.))
    assert not history.add(state(100.))
    def produce():
        time.sleep(.01)
        history.add(state(100.04, (.04, 0, 0)))
    producer = threading.Thread(target=produce)
    producer.start()
    result = history.pair(100.02, wait_s=.2)
    producer.join()
    assert result['status'] == 'matched'
    assert result['B_T_E'][0][3] == pytest.approx(.02)
    assert history.pair(100.2, wait_s=0)['status'] == 'unmatched'
    assert StateHistory().pair(100, wait_s=0)['status'] == 'unmatched'


def record(index, stamp, xyz=(0., 0., 0.), matched=True):
    pose = state(stamp, xyz)
    return {'index': index, 'video_frame_index': index, 'timestamp_valid': True, 'image_stamp_unix_s': stamp, 'robot': {'left': {'status': 'matched' if matched else 'unmatched', 'B_T_E': pose['B_T_E'], 'F_T_E': pose['F_T_E'], 'state_image_gap_s': .02}}}


def test_camera_selection_does_not_require_inactive_wrist():
    assert list(select_cameras('left')) == ['left']
    assert list(select_cameras('right')) == ['right']
    assert list(select_cameras('both')) == ['left', 'right']
    assert list(select_cameras('left', ['left', 'front', 'left'])) == ['left', 'front']
    with pytest.raises(ValueError, match='required wrist'):
        select_cameras('left', ['right', 'front'])


def test_five_fps_selection_accepts_jitter_but_splits_a_dropped_frame():
    # 260 ms can occur at 5 FPS; an entire missing frame (400 ms) splits a stop.
    stamps = [100 + i * .2 for i in range(5)] + [101.06 + i * .2 for i in range(5)]
    assert len(select_windows([record(i, t) for i, t in enumerate(stamps)], 'left', fps=5)) == 1
    stamps = [100 + i * .2 for i in range(5)] + [101.2 + i * .2 for i in range(5)]
    assert select_windows([record(i, t) for i, t in enumerate(stamps)], 'left', fps=5) == []


def test_selects_stops_and_omits_moving_duplicate_or_unmatched_intervals():
    records = []
    for i in range(100):
        # Two different stops, one moving interval, then a repeated stop.
        xyz = (0, 0, 0) if i < 20 else (((i-20)*.002, 0, 0) if i < 40 else ((.1, 0, 0) if i < 65 else (0, 0, 0)))
        records.append(record(i, 100+i/15, xyz))
    windows = select_windows(records, 'left')
    assert len(windows) == 2
    assert all(8 <= len(w) <= 15 for w in windows)
    assert select_windows([record(i, 100+i/15, matched=False) for i in range(30)], 'left') == []


@pytest.mark.parametrize('fps', [5, 15])
def test_video_frame_indices_survive_lossless_recording_and_sample_export(tmp_path, fps):
    """Encode, decode, select, and verify that each exported image has its own pose."""
    directory = tmp_path / 'left'
    directory.mkdir()
    video_path = directory / 'color.mkv'
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'FFV1'), fps, (128, 96))
    assert writer.isOpened(), 'FFV1 lossless video support is required'
    records = []
    images = {}
    for i in range(60):
        image = np.full((96, 128, 3), [i, i+1, i+2], dtype=np.uint8)
        writer.write(image)
        images[i] = image
        records.append(record(i, 100+i/fps, (0, 0, 0) if i < 30 else (.1, 0, 0)))
    writer.release()
    (directory / 'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    manifest = {'format': 'real_exp_continuous_wrist_recording_v1', 'status': 'complete', 'cameras': {'left': {'serial': 'synthetic', 'color_intrinsics': {'width': 128, 'height': 96}, 'video_file': 'left/color.mkv', 'frames_file': 'left/frames.jsonl'}}, 'tag': {'family': 'tag36h11', 'id': 0, 'size_m': .095}, 'world_T_tag': np.eye(4).tolist()}
    (tmp_path / 'session.json').write_text(json.dumps(manifest))
    manifest['cameras']['left']['stream'] = [128, 96, fps]
    (tmp_path / 'session.json').write_text(json.dumps(manifest))
    script = Path(__file__).resolve().parents[1] / 'prepare_wrist_recording.py'
    result = subprocess.run([sys.executable, str(script), '--input', str(tmp_path), '--side', 'left'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    samples = sorted((tmp_path/'samples_left').glob('sample_*'))
    assert len(samples) == 2
    for sample in samples:
        for frame in json.loads((sample/'sample.json').read_text())['frames']:
            index = frame['source_video_frame_index']
            np.testing.assert_array_equal(np.load(sample/frame['rgb_file']), images[index])
            np.testing.assert_allclose(frame['B_T_E'], records[index]['robot']['left']['B_T_E'])
