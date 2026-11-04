#!/usr/bin/env python3
"""Select distinct settled poses from continuous video for wrist calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.calibrate_wrist_camera import rigid  # noqa: E402


def pose_distance(first, last):
    delta = np.linalg.inv(first) @ last
    return np.linalg.norm(delta[:3, 3]), Rotation.from_matrix(delta[:3, :3]).magnitude()


def select_windows(records, side, *, min_duration_s=.7, min_frames=8, max_frames=15, fps=15):
    """Select stationary, synchronized windows and omit duplicate arm poses."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Recorded camera FPS must be positive and finite')
    max_frame_gap_s = max(.25, 1.5 / fps)
    windows, current = [], []

    def finish():
        if len(current) < min_frames or current[-1]['image_stamp_unix_s'] - current[0]['image_stamp_unix_s'] < min_duration_s:
            return
        centre = rigid(current[len(current) // 2]['robot'][side]['B_T_E'], 'B_T_E')
        for previous in windows:
            other = rigid(previous[len(previous) // 2]['robot'][side]['B_T_E'], 'B_T_E')
            translation, rotation = pose_distance(other, centre)
            if translation < .01 and rotation < np.deg2rad(5):
                return
        indices = np.linspace(0, len(current) - 1, min(max_frames, len(current))).round().astype(int)
        windows.append([current[i] for i in indices])

    for record in records:
        match = record.get('robot', {}).get(side, {})
        stamp = record.get('image_stamp_unix_s')
        valid = record.get('timestamp_valid') and stamp is not None and np.isfinite(stamp) and match.get('status') == 'matched' and np.isfinite(match.get('state_image_gap_s', np.nan)) and 0 <= match.get('state_image_gap_s', 1) <= .1
        if not valid:
            finish(); current = []
            continue
        B = rigid(match['B_T_E'], 'B_T_E')
        if current:
            initial = current[0]['robot'][side]
            distance, rotation = pose_distance(rigid(initial['B_T_E'], 'B_T_E'), B)
            dt = stamp - current[-1]['image_stamp_unix_s']
            if not np.allclose(rigid(initial['F_T_E'], 'F_T_E'), rigid(match['F_T_E'], 'F_T_E'), atol=1e-6):
                raise ValueError('Controller EE definition changed during recording; use a new recording')
            if distance > .0015 or rotation > np.deg2rad(.3) or not 0 < dt <= max_frame_gap_s:
                finish(); current = []
        current.append(record)
    finish()
    return windows


def main():
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--side', choices=['left', 'right'], required=True)
    parser.add_argument('--output', type=Path, help='Default: INPUT/samples_SIDE')
    args = parser.parse_args()
    manifest = json.loads((args.input / 'session.json').read_text())
    if manifest.get('format') != 'real_exp_continuous_wrist_recording_v1':
        raise ValueError('Expected a record_wrist_calibration.py recording')
    if manifest.get('status') != 'complete':
        raise ValueError('Finish recording successfully with Ctrl-C before preparing calibration samples')
    output = args.output or args.input / f'samples_{args.side}'
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}')
    camera = manifest['cameras'][args.side]
    records = [json.loads(line) for line in (args.input / camera['frames_file']).read_text().splitlines() if line.strip()]
    windows = select_windows(records, args.side, fps=camera.get('stream', [0, 0, 15])[2])
    if not windows:
        raise ValueError('No distinct settled configurations with synchronized EE states. At 5 FPS, pause about three seconds at each pose for at least eight images.')
    expected_tool = rigid(windows[0][0]['robot'][args.side]['F_T_E'], 'F_T_E')
    if any(not np.allclose(rigid(f['robot'][args.side]['F_T_E'], 'F_T_E'), expected_tool, atol=1e-6) for window in windows for f in window):
        raise ValueError('Controller EE definition changed between selected configurations')
    output.mkdir(parents=True, exist_ok=False)
    session = {'format': 'real_exp_wrist_samples_v1', 'side': args.side, 'camera_serial': camera['serial'], 'color_intrinsics': camera['color_intrinsics'], 'tag': manifest['tag'], 'world_T_tag': manifest['world_T_tag'], 'F_T_E': expected_tool.tolist(), 'source_recording': str(args.input.resolve()), 'selection': 'Distinct stationary configurations; video and measured robot timestamps must be synchronized'}
    (output / 'session.json').write_text(json.dumps(session, indent=2) + '\n')
    wanted = {}
    for sample_id, window in enumerate(windows):
        directory = output / f'sample_{sample_id:06d}'
        directory.mkdir()
        frames = []
        for frame_id, frame in enumerate(window):
            image_name = f'rgb_{frame_id:03d}.npy'
            wanted[frame['video_frame_index']] = directory / image_name
            matched = frame['robot'][args.side]
            frames.append({'rgb_file': image_name, 'B_T_E': matched['B_T_E'], 'F_T_E': matched['F_T_E'], 'state_image_gap_s': matched['state_image_gap_s'], 'image_stamp_unix_s': frame['image_stamp_unix_s'], 'source_video_frame_index': frame['video_frame_index']})
        (directory / 'sample.json').write_text(json.dumps({'side': args.side, 'camera_serial': camera['serial'], 'sample_id': sample_id, 'frames': frames}, indent=2) + '\n')
    video = cv2.VideoCapture(str(args.input / camera['video_file']))
    if not video.isOpened():
        raise ValueError('Cannot read recorded video')
    saved = 0
    try:
        for i in range(max(wanted) + 1):
            ok, image = video.read()
            if not ok:
                raise ValueError(f'Video ended at frame {i}; timestamp index and video do not agree')
            if i in wanted:
                np.save(wanted[i], image)
                saved += 1
    finally:
        video.release()
    (output / 'selection.json').write_text(json.dumps({'source_frames': len(records), 'distinct_settled_poses': len(windows), 'exported_images': saved, 'recommended_poses': '15-20; last 3 used for held-out validation'}, indent=2) + '\n')
    print(f'Saved {len(windows)} distinct configurations ({saved} images) to {output}')
    print(f'Next: python calibration/calibrate_wrist_camera.py --input {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
