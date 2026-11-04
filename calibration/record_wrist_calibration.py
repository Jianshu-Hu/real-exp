#!/usr/bin/env python3
"""Continuously record selected color cameras and measured EE states locally.

Every image retains its device timestamp and both arms' state matches. A match
uses bracketing measured robot states and interpolates to image capture time;
missing/stale/unbracketed data is explicitly marked instead of reused.

Robot-control host: scripts/start_data_collection_client.sh --left --arm
Camera host: /usr/bin/python3 calibration/record_wrist_calibration.py --side left
This recorder owns the cameras; it replaces the data-collection server launcher.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.calibrate_wrist_camera import rigid  # noqa: E402
from calibration.collect_camera_samples import pose_to_matrix  # noqa: E402
from calibration.collect_camera_world_data import WORLD_T_TAG, _intrinsics  # noqa: E402

CAMERAS = {'front': '401622071701', 'left': '260322270745', 'right': '260322272504'}


def select_cameras(side, names=None):
    """Default to the required wrist cameras; other cameras are opt-in."""
    required = ['left', 'right'] if side == 'both' else [side]
    selected = list(dict.fromkeys(required if names is None else names))
    if any(name not in selected for name in required):
        raise ValueError(f'Camera selection must include the required wrist cameras: {required}')
    return {name: CAMERAS[name] for name in selected}


def interpolate_state(before, after, stamp, max_gap_s=.1):
    """Interpolate only bracketed poses; never extrapolate or cross tool changes."""
    t0, t1 = before['stamp'], after['stamp']
    if not np.isfinite([t0, t1, stamp]).all() or not t0 <= stamp <= t1 or t1 - t0 > max_gap_s:
        raise ValueError('No robot states bracketing image time within allowed gap')
    first = rigid(before['B_T_E'], 'before B_T_E')
    last = rigid(after['B_T_E'], 'after B_T_E')
    tool = rigid(before['F_T_E'], 'F_T_E')
    if not np.allclose(tool, rigid(after['F_T_E'], 'F_T_E'), atol=1e-6):
        raise ValueError('Controller EE definition changed between states')
    alpha = 0. if t0 == t1 else (stamp - t0) / (t1 - t0)
    relative = Rotation.from_matrix(first[:3, :3].T @ last[:3, :3])
    pose = np.eye(4)
    pose[:3, :3] = first[:3, :3] @ Rotation.from_rotvec(relative.as_rotvec() * alpha).as_matrix()
    pose[:3, 3] = first[:3, 3] * (1 - alpha) + last[:3, 3] * alpha
    return {'status': 'matched', 'method': 'linear_translation_slerp_rotation', 'B_T_E': pose.tolist(), 'F_T_E': tool.tolist(), 'before_stamp_unix_s': t0, 'after_stamp_unix_s': t1, 'bracket_gap_s': t1 - t0, 'state_image_gap_s': max(stamp - t0, t1 - stamp)}


class StateHistory:
    def __init__(self):
        self.condition = threading.Condition()
        self.states = deque(maxlen=10000)

    def add(self, state):
        with self.condition:
            if self.states and state['stamp'] <= self.states[-1]['stamp']:
                return False
            self.states.append(state)
            self.condition.notify_all()
            return True

    def pair(self, stamp, *, wait_s=.15, max_gap_s=.1):
        deadline = time.monotonic() + wait_s
        with self.condition:
            while True:
                values = list(self.states)
                stamps = [s['stamp'] for s in values]
                index = bisect_right(stamps, stamp)
                if index and stamps[index - 1] == stamp:
                    return interpolate_state(values[index - 1], values[index - 1], stamp, max_gap_s)
                if 0 < index < len(values):
                    try:
                        return interpolate_state(values[index - 1], values[index], stamp, max_gap_s)
                    except ValueError as exc:
                        return {'status': 'unmatched', 'reason': str(exc)}
                if time.monotonic() >= deadline or (values and stamp < values[0]['stamp']):
                    return {'status': 'unmatched', 'reason': 'Missing measured states bracketing capture time'}
                self.condition.wait(min(.02, max(0., deadline - time.monotonic())))

    def fresh(self):
        with self.condition:
            return bool(self.states) and time.time() - self.states[-1]['received'] < .25 and abs(time.time() - self.states[-1]['stamp']) < .25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=['right', 'left', 'both'], default='right', help='Required measured robot states and default wrist camera selection')
    parser.add_argument('--cameras', nargs='+', choices=list(CAMERAS), help='Default: selected wrist camera(s). Optionally include front and/or the other wrist')
    parser.add_argument('--output', type=Path, help='Default: calibration/runs/wrist_recording_TIMESTAMP')
    parser.add_argument('--duration-s', type=float, default=0., help='0 records until Ctrl-C')
    parser.add_argument('--wait-for-state-s', type=float, default=30.)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=5, help='Camera capture rate (default: 5); each selected camera must support this profile')
    parser.add_argument('--video-codec', choices=['FFV1', 'MJPG'], default='FFV1', help='FFV1 is lossless Matroska; MJPG is smaller AVI')
    parser.add_argument('--tag-size-m', type=float, default=.095)
    parser.add_argument('--tag-family', choices=['tag36h11', 'tag25h9', 'tag16h5'], default='tag36h11')
    parser.add_argument('--tag-id', type=int, default=0)
    args = parser.parse_args()
    if not np.isfinite([args.duration_s, args.wait_for_state_s, args.tag_size_m]).all() or args.duration_s < 0 or args.wait_for_state_s <= 0 or args.tag_size_m <= 0 or min(args.width, args.height, args.fps) <= 0:
        parser.error('Invalid duration, timeout, tag size, or stream configuration')
    try:
        cameras = select_cameras(args.side, args.cameras)
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output or Path('calibration/runs') / ('wrist_recording_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output.resolve() in [Path('/left'), Path('/right'), Path('/both')]:
        parser.error(f'Output {output} looks like an unset $calib_run variable. Set calib_run to a directory under calibration/runs, or omit --output for an automatically named session.')
    if output.exists():
        parser.error(f'Refusing to overwrite {output}')
    print(f'Recording output: {output.resolve()}', flush=True)
    import cv2
    import pyrealsense2 as rs
    import rclpy
    from franka_msgs.msg import FrankaRobotState
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data

    # Check profile support before creating output or waiting for the robot.
    for name, serial in cameras.items():
        try:
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
            config.resolve(rs.pipeline_wrapper(rs.pipeline()))
        except RuntimeError as exc:
            parser.error(f'{name} camera ({serial}) cannot resolve {args.width}x{args.height}@{args.fps}: {exc}')

    cv2.setNumThreads(1)
    required = ['left', 'right'] if args.side == 'both' else [args.side]
    histories = {side: StateHistory() for side in ['left', 'right']}
    stop = threading.Event()
    manifest_lock = threading.Lock()
    error_lock = threading.Lock()
    errors = []
    manifest = {'format': 'real_exp_continuous_wrist_recording_v1', 'created_unix_s': time.time(), 'required_robot_sides': required, 'tag': {'family': args.tag_family, 'id': args.tag_id, 'size_m': args.tag_size_m}, 'world_T_tag': WORLD_T_TAG.tolist(), 'world_definition': 'Fixed tabletop tag centre, +X forward, +Y left, +Z up', 'ee_definition': 'Measured controller EE, Franka o_t_ee; fixed F_T_E required', 'camera_frames': 'color optical frames, +X right, +Y down, +Z forward', 'video_codec': args.video_codec, 'cameras': {}, 'state_topics': {side: f'/{side}/franka_robot_state_broadcaster/robot_state' for side in histories}, 'timestamp_policy': 'RealSense global/system capture timestamps in Unix seconds; host clocks must be synchronized; interpolate only bracketed measured robot states. Video playback uses nominal FPS; frames.jsonl holds the actual capture times.', 'status': 'starting'}
    manifest['selected_cameras'] = list(cameras)
    ready = {name: threading.Event() for name in cameras}
    stats = {name: {'frames': 0, 'matched': {side: 0 for side in histories}, 'timestamp_rejections': 0} for name in cameras}
    raw_files = {}
    raw_counts = {side: 0 for side in histories}
    rclpy.init()
    node = Node('continuous_wrist_calibration_recorder')

    def callback(side, msg):
        stamp = msg.o_t_ee.header.stamp
        seconds = stamp.sec + stamp.nanosec * 1e-9
        received = time.time()
        if seconds <= 0 or abs(received - seconds) > .25:
            return
        try:
            state = {'stamp': seconds, 'received': received, 'B_T_E': rigid(pose_to_matrix(msg.o_t_ee.pose), 'B_T_E').tolist(), 'F_T_E': rigid(pose_to_matrix(msg.f_t_ee.pose), 'F_T_E').tolist()}
        except (ValueError, AttributeError):
            return
        if histories[side].add(state) and side in raw_files:
            raw_files[side].write(json.dumps(state) + '\n')
            raw_counts[side] += 1

    for side in histories:
        node.create_subscription(FrankaRobotState, manifest['state_topics'][side], lambda msg, side=side: callback(side, msg), qos_profile_sensor_data)

    def spin():
        try:
            while not stop.is_set() and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=.02)
        except Exception as exc:
            if not stop.is_set() and rclpy.ok():
                with error_lock:
                    errors.append(f'ROS state reception: {exc}')
                stop.set()

    def camera_worker(name, serial):
        pipe = rs.pipeline()
        started = False
        writer = None
        phase = 'starting stream'
        try:
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
            profile = pipe.start(config)
            started = True
            device = profile.get_device()
            for sensor in device.query_sensors():
                if sensor.supports(rs.option.global_time_enabled):
                    sensor.set_option(rs.option.global_time_enabled, 1)
            directory = output / name
            directory.mkdir()
            video_file = 'color.mkv' if args.video_codec == 'FFV1' else 'color.avi'
            writer = cv2.VideoWriter(str(directory / video_file), cv2.VideoWriter_fourcc(*args.video_codec), args.fps, (args.width, args.height))
            if not writer.isOpened():
                raise RuntimeError(f'Cannot open {args.video_codec} video encoder')
            metadata = {'serial': serial, 'name': device.get_info(rs.camera_info.name), 'color_intrinsics': _intrinsics(profile.get_stream(rs.stream.color).as_video_stream_profile()), 'stream': [args.width, args.height, args.fps], 'frames_file': f'{name}/frames.jsonl', 'video_file': f'{name}/{video_file}'}
            with manifest_lock:
                manifest['cameras'][name] = metadata
            with (directory / 'frames.jsonl').open('x', buffering=1) as index_file:
                # Warm-up also allows global timestamps and auto-exposure to settle.
                phase = 'warming up'
                for _ in range(max(5, args.fps * 3)):
                    if stop.is_set():
                        return
                    pipe.wait_for_frames(5000)
                phase = 'recording'
                while not stop.is_set():
                    frames = pipe.wait_for_frames(5000)
                    color = frames.get_color_frame()
                    if not color:
                        continue
                    received = time.time()
                    raw_stamp_ms = float(color.get_timestamp())
                    domain = color.get_frame_timestamp_domain()
                    stamp = raw_stamp_ms / 1000
                    valid_stamp = domain in [rs.timestamp_domain.system_time, rs.timestamp_domain.global_time] and abs(received - stamp) <= .25
                    image = np.asanyarray(color.get_data()).copy()
                    matches = {}
                    # Check the required arm first; inactive arms never delay frames.
                    for side in required + [side for side in histories if side not in required]:
                        if valid_stamp:
                            matches[side] = histories[side].pair(stamp, wait_s=.10 if histories[side].fresh() else 0.)
                        else:
                            matches[side] = {'status': 'unmatched', 'reason': 'Camera timestamp is not a current Unix capture timestamp'}
                        if matches[side]['status'] == 'matched':
                            stats[name]['matched'][side] += 1
                    number = stats[name]['frames']
                    writer.write(image)
                    record = {'index': number, 'video_frame_index': number, 'device_frame_number': color.get_frame_number(), 'camera_timestamp_ms': raw_stamp_ms, 'camera_timestamp_domain': str(domain), 'image_stamp_unix_s': stamp if valid_stamp else None, 'image_received_unix_s': received, 'timestamp_valid': valid_stamp, 'exposure_us': color.get_frame_metadata(rs.frame_metadata_value.actual_exposure) if color.supports_frame_metadata(rs.frame_metadata_value.actual_exposure) else None, 'robot': matches}
                    index_file.write(json.dumps(record) + '\n')
                    stats[name]['frames'] += 1
                    ready[name].set()
                    if not valid_stamp:
                        stats[name]['timestamp_rejections'] += 1
        except Exception as exc:
            with error_lock:
                errors.append(f'{name} camera ({serial}) while {phase}: {exc}. Check camera ownership and USB connection/bandwidth; use only the required wrist and a supported low capture rate.')
            stop.set()
        finally:
            if writer is not None:
                writer.release()
            if started:
                pipe.stop()

    spin_thread = threading.Thread(target=spin, name='robot_states')
    workers = []
    created = False
    try:
        spin_thread.start()
        print(f'Waiting for fresh measured EE states for {required}; keep the robot-control stack running.', flush=True)
        deadline = time.monotonic() + args.wait_for_state_s
        while not all(histories[side].fresh() for side in required):
            if time.monotonic() >= deadline or stop.wait(.1):
                raise RuntimeError('Required measured EE state stream is unavailable. Restore the robot broadcaster/DDS connection, then retry.')
        output.mkdir(parents=True, exist_ok=False)
        created = True
        for side in histories:
            raw_files[side] = (output / f'robot_{side}.jsonl').open('x', buffering=1)
        (output / 'session.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'Saving to {output}; warming up {list(cameras)} at {args.width}x{args.height}@{args.fps}. Keep the arm still.', flush=True)
        for name, serial in cameras.items():
            worker = threading.Thread(target=camera_worker, args=(name, serial), name=f'camera_{name}')
            workers.append(worker)
            worker.start()
            # Stagger startup so devices do not negotiate streams simultaneously.
            deadline = time.monotonic() + 30
            while not ready[name].is_set():
                if stop.wait(.1):
                    raise RuntimeError(f'{name} camera did not become ready')
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'{name} camera startup exceeded 30 seconds')
            print(f'{name} camera is saving frames.', flush=True)
        started = time.monotonic()
        next_report = started
        print(f'All selected cameras are saving frames. Check increasing matched counts for {required}.', flush=True)
        print('Move slowly, pause 3 seconds at varied poses; Ctrl-C finishes recording.', flush=True)
        while not stop.wait(.1):
            elapsed = time.monotonic() - started
            if args.duration_s and elapsed >= args.duration_s:
                break
            if time.monotonic() >= next_report:
                with manifest_lock:
                    manifest['status'] = 'recording'
                    (output / 'session.json').write_text(json.dumps(manifest, indent=2) + '\n')
                status = {'elapsed_s': round(elapsed, 1), 'cameras': stats, 'robot_states_received': raw_counts, 'required_states_fresh': {side: histories[side].fresh() for side in required}}
                (output / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
                print(json.dumps(status), flush=True)
                next_report = time.monotonic() + 5
    except KeyboardInterrupt:
        print('Finishing recording...', flush=True)
    except Exception as exc:
        errors.append(str(exc))
    finally:
        stop.set()
        if spin_thread.is_alive():
            spin_thread.join(timeout=3)
        for worker in workers:
            worker.join(timeout=12)
            if worker.is_alive():
                errors.append(f'{worker.name} did not finish')
        for handle in raw_files.values():
            handle.close()
        if created:
            if not errors and any(stats[name]['frames'] == 0 for name in cameras):
                errors.append('Recording ended before every camera saved a frame')
            for side in required:
                if stats[side]['frames'] and not stats[side]['matched'][side]:
                    errors.append(f'No synchronized {side} wrist images and measured EE states; check DDS connectivity and host clock synchronization')
            manifest.update({'finished_unix_s': time.time(), 'status': 'failed' if errors else 'complete', 'errors': errors, 'stats': stats, 'robot_states_received': raw_counts})
            (output / 'session.json').write_text(json.dumps(manifest, indent=2) + '\n')
            print(f'Saved {output}; status={manifest["status"]}', flush=True)
        node.destroy_node()
        rclpy.try_shutdown()
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
