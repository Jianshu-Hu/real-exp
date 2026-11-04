#!/usr/bin/env python3
"""Record settled wrist-camera images and measured Franka EE poses on Enter.

Run on the camera host after sourcing ROS and the franka_msgs overlay. This
collector subscribes to robot states; it never commands the arm.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.collect_camera_world_data import WORLD_T_TAG, _intrinsics  # noqa: E402
from calibration.collect_camera_samples import pose_to_matrix  # noqa: E402
from calibration.calibrate_wrist_camera import rigid  # noqa: E402

SERIALS = {'left': '260322270745', 'right': '260322272504'}


def check_stationary(states, *, max_translation_m=.0015, max_rotation_deg=.3):
    """Require fresh states covering a settled interval, with no EE motion."""
    from scipy.spatial.transform import Rotation

    if len(states) < 3 or states[-1]['stamp'] - states[0]['stamp'] < .6:
        raise ValueError('Need at least 0.6 seconds of fresh robot states while settled')
    stamps = np.asarray([s['stamp'] for s in states])
    if np.any(np.diff(stamps) <= 0) or np.max(np.diff(stamps)) > .25:
        raise ValueError('Robot states are stale, repeated, or interrupted')
    matrices = np.asarray([rigid(s['B_T_E'], 'B_T_E') for s in states])
    offset = np.linalg.inv(matrices[0])[None, :, :] @ matrices
    if np.max(np.linalg.norm(offset[:, :3, 3], axis=1)) > max_translation_m:
        raise ValueError('Arm is moving: hold it still before pressing Enter')
    if np.max(Rotation.from_matrix(offset[:, :3, :3]).magnitude()) > np.deg2rad(max_rotation_deg):
        raise ValueError('Arm orientation is changing: hold it still')
    first_tool = rigid(states[0]['F_T_E'], 'F_T_E')
    if any(not np.allclose(s['F_T_E'], first_tool, atol=1e-6) for s in states):
        raise ValueError('Controller EE definition F_T_E changed during capture')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=['left', 'right'], required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--camera-serial', help='Default is the installed D405 for --side')
    parser.add_argument('--state-topic', help='Default: /SIDE/franka_robot_state_broadcaster/robot_state')
    parser.add_argument('--tag-size-m', type=float, default=.095)
    parser.add_argument('--tag-id', type=int, default=0)
    parser.add_argument('--tag-family', choices=['tag36h11', 'tag25h9', 'tag16h5'], default='tag36h11')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--frames-per-pose', type=int, default=15)
    parser.add_argument('--state-timeout-s', type=float, default=10)
    parser.add_argument('--check-state', action='store_true', help='Check measured EE state without opening a camera')
    args = parser.parse_args()
    if not args.check_state and args.output is None:
        parser.error('--output is required for collection')
    if not np.isfinite(args.tag_size_m) or args.tag_size_m <= 0 or args.frames_per_pose < 5 or min(args.width, args.height, args.fps) <= 0 or not np.isfinite(args.state_timeout_s) or args.state_timeout_s <= 0:
        parser.error('Invalid size, stream configuration, frame count, or timeout')
    if args.output and args.output.exists() and not args.check_state:
        parser.error('Output already exists; choose a new session directory')
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from franka_msgs.msg import FrankaRobotState
    except ImportError as exc:
        raise SystemExit(f'Source ROS and the Franka overlay first; missing {exc.name}') from exc

    rclpy.init()
    node = Node('wrist_camera_calibration_collector')
    states = deque(maxlen=3000)
    errors = deque(maxlen=1)
    topic = args.state_topic or f'/{args.side}/franka_robot_state_broadcaster/robot_state'

    def callback(msg):
        try:
            stamp = msg.o_t_ee.header.stamp
            seconds = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            if seconds <= 0:
                raise ValueError('Robot EE state has no valid source timestamp')
            if states and seconds <= states[-1]['stamp']:
                return
            states.append({'stamp': seconds, 'received': time.time(), 'B_T_E': rigid(pose_to_matrix(msg.o_t_ee.pose), 'B_T_E').tolist(), 'F_T_E': rigid(pose_to_matrix(msg.f_t_ee.pose), 'F_T_E').tolist()})
        except (ValueError, AttributeError) as exc:
            errors.append(str(exc))

    node.create_subscription(FrankaRobotState, topic, callback, qos_profile_sensor_data)
    pipe = None
    try:
        print(f'Waiting for measured EE states: {topic}', flush=True)
        deadline = time.monotonic() + args.state_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            if states and time.time() - states[-1]['received'] < .25 and abs(time.time() - states[-1]['stamp']) < .25 and len(states) >= 3:
                break
        else:
            raise SystemExit(f'No fresh measured robot states on {topic}. Start the robot state broadcaster, match ROS_DOMAIN_ID/network, and synchronize host clocks. {list(errors)}')
        print('Fresh measured EE state received. B_T_E =\n', np.asarray(states[-1]['B_T_E']), flush=True)
        if args.check_state:
            return 0
        import pyrealsense2 as rs

        pipe = rs.pipeline()
        config = rs.config()
        serial = args.camera_serial or SERIALS[args.side]
        config.enable_device(serial)
        # Color alone supports D405 HD mode; depth is not needed for tag PnP.
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
        profile = pipe.start(config)
        for sensor in profile.get_device().query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {'format': 'real_exp_wrist_samples_v1', 'side': args.side, 'camera_serial': serial, 'camera_name': profile.get_device().get_info(rs.camera_info.name), 'created_unix_s': time.time(), 'state_topic': topic, 'tag': {'family': args.tag_family, 'id': args.tag_id, 'size_m': args.tag_size_m}, 'world_T_tag': WORLD_T_TAG.tolist(), 'color_intrinsics': _intrinsics(profile.get_stream(rs.stream.color).as_video_stream_profile()), 'stream': [args.width, args.height, args.fps], 'F_T_E': states[-1]['F_T_E'], 'world_definition': 'Fixed tabletop tag centre, X forward, Y left, Z up', 'ee_definition': 'Measured controller EE from Franka o_t_ee; F_T_E must remain fixed'}
        (args.output / 'session.json').write_text(json.dumps(manifest, indent=2) + '\n')
        for _ in range(45):
            pipe.wait_for_frames(5000)
            rclpy.spin_once(node, timeout_sec=0)
        print('Move to a varied pose, hold still, then press Enter to record. Type q to finish. Keep the tabletop tag fixed.', flush=True)
        index = 0
        while rclpy.ok():
            import select

            rclpy.spin_once(node, timeout_sec=.02)
            if not select.select([sys.stdin], [], [], 0)[0]:
                continue
            line = sys.stdin.readline()
            if not line or line.strip().lower() in ['q', 'quit', 'exit']:
                break
            try:
                # Empty the camera queue and collect fresh frames while spinning ROS.
                for _ in range(100):
                    if not pipe.poll_for_frames():
                        break
                recent = [s for s in states if s['stamp'] >= time.time() - 1.0]
                check_stationary(recent)
                if not np.allclose(recent[-1]['F_T_E'], manifest['F_T_E'], atol=1e-6):
                    raise ValueError('F_T_E changed since session start; restart calibration')
                frames, images = [], []
                start = time.time()
                for i in range(args.frames_per_pose):
                    color = pipe.wait_for_frames(5000).get_color_frame()
                    arrival = time.time()
                    image = np.asanyarray(color.get_data()).copy()
                    # Allow a state after the image, then pair with nearest source timestamp.
                    end = time.monotonic() + .03
                    while time.monotonic() < end:
                        rclpy.spin_once(node, timeout_sec=.005)
                    stamp = float(color.get_timestamp()) / 1000
                    domain = color.get_frame_timestamp_domain()
                    if domain not in [rs.timestamp_domain.system_time, rs.timestamp_domain.global_time] or abs(arrival - stamp) > .25:
                        raise ValueError('Camera global/system timestamp unavailable or stale; wait and retry')
                    state = min(states, key=lambda s: abs(s['stamp'] - stamp))
                    gap = abs(state['stamp'] - stamp)
                    if gap > .1 or time.time() - state['received'] > .25:
                        raise ValueError('Robot state is stale or more than 100 ms from image; sample discarded')
                    frames.append({'rgb_file': f'rgb_{i:03d}.npy', 'B_T_E': state['B_T_E'], 'F_T_E': state['F_T_E'], 'image_stamp_unix_s': stamp, 'image_received_unix_s': arrival, 'robot_state_stamp_unix_s': state['stamp'], 'state_image_gap_s': gap, 'camera_timestamp_domain': str(domain)})
                    images.append(image)
                check_stationary([s for s in states if s['stamp'] >= start - .7])
                directory = args.output / f'sample_{index:06d}'
                directory.mkdir(exist_ok=False)
                for frame, image in zip(frames, images):
                    np.save(directory / frame['rgb_file'], image)
                (directory / 'sample.json').write_text(json.dumps({'side': args.side, 'camera_serial': serial, 'sample_id': index, 'captured_unix_s': start, 'frames': frames}, indent=2) + '\n')
                print(f'Saved {directory}: {len(frames)} synchronized images; move to the next pose.', flush=True)
                index += 1
            except (ValueError, RuntimeError) as exc:
                print(f'Sample rejected: {exc}', flush=True)
        return 0
    finally:
        if pipe is not None:
            try:
                pipe.stop()
            except RuntimeError:
                pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
