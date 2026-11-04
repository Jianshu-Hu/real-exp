#!/usr/bin/env python3
"""Solve W_T_B and E_T_D for a wrist camera observing a fixed tabletop tag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration.calibrate_camera_to_world import (  # noqa: E402
    PHYSICAL_TAG_T_PNP_TAG, dominant_pose_cluster, invert,
    world_t_tag_from_metadata,
)


def rigid(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} has an invalid bottom row")
    R = matrix[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1):
        raise ValueError(f"{name} must have a proper rotation")
    return matrix


def estimate_tag_pose(image, intrinsics, tag, world_T_tag):
    """Return a color-camera pose in W and its corner reprojection error."""
    import cv2

    names = {"tag36h11": "DICT_APRILTAG_36h11", "tag25h9": "DICT_APRILTAG_25h9", "tag16h5": "DICT_APRILTAG_16h5"}
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, names[tag['family']]))
    # OpenCV 4.6 exposes a broken direct constructor; its factory is required.
    parameters = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'ArucoDetector') else cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, 'ArucoDetector'):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, parameters).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    if ids is None or tag['id'] not in ids:
        return None
    corners = np.asarray(corners[int(np.flatnonzero(ids.reshape(-1) == tag['id'])[0])], dtype=float).reshape(4, 2)
    if image.shape[:2] != (intrinsics['height'], intrinsics['width']):
        raise ValueError('Image dimensions do not match recorded intrinsics')
    K = np.array([[intrinsics['fx'], 0, intrinsics['cx']], [0, intrinsics['fy'], intrinsics['cy']], [0, 0, 1.]])
    distortion = np.asarray(intrinsics['coeffs'], dtype=float)
    half = tag['size_m'] / 2
    points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]])
    ok, rvec, tvec = cv2.solvePnP(points, corners, K, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or float(tvec[2, 0]) <= 0:
        return None
    camera_T_pnp = np.eye(4)
    camera_T_pnp[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_T_pnp[:3, 3] = tvec.reshape(3)
    projected = cv2.projectPoints(points, rvec, tvec, K, distortion)[0].reshape(4, 2)
    rmse = float(np.sqrt(np.mean(np.sum((projected - corners) ** 2, axis=1))))
    return world_T_tag @ PHYSICAL_TAG_T_PNP_TAG @ invert(camera_T_pnp), rmse


def solve_transforms(base_T_ee, world_T_camera):
    """Solve W_T_D_i = W_T_B @ B_T_E_i @ E_T_D; reject unobservable motion."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    B = np.asarray([rigid(v, 'base_T_ee') for v in base_T_ee])
    W = np.asarray([rigid(v, 'world_T_camera') for v in world_T_camera])
    if len(B) != len(W) or len(B) < 4:
        raise ValueError('Need at least four paired poses')
    motions = Rotation.from_matrix(B[0, :3, :3].T @ B[:, :3, :3]).as_rotvec()
    excitation = np.linalg.svd(motions, compute_uv=False)
    if excitation[1] < 0.10:
        raise ValueError('Insufficient rotation diversity: rotate around at least two axes, by about 15-30 degrees')
    camera_T_world = np.asarray([invert(v) for v in W])
    R, t = cv2.calibrateHandEye(
        list(B[:, :3, :3]), list(B[:, :3, 3]),
        list(camera_T_world[:, :3, :3]), list(camera_T_world[:, :3, 3]),
        method=cv2.CALIB_HAND_EYE_PARK,
    )
    ee_T_camera = np.eye(4)
    ee_T_camera[:3, :3], ee_T_camera[:3, 3] = R, np.asarray(t).reshape(3)
    rigid(ee_T_camera, 'initial ee_T_camera')
    candidates = np.asarray([w @ invert(ee_T_camera) @ invert(b) for b, w in zip(B, W)])
    world_T_base = np.eye(4)
    world_T_base[:3, :3] = Rotation.from_matrix(candidates[:, :3, :3]).mean().as_matrix()
    world_T_base[:3, 3] = np.median(candidates[:, :3, 3], axis=0)

    def pack(T):
        return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]

    def unpack(x):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
        T[:3, 3] = x[3:]
        return T

    def residual(x):
        world_base, ee_camera = unpack(x[:6]), unpack(x[6:])
        delta = np.asarray([invert(w) @ world_base @ b @ ee_camera for b, w in zip(B, W)])
        # Translation and rotation normalized by useful calibration tolerances.
        return np.c_[delta[:, :3, 3] / .01, Rotation.from_matrix(delta[:, :3, :3]).as_rotvec() / np.deg2rad(2)].ravel()

    fit = least_squares(residual, np.r_[pack(world_T_base), pack(ee_T_camera)], loss='soft_l1', max_nfev=1000)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if not fit.success or singular[-1] / singular[0] < 1e-5:
        raise ValueError('Calibration is unobservable or did not converge; collect more varied poses')
    return unpack(fit.x[:6]), unpack(fit.x[6:]), {'jacobian_singular_values': singular.tolist(), 'rotation_excitation_singular_values_rad': excitation.tolist()}


def pose_residual(world_T_base, base_T_ee, ee_T_camera, observed_world_T_camera):
    from scipy.spatial.transform import Rotation

    delta = invert(observed_world_T_camera) @ world_T_base @ base_T_ee @ ee_T_camera
    return float(np.linalg.norm(delta[:3, 3])), float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude()))


def main():
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--front-calibration', type=Path, help='Front camera_to_world.json for composing front_camera_T_base')
    parser.add_argument('--min-samples', type=int, default=10)
    parser.add_argument('--holdout-samples', type=int, default=3)
    args = parser.parse_args()
    if args.min_samples < 6 or args.holdout_samples < 1:
        parser.error('min-samples must be >=6 and holdout-samples must be >=1')
    manifest = json.loads((args.input / 'session.json').read_text())
    if manifest.get('format') != 'real_exp_wrist_samples_v1':
        raise ValueError('Expected a collect_wrist_camera_samples.py session')
    world_T_tag = world_t_tag_from_metadata(manifest)
    tag = manifest['tag']
    if not np.isfinite(tag['size_m']) or tag['size_m'] <= 0:
        raise ValueError('Tag size must be positive')
    output = args.output or args.input / 'wrist_calibration.json'
    if output.exists():
        raise ValueError(f'Refusing to overwrite {output}; choose --output')
    front = None
    if args.front_calibration:
        front = json.loads(args.front_calibration.read_text())
        if front['tag'] != tag or not np.allclose(world_t_tag_from_metadata(front), world_T_tag):
            raise ValueError('Front calibration has different tag parameters or world axes')
        rigid(front['world_T_camera'], 'front world_T_camera')
    accepted, rejected = [], []
    for path in sorted(args.input.glob('sample_*/sample.json')):
        sample = json.loads(path.read_text())
        if sample['side'] != manifest['side'] or sample['camera_serial'] != manifest['camera_serial']:
            raise ValueError(f'Camera/arm mismatch in {path}')
        poses, errors, robot_poses = [], [], []
        for frame in sample['frames']:
            if not np.isfinite(frame['state_image_gap_s']) or not 0 <= frame['state_image_gap_s'] <= .1:
                raise ValueError(f'Unsynchronized robot state in {path}')
            if not np.allclose(rigid(frame['F_T_E'], 'F_T_E'), rigid(manifest['F_T_E'], 'session F_T_E'), atol=1e-6):
                raise ValueError(f'Controller EE definition changed in {path}')
            result = estimate_tag_pose(np.load(path.parent / frame['rgb_file']), manifest['color_intrinsics'], tag, world_T_tag)
            if result is None or result[1] > .75:
                continue
            poses.append(result[0]); errors.append(result[1]); robot_poses.append(rigid(frame['B_T_E'], 'B_T_E'))
        if len(poses) < 5:
            rejected.append({'sample': path.parent.name, 'reason': 'fewer than five clear tag observations'})
            continue
        robot_delta = np.linalg.inv(robot_poses[0])[None, :, :] @ np.asarray(robot_poses)
        if np.max(np.linalg.norm(robot_delta[:, :3, 3], axis=1)) > .0015 or np.max(Rotation.from_matrix(robot_delta[:, :3, :3]).magnitude()) > np.deg2rad(.3):
            rejected.append({'sample': path.parent.name, 'reason': 'robot moved during capture'})
            continue
        indices = dominant_pose_cluster(np.asarray(poses), Rotation, translation_threshold_m=.01, rotation_threshold_rad=np.deg2rad(1))
        if len(indices) < 5:
            rejected.append({'sample': path.parent.name, 'reason': 'unstable tag pose'})
            continue
        def aggregate(values):
            T = np.eye(4)
            selected = np.asarray(values)[indices]
            T[:3, :3] = Rotation.from_matrix(selected[:, :3, :3]).mean().as_matrix()
            T[:3, 3] = np.median(selected[:, :3, 3], axis=0)
            return T
        accepted.append({'sample_id': path.parent.name, 'world_T_camera': aggregate(poses), 'base_T_ee': aggregate(robot_poses), 'median_reprojection_rmse_px': float(np.median(np.asarray(errors)[indices])), 'cluster_frames': len(indices)})
    if len(accepted) < args.min_samples or len(accepted) - args.holdout_samples < 6:
        raise ValueError(f'Only {len(accepted)} valid settled poses; need >= {args.min_samples} and >=6 training poses after holdout; rejected={rejected}')
    train, holdout = accepted[:-args.holdout_samples], accepted[-args.holdout_samples:]
    world_T_base, ee_T_camera, diagnostics = solve_transforms([s['base_T_ee'] for s in train], [s['world_T_camera'] for s in train])
    samples = []
    for index, s in enumerate(accepted):
        translation, rotation = pose_residual(world_T_base, s['base_T_ee'], ee_T_camera, s['world_T_camera'])
        samples.append({**{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}, 'split': 'train' if index < len(train) else 'holdout', 'translation_residual_m': translation, 'rotation_residual_deg': rotation, 'base_T_camera': (s['base_T_ee'] @ ee_T_camera).tolist()})
    quality = {}
    for split in ['train', 'holdout']:
        group = [s for s in samples if s['split'] == split]
        quality[split] = {'count': len(group), 'median_translation_m': float(np.median([s['translation_residual_m'] for s in group])), 'max_translation_m': max(s['translation_residual_m'] for s in group), 'median_rotation_deg': float(np.median([s['rotation_residual_deg'] for s in group])), 'max_rotation_deg': max(s['rotation_residual_deg'] for s in group)}
    passed = all(q['median_translation_m'] <= .015 and q['max_translation_m'] <= .03 and q['median_rotation_deg'] <= 3 and q['max_rotation_deg'] <= 6 for q in quality.values())
    result = {'format': 'real_exp_wrist_calibration_v1', 'status': 'validated_on_holdout' if passed else 'failed_residual_limits', 'input': str(args.input), 'side': manifest['side'], 'camera_serial': manifest['camera_serial'], 'tag': tag, 'world_T_tag': world_T_tag.tolist(), 'ee_frame': 'controller EE from Franka o_t_ee; fixed F_T_E required throughout collection', 'world_T_base': world_T_base.tolist(), 'ee_T_camera': ee_T_camera.tolist(), 'camera_T_ee': invert(ee_T_camera).tolist(), 'base_T_camera_definition': 'B_T_D(q) = B_T_E(q) @ E_T_D; depends on arm pose', 'quality': quality, 'diagnostics': diagnostics, 'samples': samples, 'rejected_samples': rejected}
    if front:
        result['front_camera_T_base'] = (invert(np.asarray(front['world_T_camera'])) @ world_T_base).tolist()
        result['front_calibration'] = str(args.front_calibration)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {output}: {result["status"]}; {len(train)} training / {len(holdout)} held-out poses')
    for name in ['world_T_base', 'ee_T_camera']:
        print(name, '\n', np.asarray(result[name]))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
