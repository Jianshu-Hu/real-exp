"""Checks of transform direction, identifiability, and capture motion rejection."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration.calibrate_camera_to_world import invert
from calibration.calibrate_wrist_camera import pose_residual, rigid, solve_transforms
from calibration.collect_wrist_camera_samples import check_stationary


def transform(rpy, xyz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', rpy).as_matrix()
    T[:3, 3] = xyz
    return T


@pytest.mark.parametrize('noisy', [False, True])
def test_recovers_base_and_mount_and_predicts_unseen_arm_poses(noisy):
    rng = np.random.default_rng(371)
    world_T_base = transform([.3, -.2, 1.1], [.25, -.45, .12])
    ee_T_camera = transform([1.2, -.3, .7], [.04, -.02, .09])
    robot = [transform(rng.uniform(-1, 1, 3), rng.uniform(-.6, .6, 3)) for _ in range(24)]
    observed = [world_T_base @ B @ ee_T_camera for B in robot]
    if noisy:
        observed = [T @ transform(rng.normal(0, .001, 3), rng.normal(0, .0005, 3)) for T in observed]
    fitted_base, fitted_mount, _ = solve_transforms(robot[:20], observed[:20])
    translation_tolerance = .002 if noisy else 1e-8
    rotation_tolerance_deg = .2 if noisy else 1e-5
    for fitted, truth in [(fitted_base, world_T_base), (fitted_mount, ee_T_camera)]:
        error = invert(truth) @ fitted
        assert np.linalg.norm(error[:3, 3]) < translation_tolerance
        assert np.rad2deg(Rotation.from_matrix(error[:3, :3]).magnitude()) < rotation_tolerance_deg
    for B, W in zip(robot[20:], observed[20:]):
        t, r = pose_residual(fitted_base, B, fitted_mount, W)
        assert t < translation_tolerance
        assert r < rotation_tolerance_deg
        # Composition must map camera points through the moving base correctly.
        point_camera = np.array([.03, -.02, .5, 1])
        np.testing.assert_allclose(fitted_base @ (B @ fitted_mount @ point_camera), W @ point_camera, atol=.003 if noisy else 1e-8)


@pytest.mark.parametrize('rotating', [False, True])
def test_rejects_translation_only_or_single_axis_data(rotating):
    robot = [transform([0, 0, .1*i if rotating else 0], [i*.02, 0, .4]) for i in range(15)]
    with pytest.raises(ValueError, match='rotation diversity'):
        solve_transforms(robot, robot)


def test_stationarity_and_source_timestamp_checks():
    states = [{'stamp': 100+i*.01, 'B_T_E': np.eye(4).tolist(), 'F_T_E': np.eye(4).tolist()} for i in range(101)]
    check_stationary(states)
    moved = [{**s, 'B_T_E': transform([0, 0, 0], [i*.0001, 0, 0]).tolist()} for i, s in enumerate(states)]
    with pytest.raises(ValueError, match='moving'):
        check_stationary(moved)
    rotated = [{**s, 'B_T_E': transform([i*.0001, 0, 0], [0, 0, 0]).tolist()} for i, s in enumerate(states)]
    with pytest.raises(ValueError, match='orientation'):
        check_stationary(rotated)
    with pytest.raises(ValueError, match='stale'):
        check_stationary(states[:10] + states[80:])
    with pytest.raises(ValueError, match='0.6 seconds'):
        check_stationary(states[:5])


def test_rejects_nonrigid_transform():
    T = np.eye(4)
    T[0, 0] = -1
    with pytest.raises(ValueError, match='proper rotation'):
        rigid(T, 'camera')


def test_image_samples_through_solver_cli(tmp_path):
    """Render actual tags and solve through detection, clustering, holdout and JSON."""
    import json
    from pathlib import Path
    import subprocess
    import sys
    import cv2
    from calibration.calibrate_camera_to_world import DEFAULT_WORLD_T_TAG, PHYSICAL_TAG_T_PNP_TAG

    rng = np.random.default_rng(71)
    intrinsics = {'width': 1280, 'height': 720, 'fx': 900., 'fy': 900., 'cx': 640., 'cy': 360., 'coeffs': [0.] * 5}
    manifest = {'format': 'real_exp_wrist_samples_v1', 'side': 'right', 'camera_serial': 'synthetic', 'color_intrinsics': intrinsics, 'F_T_E': np.eye(4).tolist(), 'world_T_tag': DEFAULT_WORLD_T_TAG.tolist(), 'tag': {'family': 'tag36h11', 'id': 0, 'size_m': .095}}
    (tmp_path / 'session.json').write_text(json.dumps(manifest))
    truth_base = transform([.1, -.15, .7], [.2, -.4, .1])
    truth_mount = transform([-.3, .2, 1.2], [.03, -.04, .08])
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    draw_marker = getattr(cv2.aruco, 'generateImageMarker', None) or cv2.aruco.drawMarker
    marker = draw_marker(dictionary, 0, 400)
    half = .095 / 2
    points = np.array([[-half, half, 0.], [half, half, 0.], [half, -half, 0.], [-half, -half, 0.]])
    K = np.array([[900., 0, 640.], [0, 900., 360.], [0, 0, 1.]])
    for i in range(15):
        camera_T_pnp = transform([np.pi + rng.uniform(-.4, .4), rng.uniform(-.4, .4), rng.uniform(-.5, .5)], [rng.uniform(-.05, .05), rng.uniform(-.04, .04), rng.uniform(.45, .65)])
        W_D = DEFAULT_WORLD_T_TAG @ PHYSICAL_TAG_T_PNP_TAG @ invert(camera_T_pnp)
        B_E = invert(truth_base) @ W_D @ invert(truth_mount)
        rvec = cv2.Rodrigues(camera_T_pnp[:3, :3])[0]
        projected = cv2.projectPoints(points, rvec, camera_T_pnp[:3, 3], K, np.zeros(5))[0].reshape(4, 2).astype(np.float32)
        homography = cv2.getPerspectiveTransform(np.array([[0., 0.], [399., 0.], [399., 399.], [0., 399.]], dtype=np.float32), projected)
        gray = cv2.warpPerspective(marker, homography, (1280, 720), borderValue=255)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        directory = tmp_path / f'sample_{i:06d}'
        directory.mkdir()
        np.save(directory / 'rgb.npy', image)
        frames = [{'rgb_file': 'rgb.npy', 'B_T_E': B_E.tolist(), 'F_T_E': np.eye(4).tolist(), 'state_image_gap_s': .001}] * 5
        (directory / 'sample.json').write_text(json.dumps({'side': 'right', 'camera_serial': 'synthetic', 'frames': frames}))
    script = Path(__file__).resolve().parents[1] / 'calibrate_wrist_camera.py'
    completed = subprocess.run([sys.executable, str(script), '--input', str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((tmp_path / 'wrist_calibration.json').read_text())
    assert result['status'] == 'validated_on_holdout'
    assert result['quality']['holdout']['count'] == 3
    for key, truth in [('world_T_base', truth_base), ('ee_T_camera', truth_mount)]:
        error = invert(truth) @ np.asarray(result[key])
        assert np.linalg.norm(error[:3, 3]) < .015
        assert np.rad2deg(Rotation.from_matrix(error[:3, :3]).magnitude()) < 2
    # A camera-in-base result must follow the robot pose, rather than be a constant.
    assert not np.allclose(result['samples'][0]['base_T_camera'], result['samples'][1]['base_T_camera'])
