from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "calibration"))
import build_simtoolreal_policy_frame as calibration


def test_mount_is_read_from_slanted_urdf() -> None:
    mount = calibration.read_urdf_mount(ROOT / "simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf")
    assert mount[0, 3] == pytest.approx(0.045)
    assert mount[1, 3] == pytest.approx(-0.2093065)
    assert mount[2, 3] == pytest.approx(0.65565)
    assert mount[:3, :3] @ mount[:3, :3].T == pytest.approx(np.eye(3), abs=1e-6)


def test_scheme2_compositions_match_calibrated_values() -> None:
    world, camera_base = calibration.load_inputs(ROOT / "calibration/simtoolreal_scheme2_input.json")
    result = calibration.build(world, camera_base, calibration.read_urdf_mount(ROOT / "simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf"), calibration.DEFAULT_POLICY_T_U)
    np.testing.assert_allclose(result["Wp_T_Wreal"], [[0.043334780, 0.998938817, 0.015599219, -0.025495724], [-0.997610266, 0.042425573, 0.054532804, -0.229389650], [0.053813130, -0.017925108, 0.998390123, 0.569236009], [0.0, 0.0, 0.0, 1.0]], atol=2e-7)
    np.testing.assert_allclose(result["Wp_T_C"], [[-0.997701365, -0.045854410, 0.049893481, -0.066035271], [-0.061077178, 0.927423222, -0.368992879, 0.226006827], [-0.029352422, -0.371192053, -0.928092072, 1.706007177], [0.0, 0.0, 0.0, 1.0]], atol=2e-7)
    np.testing.assert_allclose(result["Wp_T_C"], result["Wp_T_Wreal"] @ world, atol=2e-8)
    np.testing.assert_allclose(result["Wp_T_Wreal"] @ result["Wreal_T_B_R"], result["Wp_T_B_R"], atol=2e-8)


def test_improper_rotation_is_rejected() -> None:
    bad = np.eye(4)
    bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="right-handed"):
        calibration.checked_transform(bad, "bad")


def test_missing_goal_path_has_actionable_error() -> None:
    missing = ROOT / "calibration/definitely_missing_real_world_goal.json"
    with pytest.raises(FileNotFoundError, match="Wreal_T_G input file does not exist"):
        calibration.read_matrix_arg(str(missing), "Wreal_T_G")
