#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./calibration/move_right_ee_from_world.sh \
  --world-ee-pose X Y Z ROLL PITCH YAW \
  [--controller-dry-run|--execute] [--conda-env NAME]

Convert a controller-EE pose from the tabletop world frame to the right FR3
base frame. Pose values are x,y,z,roll,pitch,yaw in meters/radians and use:

  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

Modes:
  With neither mode option, only print the converted pose and command.
  --controller-dry-run  Start the ROS controller, validate live state and IK,
                        but do not command motion.
  --execute             Pass the target to the controller. The existing
                        interactive y/yes confirmation is still required.

Options:
  --conda-env NAME      Conda environment used only for coordinate conversion.
                        Default: CALIBRATION_CONDA_ENV, then LEROBOT_CONDA_ENV,
                        then "lerobot".
  --help                Show this help.

The conversion runs in the selected Conda environment. ROS control is then
delegated to scripts/move_to_target_ee.sh, which uses system /usr/bin/python3.

Examples:
  ./calibration/move_right_ee_from_world.sh \
    --world-ee-pose 0.20 -0.10 0.30 3.14159 0 0

  ./calibration/move_right_ee_from_world.sh \
    --world-ee-pose 0.20 -0.10 0.30 3.14159 0 0 \
    --controller-dry-run
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi
if [[ "$#" -eq 1 && "$1" == "--help" ]]; then
  usage
  exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
move_script="${repository_root}/scripts/move_to_target_ee.sh"
conda_helper="${repository_root}/scripts/conda_env.sh"
conda_environment="${CALIBRATION_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"

declare -a world_pose=()
mode="convert"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --world-ee-pose)
      [[ "${#world_pose[@]}" -eq 0 ]] || die "--world-ee-pose may only be passed once"
      [[ "$#" -ge 7 ]] || die "--world-ee-pose requires exactly six XYZ/RPY values"
      world_pose=("$2" "$3" "$4" "$5" "$6" "$7")
      shift 7
      ;;
    --controller-dry-run)
      [[ "${mode}" == "convert" ]] || die "--controller-dry-run and --execute are mutually exclusive"
      mode="dry-run"
      shift
      ;;
    --execute)
      [[ "${mode}" == "convert" ]] || die "--controller-dry-run and --execute are mutually exclusive"
      mode="execute"
      shift
      ;;
    --conda-env)
      [[ "$#" -ge 2 && -n "$2" ]] || die "--conda-env requires a non-empty environment name"
      conda_environment="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "${#world_pose[@]}" -eq 6 ]] || die "--world-ee-pose is required with six XYZ/RPY values"
[[ -r "${conda_helper}" ]] || die "Conda helper not found: ${conda_helper}"

# shellcheck source=scripts/conda_env.sh
source "${conda_helper}"
declare -a conda_python=()
real_exp_build_conda_python_command "${conda_environment}" conda_python || exit 1
real_exp_require_conda_python_modules "${conda_environment}" numpy || die \
  "the '${conda_environment}' Conda environment cannot run the coordinate conversion"

# Keep numerical dependencies in Conda. This child emits only the six values
# consumed by the shell launcher; it never imports ROS or opens robot hardware.
converted_pose="$("${conda_python[@]}" - "${world_pose[@]}" <<'PY'
import math
import sys

import numpy as np


WORLD_T_CAMERA = np.asarray(
    [
        [0.016116505, -0.947169025, 0.320329670, -0.394891761],
        [-0.998707711, 0.000194370, 0.050821951, -0.041552817],
        [-0.048199240, -0.320734783, -0.945941876, 1.159142768],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
CAMERA_T_RIGHT_BASE = np.asarray(
    [
        [0.061077178, -0.724658647, 0.686395967, 0.162840030],
        [-0.927423222, -0.295425134, -0.229369041, 0.562456279],
        [0.368992880, -0.622570346, -0.690108991, 0.901632663],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def rpy_to_rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def rotation_to_rpy(rotation):
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


try:
    world_pose = np.asarray([float(value) for value in sys.argv[1:]], dtype=np.float64)
except ValueError as exc:
    raise SystemExit(f"Error: --world-ee-pose contains a non-numeric value: {exc}") from exc
if world_pose.shape != (6,) or not np.all(np.isfinite(world_pose)):
    raise SystemExit("Error: --world-ee-pose must contain six finite XYZ/RPY values")

world_t_ee = np.eye(4, dtype=np.float64)
world_t_ee[:3, :3] = rpy_to_rotation(*world_pose[3:])
world_t_ee[:3, 3] = world_pose[:3]

# A_T_B maps B coordinates into A coordinates:
# W_T_B_R = W_T_C @ C_T_B_R
# B_R_T_E = inverse(W_T_B_R) @ W_T_E
world_t_right_base = WORLD_T_CAMERA @ CAMERA_T_RIGHT_BASE
right_base_t_world = np.eye(4, dtype=np.float64)
right_base_t_world[:3, :3] = world_t_right_base[:3, :3].T
right_base_t_world[:3, 3] = (
    -right_base_t_world[:3, :3] @ world_t_right_base[:3, 3]
)
right_base_t_ee = right_base_t_world @ world_t_ee
right_base_pose = np.concatenate(
    (right_base_t_ee[:3, 3], rotation_to_rpy(right_base_t_ee[:3, :3]))
)
print(" ".join(f"{value:.12g}" for value in right_base_pose))
PY
)"

read -r -a right_base_pose <<< "${converted_pose}"
[[ "${#right_base_pose[@]}" -eq 6 ]] || die \
  "coordinate converter returned an invalid pose: ${converted_pose}"

printf 'world EE pose       [x y z r p y]:'
printf ' %.9f' "${world_pose[@]}"
printf '\n'
printf 'right-base EE pose  [x y z r p y]:'
printf ' %.9f' "${right_base_pose[@]}"
printf '\n'

declare -a move_command=(
  "${move_script}" --right --arm --target-ee-pose "${right_base_pose[@]}"
)
if [[ "${mode}" == "dry-run" ]]; then
  move_command+=(--dry-run)
fi
printf 'controller command:'
printf ' %q' "${move_command[@]}"
printf '\n'

if [[ "${mode}" == "convert" ]]; then
  echo "Conversion only: controller was not started."
  exit 0
fi

[[ -x "${move_script}" ]] || die "move utility is missing or not executable: ${move_script}"

# move_to_target_ee.sh unsets Conda/ROS path variables, sources the ROS setup,
# and invokes its ROS program with /usr/bin/python3. Replacing this process also
# preserves its signal handling and interactive hardware confirmation behavior.
exec "${move_command[@]}"
