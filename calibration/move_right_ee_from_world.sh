#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./calibration/move_right_ee_from_world.sh \
  (--world-flange-pose X Y Z ROLL PITCH YAW | \
   --world-ee-pose X Y Z ROLL PITCH YAW | \
   --world-hand-pose X Y Z ROLL PITCH YAW) \
  [--controller-dry-run|--execute] [--conda-env NAME]

Convert an fr3_link8/flange, default controller-EE, or Wuji hand-root pose from
the tabletop world frame to the controller-EE target expected by
scripts/move_to_target_ee.sh. Exactly one pose option is required. An EE target
is used directly without any tool-coordinate transform. Hand targets use
grasp/ee_to_wuji_nominal.json for EE_T_hand. Pose values are
x,y,z,roll,pitch,yaw in meters/radians and use:

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
    --world-flange-pose 0.20 -0.10 0.30 3.14159 0 0

  ./calibration/move_right_ee_from_world.sh \
    --world-ee-pose 0.20 -0.10 0.30 3.14159 0 0

  ./calibration/move_right_ee_from_world.sh \
    --world-hand-pose 0.20 -0.10 0.30 3.14159 0 0

  ./calibration/move_right_ee_from_world.sh \
    --world-hand-pose 0.20 -0.10 0.30 3.14159 0 0 \
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
mount_calibration="${repository_root}/grasp/ee_to_wuji_nominal.json"
conda_environment="${CALIBRATION_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"

declare -a world_flange_pose=()
declare -a world_ee_pose=()
declare -a world_hand_pose=()
mode="convert"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --world-flange-pose)
      [[ "${#world_flange_pose[@]}" -eq 0 ]] || die "--world-flange-pose may only be passed once"
      [[ "$#" -ge 7 ]] || die "--world-flange-pose requires exactly six XYZ/RPY values"
      world_flange_pose=("$2" "$3" "$4" "$5" "$6" "$7")
      shift 7
      ;;
    --world-ee-pose)
      [[ "${#world_ee_pose[@]}" -eq 0 ]] || die "--world-ee-pose may only be passed once"
      [[ "$#" -ge 7 ]] || die "--world-ee-pose requires exactly six XYZ/RPY values"
      world_ee_pose=("$2" "$3" "$4" "$5" "$6" "$7")
      shift 7
      ;;
    --world-hand-pose)
      [[ "${#world_hand_pose[@]}" -eq 0 ]] || die "--world-hand-pose may only be passed once"
      [[ "$#" -ge 7 ]] || die "--world-hand-pose requires exactly six XYZ/RPY values"
      world_hand_pose=("$2" "$3" "$4" "$5" "$6" "$7")
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

pose_option_count=0
[[ "${#world_flange_pose[@]}" -eq 6 ]] && ((pose_option_count += 1))
[[ "${#world_ee_pose[@]}" -eq 6 ]] && ((pose_option_count += 1))
[[ "${#world_hand_pose[@]}" -eq 6 ]] && ((pose_option_count += 1))
[[ "${pose_option_count}" -eq 1 ]] || die \
  "exactly one of --world-flange-pose, --world-ee-pose, or --world-hand-pose is required"

if [[ "${#world_flange_pose[@]}" -eq 6 ]]; then
  input_pose_kind="flange"
  input_pose=("${world_flange_pose[@]}")
elif [[ "${#world_ee_pose[@]}" -eq 6 ]]; then
  input_pose_kind="ee"
  input_pose=("${world_ee_pose[@]}")
elif [[ "${#world_hand_pose[@]}" -eq 6 ]]; then
  input_pose_kind="hand"
  input_pose=("${world_hand_pose[@]}")
fi
[[ -r "${conda_helper}" ]] || die "Conda helper not found: ${conda_helper}"
if [[ "${input_pose_kind}" == "hand" ]]; then
  [[ -r "${mount_calibration}" ]] || die \
    "Wuji mount calibration not found: ${mount_calibration}"
fi

# shellcheck source=scripts/conda_env.sh
source "${conda_helper}"
declare -a conda_python=()
real_exp_build_conda_python_command "${conda_environment}" conda_python || exit 1
real_exp_require_conda_python_modules "${conda_environment}" numpy || die \
  "the '${conda_environment}' Conda environment cannot run the coordinate conversion"

# Keep numerical dependencies in Conda. This child emits the six-value world
# flange pose, right-base flange pose, and right-base controller-EE pose; it
# never imports ROS or opens robot hardware.
converted_values="$("${conda_python[@]}" - \
  "${input_pose_kind}" "${mount_calibration}" "${input_pose[@]}" <<'PY'
import json
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

# Fixed transform used by the current FR3 controller configuration. It maps
# controller-EE coordinates into the fr3_link8/flange frame:
# p_F = F_T_EE @ p_EE. Keep this aligned with the robot's reported F_T_EE.
FLANGE_T_CONTROLLER_EE = np.eye(4, dtype=np.float64)
FLANGE_T_CONTROLLER_EE[:3, :3] = np.asarray(
    [
        [math.cos(-math.pi / 4.0), -math.sin(-math.pi / 4.0), 0.0],
        [math.sin(-math.pi / 4.0), math.cos(-math.pi / 4.0), 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
FLANGE_T_CONTROLLER_EE[:3, 3] = [0.0, 0.0, 0.1034]


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


def invert_transform(transform):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def validate_transform(value, name):
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise SystemExit(f"Error: {name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise SystemExit(
            f"Error: {name} must have homogeneous bottom row [0, 0, 0, 1]"
        )
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise SystemExit(f"Error: {name} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise SystemExit(f"Error: {name} rotation determinant must be +1")
    return transform


input_pose_kind = sys.argv[1]
mount_calibration_path = sys.argv[2]
try:
    input_pose = np.asarray(
        [float(value) for value in sys.argv[3:]], dtype=np.float64
    )
except ValueError as exc:
    raise SystemExit(
        f"Error: --world-{input_pose_kind}-pose contains a non-numeric value: {exc}"
    ) from exc
if input_pose.shape != (6,) or not np.all(np.isfinite(input_pose)):
    raise SystemExit(
        f"Error: --world-{input_pose_kind}-pose must contain six finite XYZ/RPY values"
    )

world_t_input = np.eye(4, dtype=np.float64)
world_t_input[:3, :3] = rpy_to_rotation(*input_pose[3:])
world_t_input[:3, 3] = input_pose[:3]

if input_pose_kind == "flange":
    world_t_flange = world_t_input
    world_t_controller_ee = world_t_flange @ FLANGE_T_CONTROLLER_EE
elif input_pose_kind == "ee":
    # The input already describes the default controller EE. Do not apply a
    # hand/tool transform; only recover the flange pose for diagnostics.
    world_t_controller_ee = world_t_input
    world_t_flange = world_t_controller_ee @ invert_transform(
        FLANGE_T_CONTROLLER_EE
    )
elif input_pose_kind == "hand":
    try:
        with open(mount_calibration_path, encoding="utf-8") as calibration_file:
            mount_data = json.load(calibration_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"Error: cannot read Wuji mount calibration {mount_calibration_path}: {exc}"
        ) from exc
    if "ee_T_hand" not in mount_data:
        raise SystemExit(
            f"Error: {mount_calibration_path} must define ee_T_hand"
        )
    ee_t_hand = validate_transform(mount_data["ee_T_hand"], "ee_T_hand")
    # W_T_H = W_T_EE @ EE_T_H, so W_T_EE = W_T_H @ inverse(EE_T_H).
    world_t_controller_ee = world_t_input @ invert_transform(ee_t_hand)
    # W_T_EE = W_T_F @ F_T_EE, so recover the corresponding flange target.
    world_t_flange = world_t_controller_ee @ invert_transform(
        FLANGE_T_CONTROLLER_EE
    )
else:
    raise SystemExit(f"Error: unsupported input pose kind: {input_pose_kind}")

# A_T_B maps B coordinates into A coordinates:
# W_T_B_R = W_T_C @ C_T_B_R
# B_R_T_F = inverse(W_T_B_R) @ W_T_F
# B_R_T_EE = inverse(W_T_B_R) @ W_T_EE
world_t_right_base = WORLD_T_CAMERA @ CAMERA_T_RIGHT_BASE
right_base_t_world = invert_transform(world_t_right_base)
right_base_t_flange = right_base_t_world @ world_t_flange
right_base_t_controller_ee = right_base_t_world @ world_t_controller_ee
world_flange_pose = np.concatenate(
    (
        world_t_flange[:3, 3],
        rotation_to_rpy(world_t_flange[:3, :3]),
    )
)
right_base_flange_pose = np.concatenate(
    (
        right_base_t_flange[:3, 3],
        rotation_to_rpy(right_base_t_flange[:3, :3]),
    )
)
right_base_controller_ee_pose = np.concatenate(
    (
        right_base_t_controller_ee[:3, 3],
        rotation_to_rpy(right_base_t_controller_ee[:3, :3]),
    )
)
print(
    " ".join(
        f"{value:.12g}"
        for value in np.concatenate(
            (world_flange_pose, right_base_flange_pose, right_base_controller_ee_pose)
        )
    )
)
PY
)"

read -r -a converted_pose_values <<< "${converted_values}"
[[ "${#converted_pose_values[@]}" -eq 18 ]] || die \
  "coordinate converter returned invalid poses: ${converted_values}"
converted_world_flange_pose=("${converted_pose_values[@]:0:6}")
right_base_flange_pose=("${converted_pose_values[@]:6:6}")
right_base_controller_ee_pose=("${converted_pose_values[@]:12:6}")

if [[ "${input_pose_kind}" == "ee" ]]; then
  printf 'world controller EE pose    [x y z r p y]:'
  printf ' %.9f' "${world_ee_pose[@]}"
  printf '\n'
elif [[ "${input_pose_kind}" == "hand" ]]; then
  printf 'world hand pose             [x y z r p y]:'
  printf ' %.9f' "${world_hand_pose[@]}"
  printf '\n'
fi
printf 'world flange pose          [x y z r p y]:'
printf ' %.9f' "${converted_world_flange_pose[@]}"
printf '\n'
printf 'right-base flange pose     [x y z r p y]:'
printf ' %.9f' "${right_base_flange_pose[@]}"
printf '\n'
printf 'right-base controller EE   [x y z r p y]:'
printf ' %.9f' "${right_base_controller_ee_pose[@]}"
printf '\n'

declare -a move_command=(
  "${move_script}" --right --arm --target-ee-pose "${right_base_controller_ee_pose[@]}"
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
