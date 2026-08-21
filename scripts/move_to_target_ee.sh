#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/move_to_target_ee.sh --duo|--left|--right --arm|--gripper|--hand \
  --target-ee-pose VALUES [--target-ee-joint VALUES] [options]

Pose format:
  Each pose is x,y,z,roll,pitch,yaw in the FR3 base frame (meters/radians).
  Always pass 6 comma- or space-separated values. --duo sends the same pose
  to both sides. Brackets and mixed comma/space NumPy output are accepted.
  Keep a vector on one shell line, or end each continued line with a backslash.

End-effector target format:
  --arm       Omit --target-ee-joint.
  --gripper   Pass 1 physical width in meters.
  --hand      Pass 20 Wuji Hand 2 joint angles in radians.
  With --duo, the same end-effector target is sent to both sides.

Options:
  --ip-left ADDR       Left FR3 address (default: 172.16.0.3).
  --ip-right ADDR      Right FR3 address (default: 172.16.0.2).
  --left-hand-ip ADDR  Left Wuji Hand 2 SDK address (or WUJI_LEFT_HAND_IP).
  --right-hand-ip ADDR Right Wuji Hand 2 SDK address (or WUJI_RIGHT_HAND_IP).
  --dry-run            Read current state and print it with the targets; do not move.
  --help               Show this help.

Examples:
  ./scripts/move_to_target_ee.sh --left --arm \
    --target-ee-pose 0.45 0.20 0.35 3.14 0 0
  ./scripts/move_to_target_ee.sh --right --gripper \
    --target-ee-pose 0.45 -0.20 0.35 3.14 0 0 --target-ee-joint 0.06

For a real motion, the utility reads and prints the current pose and selected
end-effector state, then requires an explicit y/yes confirmation.
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
program="${repository_root}/data_collection/move_to_target_ee.py"
[[ -r "${program}" ]] || die "move utility not found: ${program}"

# Use the same project environment selection as Wuji teleoperation. It carries
# both pylibfranka and the Wuji SDK, while the Python entrypoint owns validation.
# shellcheck source=scripts/conda_env.sh
source "${script_dir}/conda_env.sh"
control_conda_env="${WUJI_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"
declare -a control_python=()
real_exp_build_conda_python_command "${control_conda_env}" control_python || exit 1

exec "${control_python[@]}" "${program}" "$@"
