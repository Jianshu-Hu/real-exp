#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/move_to_target_ee.sh --left|--right --arm|--gripper|--hand \
  --target-ee-pose VALUES [--target-ee-joint VALUES] [options]

Pose format:
  Each pose is x,y,z,roll,pitch,yaw in the FR3 base frame (meters/radians).
  Always pass 6 comma- or space-separated values. Brackets and mixed
  comma/space NumPy output are accepted.
  Keep a vector on one shell line, or end each continued line with a backslash.

End-effector target format:
  --arm       Omit --target-ee-joint.
  --gripper   Pass 1 physical width in meters.
  --hand      Pass 20 Wuji Hand 2 joint angles in radians.
Options:
  --ip-left ADDR       Left FR3 address (default: 172.16.0.3).
  --ip-right ADDR      Right FR3 address (default: 172.16.0.2).
  --left-hand-ip ADDR  Left Wuji Hand 2 SDK address (or WUJI_LEFT_HAND_IP).
  --right-hand-ip ADDR Right Wuji Hand 2 SDK address (or WUJI_RIGHT_HAND_IP).
  --dry-run            Plan and preview the motion, but do not execute it.
  --help               Show this help.

Examples:
  ./scripts/move_to_target_ee.sh --left --arm \
    --target-ee-pose 0.45 0.20 0.35 3.14 0 0
  ./scripts/move_to_target_ee.sh --right --gripper \
    --target-ee-pose 0.45 -0.20 0.35 3.14 0 0 --target-ee-joint 0.06

The utility deterministically solves a collision-checked Cartesian path: EE
translation is linearly interpolated and orientation follows the shortest rotation. It
rejects partial paths, large joint jumps, excessive joint travel, and paths
that depart from the requested EE interpolation. Joint travel is constrained
during bounded, previous-solution-seeded Cartesian IK relative to the live
configuration. Every generated state is checked against MoveIt's planning scene,
then independently audited afterward. It previews the plan and requires explicit
y/yes confirmation before FollowJointTrajectory execution.

Measured execution succeeds within 20 mm and 0.08 rad. If the first execution
does not reach that tolerance, the utility replans once from the measured
current pose, previews the correction, and requires a second confirmation.

Only obstacles already published into the selected arm's MoveIt planning scene
are checked. This launcher does not infer tables, fixtures, people, or the other
FR3 from camera data. Keep the real workspace clear unless those objects have
been added to the planning scene.
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
replay_program="${repository_root}/data_collection/replay_lerobot_episode.py"
[[ -r "${program}" ]] || die "move utility not found: ${program}"
[[ -r "${replay_program}" ]] || die "replay utility not found: ${replay_program}"

# The arm utility deliberately runs in the ROS Python environment: it uses the
# same rclpy/Pinocchio model and controller topics as replay_lerobot_episode.
# Wuji hand workers, when requested, are started separately by this launcher in
# the project Conda environment because the SDK is not installed in ROS Python.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
setup_files=(
  "/opt/ros/humble/setup.bash"
  "${HOME}/franka_ros2_ws/install/local_setup.bash"
  "${repository_root}/gello_software/ros2/install/local_setup.bash"
)
set +u
for setup_file in "${setup_files[@]}"; do
  [[ -r "${setup_file}" ]] || die "ROS setup file is missing or unreadable: ${setup_file}"
  # shellcheck disable=SC1090
  source "${setup_file}"
done
set -u

ros_python="/usr/bin/python3"
[[ -x "${ros_python}" ]] || die "ROS Python interpreter is missing: ${ros_python}"
for command_name in setsid timeout ss; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command not found: ${command_name}"
done

declare -a passthrough_args=("$@")
arm_mode=""
end_effector=""
left_hand_ip="${WUJI_LEFT_HAND_IP:-}"
right_hand_ip="${WUJI_RIGHT_HAND_IP:-}"
for ((index = 0; index < ${#passthrough_args[@]}; index++)); do
  case "${passthrough_args[index]}" in
    --left|--right) arm_mode="${passthrough_args[index]#--}" ;;
    --arm|--gripper|--hand) end_effector="${passthrough_args[index]#--}" ;;
    --left-hand-ip) left_hand_ip="${passthrough_args[index + 1]:-}"; ((index++)) ;;
    --right-hand-ip) right_hand_ip="${passthrough_args[index + 1]:-}"; ((index++)) ;;
  esac
done
[[ -n "${arm_mode}" ]] || die "one of --left or --right is required"
[[ -n "${end_effector}" ]] || die "one of --arm, --gripper, or --hand is required"

if [[ "${arm_mode}" == "right" ]]; then robot_config="example_fr3_right_config_no_gripper.yaml"
else robot_config="example_fr3_config_no_gripper.yaml"; fi

if [[ "${end_effector}" == "hand" ]]; then
  # shellcheck source=scripts/conda_env.sh
  source "${script_dir}/conda_env.sh"
  wuji_conda_env="${WUJI_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"
  declare -a wuji_python=()
  real_exp_build_conda_python_command "${wuji_conda_env}" wuji_python || exit 1
  real_exp_require_conda_python_modules "${wuji_conda_env}" wuji_sdk wujihandpy zmq numpy || die \
    "the '${wuji_conda_env}' Conda environment cannot run Wuji hand control"
fi

declare -a child_pids=()
shutdown_started=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  [[ "${shutdown_started}" -eq 1 ]] && return
  shutdown_started=1
  for pid in "${child_pids[@]}"; do kill -INT -- "-${pid}" 2>/dev/null || true; done
  sleep 0.5
  for pid in "${child_pids[@]}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
  for pid in "${child_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT INT TERM
start_process() {
  setsid -- bash -c 'trap - INT QUIT; exec "$@"' _ "$@" &
  child_pids+=("$!")
}
wait_for_topic() {
  local topic="$1" pid="$2" deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    kill -0 "${pid}" 2>/dev/null || die "FR3 controller exited before ${topic} became available"
    if timeout 5s ros2 topic echo "${topic}" --once --no-daemon >/dev/null 2>&1; then return 0; fi
  done
  die "timed out waiting for ${topic}"
}
wait_for_subscription() {
  local topic="$1" pid="$2" deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    kill -0 "${pid}" 2>/dev/null || die "FR3 controller exited before subscribing to ${topic}"
    if timeout 5s ros2 topic info "${topic}" --no-daemon 2>/dev/null \
      | grep -Eq 'Subscription count: [1-9][0-9]*'; then return 0; fi
    sleep 0.5
  done
  die "timed out waiting for a subscription to ${topic}"
}
wait_for_tcp_port() {
  local port="$1" pid="$2" deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    kill -0 "${pid}" 2>/dev/null || die "Wuji hand worker exited before binding port ${port}"
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then return 0; fi
    sleep 0.5
  done
  die "timed out waiting for Wuji hand worker port ${port}"
}

echo "Starting ${arm_mode} FR3 ROS controller (${robot_config})."
start_process ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  "robot_config_file:=${robot_config}" motion_controller:=trajectory
controller_pid="${child_pids[0]}"
if [[ "${arm_mode}" == "left" ]]; then
  wait_for_topic /left/franka/joint_states "${controller_pid}"
  wait_for_subscription /left/fr3_arm_controller/joint_trajectory "${controller_pid}"
fi
if [[ "${arm_mode}" == "right" ]]; then
  wait_for_topic /right/franka/joint_states "${controller_pid}"
  wait_for_subscription /right/fr3_arm_controller/joint_trajectory "${controller_pid}"
fi

if [[ "${end_effector}" == "hand" ]]; then
  if [[ "${arm_mode}" == "left" ]]; then
    start_process "${wuji_python[@]}" "${replay_program}" \
      --internal-wuji-hand left --left-hand-command-port 5561 --left-hand-status-port 5563 --hand-ip "${left_hand_ip}"
    wait_for_tcp_port 5563 "${child_pids[${#child_pids[@]} - 1]}"
  fi
  if [[ "${arm_mode}" == "right" ]]; then
    start_process "${wuji_python[@]}" "${replay_program}" \
      --internal-wuji-hand right --right-hand-command-port 5562 --right-hand-status-port 5564 --hand-ip "${right_hand_ip}"
    wait_for_tcp_port 5564 "${child_pids[${#child_pids[@]} - 1]}"
  fi
fi

set +e
"${ros_python}" "${program}" "${passthrough_args[@]}"
status=$?
set -e
exit "${status}"
