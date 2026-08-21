#!/usr/bin/env bash
set -euo pipefail

# Start the matching FR3 replay stack and replay one LeRobot episode.
# Usage: bash scripts/replay.sh --dataset-root data/my_dataset --episode 0 [replay options]

usage() {
  cat <<'EOF'
Usage: bash scripts/replay.sh --dataset-root DATASET --episode N [setting] [options]

The script reads trajectory hardware metadata before starting ROS. Pass the
current setting with --arm/--gripper/--hand and --duo/--left/--right; omitting
them selects the recorded setting. A mismatch aborts before hardware startup.
All trajectory replay code lives in data_collection/replay_lerobot_episode.py.
Setting (optional; defaults to trajectory metadata):
  --arm | --gripper | --hand
  --duo | --left | --right
  --left-hand-ip ADDR       Wuji Hand 2 address for left-hand replay.
  --right-hand-ip ADDR      Wuji Hand 2 address for right-hand replay.

Use --no-gripper as a compatibility alias for --arm. A mismatch is rejected
before ROS controllers or Wuji SDK sessions are started.
All other options are passed to replay_lerobot_episode.py. The replay process
waits for actual arm/gripper state samples before accepting `s`.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

if [[ "$#" -eq 1 && "$1" == "--help" ]]; then
  usage
  exit 0
fi

[[ "$#" -gt 0 ]] || {
  usage >&2
  exit 2
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
# shellcheck source=scripts/conda_env.sh
source "${script_dir}/conda_env.sh"
replay_script="${repository_root}/data_collection/replay_lerobot_episode.py"
[[ -f "${replay_script}" ]] || die "replay script not found: ${replay_script}"

replay_args=()
end_effector=""
arm_mode=""
left_hand_ip=""
right_hand_ip=""
use_dry_run=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --arm|--gripper|--hand)
      [[ -z "${end_effector}" ]] || die "choose only one of --arm, --gripper, or --hand"
      end_effector="${1#--}"; shift ;;
    --no-gripper)
      [[ -z "${end_effector}" ]] || die "choose only one end-effector setting"
      end_effector="arm"; shift ;;
    --duo|--left|--right)
      [[ -z "${arm_mode}" ]] || die "choose only one of --duo, --left, or --right"
      arm_mode="${1#--}"; shift ;;
    --left-hand-ip)
      [[ "$#" -ge 2 ]] || die "--left-hand-ip requires an address"
      left_hand_ip="$2"; shift 2 ;;
    --right-hand-ip)
      [[ "$#" -ge 2 ]] || die "--right-hand-ip requires an address"
      right_hand_ip="$2"; shift 2 ;;
    --dry-run) use_dry_run=1; replay_args+=("$1"); shift ;;
    *) replay_args+=("$1"); shift ;;
  esac
done

dataset_root=""
for ((index = 0; index < ${#replay_args[@]}; index++)); do
  case "${replay_args[index]}" in
    --dataset-root)
      [[ "${#replay_args[@]}" -gt $((index + 1)) ]] || die "--dataset-root requires a path"
      dataset_root="${replay_args[index + 1]}" ;;
  esac
done
[[ -n "${dataset_root}" ]] || die "--dataset-root is required"

unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH

setup_files=(
  "/opt/ros/humble/setup.bash"
  "${HOME}/franka_ros2_ws/install/local_setup.bash"
  "${repository_root}/gello_software/ros2/install/local_setup.bash"
)

# Generated ROS environment hooks may inspect optional variables before defining them.
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

# Background jobs may receive /dev/null on stdin when job control is enabled.
# Prefer a direct terminal handle so the replay prompt remains interactive
# after the child is placed in its own process group. Fall back to the caller's
# stdin for redirected/non-interactive invocations.
if [[ -t 0 && -r /dev/tty && -w /dev/tty ]]; then
  exec 10<>/dev/tty || exec 10<&0
else
  exec 10<&0
fi

recorded_end_effector="$("${ros_python}" "${repository_root}/utils/trajectory_metadata.py" --dataset-root "${dataset_root}" --field end_effector)" \
  || die "could not read trajectory metadata"
recorded_arm_mode="$("${ros_python}" "${repository_root}/utils/trajectory_metadata.py" --dataset-root "${dataset_root}" --field arm_mode)" \
  || die "could not read trajectory metadata"
[[ -n "${end_effector}" ]] || end_effector="${recorded_end_effector}"
[[ -n "${arm_mode}" ]] || arm_mode="${recorded_arm_mode}"
replay_setting_args=(--robot-end-effector "${end_effector}" --robot-arm-mode "${arm_mode}")
echo "Checking trajectory setting: ${recorded_end_effector}/${recorded_arm_mode}"
"${ros_python}" "${replay_script}" "${replay_args[@]}" "${replay_setting_args[@]}" --dry-run >/dev/null \
  || die "current replay setting ${end_effector}/${arm_mode} does not satisfy trajectory metadata"

if [[ "${use_dry_run}" -eq 1 ]]; then
  echo "Dry-run requested; skipping ROS controller and gripper/hand startup."
  exec "${ros_python}" "${replay_script}" "${replay_args[@]}" "${replay_setting_args[@]}"
fi

declare -a child_pids=()
declare -A child_names=()
shutdown_started=0

start_process() {
  local process_name="$1"
  shift

  echo "Starting ${process_name}: $*"
  setsid -- bash -c 'trap - INT QUIT; exec "$@"' _ "$@" <&10 &
  local child_pid=$!
  child_pids+=("${child_pid}")
  child_names["${child_pid}"]="${process_name}"
}

signal_running_groups() {
  local signal_name="$1"

  for child_pid in "${child_pids[@]}"; do
    if kill -0 -- "-${child_pid}" 2>/dev/null; then
      kill -s "${signal_name}" -- "-${child_pid}" 2>/dev/null || true
    fi
  done
}

wait_for_groups_to_stop() {
  local attempts="$1"
  local attempt

  for ((attempt = 0; attempt < attempts; attempt++)); do
    local running_group=0
    for child_pid in "${child_pids[@]}"; do
      if ps -o stat= --sid "${child_pid}" 2>/dev/null | grep -qv '^[[:space:]]*Z'; then
        running_group=1
        break
      fi
    done
    [[ "${running_group}" -eq 0 ]] && return 0
    sleep 0.1
  done

  return 1
}

shutdown() {
  local exit_status=$?

  if [[ "${shutdown_started}" -eq 1 ]]; then
    return
  fi
  shutdown_started=1
  trap - EXIT INT TERM

  if [[ "${#child_pids[@]}" -gt 0 ]]; then
    echo "Stopping replay processes..."
    signal_running_groups INT
    if ! wait_for_groups_to_stop 20; then
      signal_running_groups TERM
    fi
    if ! wait_for_groups_to_stop 30; then
      echo "Force-stopping unresponsive replay processes..." >&2
      signal_running_groups KILL
    fi
    for child_pid in "${child_pids[@]}"; do
      wait "${child_pid}" 2>/dev/null || true
    done
  fi

  exit "${exit_status}"
}

trap shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_for_topic() {
  local topic="$1"
  local process_pid="$2"
  local timeout_seconds=90
  local deadline=$((SECONDS + timeout_seconds))

  while ((SECONDS < deadline)); do
    if ! kill -0 "${process_pid}" 2>/dev/null; then
      wait "${process_pid}" || true
      die "${child_names[${process_pid}]} exited before ${topic} became available"
    fi
    if timeout 5s ros2 topic echo "${topic}" --once --no-daemon >/dev/null 2>&1; then
      echo "Ready: ${topic}"
      return 0
    fi
  done

  die "timed out after ${timeout_seconds}s waiting for ${topic}"
}

wait_for_subscription() {
  local topic="$1"
  local process_pid="$2"
  local timeout_seconds=90
  local deadline=$((SECONDS + timeout_seconds))

  while ((SECONDS < deadline)); do
    if ! kill -0 "${process_pid}" 2>/dev/null; then
      wait "${process_pid}" || true
      die "${child_names[${process_pid}]} exited before subscribing to ${topic}"
    fi
    if timeout 5s ros2 topic info "${topic}" --no-daemon 2>/dev/null \
      | grep -Eq 'Subscription count: [1-9][0-9]*'; then
      echo "Ready: subscription ${topic}"
      return 0
    fi
    sleep 0.5
  done

  die "timed out after ${timeout_seconds}s waiting for subscription ${topic}"
}

wait_for_tcp_port() {
  local port="$1" process_pid="$2" timeout_seconds=90 deadline
  deadline=$((SECONDS + timeout_seconds))
  while ((SECONDS < deadline)); do
    if ! kill -0 "${process_pid}" 2>/dev/null; then
      wait "${process_pid}" || true
      die "Wuji hand replay process exited before binding port ${port}"
    fi
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
      echo "Ready: Wuji hand replay command port ${port}"
      return 0
    fi
    sleep 0.5
  done
  die "timed out waiting for Wuji hand replay command port ${port}"
}

echo "Starting ${arm_mode} FR3 replay stack for ${end_effector} trajectory."
echo "The robot will receive recorded commands after you confirm replay with 's'."

if [[ "${end_effector}" == "gripper" ]]; then
  if [[ "${arm_mode}" == "duo" ]]; then
    robot_config="example_fr3_duo_config.yaml"
  elif [[ "${arm_mode}" == "right" ]]; then
    robot_config="example_fr3_right_config.yaml"
  else
    robot_config="example_fr3_config.yaml"
  fi
elif [[ "${arm_mode}" == "duo" ]]; then
  robot_config="example_fr3_duo_config_no_gripper.yaml"
elif [[ "${arm_mode}" == "right" ]]; then
  robot_config="example_fr3_right_config_no_gripper.yaml"
else
  robot_config="example_fr3_config_no_gripper.yaml"
fi

start_process \
  "FR3 controllers" \
  ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  "robot_config_file:=${robot_config}"
controller_pid="${child_pids[0]}"

if [[ "${arm_mode}" == "duo" || "${arm_mode}" == "left" ]]; then
  wait_for_topic "/left/franka/joint_states" "${controller_pid}"
  wait_for_subscription "/left/gello/raw_joint_states" "${controller_pid}"
fi
if [[ "${arm_mode}" == "duo" || "${arm_mode}" == "right" ]]; then
  wait_for_topic "/right/franka/joint_states" "${controller_pid}"
  wait_for_subscription "/right/gello/raw_joint_states" "${controller_pid}"
fi
echo "FR3 controllers are ready."

if [[ "${end_effector}" == "gripper" ]]; then
  gripper_config="example_fr3_duo_config_franka_hand.yaml"
  [[ "${arm_mode}" == "left" ]] && gripper_config="example_fr3_config_franka_hand.yaml"
  [[ "${arm_mode}" == "right" ]] && gripper_config="example_fr3_right_config_franka_hand.yaml"
  start_process \
    "Franka-hand managers" \
    ros2 launch franka_gripper_manager franka_gripper_client.launch.py \
    "config_file:=${gripper_config}"
  gripper_pid="${child_pids[${#child_pids[@]} - 1]}"
  # Readiness is based on the state sample and command subscription actually
  # needed by replay. Avoid global node-list discovery here: one malformed
  # participant elsewhere in the DDS graph can make `ros2 node list` fail even
  # while these topic endpoints are healthy.
  if [[ "${arm_mode}" == "duo" || "${arm_mode}" == "left" ]]; then
    wait_for_topic "/left/franka_gripper/joint_states" "${gripper_pid}"
    wait_for_subscription "/left/gripper/gripper_client/target_gripper_width_percent" "${gripper_pid}"
  fi
  if [[ "${arm_mode}" == "duo" || "${arm_mode}" == "right" ]]; then
    wait_for_topic "/right/franka_gripper/joint_states" "${gripper_pid}"
    wait_for_subscription "/right/gripper/gripper_client/target_gripper_width_percent" "${gripper_pid}"
  fi
  echo "Franka-hand managers are ready; starting episode replay."
elif [[ "${end_effector}" == "hand" ]]; then
  wuji_conda_env="${WUJI_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"
  declare -a wuji_python=()
  real_exp_build_conda_python_command "${wuji_conda_env}" wuji_python || exit 1
  real_exp_require_conda_python_modules "${wuji_conda_env}" wuji_sdk wujihandpy zmq numpy || die \
    "the '${wuji_conda_env}' Conda environment cannot run Wuji hand replay"
  if [[ "${arm_mode}" == "duo" || "${arm_mode}" == "left" ]]; then
    start_process "Left Wuji hand replay" "${wuji_python[@]}" \
      "${replay_script}" --internal-wuji-hand left --left-hand-command-port 5561 \
      --hand-ip "${left_hand_ip}"
    wait_for_tcp_port 5561 "${child_pids[${#child_pids[@]} - 1]}"
  fi
  if [[ "${arm_mode}" == "duo" || "${arm_mode}" == "right" ]]; then
    start_process "Right Wuji hand replay" "${wuji_python[@]}" \
      "${replay_script}" --internal-wuji-hand right --right-hand-command-port 5562 \
      --hand-ip "${right_hand_ip}"
    wait_for_tcp_port 5562 "${child_pids[${#child_pids[@]} - 1]}"
  fi
  echo "Wuji hand replay is ready; starting episode replay."
else
  echo "End-effector management disabled; starting episode replay."
fi

start_process "LeRobot episode replay" "${ros_python}" "${replay_script}" \
  "${replay_args[@]}" "${replay_setting_args[@]}" \
  --left-hand-command-port 5561 --right-hand-command-port 5562

echo "Replay stack is running. Type 's' + Enter in the replay prompt to begin, or 'q' + Enter to abort."

set +e
completed_pid=""
wait -n -p completed_pid "${child_pids[@]}"
child_status=$?
set -e

echo "${child_names[${completed_pid}]} exited with status ${child_status}." >&2
exit "${child_status}"
