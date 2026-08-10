#!/usr/bin/env bash
set -euo pipefail

# Start the bimanual FR3 replay stack and replay one LeRobot episode.
# Usage: bash scripts/replay.sh --dataset-root data/my_dataset --episode 0 [replay options]

usage() {
  cat <<'EOF'
Usage: bash scripts/replay.sh --dataset-root DATASET --episode N [options]

The script starts the dual-arm FR3 controllers, both Franka-hand managers, and
then runs data_collection/replay_lerobot_episode.py with the supplied options.
Use --no-gripper to skip the Franka-hand managers and gripper replay.
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
replay_script="${repository_root}/data_collection/replay_lerobot_episode.py"
[[ -f "${replay_script}" ]] || die "replay script not found: ${replay_script}"

replay_args=("$@")
use_gripper=1
use_dry_run=0
for argument in "${replay_args[@]}"; do
  if [[ "${argument}" == "--no-gripper" ]]; then
    use_gripper=0
  elif [[ "${argument}" == "--dry-run" ]]; then
    use_dry_run=1
  fi
done

setup_files=(
  "/opt/ros/humble/setup.bash"
  "${HOME}/franka_ros2_ws/install/setup.bash"
  "${repository_root}/gello_software/ros2/install/setup.bash"
)

# Generated ROS environment hooks may inspect optional variables before defining them.
set +u
for setup_file in "${setup_files[@]}"; do
  [[ -r "${setup_file}" ]] || die "ROS setup file is missing or unreadable: ${setup_file}"
  # shellcheck disable=SC1090
  source "${setup_file}"
done
set -u

for command_name in setsid timeout python3; do
  command -v "${command_name}" >/dev/null 2>&1 || die "required command not found: ${command_name}"
done

# Background jobs may receive /dev/null on stdin when job control is enabled.
# Prefer a direct terminal handle so the replay prompt remains interactive
# after the child is placed in its own process group. Fall back to the caller's
# stdin for redirected/non-interactive invocations.
if [[ -r /dev/tty && -w /dev/tty ]]; then
  exec 10<>/dev/tty
else
  exec 10<&0
fi

if [[ "${use_dry_run}" -eq 1 ]]; then
  echo "Dry-run requested; skipping ROS controller and gripper startup."
  exec python3 "${replay_script}" "${replay_args[@]}"
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

wait_for_node() {
  local node_name="$1"
  local process_pid="$2"
  local process_description="$3"
  local timeout_seconds=90
  local deadline=$((SECONDS + timeout_seconds))

  while ((SECONDS < deadline)); do
    if ! kill -0 "${process_pid}" 2>/dev/null; then
      wait "${process_pid}" || true
      die "${process_description} exited before ROS node ${node_name} became available"
    fi
    if timeout 5s ros2 node list --no-daemon 2>/dev/null | grep -Fxq "${node_name}"; then
      echo "Ready: ROS node ${node_name}"
      return 0
    fi
    sleep 0.5
  done

  die "timed out after ${timeout_seconds}s waiting for ROS node ${node_name}"
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
      echo "Ready: gripper manager subscription ${topic}"
      return 0
    fi
    sleep 0.5
  done

  die "timed out after ${timeout_seconds}s waiting for gripper manager subscription ${topic}"
}

echo "Starting bimanual FR3 replay stack."
echo "The robot will receive recorded commands after you confirm replay with 's'."

start_process \
  "FR3 controllers" \
  ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  robot_config_file:=example_fr3_duo_config.yaml
controller_pid="${child_pids[0]}"

wait_for_node "/left/controller_manager" "${controller_pid}" "FR3 controller launch"
wait_for_node "/right/controller_manager" "${controller_pid}" "FR3 controller launch"
wait_for_topic "/left/franka/joint_states" "${controller_pid}"
wait_for_topic "/right/franka/joint_states" "${controller_pid}"
echo "FR3 controllers are ready; starting Franka-hand managers."

if [[ "${use_gripper}" -eq 1 ]]; then
  start_process \
    "Franka-hand managers" \
    ros2 launch franka_gripper_manager franka_gripper_client.launch.py \
    config_file:=example_fr3_duo_config_franka_hand.yaml
  gripper_pid="${child_pids[${#child_pids[@]} - 1]}"
  # The executable creates a node named `franka_gripper_client` inside each
  # configured namespace (see the launch output: /left.franka_gripper_client).
  wait_for_node "/left/franka_gripper_client" "${gripper_pid}" "Franka-hand manager launch"
  wait_for_node "/right/franka_gripper_client" "${gripper_pid}" "Franka-hand manager launch"
  wait_for_topic "/left/franka_gripper/joint_states" "${gripper_pid}"
  wait_for_topic "/right/franka_gripper/joint_states" "${gripper_pid}"
  wait_for_subscription "/left/gripper/gripper_client/target_gripper_width_percent" "${gripper_pid}"
  wait_for_subscription "/right/gripper/gripper_client/target_gripper_width_percent" "${gripper_pid}"
  echo "Franka-hand managers are ready; starting episode replay."
else
  echo "Gripper management disabled; starting episode replay."
fi

start_process "LeRobot episode replay" python3 "${replay_script}" "${replay_args[@]}"

echo "Replay stack is running. Type 's' + Enter in the replay prompt to begin, or 'q' + Enter to abort."

set +e
completed_pid=""
wait -n -p completed_pid "${child_pids[@]}"
child_status=$?
set -e

echo "${child_names[${completed_pid}]} exited with status ${child_status}." >&2
exit "${child_status}"
