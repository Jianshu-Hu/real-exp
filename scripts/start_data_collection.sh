#!/usr/bin/env bash
set -euo pipefail

# Start teleoperation, RealSense publishing, and the LeRobot data bridge.
# Usage: ./scripts/start_data_collection.sh --duo|--single --gripper|--no-gripper

usage() {
  cat <<'EOF'
Usage: ./scripts/start_data_collection.sh --duo|--single --gripper|--no-gripper

Arm mode:
  --duo          Start dual-arm teleoperation and the bimanual bridge.
  --single       Start left-arm teleoperation and the two-camera single-arm bridge.

Gripper mode:
  --gripper      Start teleoperation with the matching Franka-hand manager.
  --no-gripper   Start teleoperation without a gripper manager.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

arm_mode=""
gripper_mode=""

if [[ "$#" -eq 1 && "$1" == "--help" ]]; then
  usage
  exit 0
fi

[[ "$#" -eq 2 ]] || {
  usage >&2
  exit 2
}

for argument in "$@"; do
  case "${argument}" in
    --duo)
      [[ -z "${arm_mode}" ]] || die "choose only one of --duo or --single"
      arm_mode="duo"
      ;;
    --single)
      [[ -z "${arm_mode}" ]] || die "choose only one of --duo or --single"
      arm_mode="single"
      ;;
    --gripper)
      [[ -z "${gripper_mode}" ]] || die "choose only one of --gripper or --no-gripper"
      gripper_mode="gripper"
      ;;
    --no-gripper)
      [[ -z "${gripper_mode}" ]] || die "choose only one of --gripper or --no-gripper"
      gripper_mode="no-gripper"
      ;;
    *)
      die "unknown argument: ${argument}"
      ;;
  esac
done

[[ -n "${arm_mode}" ]] || die "one of --duo or --single is required"
[[ -n "${gripper_mode}" ]] || die "one of --gripper or --no-gripper is required"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
teleoperation_script="${script_dir}/start_teleoperation.sh"

if [[ "${arm_mode}" == "duo" ]]; then
  calibrated_gello_config="gello_duo.yaml"
else
  calibrated_gello_config="gello_single.yaml"
fi

[[ -x "${teleoperation_script}" ]] || die "teleoperation script is missing or not executable: ${teleoperation_script}"

# start_teleoperation.sh selects this calibrated GELLO configuration for the
# requested arm mode before starting the FR3 controller and data-collection stack.
calibrated_gello_config_path="${repository_root}/gello_software/ros2/install/franka_gello_state_publisher/share/franka_gello_state_publisher/config/${calibrated_gello_config}"
[[ -r "${calibrated_gello_config_path}" ]] || die "calibrated GELLO config is missing: ${calibrated_gello_config_path}"

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

command -v setsid >/dev/null 2>&1 || die "required command not found: setsid"

declare -a child_pids=()
declare -A child_names=()
shutdown_started=0

start_process() {
  local process_name="$1"
  shift

  echo "Starting ${process_name}: $*"
  setsid -- bash -c 'trap - INT QUIT; exec "$@"' _ "$@" &
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
    echo "Stopping data-collection processes..."
    signal_running_groups INT
    if ! wait_for_groups_to_stop 60; then
      signal_running_groups TERM
    fi
    if ! wait_for_groups_to_stop 30; then
      echo "Force-stopping unresponsive data-collection processes..." >&2
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

echo "Data collection mode: ${arm_mode}; gripper: ${gripper_mode}"

start_process \
  "teleoperation" \
  "${teleoperation_script}" "$@"

# In gripper mode, start_teleoperation.sh waits for every selected arm controller
# to publish commanded joint states before it launches the Franka-hand manager.

start_process \
  "RealSense camera publisher" \
  ros2 launch franka_realsense_camera_publisher cameras.launch.py

if [[ "${arm_mode}" == "single" ]]; then
  start_process \
    "LeRobot data bridge" \
    ros2 launch franka_lerobot_data_bridge bridge.launch.py \
    config_file:=example_single.yaml
else
  start_process \
    "LeRobot data bridge" \
    ros2 launch franka_lerobot_data_bridge bridge.launch.py
fi

echo "Data-collection support stack is running. Press Ctrl-C to stop everything."

set +e
completed_pid=""
wait -n -p completed_pid "${child_pids[@]}"
child_status=$?
set -e

echo "${child_names[${completed_pid}]} exited with status ${child_status}." >&2

exit "${child_status}"
