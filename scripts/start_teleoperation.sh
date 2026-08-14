#!/usr/bin/env bash
set -euo pipefail

# Start the GELLO publisher, FR3 controller, and optional Franka-hand manager.
# Usage: ./scripts/start_teleoperation.sh --duo|--single --gripper|--no-gripper

usage() {
  cat <<'EOF'
Usage: ./scripts/start_teleoperation.sh --duo|--single --gripper|--no-gripper

Arm mode:
  --duo          Start left and right GELLO/FR3 teleoperation.
  --single       Start left GELLO/FR3 teleoperation.

Gripper mode:
  --gripper      Start the matching Franka-hand manager.
  --no-gripper   Do not start a gripper manager.
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
command -v timeout >/dev/null 2>&1 || die "required command not found: timeout"

required_ports=("/dev/ttyUSB_left")
required_topics=("/left/gello/raw_joint_states" "/left/gello/joint_states")
arm_namespaces=("left")
if [[ "${arm_mode}" == "duo" ]]; then
  required_ports+=("/dev/ttyUSB_right")
  required_topics+=("/right/gello/raw_joint_states" "/right/gello/joint_states")
  arm_namespaces+=("right")
fi

for port in "${required_ports[@]}"; do
  [[ -e "${port}" ]] || die "GELLO serial alias not found: ${port}. Run sudo ./scripts/setup_usb_rules.sh and reconnect the device."
  [[ -r "${port}" && -w "${port}" ]] || die "GELLO serial port is not readable and writable: ${port}. Check dialout group membership."
done

if [[ "${arm_mode}" == "duo" ]]; then
  gello_config="gello_duo.yaml"
  robot_config="example_fr3_duo_config.yaml"
  gripper_config="example_fr3_duo_config_franka_hand.yaml"
else
  gello_config="gello_single.yaml"
  robot_config="example_fr3_config.yaml"
  gripper_config="example_fr3_config_franka_hand.yaml"
fi

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
    echo "Stopping teleoperation processes..."
    signal_running_groups INT
    if ! wait_for_groups_to_stop 10; then
      signal_running_groups TERM
    fi
    if ! wait_for_groups_to_stop 30; then
      echo "Force-stopping unresponsive teleoperation processes..." >&2
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
  local publisher_pid="$2"
  local timeout_seconds=20
  local deadline=$((SECONDS + timeout_seconds))

  while ((SECONDS < deadline)); do
    if ! kill -0 "${publisher_pid}" 2>/dev/null; then
      wait "${publisher_pid}" || true
      die "GELLO publisher exited before ${topic} became available"
    fi

    if ros2 topic type "${topic}" --no-daemon 2>/dev/null | grep -qx "sensor_msgs/msg/JointState"; then
      echo "Ready: ${topic}"
      return 0
    fi
    sleep 0.5
  done

  die "timed out after ${timeout_seconds}s waiting for ${topic}. Check the GELLO publisher output and USB mapping."
}

wait_for_arm_controller() {
  local namespace="$1"
  local controller_launch_pid="$2"
  local timeout_seconds=60
  local deadline=$((SECONDS + timeout_seconds))
  local commanded_topic="/${namespace}/franka/commanded_joint_states"

  while ((SECONDS < deadline)); do
    if ! kill -0 "${controller_launch_pid}" 2>/dev/null; then
      wait "${controller_launch_pid}" || true
      die "FR3 controller launch exited before ${namespace} arm became active"
    fi

    if timeout 5s ros2 topic echo \
      "${commanded_topic}" sensor_msgs/msg/JointState --once --no-daemon \
      >/dev/null 2>&1; then
      echo "Ready: ${commanded_topic} is publishing"
      return 0
    fi
  done

  die "timed out after ${timeout_seconds}s waiting for ${commanded_topic} to publish"
}

echo "Teleoperation mode: ${arm_mode}; gripper: ${gripper_mode}"
echo "Keep the GELLO arm(s) in a safe pose while the FR3 controller starts."

start_process \
  "GELLO publisher" \
  ros2 launch franka_gello_state_publisher main.launch.py "config_file:=${gello_config}"
gello_pid="${child_pids[0]}"

for topic in "${required_topics[@]}"; do
  wait_for_topic "${topic}" "${gello_pid}"
done

start_process \
  "FR3 controller" \
  ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  "robot_config_file:=${robot_config}"
arm_controller_pid="${child_pids[${#child_pids[@]} - 1]}"

if [[ "${gripper_mode}" == "gripper" ]]; then
  for namespace in "${arm_namespaces[@]}"; do
    wait_for_arm_controller "${namespace}" "${arm_controller_pid}"
  done

  start_process \
    "Franka-hand manager" \
    ros2 launch franka_gripper_manager franka_gripper_client.launch.py \
    "config_file:=${gripper_config}"
fi

echo "Teleoperation is running. Press Ctrl-C to stop the complete stack."

set +e
completed_pid=""
wait -n -p completed_pid "${child_pids[@]}"
child_status=$?
set -e

echo "${child_names[${completed_pid}]} exited with status ${child_status}." >&2

exit "${child_status}"
