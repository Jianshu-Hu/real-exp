#!/usr/bin/env bash
set -euo pipefail

# Unified GELLO/FR3 arm, Franka-gripper, and Wuji-hand teleoperation launcher.

process_start_time() {
  local process_pid="$1"
  awk '{print $22}' "/proc/${process_pid}/stat" 2>/dev/null
}

process_group_is_running() {
  local group_pid="$1"
  ps -o stat= --sid "${group_pid}" 2>/dev/null | grep -qv '^[[:space:]]*Z'
}

wait_for_process_group_to_stop() {
  local group_pid="$1"
  local attempts="$2"
  local attempt
  for ((attempt = 0; attempt < attempts; attempt++)); do
    process_group_is_running "${group_pid}" || return 0
    sleep 0.1
  done
  ! process_group_is_running "${group_pid}"
}

run_arm_watchdog() {
  local launcher_pid="$1"
  local launcher_start_time="$2"
  local group_pid="$3"
  local current_start_time
  while true; do
    current_start_time="$(process_start_time "${launcher_pid}" || true)"
    [[ -n "${current_start_time}" && "${current_start_time}" == "${launcher_start_time}" ]] || break
    sleep 0.2
  done
  process_group_is_running "${group_pid}" || return 0
  kill -s INT -- "-${group_pid}" 2>/dev/null || true
  wait_for_process_group_to_stop "${group_pid}" 10 && return 0
  kill -s TERM -- "-${group_pid}" 2>/dev/null || true
  wait_for_process_group_to_stop "${group_pid}" 30 && return 0
  kill -s KILL -- "-${group_pid}" 2>/dev/null || true
}
run_arm_stack() {
usage() {
  cat <<'EOF'
Internal arm stack mode.

Arm mode:
  --duo          Start left and right GELLO/FR3 teleoperation.
  --left         Start left GELLO/FR3 teleoperation.
  --right        Start right GELLO/FR3 teleoperation.

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
      [[ -z "${arm_mode}" ]] || die "choose only one of --duo, --left, or --right"
      arm_mode="duo"
      ;;
    --left)
      [[ -z "${arm_mode}" ]] || die "choose only one of --duo, --left, or --right"
      arm_mode="left"
      ;;
    --right)
      [[ -z "${arm_mode}" ]] || die "choose only one of --duo, --left, or --right"
      arm_mode="right"
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

[[ -n "${arm_mode}" ]] || die "one of --duo, --left, or --right is required"
[[ -n "${gripper_mode}" ]] || die "one of --gripper or --no-gripper is required"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
# shellcheck source=scripts/conda_env.sh
source "${script_dir}/conda_env.sh"
gello_config_dir="${repository_root}/gello_software/ros2/src/franka_gello_state_publisher/config"
arm_config_dir="${repository_root}/gello_software/ros2/src/franka_fr3_arm_controllers/config"
gripper_config_dir="${repository_root}/gello_software/ros2/src/franka_gripper_manager/config"

# Do not let ROS overlays inherited from an interactive shell select packages
# from a different checkout. The setup files below rebuild the required ROS
# paths in a deterministic order.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH

setup_files=(
  "/opt/ros/humble/setup.bash"
  # Use local_setup for overlays so a generated setup.bash cannot pull in a
  # stale workspace path from the machine where that workspace was built.
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

command -v setsid >/dev/null 2>&1 || die "required command not found: setsid"
command -v env >/dev/null 2>&1 || die "required command not found: env"
command -v timeout >/dev/null 2>&1 || die "required command not found: timeout"
command -v flock >/dev/null 2>&1 || die "required command not found: flock"
command -v fuser >/dev/null 2>&1 || die "required command not found: fuser"

runtime_dir="${XDG_RUNTIME_DIR:-/tmp}"
launcher_lock_path="${runtime_dir}/real-exp-teleoperation-${UID}.lock"
exec 9>"${launcher_lock_path}"
flock -n 9 || die "another arm teleoperation instance is already running"

case "${arm_mode}" in
  duo)
    required_ports=("/dev/ttyUSB_left" "/dev/ttyUSB_right")
    required_topics=(
      "/left/gello/raw_joint_states"
      "/left/gello/accepted_joint_states"
      "/right/gello/raw_joint_states"
      "/right/gello/accepted_joint_states"
    )
    arm_namespaces=("left" "right")
    gello_config="gello_duo.yaml"
    if [[ "${gripper_mode}" == "gripper" ]]; then
      robot_config="example_fr3_duo_config.yaml"
    else
      robot_config="${arm_config_dir}/example_fr3_duo_config_no_gripper.yaml"
    fi
    gripper_config="example_fr3_duo_config_franka_hand.yaml"
    ;;
  left)
    required_ports=("/dev/ttyUSB_left")
    required_topics=("/left/gello/raw_joint_states" "/left/gello/accepted_joint_states")
    arm_namespaces=("left")
    gello_config="gello_single.yaml"
    if [[ "${gripper_mode}" == "gripper" ]]; then
      robot_config="example_fr3_config.yaml"
    else
      robot_config="${arm_config_dir}/example_fr3_config_no_gripper.yaml"
    fi
    gripper_config="example_fr3_config_franka_hand.yaml"
    ;;
  right)
    required_ports=("/dev/ttyUSB_right")
    required_topics=("/right/gello/raw_joint_states" "/right/gello/accepted_joint_states")
    arm_namespaces=("right")
    gello_config="${gello_config_dir}/gello_right.yaml"
    if [[ "${gripper_mode}" == "gripper" ]]; then
      robot_config="${arm_config_dir}/example_fr3_right_config.yaml"
    else
      robot_config="${arm_config_dir}/example_fr3_right_config_no_gripper.yaml"
    fi
    gripper_config="${gripper_config_dir}/example_fr3_right_config_franka_hand.yaml"
    ;;
esac

for config_file in "${gello_config}" "${robot_config}"; do
  if [[ "${config_file}" == /* ]]; then
    [[ -r "${config_file}" ]] || die "teleoperation config is missing or unreadable: ${config_file}"
  fi
done
if [[ "${gripper_mode}" == "gripper" && "${gripper_config}" == /* ]]; then
  [[ -r "${gripper_config}" ]] || die "gripper config is missing or unreadable: ${gripper_config}"
fi

for port in "${required_ports[@]}"; do
  [[ -e "${port}" ]] || die "GELLO serial alias not found: ${port}. Run sudo ./scripts/setup_usb_rules.sh and reconnect the device."
  [[ -r "${port}" && -w "${port}" ]] || die "GELLO serial port is not readable and writable: ${port}. Check dialout group membership."
  port_owners="$(fuser "${port}" 2>/dev/null || true)"
  [[ -z "${port_owners//[[:space:]]/}" ]] || die \
    "GELLO serial port is already in use: ${port} (PID(s):${port_owners}). Stop the existing publisher before retrying."
done

declare -a child_pids=()
declare -A child_names=()
shutdown_started=0
launcher_start_time="$(process_start_time "$$")"
[[ -n "${launcher_start_time}" ]] || die "could not determine launcher process identity"

start_process_group_watchdog() {
  local group_pid="$1"

  setsid -- "${script_dir}/start_teleoperation.sh" \
    --internal-arm-watchdog "$$" "${launcher_start_time}" "${group_pid}" \
    </dev/null >/dev/null 2>&1 &
}

start_process() {
  local process_name="$1"
  shift

  echo "Starting ${process_name}: $*"
  # Background jobs inherit SIGINT/SIGQUIT as ignored from Bash. Reset their
  # dispositions before Bash starts so ROS launch can handle graceful signals.
  setsid -- env --default-signal=INT,QUIT,TERM -- \
    bash -c 'exec 9>&-; exec "$@"' _ "$@" &
  local child_pid=$!
  child_pids+=("${child_pid}")
  child_names["${child_pid}"]="${process_name}"
  start_process_group_watchdog "${child_pid}"
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
  "FR3 controller" \
  ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  "robot_config_file:=${robot_config}"
arm_controller_pid="${child_pids[${#child_pids[@]} - 1]}"

# The accepted Gello target topics are published by the FR3 controllers. Start
# and verify those controllers before starting Gello; otherwise waiting for an
# accepted target before launching the controller creates a startup deadlock.
for namespace in "${arm_namespaces[@]}"; do
  wait_for_arm_controller "${namespace}" "${arm_controller_pid}"
done

start_process \
  "GELLO publisher" \
  ros2 launch franka_gello_state_publisher main.launch.py "config_file:=${gello_config}"
gello_pid="${child_pids[${#child_pids[@]} - 1]}"

for topic in "${required_topics[@]}"; do
  wait_for_topic "${topic}" "${gello_pid}"
done

if [[ "${gripper_mode}" == "gripper" ]]; then
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

}

run_wuji_stack() {
usage() {
  cat <<'EOF'
Internal Wuji stack mode.

Side selection:
  --duo                    Start left and right Wuji teleoperation.
  --left                   Start only the left glove/hand.
  --right                  Start only the right glove/hand.
                           With no arguments, --right is used for compatibility.

Device selection:
  --left-glove-sn SN       Left Wuji Glove serial.
  --right-glove-sn SN      Right Wuji Glove serial.
  --left-hand-ip ADDR      Left Wuji Hand 2 SDK address (IP:port).
  --right-hand-ip ADDR     Right Wuji Hand 2 SDK address (IP:port).
  --telemetry-host IP      Data-server host for hand telemetry (default:
                           DATA_COLLECTION_SERVER_IP or 192.168.50.13).
  --telemetry-port PORT    Data-server ZMQ port for hand telemetry (default:
                           HAND_TELEMETRY_PORT or 5558).

Environment defaults:
  WUJI_LEFT_GLOVE_SN
  WUJI_RIGHT_GLOVE_SN      Defaults to WG1KA06260623515 for this setup.
  WUJI_LEFT_HAND_IP
  WUJI_RIGHT_HAND_IP
  HAND_TELEMETRY_RATE_HZ   Hand telemetry publish rate (default: 15 Hz; keep
                           this equal to the bridge sample_rate_hz).

For a single side, an omitted glove serial or hand address lets wuji_sdk
auto-discover the device. For --duo, both glove serials and both hand addresses
are required so the two SDK processes cannot select or probe each other's device.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

side_mode=""
left_option_given=0
right_option_given=0
validate_only=0
left_glove_sn="${WUJI_LEFT_GLOVE_SN:-}"
right_glove_sn="${WUJI_RIGHT_GLOVE_SN:-WG1KA06260623515}"
left_hand_ip="${WUJI_LEFT_HAND_IP:-}"
right_hand_ip="${WUJI_RIGHT_HAND_IP:-}"
telemetry_host="${DATA_COLLECTION_SERVER_IP:-192.168.50.13}"
telemetry_port="${HAND_TELEMETRY_PORT:-5558}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --duo|--left|--right)
      [[ -z "${side_mode}" ]] || die "choose only one of --duo, --left, or --right"
      side_mode="${1#--}"
      shift
      ;;
    --left-glove-sn)
      [[ "$#" -ge 2 ]] || die "--left-glove-sn requires a serial number"
      left_glove_sn="$2"
      left_option_given=1
      shift 2
      ;;
    --right-glove-sn)
      [[ "$#" -ge 2 ]] || die "--right-glove-sn requires a serial number"
      right_glove_sn="$2"
      right_option_given=1
      shift 2
      ;;
    --left-hand-ip)
      [[ "$#" -ge 2 ]] || die "--left-hand-ip requires an IP:port address"
      left_hand_ip="$2"
      left_option_given=1
      shift 2
      ;;
    --right-hand-ip)
      [[ "$#" -ge 2 ]] || die "--right-hand-ip requires an IP:port address"
      right_hand_ip="$2"
      right_option_given=1
      shift 2
      ;;
    --telemetry-host)
      [[ "$#" -ge 2 ]] || die "--telemetry-host requires an address"
      telemetry_host="$2"
      shift 2
      ;;
    --telemetry-port)
      [[ "$#" -ge 2 ]] || die "--telemetry-port requires a port"
      telemetry_port="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    --validate-only)
      validate_only=1
      shift
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ -z "${side_mode}" ]]; then
  if [[ "${left_option_given}" -eq 0 && "${right_option_given}" -eq 0 ]]; then
    side_mode="right"
  else
    die "one of --duo, --left, or --right is required when passing device options"
  fi
fi

[[ "${side_mode}" != "left" || "${right_option_given}" -eq 0 ]] || die \
  "right-side device options cannot be used with --left"
[[ "${side_mode}" != "right" || "${left_option_given}" -eq 0 ]] || die \
  "left-side device options cannot be used with --right"

if [[ "${side_mode}" == "duo" ]]; then
  [[ -n "${left_glove_sn}" && -n "${right_glove_sn}" ]] || die \
    "--duo requires both glove serials; pass --left-glove-sn/--right-glove-sn or set WUJI_LEFT_GLOVE_SN/WUJI_RIGHT_GLOVE_SN"
  [[ -n "${left_hand_ip}" && -n "${right_hand_ip}" ]] || die \
    "--duo requires both hand addresses; pass --left-hand-ip/--right-hand-ip or set WUJI_LEFT_HAND_IP/WUJI_RIGHT_HAND_IP"
  [[ "${left_glove_sn}" != "${right_glove_sn}" ]] || die "left and right glove serials must differ"
  [[ "${left_hand_ip}" != "${right_hand_ip}" ]] || die "left and right hand addresses must differ"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
# shellcheck source=scripts/conda_env.sh
source "${script_dir}/conda_env.sh"
example_dir="${repository_root}/libs/wuji-retargeting/example"
teleop_program="${repository_root}/utils/wuji_telemetry_proxy.py"

[[ -r "${teleop_program}" ]] || die "Wuji teleoperation adapter not found: ${teleop_program}"
wuji_conda_env="${WUJI_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"
declare -a wuji_python=()
real_exp_build_conda_python_command "${wuji_conda_env}" wuji_python || exit 1
command -v setsid >/dev/null 2>&1 || die "required command not found: setsid"
command -v env >/dev/null 2>&1 || die "required command not found: env"
command -v flock >/dev/null 2>&1 || die "required command not found: flock"

declare -a selected_sides=()
case "${side_mode}" in
  duo) selected_sides=(left right) ;;
  left) selected_sides=(left) ;;
  right) selected_sides=(right) ;;
esac

for side in "${selected_sides[@]}"; do
  config_path="${example_dir}/config/adaptive_analytical_wuji_glove_wuji_hand_2_${side}.yaml"
  [[ -r "${config_path}" ]] || die "Wuji ${side} config not found: ${config_path}"
done

if [[ "${validate_only}" -eq 1 ]]; then
  real_exp_require_conda_python_modules "${wuji_conda_env}" \
    nlopt pinocchio wuji_sdk wujihandpy yaml zmq || die \
    "the '${wuji_conda_env}' Conda environment cannot import Wuji teleoperation dependencies"
  exit 0
fi

runtime_dir="${XDG_RUNTIME_DIR:-/tmp}"
exec 9>"${runtime_dir}/real-exp-wuji-teleoperation-${UID}.lock"
flock -n 9 || die "another Wuji teleoperation instance is already running"

declare -a child_pids=()
declare -A child_names=()
shutdown_started=0

start_wuji_side() {
  local side="$1"
  local glove_sn hand_ip device_name config
  local -a command

  if [[ "${side}" == "left" ]]; then
    glove_sn="${left_glove_sn}"
    hand_ip="${left_hand_ip}"
  else
    glove_sn="${right_glove_sn}"
    hand_ip="${right_hand_ip}"
  fi

  device_name="${side}_glove"
  config="config/adaptive_analytical_wuji_glove_wuji_hand_2_${side}.yaml"
  [[ -r "${example_dir}/${config}" ]] || die "Wuji ${side} config not found: ${example_dir}/${config}"

  command=(
    "${wuji_python[@]}" "${teleop_program}"
    --input wuji_glove
    --hand "${side}"
    --device-name "${device_name}"
    --config "${config}"
  )
  [[ -z "${glove_sn}" ]] || command+=(--glove-sn "${glove_sn}")
  [[ -z "${hand_ip}" ]] || command+=(--wuji-hand-2-ip "${hand_ip}")
  [[ -z "${telemetry_host}" ]] || command+=(--telemetry-host "${telemetry_host}")
  [[ "${telemetry_port}" == "0" ]] || command+=(--telemetry-port "${telemetry_port}")

  echo "Starting ${side} Wuji glove/hand teleoperation"
  (
    cd -- "${example_dir}"
    exec 9>&-
    # Restore signals ignored by Bash for asynchronous jobs before starting
    # the per-hand process in its own session.
    # Wuji standalone teleoperation does not import ROS Python modules. A
    # caller may have sourced ROS setup files, which prepend the ROS Humble
    # Python 3.10 Pinocchio package and libraries ahead of this Python 3.12
    # environment. Clear both paths so imports resolve from the active
    # environment only.
    exec setsid -- env --default-signal=INT,QUIT,TERM PYTHONPATH= LD_LIBRARY_PATH= "${command[@]}"
  ) &
  local child_pid=$!
  child_pids+=("${child_pid}")
  child_names["${child_pid}"]="${side} Wuji teleoperation"
}

signal_running_groups() {
  local signal_name="$1"
  local child_pid
  for child_pid in "${child_pids[@]}"; do
    kill -s "${signal_name}" -- "-${child_pid}" 2>/dev/null || true
  done
}

groups_are_running() {
  local child_pid
  for child_pid in "${child_pids[@]}"; do
    if ps -o stat= --sid "${child_pid}" 2>/dev/null | grep -qv '^[[:space:]]*Z'; then
      return 0
    fi
  done
  return 1
}

wait_for_groups_to_stop() {
  local attempts="$1"
  local attempt
  for ((attempt = 0; attempt < attempts; attempt++)); do
    groups_are_running || return 0
    sleep 0.1
  done
  return 1
}

shutdown() {
  local exit_status=$?
  local child_pid
  [[ "${shutdown_started}" -eq 0 ]] || return
  shutdown_started=1
  trap - EXIT INT TERM

  if [[ "${#child_pids[@]}" -gt 0 ]]; then
    echo "Stopping Wuji teleoperation processes..."
    signal_running_groups INT
    wait_for_groups_to_stop 20 || signal_running_groups TERM
    wait_for_groups_to_stop 30 || signal_running_groups KILL
    for child_pid in "${child_pids[@]}"; do
      wait "${child_pid}" 2>/dev/null || true
    done
  fi
  exit "${exit_status}"
}

trap shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for side in "${selected_sides[@]}"; do
  start_wuji_side "${side}"
done

echo "Wuji teleoperation is running. Press Ctrl-C to stop the selected hand(s)."
set +e
completed_pid=""
wait -n -p completed_pid "${child_pids[@]}"
child_status=$?
set -e
echo "${child_names[${completed_pid}]} exited with status ${child_status}." >&2
exit "${child_status}"

}

case "${1:-}" in
  --internal-arm-watchdog)
    [[ "$#" -eq 4 ]] || exit 2
    exec 9>&-
    run_arm_watchdog "$2" "$3" "$4"
    exit 0
    ;;
  --internal-arm-stack)
    shift
    run_arm_stack "$@"
    exit $?
    ;;
  --internal-wuji-stack)
    shift
    run_wuji_stack "$@"
    exit $?
    ;;
esac


usage() {
  cat <<'EOF'
Usage: ./scripts/start_teleoperation.sh --arm|--gripper|--hand --duo|--left|--right [Wuji options]

Arm selection:
  --duo                    Use both left and right arms.
  --left                   Use only the left arm.
  --right                  Use only the right arm.

End effector:
  --arm                    Use the arm only; no end-effector process.
  --gripper                Use the matching Franka gripper(s).
  --hand                   Use the matching Wuji glove/hand pair(s).

Wuji options (used only with --hand):
  --left-glove-sn SN       Left Wuji Glove serial.
  --right-glove-sn SN      Right Wuji Glove serial.
  --left-hand-ip ADDR      Left Wuji Hand 2 SDK address (IP:port).
  --right-hand-ip ADDR     Right Wuji Hand 2 SDK address (IP:port).
  --telemetry-host IP      Data-server host for hand telemetry (default:
                           DATA_COLLECTION_SERVER_IP or 192.168.50.13).
  --telemetry-port PORT    Data-server ZMQ port for hand telemetry (default:
                           HAND_TELEMETRY_PORT or 5558).

Examples:
  ./scripts/start_teleoperation.sh --gripper --left
  ./scripts/start_teleoperation.sh --hand --right
  ./scripts/start_teleoperation.sh --arm --duo
  ./scripts/start_teleoperation.sh --hand --duo \
    --left-glove-sn <LEFT_SN> --right-glove-sn <RIGHT_SN> \
    --left-hand-ip <LEFT_IP:PORT> --right-hand-ip <RIGHT_IP:PORT>

The Wuji options can also come from WUJI_LEFT_GLOVE_SN,
WUJI_RIGHT_GLOVE_SN, WUJI_LEFT_HAND_IP, and WUJI_RIGHT_HAND_IP.
Hand telemetry defaults to 15 Hz; set HAND_TELEMETRY_RATE_HZ when changing the
bridge sample rate.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

side_mode=""
end_effector=""
declare -a wuji_options=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --duo|--left|--right)
      [[ -z "${side_mode}" ]] || die "choose only one of --duo, --left, or --right"
      side_mode="${1#--}"
      shift
      ;;
    --arm|--gripper|--hand)
      [[ -z "${end_effector}" ]] || die "choose only one of --arm, --gripper, or --hand"
      end_effector="${1#--}"
      shift
      ;;
    --left-glove-sn|--right-glove-sn|--left-hand-ip|--right-hand-ip|--telemetry-host|--telemetry-port)
      [[ "$#" -ge 2 ]] || die "$1 requires a value"
      wuji_options+=("$1" "$2")
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

[[ -n "${side_mode}" && -n "${end_effector}" ]] || {
  usage >&2
  exit 2
}
[[ "${end_effector}" == "hand" || "${#wuji_options[@]}" -eq 0 ]] || die \
  "Wuji and telemetry options are valid only with --hand"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v setsid >/dev/null 2>&1 || die "required command not found: setsid"
command -v env >/dev/null 2>&1 || die "required command not found: env"

declare -a child_pids=()
declare -A child_names=()
shutdown_started=0

start_component() {
  local component_name="$1"
  shift
  echo "Starting ${component_name}: $*"
  # Bash starts asynchronous jobs with SIGINT and SIGQUIT ignored. That signal
  # disposition survives setsid and exec, and an ignored signal cannot be
  # restored from inside a newly started Bash script. Reset it in env before
  # executing the component so the nested supervisor can handle our SIGINT.
  setsid -- env --default-signal=INT,QUIT,TERM -- "$@" &
  local child_pid=$!
  child_pids+=("${child_pid}")
  child_names["${child_pid}"]="${component_name}"
}

signal_running_groups() {
  local signal_name="$1"
  local child_pid

  for child_pid in "${child_pids[@]}"; do
    if kill -0 -- "-${child_pid}" 2>/dev/null; then
      kill -s "${signal_name}" -- "-${child_pid}" 2>/dev/null || true
    fi
  done
}

groups_are_running() {
  local child_pid

  for child_pid in "${child_pids[@]}"; do
    if ps -o stat= --sid "${child_pid}" 2>/dev/null | grep -qv '^[[:space:]]*Z'; then
      return 0
    fi
  done
  return 1
}

wait_for_groups_to_stop() {
  local attempts="$1"
  local attempt

  for ((attempt = 0; attempt < attempts; attempt++)); do
    groups_are_running || return 0
    sleep 0.1
  done
  return 1
}

shutdown() {
  local exit_status=$?
  local child_pid
  [[ "${shutdown_started}" -eq 0 ]] || return
  shutdown_started=1
  trap - EXIT INT TERM

  if [[ "${#child_pids[@]}" -gt 0 ]]; then
    echo "Stopping the complete teleoperation stack..."
    signal_running_groups INT
    if ! wait_for_groups_to_stop 60; then
      echo "Teleoperation processes did not stop after SIGINT; sending SIGTERM..." >&2
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

if [[ "${end_effector}" == "arm" ]]; then
  start_component "FR3 arm" \
    bash "${script_dir}/start_teleoperation.sh" \
    --internal-arm-stack "--${side_mode}" --no-gripper
elif [[ "${end_effector}" == "gripper" ]]; then
  start_component "FR3 arm and Franka gripper" \
    bash "${script_dir}/start_teleoperation.sh" \
    --internal-arm-stack "--${side_mode}" --gripper
else
  bash "${script_dir}/start_teleoperation.sh" \
    --internal-wuji-stack "--${side_mode}" "${wuji_options[@]}" --validate-only
  start_component "FR3 arm" \
    bash "${script_dir}/start_teleoperation.sh" \
    --internal-arm-stack "--${side_mode}" --no-gripper
  start_component "Wuji glove and hand" \
    bash "${script_dir}/start_teleoperation.sh" \
    --internal-wuji-stack "--${side_mode}" "${wuji_options[@]}"
fi

echo "Unified teleoperation is running. Press Ctrl-C to stop the complete stack."
set +e
completed_pid=""
wait -n -p completed_pid "${child_pids[@]}"
child_status=$?
set -e
echo "${child_names[${completed_pid}]} exited with status ${child_status}." >&2
exit "${child_status}"
