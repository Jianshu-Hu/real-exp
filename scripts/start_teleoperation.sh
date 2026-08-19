#!/usr/bin/env bash
set -euo pipefail

# Unified GELLO/FR3 arm, Franka-gripper, and Wuji-hand teleoperation launcher.

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
  --telemetry-host IP      Data-server host for hand telemetry.
  --telemetry-port PORT    Data-server ZMQ port for hand telemetry.

Examples:
  ./scripts/start_teleoperation.sh --gripper --left
  ./scripts/start_teleoperation.sh --hand --right
  ./scripts/start_teleoperation.sh --arm --duo
  ./scripts/start_teleoperation.sh --hand --duo \
    --left-glove-sn <LEFT_SN> --right-glove-sn <RIGHT_SN> \
    --left-hand-ip <LEFT_IP:PORT> --right-hand-ip <RIGHT_IP:PORT>

The Wuji options can also come from WUJI_LEFT_GLOVE_SN,
WUJI_RIGHT_GLOVE_SN, WUJI_LEFT_HAND_IP, and WUJI_RIGHT_HAND_IP.
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
arm_launcher="${script_dir}/start_arm_only_teleop.sh"
wuji_launcher="${script_dir}/start_wuji_only_teleop.sh"
[[ -x "${arm_launcher}" ]] || die "arm launcher is missing or not executable: ${arm_launcher}"
[[ "${end_effector}" != "hand" || -x "${wuji_launcher}" ]] || die \
  "Wuji launcher is missing or not executable: ${wuji_launcher}"
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
    "${arm_launcher}" "--${side_mode}" --no-gripper
elif [[ "${end_effector}" == "gripper" ]]; then
  start_component "FR3 arm and Franka gripper" \
    "${arm_launcher}" "--${side_mode}" --gripper
else
  "${wuji_launcher}" "--${side_mode}" "${wuji_options[@]}" --validate-only
  start_component "FR3 arm" \
    "${arm_launcher}" "--${side_mode}" --no-gripper
  start_component "Wuji glove and hand" \
    "${wuji_launcher}" "--${side_mode}" "${wuji_options[@]}"
fi

echo "Unified teleoperation is running. Press Ctrl-C to stop the complete stack."
set +e
completed_pid=""
wait -n -p completed_pid "${child_pids[@]}"
child_status=$?
set -e
echo "${child_names[${completed_pid}]} exited with status ${child_status}." >&2
exit "${child_status}"
