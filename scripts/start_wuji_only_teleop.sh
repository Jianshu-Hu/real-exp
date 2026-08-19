#!/usr/bin/env bash
set -euo pipefail

# Start Wuji glove -> Wuji Hand 2 teleoperation for the selected side(s).

usage() {
  cat <<'EOF'
Usage: ./scripts/start_wuji_only_teleop.sh [--duo|--left|--right] [options]

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

Environment defaults:
  WUJI_LEFT_GLOVE_SN
  WUJI_RIGHT_GLOVE_SN      Defaults to WG1KA06260623515 for this setup.
  WUJI_LEFT_HAND_IP
  WUJI_RIGHT_HAND_IP

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
example_dir="${repository_root}/libs/wuji-retargeting/example"
teleop_program="${example_dir}/teleop_real.py"

[[ -r "${teleop_program}" ]] || die "Wuji teleoperation program not found: ${teleop_program}"
command -v python >/dev/null 2>&1 || die "python is not available in the current environment"
command -v setsid >/dev/null 2>&1 || die "required command not found: setsid"
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
  python -c 'import nlopt, pinocchio, wuji_sdk, wujihandpy, yaml' >/dev/null || die \
    "the current Python environment is missing a Wuji teleoperation dependency"
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
    python "${teleop_program}"
    --input wuji_glove
    --hand "${side}"
    --device-name "${device_name}"
    --config "${config}"
  )
  [[ -z "${glove_sn}" ]] || command+=(--glove-sn "${glove_sn}")
  [[ -z "${hand_ip}" ]] || command+=(--wuji-hand-2-ip "${hand_ip}")

  echo "Starting ${side} Wuji glove/hand teleoperation"
  (
    cd -- "${example_dir}"
    exec 9>&-
    exec setsid -- "${command[@]}"
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
