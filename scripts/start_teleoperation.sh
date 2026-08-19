#!/usr/bin/env bash
set -euo pipefail

# Unified GELLO/FR3 arm plus Franka-gripper or Wuji-hand teleoperation launcher.

usage() {
  cat <<'EOF'
Usage: ./scripts/start_teleoperation.sh --duo|--left|--right --gripper|--hand [Wuji options]

Arm selection:
  --duo                    Use both left and right arms.
  --left                   Use only the left arm.
  --right                  Use only the right arm.

End effector:
  --gripper                Use the matching Franka gripper(s).
  --hand                   Use the matching Wuji glove/hand pair(s).

Wuji options (used only with --hand):
  --left-glove-sn SN       Left Wuji Glove serial.
  --right-glove-sn SN      Right Wuji Glove serial.
  --left-hand-ip ADDR      Left Wuji Hand 2 SDK address (IP:port).
  --right-hand-ip ADDR     Right Wuji Hand 2 SDK address (IP:port).

Examples:
  ./scripts/start_teleoperation.sh --left --gripper
  ./scripts/start_teleoperation.sh --right --hand
  ./scripts/start_teleoperation.sh --duo --gripper
  ./scripts/start_teleoperation.sh --duo --hand \
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
    --gripper|--hand)
      [[ -z "${end_effector}" ]] || die "choose only one of --gripper or --hand"
      end_effector="${1#--}"
      shift
      ;;
    --left-glove-sn|--right-glove-sn|--left-hand-ip|--right-hand-ip)
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
  "Wuji device options are valid only with --hand"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
arm_launcher="${script_dir}/start_arm_only_teleop.sh"
wuji_launcher="${script_dir}/start_wuji_only_teleop.sh"
[[ -x "${arm_launcher}" ]] || die "arm launcher is missing or not executable: ${arm_launcher}"
[[ "${end_effector}" != "hand" || -x "${wuji_launcher}" ]] || die \
  "Wuji launcher is missing or not executable: ${wuji_launcher}"
command -v setsid >/dev/null 2>&1 || die "required command not found: setsid"

declare -a child_pids=()
declare -A child_names=()
shutdown_started=0

start_component() {
  local component_name="$1"
  shift
  echo "Starting ${component_name}: $*"
  setsid -- "$@" &
  local child_pid=$!
  child_pids+=("${child_pid}")
  child_names["${child_pid}"]="${component_name}"
}

shutdown() {
  local exit_status=$?
  local child_pid
  [[ "${shutdown_started}" -eq 0 ]] || return
  shutdown_started=1
  trap - EXIT INT TERM

  if [[ "${#child_pids[@]}" -gt 0 ]]; then
    echo "Stopping the complete teleoperation stack..."
    for child_pid in "${child_pids[@]}"; do
      kill -s INT -- "-${child_pid}" 2>/dev/null || true
    done
    for child_pid in "${child_pids[@]}"; do
      wait "${child_pid}" 2>/dev/null || true
    done
  fi
  exit "${exit_status}"
}

trap shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "${end_effector}" == "gripper" ]]; then
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
