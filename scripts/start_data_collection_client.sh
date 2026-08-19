#!/usr/bin/env bash
set -euo pipefail

# Start the robot-control side of a split data-collection setup.
# Cameras, the ROS bridge, and the LeRobot recorder run on the data server;
# this host runs only GELLO/Franka teleoperation and robot control.
# Usage: ./scripts/start_data_collection_client.sh --duo|--single --gripper|--no-gripper|--hand

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -eq 1 && "$1" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/start_data_collection_client.sh --duo|--single --gripper|--no-gripper|--hand

Starts only the control-host teleoperation stack. On the data server run:
  ./scripts/start_data_collection_server.sh --duo|--single --gripper|--no-gripper|--hand
EOF
  exit 0
fi

[[ "$#" -eq 2 ]] || {
  echo "Usage: $0 --duo|--single --gripper|--no-gripper|--hand" >&2
  exit 2
}

arm_mode=""
data_mode=""
for argument in "$@"; do
  case "${argument}" in
    --duo|--single)
      [[ -z "${arm_mode}" ]] || { echo "choose only one arm mode" >&2; exit 2; }
      arm_mode="${argument#--}" ;;
    --gripper|--no-gripper|--hand)
      [[ -z "${data_mode}" ]] || { echo "choose only one data mode" >&2; exit 2; }
      data_mode="${argument#--}" ;;
    *) echo "unknown argument: ${argument}" >&2; exit 2 ;;
  esac
done
[[ -n "${arm_mode}" && -n "${data_mode}" ]] || { echo "both arm and data modes are required" >&2; exit 2; }

echo "Starting control-host teleoperation only."
echo "Run scripts/start_data_collection_server.sh on 192.168.50.13 for cameras and recording."
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
arm_side="--${arm_mode}"
[[ "${arm_mode}" == "single" ]] && arm_side="--left"

if [[ "${data_mode}" != "hand" ]]; then
  exec "${script_dir}/start_arm_only_teleop.sh" "${arm_side}" "--${data_mode}"
fi

server_host="${DATA_COLLECTION_SERVER_IP:-192.168.50.13}"
telemetry_port="${HAND_TELEMETRY_PORT:-5558}"
declare -a child_pids=()
shutdown_started=0
shutdown() {
  local status=$?
  [[ "${shutdown_started}" -eq 0 ]] || return
  shutdown_started=1
  trap - EXIT INT TERM
  for child_pid in "${child_pids[@]}"; do
    kill -INT -- "-${child_pid}" 2>/dev/null || true
  done
  for child_pid in "${child_pids[@]}"; do wait "${child_pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid -- "${script_dir}/start_arm_only_teleop.sh" "${arm_side}" --no-gripper &
child_pids+=("$!")
wuji_args=("--${arm_mode}")
[[ "${arm_mode}" == "single" ]] && wuji_args=("--left")
setsid -- "${script_dir}/start_wuji_only_teleop.sh" "${wuji_args[@]}" \
  --telemetry-host "${server_host}" --telemetry-port "${telemetry_port}" &
child_pids+=("$!")
echo "Arm and Wuji hand teleoperation are running. Press Ctrl-C to stop."
set +e
wait -n -p completed_pid "${child_pids[@]}"
status=$?
set -e
exit "${status}"
