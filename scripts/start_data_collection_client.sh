#!/usr/bin/env bash
set -euo pipefail

# Start the robot-control side of a split data-collection setup.
# Cameras, the ROS bridge, and the LeRobot recorder run on the data server;
# this host runs only GELLO/Franka teleoperation and robot control.
# Usage: ./scripts/start_data_collection_client.sh --duo|--left|--right --arm|--gripper|--hand

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -eq 1 && "$1" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/start_data_collection_client.sh --duo|--left|--right --arm|--gripper|--hand

Starts only the control-host teleoperation stack. On the data server run:
  ./scripts/start_data_collection_server.sh --duo|--left|--right --arm|--gripper|--hand
EOF
  exit 0
fi

[[ "$#" -eq 2 ]] || {
  echo "Usage: $0 --duo|--left|--right --arm|--gripper|--hand" >&2
  exit 2
}

arm_mode=""
data_mode=""
for argument in "$@"; do
  case "${argument}" in
    --duo|--left|--right|--single)
      [[ -z "${arm_mode}" ]] || { echo "choose only one arm mode" >&2; exit 2; }
      arm_mode="${argument#--}"
      [[ "${arm_mode}" != "single" ]] || arm_mode="left" ;;
    --arm|--gripper|--no-gripper|--hand)
      [[ -z "${data_mode}" ]] || { echo "choose only one data mode" >&2; exit 2; }
      data_mode="${argument#--}"
      [[ "${data_mode}" != "no-gripper" ]] || data_mode="arm" ;;
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
teleop_launcher="${script_dir}/start_teleoperation.sh"

if [[ "${data_mode}" == "gripper" ]]; then
  exec "${teleop_launcher}" --gripper "${arm_side}"
elif [[ "${data_mode}" == "arm" ]]; then
  exec "${teleop_launcher}" --arm "${arm_side}"
fi

server_host="${DATA_COLLECTION_SERVER_IP:-192.168.50.13}"
telemetry_port="${HAND_TELEMETRY_PORT:-5558}"
exec "${teleop_launcher}" --hand "${arm_side}" \
  --telemetry-host "${server_host}" --telemetry-port "${telemetry_port}"
