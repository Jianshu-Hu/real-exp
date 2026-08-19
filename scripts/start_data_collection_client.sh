#!/usr/bin/env bash
set -euo pipefail

# Start the robot-control side of a split data-collection setup.
# Cameras, the ROS bridge, and the LeRobot recorder run on the data server;
# this host runs only GELLO/Franka teleoperation and robot control.
# Usage: ./scripts/start_data_collection_client.sh --duo|--single --gripper|--no-gripper

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -eq 1 && "$1" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/start_data_collection_client.sh --duo|--single --gripper|--no-gripper

Starts only the control-host teleoperation stack. On the data server run:
  ./scripts/start_data_collection_server.sh --duo|--single --gripper|--no-gripper
EOF
  exit 0
fi

[[ "$#" -eq 2 ]] || {
  echo "Usage: $0 --duo|--single --gripper|--no-gripper" >&2
  exit 2
}

echo "Starting control-host teleoperation only."
echo "Run scripts/start_data_collection_server.sh on 192.168.50.13 for cameras and recording."
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
arm_arguments=("$@")
for index in "${!arm_arguments[@]}"; do
  if [[ "${arm_arguments[${index}]}" == "--single" ]]; then
    # The data-server interface still calls this mode "single"; the arm-only
    # launcher uses the physically explicit side name.
    arm_arguments[${index}]="--left"
  fi
done
exec "${script_dir}/start_arm_only_teleop.sh" "${arm_arguments[@]}"
