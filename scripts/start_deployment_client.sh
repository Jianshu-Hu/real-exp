#!/usr/bin/env bash
set -euo pipefail

# Robot-control side of deployment. Run on the robot computer.
# The policy executor is intentionally a separate command and safety gate.

usage() {
  cat <<'EOF'
Usage: ./scripts/start_deployment_client.sh [options]

Options:
  --robot-config FILE    FR3 config (default: example_fr3_duo_config.yaml)
  --gripper-config FILE  Franka-hand config (default: example_fr3_duo_config_franka_hand.yaml)
  --no-gripper            Do not start the gripper manager
  --ros-domain-id ID      Set ROS_DOMAIN_ID
  --ros-distro NAME       ROS distribution under /opt/ros
  --help                  Show this help

This does not start franka_gello_state_publisher. Deployment targets are sent
by the bridge to deployment controllers; teleoperation publishers may conflict.
EOF
}
die() { echo "Error: $*" >&2; exit 1; }
robot_config=example_fr3_duo_config.yaml; gripper_config=example_fr3_duo_config_franka_hand.yaml
start_gripper=1; ros_domain_id="${ROS_DOMAIN_ID:-0}"; ros_distro="${DEPLOYMENT_ROS_DISTRO:-${ROS_DISTRO:-}}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --robot-config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; robot_config="$2"; shift 2 ;;
    --gripper-config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; gripper_config="$2"; shift 2 ;;
    --no-gripper) start_gripper=0; shift ;;
    --ros-domain-id) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_domain_id="$2"; shift 2 ;;
    --ros-distro) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_distro="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; repository_root="$(cd -- "${script_dir}/.." && pwd)"
if [[ -n "${ros_distro}" ]]; then ros_setup_file="/opt/ros/${ros_distro}/setup.bash"; else
  shopt -s nullglob; candidates=(/opt/ros/*/setup.bash); shopt -u nullglob
  [[ "${#candidates[@]}" -eq 1 ]] || die "select ROS distro with --ros-distro"
  ros_setup_file="${candidates[0]}"; ros_distro="$(basename "$(dirname "${ros_setup_file}")")"
fi
setup_files=("${ros_setup_file}" "${repository_root}/gello_software/ros2/install/setup.bash"); set +u
for setup_file in "${setup_files[@]}"; do
  [[ -r "${setup_file}" ]] || die "ROS setup file is missing: ${setup_file}"
  # shellcheck disable=SC1090
  source "${setup_file}"
done
set -u
# Deployment spans two computers, so keep DDS on the selected domain and do
# not inherit a localhost-only setting from an unrelated ROS shell.
export ROS_DOMAIN_ID="${ros_domain_id}"
export ROS_LOCALHOST_ONLY=0
command -v setsid >/dev/null || die "setsid is required"; command -v ros2 >/dev/null || die "ros2 is unavailable after sourcing ROS"
robot_config_source="${repository_root}/gello_software/ros2/src/franka_fr3_arm_controllers/config/${robot_config}"
gripper_config_source="${repository_root}/gello_software/ros2/src/franka_gripper_manager/config/${gripper_config}"
[[ -f "${robot_config_source}" ]] || die "robot config not found: ${robot_config}"
if [[ "${start_gripper}" -eq 1 ]]; then [[ -f "${gripper_config_source}" ]] || die "gripper config not found: ${gripper_config}"; fi
declare -a child_pids=(); declare -A child_names=(); shutdown_started=0
start_process() { local name="$1"; shift; echo "Starting ${name}: $*"; setsid -- bash -c 'trap - INT QUIT; exec "$@"' _ "$@" & child_pids+=("$!"); child_names["${child_pids[-1]}"]="${name}"; }
signal_groups() { local sig="$1"; for pid in "${child_pids[@]}"; do kill -0 -- "-${pid}" 2>/dev/null && kill -s "${sig}" -- "-${pid}" 2>/dev/null || true; done; }
shutdown() { local status=$?; [[ "${shutdown_started}" -eq 1 ]] && return; shutdown_started=1; trap - EXIT INT TERM; signal_groups INT; sleep 1; signal_groups TERM; for pid in "${child_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done; exit "${status}"; }
trap shutdown EXIT; trap 'exit 130' INT; trap 'exit 143' TERM
start_process "FR3 deployment controllers" ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py "robot_config_file:=${robot_config}" deployment_mode:=true
if [[ "${start_gripper}" -eq 1 ]]; then start_process "Franka-hand managers" ros2 launch franka_gripper_manager franka_gripper_client.launch.py "config_file:=${gripper_config}"; else echo "Gripper manager disabled."; fi
echo "Deployment client is running. Start the policy executor here; press Ctrl-C to stop controllers."
set +e; completed_pid=""; wait -n -p completed_pid "${child_pids[@]}"; status=$?; set -e
echo "${child_names[${completed_pid}]} exited with status ${status}." >&2; exit "${status}"
