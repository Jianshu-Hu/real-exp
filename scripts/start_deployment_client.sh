#!/usr/bin/env bash
set -euo pipefail

# Robot-control side of deployment. Run on the robot computer.
# The policy executor is intentionally a separate command and safety gate.

usage() {
  cat <<'EOF'
Usage: ./scripts/start_deployment_client.sh [options]

Required:
  --server-address HOST:PORT  Policy server gRPC address used to discover the checkpoint contract

Options:
  --server-ip IP         Inference/server computer IP (default: DEPLOYMENT_SERVER_IP or 192.168.50.13)
  --server-address HOST:PORT  Policy server gRPC address (default: server-ip:8080)
  --metadata-address HOST:PORT  Metadata HTTP endpoint (default: server-ip:8081)
  --robot-config FILE    Override the metadata-selected FR3 config
  --gripper-config FILE  Override the metadata-selected Franka-hand config
  --left-hand-ip ADDR    Left Wuji Hand 2 SDK address (IP:port)
  --right-hand-ip ADDR   Right Wuji Hand 2 SDK address (IP:port)
  --left-hand-command-port PORT   Local left Wuji command port (default: 5561)
  --right-hand-command-port PORT  Local right Wuji command port (default: 5562)
  --hand-telemetry-port PORT  Server bridge telemetry port (default: 5558)
  --print-config          Resolve and print metadata-selected settings, then exit
  --ros-domain-id ID      Set ROS_DOMAIN_ID
  --ros-distro NAME       ROS distribution under /opt/ros
  --help                  Show this help

The policy server metadata controls arm mode, end effector, dimensions, cameras, and FPS.
This does not start franka_gello_state_publisher. Deployment targets are sent
by the bridge to deployment controllers; teleoperation publishers may conflict.
EOF
}
die() { echo "Error: $*" >&2; exit 1; }
validate_port() { [[ "$1" =~ ^[0-9]+$ ]] && ((1 <= 10#$1 && 10#$1 <= 65535)); }
server_ip="${DEPLOYMENT_SERVER_IP:-192.168.50.13}"; server_address=""; metadata_address=""
robot_config=""; gripper_config=""; left_hand_ip="${WUJI_LEFT_HAND_IP:-}"; right_hand_ip="${WUJI_RIGHT_HAND_IP:-}"
left_hand_command_port=5561; right_hand_command_port=5562; hand_telemetry_port=5558
print_config=0
ros_domain_id="${ROS_DOMAIN_ID:-0}"; ros_distro="${DEPLOYMENT_ROS_DISTRO:-${ROS_DISTRO:-}}"
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dataset-root) die "--dataset-root is obsolete; metadata is discovered from the policy server" ;;
    --server-ip) [[ "$#" -ge 2 ]] || die "$1 requires a value"; server_ip="$2"; shift 2 ;;
    --server-address) [[ "$#" -ge 2 ]] || die "$1 requires a value"; server_address="$2"; shift 2 ;;
    --metadata-address) [[ "$#" -ge 2 ]] || die "$1 requires a value"; metadata_address="$2"; shift 2 ;;
    --robot-config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; robot_config="$2"; shift 2 ;;
    --gripper-config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; gripper_config="$2"; shift 2 ;;
    --left-hand-ip) [[ "$#" -ge 2 ]] || die "$1 requires a value"; left_hand_ip="$2"; shift 2 ;;
    --right-hand-ip) [[ "$#" -ge 2 ]] || die "$1 requires a value"; right_hand_ip="$2"; shift 2 ;;
    --left-hand-command-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; left_hand_command_port="$2"; shift 2 ;;
    --right-hand-command-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; right_hand_command_port="$2"; shift 2 ;;
    --hand-telemetry-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; hand_telemetry_port="$2"; shift 2 ;;
    --print-config) print_config=1; shift ;;
    --ros-domain-id) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_domain_id="$2"; shift 2 ;;
    --ros-distro) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_distro="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "${server_address}" ]] || server_address="${server_ip}:8080"
[[ -n "${metadata_address}" ]] || metadata_address="${server_ip}:8081"
validate_port "${left_hand_command_port}" || die "invalid left hand command port: ${left_hand_command_port}"
validate_port "${right_hand_command_port}" || die "invalid right hand command port: ${right_hand_command_port}"
validate_port "${hand_telemetry_port}" || die "invalid hand telemetry port: ${hand_telemetry_port}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; repository_root="$(cd -- "${script_dir}/.." && pwd)"
# shellcheck source=scripts/conda_env.sh
source "${script_dir}/conda_env.sh"
trajectory_output="$(PYTHONPATH="${repository_root}" python3 "${repository_root}/utils/deployment_metadata.py" --url "http://${metadata_address}/deployment-metadata" --deployment-lines 2>/dev/null)" \
  || die "could not resolve deployment metadata from ${metadata_address}"
mapfile -t trajectory_lines <<<"${trajectory_output}"
(( ${#trajectory_lines[@]} == 9 )) || die "deployment metadata resolver returned an unexpected number of fields"
arm_mode="${trajectory_lines[0]}"; end_effector="${trajectory_lines[1]}"; fps="${trajectory_lines[2]}"
state_dim="${trajectory_lines[3]}"; action_dim="${trajectory_lines[4]}"; state_action_mode="${trajectory_lines[5]}"; camera_names="${trajectory_lines[6]}"
policy_type="${trajectory_lines[7]}"; actions_per_chunk="${trajectory_lines[8]}"
case "${arm_mode}" in left|right|duo) ;; *) die "unsupported metadata arm mode: ${arm_mode}" ;; esac
case "${end_effector}" in arm|gripper|hand) ;; *) die "unsupported metadata end effector: ${end_effector}" ;; esac
case "${state_action_mode}" in joint|end_effector) ;; *) die "unsupported metadata state/action mode: ${state_action_mode}" ;; esac
if [[ -z "${robot_config}" ]]; then
  case "${arm_mode}:${end_effector}" in
    left:gripper) robot_config=example_fr3_config.yaml ;;
    right:gripper) robot_config=example_fr3_right_config.yaml ;;
    duo:gripper) robot_config=example_fr3_duo_config.yaml ;;
    left:arm|left:hand) robot_config=example_fr3_config_no_gripper.yaml ;;
    right:arm|right:hand) robot_config=example_fr3_right_config_no_gripper.yaml ;;
    duo:arm|duo:hand) robot_config=example_fr3_duo_config_no_gripper.yaml ;;
  esac
fi
if [[ -z "${gripper_config}" && "${end_effector}" == "gripper" ]]; then
  case "${arm_mode}" in
    left) gripper_config=example_fr3_config_franka_hand.yaml ;;
    right) gripper_config=example_fr3_right_config_franka_hand.yaml ;;
    duo) gripper_config=example_fr3_duo_config_franka_hand.yaml ;;
  esac
fi
start_gripper=0; [[ "${end_effector}" == "gripper" ]] && start_gripper=1
if [[ "${end_effector}" == "hand" ]]; then
  [[ "${server_ip}" != "127.0.0.1" && -n "${server_ip}" ]] || die "--server-ip must identify the inference computer for hand telemetry"
  if [[ "${arm_mode}" == "duo" && "${left_hand_command_port}" == "${right_hand_command_port}" ]]; then
    die "left and right hand command ports must differ in duo-hand mode"
  fi
fi
if [[ "${print_config}" -eq 1 ]]; then
  printf 'arm_mode=%s\nend_effector=%s\nfps=%s\nstate_dim=%s\naction_dim=%s\nstate_action_mode=%s\ncameras=%s\npolicy_type=%s\nactions_per_chunk=%s\nrobot_config=%s\ngripper_config=%s\nserver_ip=%s\nserver_address=%s\nmetadata_address=%s\n' \
    "${arm_mode}" "${end_effector}" "${fps}" "${state_dim}" "${action_dim}" "${state_action_mode}" \
    "${camera_names}" "${policy_type}" "${actions_per_chunk}" "${robot_config}" "${gripper_config}" "${server_ip}" "${server_address}" "${metadata_address}"
  exit 0
fi
if [[ -n "${ros_distro}" ]]; then ros_setup_file="/opt/ros/${ros_distro}/setup.bash"; else
  shopt -s nullglob; candidates=(/opt/ros/*/setup.bash); shopt -u nullglob
  [[ "${#candidates[@]}" -eq 1 ]] || die "select ROS distro with --ros-distro"
  ros_setup_file="${candidates[0]}"; ros_distro="$(basename "$(dirname "${ros_setup_file}")")"
fi
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
setup_files=(
  "${ros_setup_file}"
  "${HOME}/franka_ros2_ws/install/local_setup.bash"
  "${repository_root}/gello_software/ros2/install/local_setup.bash"
); set +u
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
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
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
if [[ "${end_effector}" == "hand" ]]; then
  deployment_conda_env="${DEPLOYMENT_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"
  declare -a hand_python=()
  real_exp_build_conda_python_command "${deployment_conda_env}" hand_python || exit 1
  real_exp_require_conda_python_modules "${deployment_conda_env}" wuji_sdk wujihandpy zmq numpy || die \
    "the '${deployment_conda_env}' Conda environment cannot run the Wuji deployment worker"
  if [[ "${arm_mode}" == "left" || "${arm_mode}" == "duo" ]]; then
    start_process "Left Wuji deployment worker" "${hand_python[@]}" \
      "${repository_root}/deploy/wuji_hand_command_server.py" --side left \
      --hand-ip "${left_hand_ip}" --command-address "tcp://127.0.0.1:${left_hand_command_port}" \
      --telemetry-address "tcp://${server_ip}:${hand_telemetry_port}" --telemetry-rate "${fps}"
  fi
  if [[ "${arm_mode}" == "right" || "${arm_mode}" == "duo" ]]; then
    start_process "Right Wuji deployment worker" "${hand_python[@]}" \
      "${repository_root}/deploy/wuji_hand_command_server.py" --side right \
      --hand-ip "${right_hand_ip}" --command-address "tcp://127.0.0.1:${right_hand_command_port}" \
      --telemetry-address "tcp://${server_ip}:${hand_telemetry_port}" --telemetry-rate "${fps}"
  fi
fi
echo "Deployment contract: ${end_effector}/${arm_mode}, mode=${state_action_mode}, state/action=${state_dim}/${action_dim}, cameras=${camera_names}, fps=${fps}"
echo "Deployment client is running. Start the policy executor here; press Ctrl-C to stop controllers."
set +e; completed_pid=""; wait -n -p completed_pid "${child_pids[@]}"; status=$?; set -e
echo "${child_names[${completed_pid}]} exited with status ${status}." >&2; exit "${status}"
