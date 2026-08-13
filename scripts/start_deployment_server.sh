#!/usr/bin/env bash
set -euo pipefail

# Camera/bridge/policy side of deployment. Run on the server computer.

usage() {
  cat <<'EOF'
Usage: ./scripts/start_deployment_server.sh [options]

Options:
  --server-ip IP       ZMQ bind address (default: DEPLOYMENT_SERVER_IP or 192.168.50.13)
  --publish-port PORT  Observation ZMQ port (default: 5555)
  --command-port PORT  Command ZMQ port (default: 5556)
  --policy-port PORT   gRPC policy port (default: 8080)
  --bridge-config FILE Bridge config (default: deployment_duo.yaml)
  --ros-domain-id ID   Set ROS_DOMAIN_ID
  --ros-distro NAME    ROS distribution under /opt/ros
  --python PATH        Policy Python (default: DEPLOYMENT_PYTHON or python)
  --help               Show this help

The robot executor supplies the checkpoint path during the gRPC setup handshake;
that path must exist on this server.
EOF
}
die() { echo "Error: $*" >&2; exit 1; }
validate_port() { [[ "$1" =~ ^[0-9]+$ ]] && ((1 <= 10#$1 && 10#$1 <= 65535)); }

server_ip="${DEPLOYMENT_SERVER_IP:-192.168.50.13}"
publish_port=5555; command_port=5556; policy_port=8080
bridge_config=deployment_duo.yaml; ros_domain_id=""
ros_distro="${DEPLOYMENT_ROS_DISTRO:-${ROS_DISTRO:-}}"
policy_python="${DEPLOYMENT_PYTHON:-python}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --server-ip) [[ "$#" -ge 2 ]] || die "$1 requires a value"; server_ip="$2"; shift 2 ;;
    --publish-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; publish_port="$2"; shift 2 ;;
    --command-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; command_port="$2"; shift 2 ;;
    --policy-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_port="$2"; shift 2 ;;
    --bridge-config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; bridge_config="$2"; shift 2 ;;
    --ros-domain-id) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_domain_id="$2"; shift 2 ;;
    --ros-distro) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_distro="$2"; shift 2 ;;
    --python) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_python="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
validate_port "${publish_port}" || die "invalid observation port: ${publish_port}"
validate_port "${command_port}" || die "invalid command port: ${command_port}"
validate_port "${policy_port}" || die "invalid policy port: ${policy_port}"
[[ "${publish_port}" != "${command_port}" && "${publish_port}" != "${policy_port}" && "${command_port}" != "${policy_port}" ]] || \
  die "observation, command, and policy ports must be distinct"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
if [[ -n "${ros_distro}" ]]; then
  ros_setup_file="/opt/ros/${ros_distro}/setup.bash"
else
  shopt -s nullglob; candidates=(/opt/ros/*/setup.bash); shopt -u nullglob
  [[ "${#candidates[@]}" -eq 1 ]] || die "select ROS distro with --ros-distro"
  ros_setup_file="${candidates[0]}"; ros_distro="$(basename "$(dirname "${ros_setup_file}")")"
fi
setup_files=("${ros_setup_file}" "${repository_root}/gello_software/ros2/install/setup.bash")
set +u
for setup_file in "${setup_files[@]}"; do
  [[ -r "${setup_file}" ]] || die "ROS setup file is missing: ${setup_file}"
  # shellcheck disable=SC1090
  source "${setup_file}"
done
set -u
[[ -z "${ros_domain_id}" ]] || export ROS_DOMAIN_ID="${ros_domain_id}"
command -v setsid >/dev/null || die "setsid is required"
command -v ros2 >/dev/null || die "ros2 is unavailable after sourcing ROS"
command -v "${policy_python}" >/dev/null || die "policy Python not found: ${policy_python}"
"${policy_python}" -c 'import grpc, torch, zmq' >/dev/null 2>&1 || die \
  "policy Python is missing grpc, torch, or zmq: ${policy_python}"
bridge_config_source="${repository_root}/gello_software/ros2/src/franka_lerobot_data_bridge/config/${bridge_config}"
if [[ "${bridge_config}" = /* ]]; then bridge_config_source="${bridge_config}"; fi
[[ -f "${bridge_config_source}" ]] || die "bridge config not found: ${bridge_config}"

declare -a child_pids=(); declare -A child_names=(); shutdown_started=0
start_process() {
  local name="$1"; shift; echo "Starting ${name}: $*"
  setsid -- bash -c 'trap - INT QUIT; exec "$@"' _ "$@" &
  child_pids+=("$!"); child_names["${child_pids[-1]}"]="${name}"
}
signal_groups() { local sig="$1"; for pid in "${child_pids[@]}"; do kill -0 -- "-${pid}" 2>/dev/null && kill -s "${sig}" -- "-${pid}" 2>/dev/null || true; done; }
shutdown() {
  local status=$?; [[ "${shutdown_started}" -eq 1 ]] && return; shutdown_started=1
  trap - EXIT INT TERM; signal_groups INT; sleep 1; signal_groups TERM
  for pid in "${child_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done; exit "${status}"
}
trap shutdown EXIT; trap 'exit 130' INT; trap 'exit 143' TERM

echo "Deployment server: observation tcp://${server_ip}:${publish_port}, command tcp://${server_ip}:${command_port}, gRPC :${policy_port}"
start_process "RealSense camera publisher" ros2 launch franka_realsense_camera_publisher cameras.launch.py
start_process "Deployment observation bridge" ros2 launch franka_lerobot_data_bridge bridge.launch.py \
  "config_file:=${bridge_config}" "publish_host:=${server_ip}" "publish_port:=${publish_port}" \
  "command_host:=${server_ip}" "command_port:=${command_port}"
start_process "Policy server" "${policy_python}" "${repository_root}/deploy/deploy_lerobot_policy.py" server \
  --host 0.0.0.0 --port "${policy_port}" --fps 15
echo "Deployment server is running. Press Ctrl-C to stop all children."
set +e; completed_pid=""; wait -n -p completed_pid "${child_pids[@]}"; status=$?; set -e
echo "${child_names[${completed_pid}]} exited with status ${status}." >&2; exit "${status}"
