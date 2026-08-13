#!/usr/bin/env bash
set -euo pipefail

# Start the data-server side of split collection: RealSense publishers, the
# ROS-to-ZMQ bridge, and optionally the LeRobot recorder.  The control host
# must run start_data_collection_client.sh at the same time and share the ROS 2 DDS
# domain/network with this host.
#
# Usage:
#   ./scripts/start_data_collection_server.sh --duo|--single --gripper|--no-gripper [--record]
#                                               [--local-dir PATH]
#                                               [--repo-id ID]
#
# --record starts data_collection/lerobot_collection.py in the selected Python
# environment. Without it, start the recorder separately on this server.

usage() {
  cat <<'EOF'
Usage: ./scripts/start_data_collection_server.sh --duo|--single --gripper|--no-gripper [options]

Required:
  --duo                 Record both arms and three cameras.
  --single              Record the left arm and two cameras.
  --gripper             Include gripper state/action data.
  --no-gripper          Record arm data without gripper topics.

Options:
  --record              Start the LeRobot recorder on this server.
  --local-dir PATH      Dataset directory for --record (default: ./lerobot_data).
  --repo-id ID          LeRobot repo id for --record.
  --bridge-host IP      Address on which the bridge publishes samples
                        (default: ${DATA_COLLECTION_SERVER_IP:-192.168.50.13}).
  --bridge-port PORT    ZMQ sample port (default: 5555).
  --ros-domain-id ID    Set ROS_DOMAIN_ID for this process (default: preserve env).
  --ros-distro DISTRO   ROS 2 distribution installed under /opt/ros (default:
                        DATA_SERVER_ROS_DISTRO, current ROS_DISTRO, or the only
                        locally installed distribution).
  --help                Show this help.
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

arm_mode=""
gripper_mode=""
record=0
local_dir="./lerobot_data"
repo_id="local/franka_gello_teleop"
bridge_host="${DATA_COLLECTION_SERVER_IP:-192.168.50.13}"
bridge_port="5555"
ros_domain_id=""
ros_distro="${DATA_SERVER_ROS_DISTRO:-${ROS_DISTRO:-}}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --duo|--single)
      [[ -z "${arm_mode}" ]] || die "choose only one of --duo or --single"
      arm_mode="${1#--}"
      shift
      ;;
    --gripper|--no-gripper)
      [[ -z "${gripper_mode}" ]] || die "choose only one of --gripper or --no-gripper"
      gripper_mode="${1#--}"
      shift
      ;;
    --record) record=1; shift ;;
    --local-dir)
      [[ "$#" -ge 2 ]] || die "--local-dir requires a path"
      local_dir="$2"; shift 2 ;;
    --repo-id)
      [[ "$#" -ge 2 ]] || die "--repo-id requires an id"
      repo_id="$2"; shift 2 ;;
    --bridge-host)
      [[ "$#" -ge 2 ]] || die "--bridge-host requires an address"
      bridge_host="$2"; shift 2 ;;
    --bridge-port)
      [[ "$#" -ge 2 ]] || die "--bridge-port requires a port"
      bridge_port="$2"; shift 2 ;;
    --ros-domain-id)
      [[ "$#" -ge 2 ]] || die "--ros-domain-id requires an id"
      ros_domain_id="$2"; shift 2 ;;
    --ros-distro)
      [[ "$#" -ge 2 ]] || die "--ros-distro requires a distribution name"
      ros_distro="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "${arm_mode}" && -n "${gripper_mode}" ]] || { usage >&2; exit 2; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"

if [[ -n "${ros_distro}" ]]; then
  ros_setup_file="/opt/ros/${ros_distro}/setup.bash"
else
  shopt -s nullglob
  ros_setup_candidates=(/opt/ros/*/setup.bash)
  shopt -u nullglob
  case "${#ros_setup_candidates[@]}" in
    0) die "no ROS 2 installation found under /opt/ros" ;;
    1) ros_setup_file="${ros_setup_candidates[0]}" ;;
    *) die "multiple ROS 2 distributions are installed; pass --ros-distro DISTRO" ;;
  esac
  ros_distro="$(basename -- "$(dirname -- "${ros_setup_file}")")"
fi

setup_files=(
  "${ros_setup_file}"
  "${repository_root}/gello_software/ros2/install/setup.bash"
)

# ROS setup hooks may inspect unset variables.
set +u
for setup_file in "${setup_files[@]}"; do
  [[ -r "${setup_file}" ]] || die "ROS setup file is missing or unreadable: ${setup_file}"
  # shellcheck disable=SC1090
  source "${setup_file}"
done
set -u

if [[ -n "${ros_domain_id}" ]]; then
  export ROS_DOMAIN_ID="${ros_domain_id}"
fi

command -v setsid >/dev/null 2>&1 || die "required command not found: setsid"
command -v ros2 >/dev/null 2>&1 || die "ros2 is unavailable after sourcing ROS ${ros_distro}"

ros_python="/usr/bin/python3"
[[ -x "${ros_python}" ]] || die "ROS Python interpreter is missing: ${ros_python}"
missing_ros_modules="$(${ros_python} - <<'PY'
import importlib.util

modules = ("controller_manager_msgs", "pyrealsense2", "rclpy", "zmq")
print(" ".join(module for module in modules if importlib.util.find_spec(module) is None))
PY
)"
[[ -z "${missing_ros_modules}" ]] || die \
  "missing ROS Python modules for ${ros_python}: ${missing_ros_modules}"

config_file="example_${arm_mode}.yaml"
declare -a child_pids=()
declare -A child_names=()
shutdown_started=0

start_process() {
  local process_name="$1"; shift
  echo "Starting ${process_name}: $*"
  setsid -- bash -c 'trap - INT QUIT; exec "$@"' _ "$@" &
  local child_pid=$!
  child_pids+=("${child_pid}")
  child_names["${child_pid}"]="${process_name}"
}

signal_running_groups() {
  local signal_name="$1"
  for child_pid in "${child_pids[@]}"; do
    if kill -0 -- "-${child_pid}" 2>/dev/null; then
      kill -s "${signal_name}" -- "-${child_pid}" 2>/dev/null || true
    fi
  done
}

shutdown() {
  local exit_status=$?
  [[ "${shutdown_started}" -eq 1 ]] && return
  shutdown_started=1
  trap - EXIT INT TERM
  signal_running_groups INT
  for child_pid in "${child_pids[@]}"; do wait "${child_pid}" 2>/dev/null || true; done
  exit "${exit_status}"
}
trap shutdown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Data-server mode: ${arm_mode}; ROS: ${ros_distro}; bridge: tcp://${bridge_host}:${bridge_port}"

start_process "RealSense camera publisher" \
  ros2 launch franka_realsense_camera_publisher cameras.launch.py
start_process "LeRobot data bridge" \
  ros2 launch franka_lerobot_data_bridge bridge.launch.py \
  "config_file:=${config_file}" \
  "publish_host:=${bridge_host}" "publish_port:=${bridge_port}" \
  "include_gripper:=$([[ "${gripper_mode}" == "gripper" ]] && echo true || echo false)"

if [[ "${record}" -eq 1 ]]; then
  command -v python >/dev/null 2>&1 || die "python is required for --record (activate the lerobot environment)"
  start_process "LeRobot recorder" \
    python "${repository_root}/data_collection/lerobot_collection.py" \
    --host "${bridge_host}" --port "${bridge_port}" \
    --repo-id "${repo_id}" --local-dir "${local_dir}"
else
  echo "Recorder not started. On this server run:"
  echo "  python data_collection/lerobot_collection.py --host ${bridge_host} --port ${bridge_port}"
fi

echo "Data-server stack is running. Press Ctrl-C to stop it."
set +e
completed_pid=""
wait -n -p completed_pid "${child_pids[@]}"
child_status=$?
set -e
echo "${child_names[${completed_pid}]} exited with status ${child_status}." >&2
exit "${child_status}"
