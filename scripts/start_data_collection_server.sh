#!/usr/bin/env bash
set -euo pipefail

# Start the data-server side of split collection: RealSense publishers, the
# ROS-to-ZMQ bridge, and optionally the LeRobot recorder.  The control host
# must run start_data_collection_client.sh at the same time and share the ROS 2 DDS
# domain/network with this host.
#
# Usage:
#   ./scripts/start_data_collection_server.sh --duo|--left|--right --arm|--gripper|--hand [--record]
#                                               [--local-dir PATH]
#                                               [--repo-id ID]
#
# --record starts data_collection/lerobot_collection.py in the selected Python
# environment. Without it, start the recorder separately on this server.

usage() {
  cat <<'EOF'
Usage: ./scripts/start_data_collection_server.sh --duo|--left|--right --arm|--gripper|--hand [options]

Required:
  --duo                 Record both arms and three cameras.
  --left                Record the left arm and two cameras.
  --right               Record the right arm and two cameras.
  --single              Compatibility alias for --left.
  --gripper             Include gripper state/action data.
  --arm, --no-gripper   Record arm data without end-effector topics.
  --hand                Include 20-joint Wuji hand current and target angles.

Options:
  --record              Start the LeRobot recorder on this server.
  --local-dir PATH      Dataset directory for --record (default: ./lerobot_data).
  --repo-id ID          LeRobot repo id for --record.
  --bridge-host IP      Address on which the bridge publishes samples
                        (default: ${DATA_COLLECTION_SERVER_IP:-192.168.50.13}).
  --bridge-port PORT    ZMQ sample port (default: 5555).
  --hand-telemetry-port PORT
                        ZMQ port for hand telemetry (default: 5558).
  --ros-domain-id ID    Set ROS_DOMAIN_ID for this process (default: environment or 0).
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
hand_telemetry_port="${HAND_TELEMETRY_PORT:-5558}"
ros_domain_id="${ROS_DOMAIN_ID:-0}"
ros_distro="${DATA_SERVER_ROS_DISTRO:-${ROS_DISTRO:-}}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --duo|--left|--right|--single)
      [[ -z "${arm_mode}" ]] || die "choose only one of --duo, --left, or --right"
      arm_mode="${1#--}"
      [[ "${arm_mode}" != "single" ]] || arm_mode="left"
      shift
      ;;
    --arm|--gripper|--no-gripper|--hand)
      [[ -z "${gripper_mode}" ]] || die "choose only one of --arm, --gripper, or --hand"
      gripper_mode="${1#--}"
      [[ "${gripper_mode}" != "no-gripper" ]] || gripper_mode="arm"
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
    --hand-telemetry-port)
      [[ "$#" -ge 2 ]] || die "--hand-telemetry-port requires a port"
      hand_telemetry_port="$2"; shift 2 ;;
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
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"
# shellcheck source=scripts/conda_env.sh
source "${script_dir}/conda_env.sh"

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

unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH

setup_files=(
  "${ros_setup_file}"
)
if [[ -r "${HOME}/franka_ros2_ws/install/local_setup.bash" ]]; then
  setup_files+=("${HOME}/franka_ros2_ws/install/local_setup.bash")
fi
setup_files+=("${repository_root}/gello_software/ros2/install/local_setup.bash")

# ROS setup hooks may inspect unset variables.
set +u
for setup_file in "${setup_files[@]}"; do
  [[ -r "${setup_file}" ]] || die "ROS setup file is missing or unreadable: ${setup_file}"
  # shellcheck disable=SC1090
  source "${setup_file}"
done
set -u

export ROS_DOMAIN_ID="${ros_domain_id}"
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

command -v setsid >/dev/null 2>&1 || die "required command not found: setsid"
command -v timeout >/dev/null 2>&1 || die "required command not found: timeout"
command -v pkill >/dev/null 2>&1 || die "required command not found: pkill"
command -v ps >/dev/null 2>&1 || die "required command not found: ps"
command -v ros2 >/dev/null 2>&1 || die "ros2 is unavailable after sourcing ROS ${ros_distro}"
ros2 pkg prefix franka_msgs >/dev/null 2>&1 || die \
  "required ROS package is unavailable after sourcing overlays: franka_msgs (source/build the Franka ROS 2 workspace)"

ros_python="/usr/bin/python3"
[[ -x "${ros_python}" ]] || die "ROS Python interpreter is missing: ${ros_python}"
missing_ros_modules="$(${ros_python} - <<'PY'
import importlib.util

modules = ("controller_manager_msgs", "numpy", "pyrealsense2", "rclpy", "zmq")
print(" ".join(module for module in modules if importlib.util.find_spec(module) is None))
PY
)"
[[ -z "${missing_ros_modules}" ]] || die \
  "missing ROS Python modules for ${ros_python}: ${missing_ros_modules}"

if ! "${ros_python}" - "${repository_root}" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from utils.fr3_kinematics import Fr3ForwardKinematics

kinematics = Fr3ForwardKinematics()
print(f"Ready: FR3 target-pose FK backend is {kinematics.backend}")
PY
then
  die "FR3 target-pose FK preflight failed in ${ros_python}; update the repository and restart"
fi

# The bridge package calls the left-arm configuration "example_single.yaml"
# for historical compatibility; keep the command-line mode names independent
# from package-internal filenames.
case "${arm_mode}" in
  left) config_file="example_single.yaml" ;;
  right|duo) config_file="example_${arm_mode}.yaml" ;;
  *) die "unsupported arm mode: ${arm_mode}" ;;
esac
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
    # Each child is started with setsid, so its PID is also the process-group
    # and session ID.  Signal both: ROS launch may create descendants that
    # remain in the session but are no longer in the launcher's process group.
    if kill -0 -- "-${child_pid}" 2>/dev/null; then
      kill -s "${signal_name}" -- "-${child_pid}" 2>/dev/null || true
    fi
    pkill "-${signal_name}" -s "${child_pid}" 2>/dev/null || true
  done
}

wait_for_groups_to_stop() {
  local attempts="$1"
  local attempt

  for ((attempt = 0; attempt < attempts; attempt++)); do
    local running_group=0
    for child_pid in "${child_pids[@]}"; do
      if ps -o stat= --sid "${child_pid}" 2>/dev/null | grep -qv '^[[:space:]]*Z'; then
        running_group=1
        break
      fi
    done
    [[ "${running_group}" -eq 0 ]] && return 0
    sleep 0.1
  done

  return 1
}

shutdown() {
  local exit_status=$?
  [[ "${shutdown_started}" -eq 1 ]] && return
  shutdown_started=1
  trap - EXIT INT TERM

  signal_running_groups INT
  if ! wait_for_groups_to_stop 20; then
    signal_running_groups TERM
  fi
  if ! wait_for_groups_to_stop 30; then
    echo "Force-stopping unresponsive data-server processes..." >&2
    signal_running_groups KILL
  fi
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
  "include_gripper:=$([[ "${gripper_mode}" == "gripper" ]] && echo true || echo false)" \
  "include_hand:=$([[ "${gripper_mode}" == "hand" ]] && echo true || echo false)" \
  "arm_mode:=${arm_mode}" \
  "hand_telemetry_host:=${bridge_host}" "hand_telemetry_port:=${hand_telemetry_port}"

required_bridge_topics=()
if [[ "${arm_mode}" == "duo" || "${arm_mode}" == "left" ]]; then
  required_bridge_topics+=("/left/joint_states" "/left/gello/raw_joint_states" "/left/gello/accepted_joint_states")
fi
if [[ "${arm_mode}" == "duo" || "${arm_mode}" == "right" ]]; then
  required_bridge_topics+=("/right/joint_states" "/right/gello/raw_joint_states" "/right/gello/accepted_joint_states")
fi
for topic in "${required_bridge_topics[@]}"; do
  if ! timeout 10s ros2 topic echo \
    "${topic}" sensor_msgs/msg/JointState --once --no-daemon \
    >/dev/null 2>&1; then
    die "no JointState payload received from ${topic}; check cross-host ROS 2 discovery and the control stack"
  fi
  echo "Ready: ${topic} is publishing on the data server"
done

if [[ "${record}" -eq 1 ]]; then
  lerobot_conda_env="${LEROBOT_CONDA_ENV:-lerobot}"
  declare -a recorder_python=()
  real_exp_build_conda_python_command "${lerobot_conda_env}" recorder_python || exit 1
  real_exp_require_conda_python_modules "${lerobot_conda_env}" lerobot pyarrow zmq numpy || die \
    "the '${lerobot_conda_env}' Conda environment is missing LeRobot recorder dependencies"
  start_process "LeRobot recorder" \
    "${recorder_python[@]}" "${repository_root}/data_collection/lerobot_collection.py" \
    --host "${bridge_host}" --port "${bridge_port}" \
    --repo-id "${repo_id}" --local-dir "${local_dir}"
else
  echo "Recorder not started. Restart this launcher with --record to launch it automatically."
fi

echo "Data-server stack is running. Press Ctrl-C to stop it."
set +e
completed_pid=""
wait -n -p completed_pid "${child_pids[@]}"
child_status=$?
set -e
echo "${child_names[${completed_pid}]} exited with status ${child_status}." >&2
exit "${child_status}"
