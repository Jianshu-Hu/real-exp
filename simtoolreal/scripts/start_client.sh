#!/usr/bin/env bash
set -euo pipefail

# Robot-side SimToolReal launcher. This starts deployment controllers and the
# Wuji worker, and can also own the robot-local deployment bridge. It never
# starts Gello or any teleoperation publisher.
die() { echo "Error: $*" >&2; exit 2; }
require_tcp_port_free() {
  local port="$1"
  local owner
  command -v ss >/dev/null 2>&1 || die "the 'ss' command is required for the client preflight check"
  # Check every TCP state, not only LISTEN.  A recently stopped ZMQ listener
  # can otherwise pass the preflight and then fail bind(2) with EADDRINUSE.
  owner="$(ss -H -tanp "sport = :${port}" 2>/dev/null || true)"
  if [[ -n "${owner}" ]]; then
    echo "Error: local TCP port ${port} is already in use:" >&2
    echo "${owner}" >&2
    echo "Stop the old SimToolReal process that owns this port, or wait for a" >&2
    echo "recently closed TCP connection to disappear, before restarting." >&2
    exit 2
  fi
}

wait_for_tcp_listener() {
  local port="$1"
  local child_pid="$2"
  local label="$3"
  local attempt
  for ((attempt = 0; attempt < 100; attempt += 1)); do
    if ! kill -0 "${child_pid}" 2>/dev/null; then
      wait "${child_pid}" 2>/dev/null || true
      die "${label} exited before opening local TCP port ${port}"
    fi
    if [[ -n "$(ss -H -ltn "sport = :${port}" 2>/dev/null || true)" ]]; then
      return 0
    fi
    sleep 0.05
  done
  die "${label} did not open local TCP port ${port} within 5 seconds"
}

wait_for_fr3_state() {
  local controller_pid="$1"
  local attempt
  local topic_output
  for ((attempt = 0; attempt < 150; attempt += 1)); do
    if ! kill -0 "${controller_pid}" 2>/dev/null; then
      wait "${controller_pid}" 2>/dev/null || true
      die "FR3 controller exited before receiving robot state. Enable FCI mode in Desk and verify the robot IP/network."
    fi
    if topic_output="$(timeout 0.2s ros2 topic echo --once /right/franka/joint_states 2>/dev/null)" && [[ -n "${topic_output}" ]]; then
      return 0
    fi
    sleep 0.1
  done
  die "FR3 controller did not publish /right/franka/joint_states within 30 seconds. Enable FCI mode in Desk and verify the robot IP/network."
}

stop_child_groups() {
  local child_pid
  local attempt
  # Every child below is started by setsid, so its PID is also the process
  # group ID.  Signalling the group prevents ros2/conda grandchildren from
  # surviving after their immediate launcher exits.
  for child_pid in "${child_pids[@]}"; do
    kill -TERM -- "-${child_pid}" 2>/dev/null || true
  done
  for ((attempt = 0; attempt < 20; attempt += 1)); do
    local any_alive=0
    for child_pid in "${child_pids[@]}"; do
      if kill -0 -- "-${child_pid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    [[ "${any_alive}" -eq 0 ]] && break
    sleep 0.1
  done
  for child_pid in "${child_pids[@]}"; do
    kill -KILL -- "-${child_pid}" 2>/dev/null || true
  done
  for child_pid in "${child_pids[@]}"; do
    wait "${child_pid}" 2>/dev/null || true
  done
}

require_fr3_route() {
  local robot_ip="$1"
  local route
  command -v ip >/dev/null 2>&1 || die "the 'ip' command is required for the FR3 network preflight check"
  route="$(ip -4 route get "${robot_ip}" 2>/dev/null || true)"
  if [[ -z "${route}" || "${route}" != *"src 172.16.0."* ]]; then
    echo "Error: the right FR3 at ${robot_ip} is not routed through a 172.16.0.x client interface." >&2
    if [[ -n "${route}" ]]; then
      echo "Current route: ${route}" >&2
    else
      echo "Current route: none" >&2
    fi
    echo "Configure the Ethernet adapter connected to the FR3 with a free 172.16.0.x/24" >&2
    echo "address (not ${robot_ip}), then verify: ip route get ${robot_ip}" >&2
    exit 2
  fi
}

require_no_existing_fr3_controller() {
  local matches
  matches="$(pgrep -a -f '/controller_manager/ros2_control_node' 2>/dev/null || true)"
  if [[ -n "${matches}" ]]; then
    echo "Error: an existing ros2_control_node is already running:" >&2
    echo "${matches}" >&2
    echo "Stop the old FR3 controller terminal before starting another controller." >&2
    exit 2
  fi
}
server_ip="${SIMTOOLREAL_SERVER_IP:-${DEPLOYMENT_SERVER_IP:-192.168.50.13}}"
hand_ip="${WUJI_RIGHT_HAND_IP:-}"
hand_port=5562
telemetry_port=5558
telemetry_address=""
robot_config="example_fr3_right_config_no_gripper.yaml"
bridge_config=""
ros_domain="${ROS_DOMAIN_ID:-0}"
ros_distro="${SIMTOOLREAL_ROS_DISTRO:-${ROS_DISTRO:-}}"
python_bin="${SIMTOOLREAL_PYTHON:-/usr/bin/python3}"
mock=0
no_controllers=0
local_bridge=0
connect=""
rate=30
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --server-ip) [[ "$#" -ge 2 ]] || die "$1 requires a value"; server_ip="$2"; shift 2 ;;
    --connect) [[ "$#" -ge 2 ]] || die "$1 requires a value"; connect="$2"; shift 2 ;;
    --right-hand-ip) [[ "$#" -ge 2 ]] || die "$1 requires a value"; hand_ip="$2"; shift 2 ;;
    --right-hand-command-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; hand_port="$2"; shift 2 ;;
    --hand-telemetry-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; telemetry_port="$2"; shift 2 ;;
    --hand-telemetry-address) [[ "$#" -ge 2 ]] || die "$1 requires a value"; telemetry_address="$2"; shift 2 ;;
    --robot-config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; robot_config="$2"; shift 2 ;;
    --local-bridge) local_bridge=1; shift ;;
    --bridge-config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; bridge_config="$2"; shift 2 ;;
    --ros-domain-id) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_domain="$2"; shift 2 ;;
    --ros-distro) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_distro="$2"; shift 2 ;;
    --python) [[ "$#" -ge 2 ]] || die "$1 requires a value"; python_bin="$2"; shift 2 ;;
    --rate) [[ "$#" -ge 2 ]] || die "$1 requires a value"; rate="$2"; shift 2 ;;
    --mock) mock=1; no_controllers=1; shift ;;
    --no-controllers) no_controllers=1; shift ;;
    --right) shift ;;
    --left|--duo) die "the loaded SimToolReal contract supports the right FR3/Wuji only" ;;
    --help|-h)
      echo "Usage: start_client.sh [--server-ip IP] [--right-hand-ip IP:PORT] [--local-bridge] [--mock]"
      echo "Real mode starts the right FR3 deployment controller and Wuji worker."
      echo "--local-bridge also starts the deployment bridge on this robot computer."
      echo "The ROS bridge publishes the combined right-arm/Wuji state; no teleoperation is started."
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "${connect}" ]] || connect="tcp://${server_ip}:5565"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/.." && pwd)"
repository_root="$(cd -- "${root_dir}/.." && pwd)"
[[ -n "${bridge_config}" ]] || bridge_config="${root_dir}/config/deployment_right_hand.yaml"
if [[ -z "${telemetry_address}" ]]; then
  if [[ "${local_bridge}" -eq 1 ]]; then
    telemetry_address="tcp://127.0.0.1:${telemetry_port}"
  else
    telemetry_address="tcp://${server_ip}:${telemetry_port}"
  fi
fi
# The endpoint checks and the later ZMQ binds must be one single-instance
# operation.  Without this lock, two launchers can both see free ports and then
# race each other to bind 5555.
command -v flock >/dev/null 2>&1 || die "the 'flock' command is required for the client single-instance lock"
command -v setsid >/dev/null 2>&1 || die "the 'setsid' command is required for reliable child cleanup"
client_lock_dir="${XDG_RUNTIME_DIR:-/tmp}"
[[ -d "${client_lock_dir}" && -w "${client_lock_dir}" ]] || client_lock_dir="/tmp"
client_lock_path="${client_lock_dir}/simtoolreal-client-${UID}.lock"
exec 9>"${client_lock_path}"
flock -n 9 || die "another SimToolReal client launcher is already running (lock: ${client_lock_path})"
# Fail before contacting real hardware when an older deployment stack still
# owns the local ZMQ endpoints or the single libfranka control connection.
# This avoids starting a second bridge/controller and producing misleading
# downstream interface errors.
if [[ "${mock}" -eq 0 && "${no_controllers}" -eq 0 ]]; then
  require_tcp_port_free "${hand_port}"
  require_no_existing_fr3_controller
  if [[ "${local_bridge}" -eq 1 ]]; then
    require_tcp_port_free 5555
    require_tcp_port_free 5556
    require_tcp_port_free 5557
    require_tcp_port_free "${telemetry_port}"
  fi
  require_fr3_route "172.16.0.2"
fi
# Use the same Conda isolation helper as the established deployment launcher.
# ROS setup files add Python 3.10 and native-library paths on Humble systems;
# leaking those paths into the Wuji worker's Conda Python can make it import an
# ABI-incompatible ROS Pinocchio extension.
# shellcheck source=scripts/conda_env.sh
source "${repository_root}/scripts/conda_env.sh"
if [[ "${mock}" -eq 1 ]]; then
  mock_args=(--mock --mock-joints 27 --connect "${connect}" --rate "${rate}" --hand-command-address "tcp://127.0.0.1:${hand_port}")
  exec "${python_bin}" "${root_dir}/joint_client.py" "${mock_args[@]}"
fi

if [[ -z "${ros_distro}" ]]; then
  shopt -s nullglob; candidates=(/opt/ros/*/setup.bash); shopt -u nullglob
  [[ "${#candidates[@]}" -eq 1 ]] || die "pass --ros-distro when ROS installation is ambiguous"
  ros_distro="$(basename -- "$(dirname -- "${candidates[0]}")")"
fi
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
set +u
source "/opt/ros/${ros_distro}/setup.bash"
source "${HOME}/franka_ros2_ws/install/local_setup.bash"
source "${repository_root}/gello_software/ros2/install/local_setup.bash"
set -u
export ROS_DOMAIN_ID="${ros_domain}" ROS_LOCALHOST_ONLY=0 ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
config_path="${repository_root}/gello_software/ros2/src/franka_fr3_arm_controllers/config/${robot_config}"
[[ -f "${config_path}" ]] || die "robot config not found: ${config_path}"
declare -a child_pids=()
cleanup() { local status=$?; trap - EXIT INT TERM; stop_child_groups; exit "${status}"; }
trap cleanup EXIT INT TERM
if [[ "${no_controllers}" -eq 0 ]]; then
  if [[ "${local_bridge}" -eq 1 ]]; then
    [[ -f "${bridge_config}" ]] || die "bridge config not found: ${bridge_config}"
    setsid ros2 launch franka_lerobot_data_bridge bridge.launch.py "config_file:=${bridge_config}" &
    bridge_pid="$!"
    child_pids+=("${bridge_pid}")
    # Do not contact the real robot until the local bridge has proved that it
    # owns the fixed endpoint set successfully.
    wait_for_tcp_listener 5555 "${bridge_pid}" "LeRobot deployment state publisher"
    wait_for_tcp_listener 5556 "${bridge_pid}" "LeRobot deployment command receiver"
    wait_for_tcp_listener 5557 "${bridge_pid}" "LeRobot camera-cache publisher"
    wait_for_tcp_listener "${telemetry_port}" "${bridge_pid}" "LeRobot hand-telemetry receiver"
  fi
  setsid ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py "robot_config_file:=${robot_config}" deployment_mode:=true &
  controller_pid="$!"
  child_pids+=("${controller_pid}")
  wait_for_fr3_state "${controller_pid}"
  deployment_env="${DEPLOYMENT_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"
  declare -a hand_python=()
  real_exp_build_conda_python_command "${deployment_env}" hand_python || exit 1
  real_exp_require_conda_python_modules "${deployment_env}" wuji_sdk wujihandpy zmq numpy pinocchio || die \
    "the '${deployment_env}' Conda environment cannot run the Wuji deployment worker"
  if [[ -z "${hand_ip}" ]]; then
    echo "Warning: WUJI_RIGHT_HAND_IP is empty; the worker will use SDK discovery." >&2
  fi
  setsid "${hand_python[@]}" "${repository_root}/deploy/wuji_hand_command_server.py" --side right --hand-ip "${hand_ip}" --command-address "tcp://127.0.0.1:${hand_port}" --telemetry-address "${telemetry_address}" --telemetry-rate 60 & child_pids+=("$!")
fi
# Treat the bridge, controller, and hand worker as one hardware stack. If any
# required child exits, the cleanup trap stops the remaining processes instead
# of leaving a partially running deployment stack behind.
[[ "${#child_pids[@]}" -gt 0 ]] || die "no client process was selected"
set +e
wait -n "${child_pids[@]}"
status=$?
set -e
echo "SimToolReal client child exited; stopping remaining processes" >&2
exit "${status}"
