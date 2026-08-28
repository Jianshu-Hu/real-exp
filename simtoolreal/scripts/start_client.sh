#!/usr/bin/env bash
set -euo pipefail

# Robot-side SimToolReal launcher. This starts deployment controllers and the
# Wuji worker only; it never starts Gello or any teleoperation publisher.
die() { echo "Error: $*" >&2; exit 2; }
server_ip="${SIMTOOLREAL_SERVER_IP:-${DEPLOYMENT_SERVER_IP:-192.168.50.13}}"
hand_ip="${WUJI_RIGHT_HAND_IP:-}"
hand_port=5562
telemetry_port=5558
robot_config="example_fr3_right_config_no_gripper.yaml"
ros_domain="${ROS_DOMAIN_ID:-0}"
ros_distro="${SIMTOOLREAL_ROS_DISTRO:-${ROS_DISTRO:-}}"
python_bin="${SIMTOOLREAL_PYTHON:-/usr/bin/python3}"
mock=0
no_controllers=0
connect=""
rate=30
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --server-ip) [[ "$#" -ge 2 ]] || die "$1 requires a value"; server_ip="$2"; shift 2 ;;
    --connect) [[ "$#" -ge 2 ]] || die "$1 requires a value"; connect="$2"; shift 2 ;;
    --right-hand-ip) [[ "$#" -ge 2 ]] || die "$1 requires a value"; hand_ip="$2"; shift 2 ;;
    --right-hand-command-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; hand_port="$2"; shift 2 ;;
    --hand-telemetry-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; telemetry_port="$2"; shift 2 ;;
    --robot-config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; robot_config="$2"; shift 2 ;;
    --ros-domain-id) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_domain="$2"; shift 2 ;;
    --ros-distro) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_distro="$2"; shift 2 ;;
    --python) [[ "$#" -ge 2 ]] || die "$1 requires a value"; python_bin="$2"; shift 2 ;;
    --rate) [[ "$#" -ge 2 ]] || die "$1 requires a value"; rate="$2"; shift 2 ;;
    --mock) mock=1; no_controllers=1; shift ;;
    --no-controllers) no_controllers=1; shift ;;
    --right) shift ;;
    --left|--duo) die "the loaded SimToolReal contract supports the right FR3/Wuji only" ;;
    --help|-h)
      echo "Usage: start_client.sh [--server-ip IP] [--right-hand-ip IP:PORT] [--mock]"
      echo "Real mode starts the right FR3 deployment controller and Wuji worker."
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
cleanup() { local status=$?; trap - EXIT INT TERM; for pid in "${child_pids[@]}"; do kill "${pid}" 2>/dev/null || true; done; for pid in "${child_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done; exit "${status}"; }
trap cleanup EXIT INT TERM
if [[ "${no_controllers}" -eq 0 ]]; then
  ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py "robot_config_file:=${robot_config}" deployment_mode:=true & child_pids+=("$!")
  deployment_env="${DEPLOYMENT_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"
  if [[ -z "${hand_ip}" ]]; then
    echo "Warning: WUJI_RIGHT_HAND_IP is empty; the worker will use SDK discovery." >&2
  fi
  conda run --no-capture-output -n "${deployment_env}" python "${repository_root}/deploy/wuji_hand_command_server.py" --side right --hand-ip "${hand_ip}" --command-address "tcp://127.0.0.1:${hand_port}" --telemetry-address "tcp://${server_ip}:${telemetry_port}" --telemetry-rate 60 & child_pids+=("$!")
fi
# The bridge process on the server owns the ZMQ PUB socket and combines the
# right FR3 ROS state with Wuji telemetry.  Keeping a second JSON publisher
# here would target an unused endpoint and could silently queue stale samples.
wait "${child_pids[@]}"
