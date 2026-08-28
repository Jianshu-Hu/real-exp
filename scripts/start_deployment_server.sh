#!/usr/bin/env bash
set -euo pipefail

# Camera/bridge/policy side of deployment. Run on the server computer.

usage() {
  cat <<'EOF'
Usage: ./scripts/start_deployment_server.sh [options]

Required:
  --policy-path DIR     Checkpoint whose embedded metadata configures the bridge and policy server

Options:
  --server-ip IP       ZMQ bind address (default: DEPLOYMENT_SERVER_IP or 192.168.50.13)
  --publish-port PORT  Observation ZMQ port (default: 5555)
  --command-port PORT  Command ZMQ port (default: 5556)
  --camera-cache-port PORT  Loopback full-camera cache port (default: 5557)
  --policy-port PORT   gRPC policy port (default: 8080)
  --metadata-port PORT Read-only deployment metadata HTTP port (default: 8081)
  --fps HZ             Optional override; must match checkpoint metadata
  --hand-telemetry-port PORT  Wuji hand telemetry port (default: 5558)
  --bridge-config FILE Bridge config (default: deployment_duo.yaml)
  --ros-domain-id ID   Set ROS_DOMAIN_ID
  --ros-distro NAME    ROS distribution under /opt/ros
  --python PATH        Explicit policy Python interpreter (default: lerobot Conda environment)
  --print-config       Resolve and print metadata-selected settings, then exit
  --help               Show this help

The server owns the checkpoint path and publishes its embedded deployment contract
on the metadata HTTP endpoint. The robot client does not need the dataset or checkpoint.
EOF
}
die() { echo "Error: $*" >&2; exit 1; }
validate_port() { [[ "$1" =~ ^[0-9]+$ ]] && ((1 <= 10#$1 && 10#$1 <= 65535)); }
check_port_available() {
  local port="$1" label="$2"
  command -v ss >/dev/null 2>&1 || return 0
  if ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .; then
    die "${label} port ${port} is already in use; stop the existing service or choose another port"
  fi
}
clear_port() {
  local port="$1"
  command -v fuser >/dev/null 2>&1 || return 0
  if ! fuser -n tcp "${port}" >/dev/null 2>&1; then
    return 0
  fi
  echo "Clearing existing policy service on tcp port ${port}." >&2
  fuser -TERM -k -n tcp "${port}" >/dev/null 2>&1 || true
  for _ in {1..10}; do
    fuser -n tcp "${port}" >/dev/null 2>&1 || return 0
    sleep 0.1
  done
  fuser -KILL -k -n tcp "${port}" >/dev/null 2>&1 || true
}

policy_path=""; metadata_port=8081; requested_fps=""
server_ip="${DEPLOYMENT_SERVER_IP:-192.168.50.13}"
publish_port=5555; command_port=5556; camera_cache_port=5557; policy_port=8080; hand_telemetry_port=5558
bridge_config=deployment_duo.yaml; ros_domain_id="${ROS_DOMAIN_ID:-0}"
ros_distro="${DEPLOYMENT_ROS_DISTRO:-${ROS_DISTRO:-}}"
policy_python="${DEPLOYMENT_PYTHON:-}"
print_config=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --policy-path) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_path="$2"; shift 2 ;;
    --dataset-root) die "--dataset-root is obsolete; use --policy-path" ;;
    --server-ip) [[ "$#" -ge 2 ]] || die "$1 requires a value"; server_ip="$2"; shift 2 ;;
    --publish-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; publish_port="$2"; shift 2 ;;
    --command-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; command_port="$2"; shift 2 ;;
    --camera-cache-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; camera_cache_port="$2"; shift 2 ;;
    --policy-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_port="$2"; shift 2 ;;
    --metadata-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; metadata_port="$2"; shift 2 ;;
    --fps) [[ "$#" -ge 2 ]] || die "$1 requires a value"; requested_fps="$2"; shift 2 ;;
    --hand-telemetry-port) [[ "$#" -ge 2 ]] || die "$1 requires a value"; hand_telemetry_port="$2"; shift 2 ;;
    --bridge-config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; bridge_config="$2"; shift 2 ;;
    --ros-domain-id) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_domain_id="$2"; shift 2 ;;
    --ros-distro) [[ "$#" -ge 2 ]] || die "$1 requires a value"; ros_distro="$2"; shift 2 ;;
    --python) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_python="$2"; shift 2 ;;
    --print-config) print_config=1; shift ;;
    --help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ -n "${policy_path}" ]] || die "--policy-path is required; deployment metadata is embedded in the checkpoint"
validate_port "${publish_port}" || die "invalid observation port: ${publish_port}"
validate_port "${command_port}" || die "invalid command port: ${command_port}"
validate_port "${camera_cache_port}" || die "invalid camera cache port: ${camera_cache_port}"
validate_port "${policy_port}" || die "invalid policy port: ${policy_port}"
validate_port "${hand_telemetry_port}" || die "invalid hand telemetry port: ${hand_telemetry_port}"
validate_port "${metadata_port}" || die "invalid metadata port: ${metadata_port}"
[[ "${publish_port}" != "${command_port}" && "${publish_port}" != "${camera_cache_port}" && \
   "${publish_port}" != "${policy_port}" && "${command_port}" != "${camera_cache_port}" && \
   "${command_port}" != "${policy_port}" && "${camera_cache_port}" != "${policy_port}" && \
   "${hand_telemetry_port}" != "${publish_port}" && "${hand_telemetry_port}" != "${command_port}" && \
   "${hand_telemetry_port}" != "${camera_cache_port}" && "${hand_telemetry_port}" != "${policy_port}" && \
   "${metadata_port}" != "${publish_port}" && "${metadata_port}" != "${command_port}" && \
   "${metadata_port}" != "${camera_cache_port}" && "${metadata_port}" != "${policy_port}" && \
   "${metadata_port}" != "${hand_telemetry_port}" ]] || die "deployment ports must be distinct"

# A previous policy process commonly survives an interrupted deployment. Clear
# only the configured policy port before starting any ROS or policy children.
if [[ "${print_config}" -eq 0 ]]; then
  clear_port "${policy_port}"
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
policy_path="$(cd -- "${policy_path}" 2>/dev/null && pwd)" || die "policy path not found: ${policy_path}"
trajectory_output="$(PYTHONPATH="${repository_root}" python3 "${repository_root}/utils/deployment_metadata.py" --checkpoint "${policy_path}" --deployment-lines)" \
  || die "could not resolve deployment metadata from ${policy_path}"
mapfile -t trajectory_lines <<<"${trajectory_output}"
(( ${#trajectory_lines[@]} == 9 )) || die "deployment metadata resolver returned an unexpected number of fields"
arm_mode="${trajectory_lines[0]}"; end_effector="${trajectory_lines[1]}"; fps="${trajectory_lines[2]}"
state_dim="${trajectory_lines[3]}"; action_dim="${trajectory_lines[4]}"; state_action_mode="${trajectory_lines[5]}"; camera_names="${trajectory_lines[6]}"
policy_type="${trajectory_lines[7]}"; actions_per_chunk="${trajectory_lines[8]}"
if [[ -n "${requested_fps}" && "${requested_fps}" != "${fps}" ]]; then
  die "--fps=${requested_fps} does not match checkpoint metadata fps=${fps}"
fi
case "${arm_mode}" in left|right|duo) ;; *) die "unsupported metadata arm mode: ${arm_mode}" ;; esac
case "${end_effector}" in arm|gripper|hand) ;; *) die "unsupported metadata end effector: ${end_effector}" ;; esac
case "${state_action_mode}" in joint|end_effector) ;; *) die "unsupported metadata state/action mode: ${state_action_mode}" ;; esac
[[ "${state_dim}" =~ ^[0-9]+$ && "${action_dim}" =~ ^[0-9]+$ ]] || die "metadata dimensions must be integers"
[[ "${fps}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "metadata fps must be numeric"
include_right_arm=false; [[ "${arm_mode}" == "duo" ]] && include_right_arm=true
include_gripper=false; [[ "${end_effector}" == "gripper" ]] && include_gripper=true
include_hand=false; [[ "${end_effector}" == "hand" ]] && include_hand=true
camera_enabled() { case ",${camera_names}," in *",$1,"*) echo true ;; *) echo false ;; esac; }
camera_left="$(camera_enabled cam_left)"; camera_front="$(camera_enabled cam_front)"; camera_right="$(camera_enabled cam_right)"
[[ "${camera_left}${camera_front}${camera_right}" != "falsefalsefalse" ]] || die "metadata contains no supported deployment camera names: ${camera_names}"
if [[ "${camera_names}" != "" ]]; then
  IFS=',' read -r -a metadata_cameras <<<"${camera_names}"
  for camera_name in "${metadata_cameras[@]}"; do
    case "${camera_name}" in cam_left|cam_front|cam_right) ;; *) die "unsupported deployment camera name in metadata: ${camera_name}" ;; esac
  done
fi
if [[ "${print_config}" -eq 1 ]]; then
  printf 'policy_path=%s\narm_mode=%s\nend_effector=%s\nfps=%s\nstate_dim=%s\naction_dim=%s\nstate_action_mode=%s\ncameras=%s\npolicy_type=%s\nactions_per_chunk=%s\ninclude_right_arm=%s\ninclude_gripper=%s\ninclude_hand=%s\ncamera_1_enabled=%s\ncamera_2_enabled=%s\ncamera_3_enabled=%s\nserver_ip=%s\nmetadata_port=%s\n' \
    "${policy_path}" "${arm_mode}" "${end_effector}" "${fps}" "${state_dim}" "${action_dim}" "${state_action_mode}" \
    "${camera_names}" "${policy_type}" "${actions_per_chunk}" "${include_right_arm}" "${include_gripper}" "${include_hand}" \
    "${camera_left}" "${camera_front}" "${camera_right}" "${server_ip}" "${metadata_port}"
  exit 0
fi
# Fail before starting ROS processes when a previous deployment (or another
# service) still owns one of the server-side TCP ports.
check_port_available "${publish_port}" "observation"
check_port_available "${command_port}" "command"
check_port_available "${camera_cache_port}" "camera cache"
check_port_available "${policy_port}" "policy"
check_port_available "${metadata_port}" "metadata"
check_port_available "${hand_telemetry_port}" "hand telemetry"

# shellcheck source=scripts/conda_env.sh
source "${script_dir}/conda_env.sh"
if [[ -n "${ros_distro}" ]]; then
  ros_setup_file="/opt/ros/${ros_distro}/setup.bash"
else
  shopt -s nullglob; candidates=(/opt/ros/*/setup.bash); shopt -u nullglob
  [[ "${#candidates[@]}" -eq 1 ]] || die "select ROS distro with --ros-distro"
  ros_setup_file="${candidates[0]}"; ros_distro="$(basename "$(dirname "${ros_setup_file}")")"
fi
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
setup_files=(
  "${ros_setup_file}"
)
if [[ -r "${HOME}/franka_ros2_ws/install/local_setup.bash" ]]; then
  setup_files+=("${HOME}/franka_ros2_ws/install/local_setup.bash")
fi
setup_files+=("${repository_root}/gello_software/ros2/install/local_setup.bash")
set +u
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
command -v setsid >/dev/null || die "setsid is required"
command -v ros2 >/dev/null || die "ros2 is unavailable after sourcing ROS"
for required_package in franka_realsense_camera_publisher franka_lerobot_data_bridge franka_msgs; do
  ros2 pkg prefix "${required_package}" >/dev/null 2>&1 || die \
    "required ROS package is unavailable after sourcing overlays: ${required_package}"
done
declare -a policy_command=()
if [[ -n "${policy_python}" ]]; then
  command -v "${policy_python}" >/dev/null || die "policy Python not found: ${policy_python}"
  policy_command=("${policy_python}")
  "${policy_command[@]}" -c 'import grpc, torch, zmq' >/dev/null 2>&1 || die \
    "policy Python is missing grpc, torch, or zmq: ${policy_python}"
else
  deployment_conda_env="${DEPLOYMENT_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"
  real_exp_build_conda_python_command "${deployment_conda_env}" policy_command || exit 1
  real_exp_require_conda_python_modules "${deployment_conda_env}" grpc torch zmq || die \
    "the '${deployment_conda_env}' Conda environment is missing policy dependencies"
fi
bridge_config_source="${repository_root}/gello_software/ros2/src/franka_lerobot_data_bridge/config/${bridge_config}"
if [[ "${bridge_config}" = /* ]]; then bridge_config_source="${bridge_config}"; fi
[[ -f "${bridge_config_source}" ]] || die "bridge config not found: ${bridge_config}"
runtime_dir="${XDG_RUNTIME_DIR:-/tmp}"
bridge_config_runtime="$(mktemp "${runtime_dir}/real-exp-deployment-bridge.XXXXXX.yaml")"
if ! python3 "${repository_root}/deploy/build_deployment_bridge_config.py" \
  --base-config "${bridge_config_source}" --output "${bridge_config_runtime}" \
  --sample-rate-hz "${fps}" --publish-host "${server_ip}" --publish-port "${publish_port}" \
  --command-host "${server_ip}" --command-port "${command_port}" \
  --camera-cache-host 127.0.0.1 --camera-cache-port "${camera_cache_port}" \
  --include-right-arm "${include_right_arm}" --arm-mode "${arm_mode}" \
  --state-action-mode "${state_action_mode}" \
  --include-gripper "${include_gripper}" --include-hand "${include_hand}" \
  --hand-telemetry-host "${server_ip}" --hand-telemetry-port "${hand_telemetry_port}" \
  --camera-1-enabled "${camera_left}" --camera-2-enabled "${camera_front}" \
  --camera-3-enabled "${camera_right}"; then
  rm -f -- "${bridge_config_runtime}"
  die "could not build runtime deployment bridge config"
fi

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
  for pid in "${child_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  rm -f -- "${bridge_config_runtime}"
  exit "${status}"
}
trap shutdown EXIT; trap 'exit 130' INT; trap 'exit 143' TERM

echo "Deployment contract: ${end_effector}/${arm_mode}, mode=${state_action_mode}, state/action=${state_dim}/${action_dim}, cameras=${camera_names}, fps=${fps}"
echo "Deployment server: observation tcp://${server_ip}:${publish_port}, command tcp://${server_ip}:${command_port}, hand telemetry tcp://${server_ip}:${hand_telemetry_port}, camera cache tcp://127.0.0.1:${camera_cache_port}, gRPC :${policy_port}"
start_process "RealSense camera publisher" ros2 launch franka_realsense_camera_publisher cameras.launch.py
start_process "Deployment observation bridge" ros2 launch franka_lerobot_data_bridge bridge.launch.py \
  "config_file:=${bridge_config_runtime}"
start_process "Policy server" "${policy_command[@]}" "${repository_root}/deploy/deploy_lerobot_policy.py" server \
  --host 0.0.0.0 --port "${policy_port}" --metadata-port "${metadata_port}" --policy-path "${policy_path}" --fps "${fps}" \
  --camera-cache-address "tcp://127.0.0.1:${camera_cache_port}"
echo "Deployment server is running. Press Ctrl-C to stop all children."
set +e; completed_pid=""; wait -n -p completed_pid "${child_pids[@]}"; status=$?; set -e
echo "${child_names[${completed_pid}]} exited with status ${status}." >&2; exit "${status}"
