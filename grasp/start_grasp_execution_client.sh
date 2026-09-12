#!/usr/bin/env bash
set -euo pipefail

grasp_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${grasp_dir}/.." && pwd)"
replay_program="${repository_root}/data_collection/replay_lerobot_episode.py"
readonly -a initial_ee_xyzrpy=(
  0.6217188822449426 0.5166373362592817 0.419112304865263
  -1.6543702490520547 0.6474849359530839 -1.3522691731776382
)
readonly -a initial_hand_joints=(
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
)
readonly right_hand_ip="192.168.1.111:7447"
client_python="/usr/bin/python3"

die() {
  echo "Error: $*" >&2
  exit 1
}

[[ -x "${client_python}" ]] || die "control-host Python is missing: ${client_python}"
server_ip="${GRASP_SERVER_IP:-${DATA_COLLECTION_SERVER_IP:-192.168.50.13}}"
server_port="${GRASP_INFERENCE_PORT:-5571}"
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"

# Help must never start a controller or move hardware.
for argument in "$@"; do
  if [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
    exec "${client_python}" -m grasp.grasp_execution_client \
      --server-address "tcp://${server_ip}:${server_port}" \
      "$@"
  fi
done

missing_modules="$(${client_python} - <<'PY'
import importlib.util

modules = ("numpy", "scipy", "zmq")
print(" ".join(module for module in modules if importlib.util.find_spec(module) is None))
PY
)"
[[ -z "${missing_modules}" ]] || die \
  "${client_python} is missing modules: ${missing_modules}"

# Reject invalid client arguments before starting robot-control processes.
"${client_python}" - "$@" <<'PY'
import sys

from grasp.grasp_execution_client import build_parser

parser = build_parser()
args = parser.parse_args(sys.argv[1:])
if args.request_timeout_s <= 0 or args.max_command_age_s <= 0:
    parser.error("timeouts and maximum command age must be positive")
PY

control_mode=""
for argument in "$@"; do
  case "${argument}" in
    --arm-only) control_mode="arm_only" ;;
    --arm-with-hand) control_mode="arm_with_hand" ;;
  esac
done

unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
setup_files=(
  "/opt/ros/humble/setup.bash"
  "${HOME}/franka_ros2_ws/install/local_setup.bash"
  "${repository_root}/gello_software/ros2/install/local_setup.bash"
)
set +u
for setup_file in "${setup_files[@]}"; do
  [[ -r "${setup_file}" ]] || die "ROS setup file is missing or unreadable: ${setup_file}"
  # shellcheck disable=SC1090
  source "${setup_file}"
done
set -u
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"

for command_name in setsid timeout ss; do
  command -v "${command_name}" >/dev/null 2>&1 || die \
    "required command not found: ${command_name}"
done

declare -a wuji_python=()
if [[ "${control_mode}" == "arm_with_hand" ]]; then
  # shellcheck source=scripts/conda_env.sh
  source "${repository_root}/scripts/conda_env.sh"
  wuji_conda_env="${WUJI_CONDA_ENV:-${LEROBOT_CONDA_ENV:-lerobot}}"
  real_exp_build_conda_python_command "${wuji_conda_env}" wuji_python || exit 1
  real_exp_require_conda_python_modules \
    "${wuji_conda_env}" wuji_sdk wujihandpy zmq numpy || die \
    "the '${wuji_conda_env}' Conda environment cannot run Wuji hand control"
fi

declare -a child_pids=()
shutdown_started=0
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  [[ "${shutdown_started}" -eq 1 ]] && return
  shutdown_started=1
  for pid in "${child_pids[@]}"; do kill -INT -- "-${pid}" 2>/dev/null || true; done
  sleep 0.5
  for pid in "${child_pids[@]}"; do kill -TERM -- "-${pid}" 2>/dev/null || true; done
  for pid in "${child_pids[@]}"; do wait "${pid}" 2>/dev/null || true; done
  exit "${status}"
}
trap cleanup EXIT INT TERM

start_process() {
  setsid -- bash -c 'trap - INT QUIT; exec "$@"' _ "$@" &
  child_pids+=("$!")
}

wait_for_topic() {
  local topic="$1" pid="$2" deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    kill -0 "${pid}" 2>/dev/null || die \
      "FR3 controller exited before ${topic} became available"
    if timeout 5s ros2 topic echo "${topic}" --once --no-daemon >/dev/null 2>&1; then
      return 0
    fi
  done
  die "timed out waiting for ${topic}"
}

wait_for_subscription() {
  local topic="$1" pid="$2" deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    kill -0 "${pid}" 2>/dev/null || die \
      "FR3 controller exited before subscribing to ${topic}"
    if timeout 5s ros2 topic info "${topic}" --no-daemon 2>/dev/null \
      | grep -Eq 'Subscription count: [1-9][0-9]*'; then
      return 0
    fi
    sleep 0.5
  done
  die "timed out waiting for a subscription to ${topic}"
}

wait_for_tcp_port() {
  local port="$1" pid="$2" deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    kill -0 "${pid}" 2>/dev/null || die \
      "Wuji hand worker exited before binding port ${port}"
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${port}$"; then
      return 0
    fi
    sleep 0.5
  done
  die "timed out waiting for Wuji hand worker port ${port}"
}

echo "Starting right FR3 ROS controller (example_fr3_right_config_no_gripper.yaml)."
start_process ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  robot_config_file:=example_fr3_right_config_no_gripper.yaml motion_controller:=trajectory
controller_pid="${child_pids[0]}"
wait_for_topic /right/franka/joint_states "${controller_pid}"
wait_for_subscription /right/fr3_arm_controller/joint_trajectory "${controller_pid}"

initial_end_effector_args=(--arm)
if [[ "${control_mode}" == "arm_with_hand" ]]; then
  start_process "${wuji_python[@]}" "${replay_program}" \
    --internal-wuji-hand right \
    --right-hand-command-port 5562 \
    --right-hand-status-port 5564 \
    --hand-ip "${right_hand_ip}"
  wait_for_tcp_port 5564 "${child_pids[${#child_pids[@]} - 1]}"
  initial_end_effector_args=(
    --hand
    --target-ee-joint "${initial_hand_joints[@]}"
  )
fi

echo "Moving the right-arm EE to the configured initial xyzrpy:"
echo "  ${initial_ee_xyzrpy[*]}"
if [[ "${control_mode}" == "arm_with_hand" ]]; then
  echo "Resetting all right-hand joints to zero during the initial move."
fi
"${client_python}" -m grasp.grasp_motion \
  --right "${initial_end_effector_args[@]}" \
  --target-ee-pose "${initial_ee_xyzrpy[@]}"

set +e
"${client_python}" -m grasp.grasp_execution_client \
  --server-address "tcp://${server_ip}:${server_port}" \
  "$@" \
  --return-ee-pose "${initial_ee_xyzrpy[@]}"
status=$?
set -e
exit "${status}"
