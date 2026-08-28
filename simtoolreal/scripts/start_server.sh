#!/usr/bin/env bash
set -euo pipefail

# Server computer: right-hand bridge + FoundationPose++ + checkpoint host.
die() { echo "Error: $*" >&2; exit 2; }
server_ip="${SIMTOOLREAL_SERVER_IP:-${DEPLOYMENT_SERVER_IP:-192.168.50.13}}"
bridge_config="${SIMTOOLREAL_BRIDGE_CONFIG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../config" && pwd)/deployment_right_hand.yaml}"
pose_address="${SIMTOOLREAL_POSE_ADDRESS:-tcp://0.0.0.0:5570}"
policy_bind="${SIMTOOLREAL_POLICY_ADDRESS:-tcp://0.0.0.0:5571}"
ros_distro="${SIMTOOLREAL_ROS_DISTRO:-${ROS_DISTRO:-}}"
policy_python="${SIMTOOLREAL_POLICY_PYTHON:-python3}"
pose_python="${SIMTOOLREAL_POSE_PYTHON:-python3}"
repository_root_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
policy_config="${SIMTOOLREAL_POLICY_CONFIG:-${repository_root_default}/libs/SimToolReal-Franka-Wuji2/pretrained_policy/config.yaml}"
policy_checkpoint="${SIMTOOLREAL_POLICY_CHECKPOINT:-${repository_root_default}/libs/SimToolReal-Franka-Wuji2/pretrained_policy/model.pth}"
pose_mode=""; pose_mesh=""; pose_roi=(); pose_no_display=0; mock_policy=0; start_bridge=1; wait_only=1; policy_upstream=""; policy_device="cpu"; pose_file=""

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --server-ip) [[ "$#" -ge 2 ]] || die "$1 requires a value"; server_ip="$2"; shift 2 ;;
    --bridge-config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; bridge_config="$2"; shift 2 ;;
    --no-bridge) start_bridge=0; shift ;;
    --foundationpose-mesh) [[ "$#" -ge 2 ]] || die "$1 requires a value"; pose_mode=live; pose_mesh="$2"; shift 2 ;;
    --foundationpose-pose-file) [[ "$#" -ge 2 ]] || die "$1 requires a value"; pose_mode=file; pose_file="$2"; shift 2 ;;
    --foundationpose-mock) pose_mode=mock; shift ;;
    --foundationpose-roi) [[ "$#" -ge 5 ]] || die "$1 requires X Y W H"; pose_roi=("$2" "$3" "$4" "$5"); shift 5 ;;
    --foundationpose-no-display) pose_no_display=1; shift ;;
    --pose-address) [[ "$#" -ge 2 ]] || die "$1 requires a value"; pose_address="$2"; shift 2 ;;
    --policy-bind) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_bind="$2"; shift 2 ;;
    --config) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_config="$2"; shift 2 ;;
    --checkpoint) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_checkpoint="$2"; shift 2 ;;
    --upstream-root) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_upstream="$2"; shift 2 ;;
    --device) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_device="$2"; shift 2 ;;
    --policy-python) [[ "$#" -ge 2 ]] || die "$1 requires a value"; policy_python="$2"; shift 2 ;;
    --pose-python) [[ "$#" -ge 2 ]] || die "$1 requires a value"; pose_python="$2"; shift 2 ;;
    --mock-policy) mock_policy=1; shift ;;
    --wait-only) wait_only=1; shift ;;
    --no-wait-only) wait_only=0; shift ;;
    --help|-h) echo "Usage: start_server.sh [--config CONFIG --checkpoint CHECKPOINT] [--foundationpose-mesh MESH --foundationpose-roi X Y W H]"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
if [[ "${mock_policy}" -eq 0 ]]; then
  [[ -n "${policy_config}" && -n "${policy_checkpoint}" ]] || die "--config and --checkpoint are required (or use --mock-policy)"
fi
[[ -f "${bridge_config}" ]] || die "bridge config not found: ${bridge_config}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; root_dir="$(cd -- "${script_dir}/.." && pwd)"; repository_root="$(cd -- "${root_dir}/.." && pwd)"
if [[ -z "${ros_distro}" && "${start_bridge}" -eq 1 ]]; then
  shopt -s nullglob; ros_setup=(/opt/ros/*/setup.bash); shopt -u nullglob
  [[ "${#ros_setup[@]}" -eq 1 ]] || die "source ROS or set SIMTOOLREAL_ROS_DISTRO"
  ros_distro="$(basename -- "$(dirname -- "${ros_setup[0]}")")"
fi
declare -a pids=()
cleanup() { local status=$?; trap - EXIT INT TERM; for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done; exit "$status"; }
trap cleanup EXIT INT TERM
if [[ "${start_bridge}" -eq 1 ]]; then
  set +u
  source "/opt/ros/${ros_distro}/setup.bash"
  [[ -r "${HOME}/franka_ros2_ws/install/local_setup.bash" ]] && source "${HOME}/franka_ros2_ws/install/local_setup.bash"
  [[ -r "${repository_root}/gello_software/ros2/install/local_setup.bash" ]] && source "${repository_root}/gello_software/ros2/install/local_setup.bash"
  set -u
  export ROS_LOCALHOST_ONLY=0 ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
  ros2 launch franka_lerobot_data_bridge bridge.launch.py "config_file:=${bridge_config}" & pids+=("$!")
fi
export PYTHONPATH="${root_dir}:${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${pose_mode}" ]]; then
  pose_args=(--connect "${pose_address}")
  [[ "${pose_mode}" == mock ]] && pose_args+=(--mock)
  [[ "${pose_mode}" == file ]] && pose_args+=(--pose-file "${pose_file}")
  [[ "${pose_mode}" == live ]] && pose_args+=(--mesh "${pose_mesh}")
  [[ "${#pose_roi[@]}" -eq 4 ]] && pose_args+=(--roi "${pose_roi[@]}")
  [[ "${pose_no_display}" -eq 1 ]] && pose_args+=(--no-display)
  "${pose_python}" "${root_dir}/foundation_pose_runner.py" "${pose_args[@]}" & pids+=("$!")
fi
if [[ -z "${pose_mode}" ]]; then
  die "select a FoundationPose++ source with --foundationpose-mesh, --foundationpose-pose-file, or --foundationpose-mock"
fi
policy_args=(--bind "${policy_bind}")
[[ "${mock_policy}" -eq 1 ]] && policy_args+=(--mock-policy)
[[ -n "${policy_config}" ]] && policy_args+=(--config "${policy_config}")
[[ -n "${policy_checkpoint}" ]] && policy_args+=(--checkpoint "${policy_checkpoint}")
[[ -n "${policy_upstream}" ]] && policy_args+=(--upstream-root "${policy_upstream}")
policy_args+=(--device "${policy_device}")
"${policy_python}" "${root_dir}/policy_server.py" "${policy_args[@]}" & pids+=("$!")
if [[ "${wait_only}" -eq 1 ]]; then
  "${policy_python}" "${root_dir}/state_server.py" --bridge-address "tcp://127.0.0.1:5555" --wait-only & pids+=("$!")
fi
# Keep all managed children supervised. If any required process exits, stop the
# remaining stack through the cleanup trap.
wait -n "${pids[@]}"
status=$?
echo "SimToolReal server child exited; stopping remaining processes" >&2
exit "${status}"
