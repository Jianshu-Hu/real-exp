#!/usr/bin/env bash
set -euo pipefail

# Robot-control host: request inference and invoke the local guarded move tool.
grasp_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${grasp_dir}/.." && pwd)"
initial_move_script="${repository_root}/scripts/move_to_target_ee.sh"
readonly -a initial_ee_xyzrpy=(
  0.682977 0.154027 0.452649 -2.134387 0.498717 -2.334388
)
readonly -a initial_hand_joints=(
  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
)
# Fixed right Wuji Hand 2 SDK endpoint for this setup. The hand uses port 7447;
# port 50001 belongs to the right glove at 192.168.1.101. It is used only in
# --arm-with-hand mode; --arm-only never starts or contacts a hand process.
readonly right_hand_ip="192.168.1.111:7447"
export GRASP_FIXED_RIGHT_HAND_IP="${right_hand_ip}"
client_python="/usr/bin/python3"
[[ -x "${client_python}" ]] || {
  echo "Error: control-host Python is missing: ${client_python}" >&2
  exit 1
}

missing_modules="$(${client_python} - <<'PY'
import importlib.util

modules = ("numpy", "scipy", "zmq")
print(" ".join(module for module in modules if importlib.util.find_spec(module) is None))
PY
)"
[[ -z "${missing_modules}" ]] || {
  echo "Error: ${client_python} is missing modules: ${missing_modules}" >&2
  exit 1
}

server_ip="${GRASP_SERVER_IP:-${DATA_COLLECTION_SERVER_IP:-192.168.50.13}}"
server_port="${GRASP_INFERENCE_PORT:-5571}"
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"

# Help must never have the side effect of moving hardware.
for argument in "$@"; do
  if [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
    exec "${client_python}" -m grasp.grasp_execution_client \
      --server-address "tcp://${server_ip}:${server_port}" \
      "$@"
  fi
done

# Reject invalid client arguments before offering to move the real robot.
"${client_python}" - "$@" <<'PY'
import os
import sys

from grasp.grasp_execution_client import build_parser

parser = build_parser()
args = parser.parse_args(sys.argv[1:])
if args.request_timeout_s <= 0 or args.max_command_age_s <= 0:
    parser.error("timeouts and maximum command age must be positive")
if not args.move_script.is_file():
    parser.error(f"move script does not exist: {args.move_script}")
if args.control_mode == "arm_with_hand" and not os.environ.get(
    "GRASP_FIXED_RIGHT_HAND_IP"
):
    parser.error(
        "arm-with-hand mode must be started through "
        "start_grasp_execution_client.sh"
    )
PY

[[ -x "${initial_move_script}" ]] || {
  echo "Error: initial EE move script is missing or not executable: ${initial_move_script}" >&2
  exit 1
}

initial_end_effector_args=(--arm)
for argument in "$@"; do
  if [[ "${argument}" == "--arm-with-hand" ]]; then
    initial_end_effector_args=(
      --hand
      --target-ee-joint "${initial_hand_joints[@]}"
      --right-hand-ip "${right_hand_ip}"
    )
    break
  fi
done

echo "Moving the right-arm EE to the configured initial xyzrpy:"
echo "  ${initial_ee_xyzrpy[*]}"
if [[ "${initial_end_effector_args[0]}" == "--hand" ]]; then
  echo "Resetting all right-hand joints to zero during the initial move."
fi
"${initial_move_script}" --right "${initial_end_effector_args[@]}" \
  --target-ee-pose "${initial_ee_xyzrpy[@]}"

exec "${client_python}" -m grasp.grasp_execution_client \
  --server-address "tcp://${server_ip}:${server_port}" \
  "$@" \
  --return-ee-pose "${initial_ee_xyzrpy[@]}"
