#!/usr/bin/env bash
set -euo pipefail

# Robot-control host: request inference and invoke the local guarded move tool.
grasp_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${grasp_dir}/.." && pwd)"
# Fixed right Wuji Hand 2 SDK endpoint for this setup. It is used only in
# --arm-with-hand mode; --arm-only never starts or contacts a hand process.
readonly right_hand_ip="192.168.1.111:50001"
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
exec "${client_python}" -m grasp.grasp_execution_client \
  --server-address "tcp://${server_ip}:${server_port}" \
  "$@"
