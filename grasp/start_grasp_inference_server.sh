#!/usr/bin/env bash
set -euo pipefail

# Camera/data server: wait for control-host triggers, then capture and infer.
grasp_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${grasp_dir}/.." && pwd)"
# shellcheck source=scripts/conda_env.sh
source "${repository_root}/scripts/conda_env.sh"

inference_env="${GRASP_CONDA_ENV:-wjh_grasp}"
declare -a inference_python=()
real_exp_build_conda_python_command "${inference_env}" inference_python || exit 1
real_exp_require_conda_python_modules "${inference_env}" chumpy numpy scipy smplx torch zmq || {
  echo "Error: the '${inference_env}' Conda environment lacks grasp-server dependencies." >&2
  echo "Install them with:" >&2
  echo "  conda run -n '${inference_env}' python -m pip install chumpy smplx" >&2
  exit 1
}
echo "Using Conda environment '${inference_env}' for grasp inference."

server_ip="${GRASP_SERVER_IP:-${DATA_COLLECTION_SERVER_IP:-192.168.50.13}}"
server_port="${GRASP_INFERENCE_PORT:-5571}"
cd -- "${repository_root}"
exec "${inference_python[@]}" -m grasp.camera_inference_server \
  --bind "tcp://${server_ip}:${server_port}" \
  --runs-dir "${GRASP_RUNS_DIR:-${repository_root}/grasp/runs}" \
  "$@"
