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
declare -a camera_args=()
has_d435i_only=0
has_secondary_cli=0
has_camera_python=0
has_camera_pythonpath=0
for argument in "$@"; do
  case "${argument}" in
    --d435i-only) has_d435i_only=1 ;;
    --secondary-camera-serial|--secondary-camera-serial=*) has_secondary_cli=1 ;;
    --camera-python|--camera-python=*) has_camera_python=1 ;;
    --camera-pythonpath|--camera-pythonpath=*) has_camera_pythonpath=1 ;;
  esac
done

secondary_serial="${GRASP_SECONDARY_CAMERA_SERIAL:-}"
has_secondary_option=0
if [[ "${has_secondary_cli}" -eq 1 || -n "${secondary_serial}" ]]; then
  has_secondary_option=1
fi
declare -a environment_camera_args=()
if [[ "${has_secondary_cli}" -eq 0 && -n "${secondary_serial}" ]]; then
  environment_camera_args+=(--secondary-camera-serial "${secondary_serial}")
fi

# L515 is the default capture source. Its older compatible binding is kept
# isolated from the inference environment and is also required for dual mode.
if [[ "${has_d435i_only}" -eq 0 ]]; then
  l515_python="${GRASP_L515_CAMERA_PYTHON:-${HOME}/miniconda3/envs/pose/bin/python}"
  l515_pythonpath="${GRASP_L515_PYTHONPATH:-${repository_root}/.vendor/l515_realsense}"
  [[ -x "${l515_python}" ]] || {
    echo "Error: L515-compatible Python is missing: ${l515_python}" >&2
    exit 1
  }
  [[ -d "${l515_pythonpath}" ]] || {
    echo "Error: L515-compatible pyrealsense2 directory is missing: ${l515_pythonpath}" >&2
    echo "Install pyrealsense2 2.54.2 as described in calibration/README.md." >&2
    exit 1
  }
  camera_args+=("${environment_camera_args[@]}")
  [[ "${has_camera_python}" -eq 1 ]] || camera_args+=(--camera-python "${l515_python}")
  [[ "${has_camera_pythonpath}" -eq 1 ]] || camera_args+=(--camera-pythonpath "${l515_pythonpath}")
elif [[ "${has_secondary_option}" -eq 0 ]]; then
  # The system SDK supports the D435i and avoids importing the L515 vendor tree.
  [[ "${has_camera_python}" -eq 1 ]] || camera_args+=(--camera-python /usr/bin/python3)
else
  # Let argparse report an explicit --d435i-only/secondary-camera conflict,
  # while preserving an environment-selected secondary camera in the command.
  camera_args+=("${environment_camera_args[@]}")
fi
cd -- "${repository_root}"
exec "${inference_python[@]}" -m grasp.camera_inference_server \
  --bind "tcp://${server_ip}:${server_port}" \
  --runs-dir "${GRASP_RUNS_DIR:-${repository_root}/grasp/runs}" \
  "${camera_args[@]}" \
  "$@"
