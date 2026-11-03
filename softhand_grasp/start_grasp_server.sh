#!/usr/bin/env bash
set -euo pipefail

module_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${module_dir}/.." && pwd)"
cd "${repository_root}"
# Server needs NumPy/pyzmq only; SAM3 and the camera may use separate interpreters.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH PYTHONHOME
export PYTHONPATH="${repository_root}"
exec "${SOFTHAND_SERVER_PYTHON:-python}" -m softhand_grasp.server "$@"
