#!/usr/bin/env bash
set -euo pipefail

module_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${module_dir}/.." && pwd)"
cd "${repository_root}"
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"
execute=0
for argument in "$@"; do
  case "${argument}" in
    -h|--help|--validate-only) exec /usr/bin/python3 -m softhand_grasp.client "$@" ;;
    --execute) execute=1 ;;
  esac
done
if [[ "${execute}" == 0 ]]; then
  exec /usr/bin/python3 -m softhand_grasp.client "$@"
fi

# Validate before starting the arm controller.
/usr/bin/python3 -m softhand_grasp.client "$@" --validate-only

# Match scripts/start_teleoperation.sh's ROS overlays and controller config.
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
set +u
for setup_file in /opt/ros/humble/setup.bash \
  "${HOME}/franka_ros2_ws/install/local_setup.bash" \
  "${repository_root}/gello_software/ros2/install/local_setup.bash"; do
  [[ -r "${setup_file}" ]] || { echo "Missing ROS setup: ${setup_file}" >&2; exit 1; }
  source "${setup_file}"
done
set -u
export PYTHONPATH="${repository_root}${PYTHONPATH:+:${PYTHONPATH}}"

# Share the teleoperation lock to prevent competing commands to the right arm.
exec 9>"${XDG_RUNTIME_DIR:-/tmp}/real-exp-teleoperation-${UID}.lock"
flock -n 9 || { echo "Stop arm teleoperation before grasp execution." >&2; exit 1; }

controller_pid=""
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${controller_pid}" ]]; then
    kill -INT -- "-${controller_pid}" 2>/dev/null || true
    for ((attempt=0; attempt<50; attempt++)); do
      kill -0 -- "-${controller_pid}" 2>/dev/null || break
      sleep 0.1
    done
    kill -TERM -- "-${controller_pid}" 2>/dev/null || true
    for ((attempt=0; attempt<30; attempt++)); do
      kill -0 -- "-${controller_pid}" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-${controller_pid}" 2>/dev/null || true
    wait "${controller_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid -- env --default-signal=INT,QUIT,TERM -- \
  bash -c 'exec 9>&-; exec "$@"' _ \
  ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  robot_config_file:=example_fr3_right_config_no_gripper.yaml motion_controller:=trajectory &
controller_pid=$!

deadline=$((SECONDS + 90))
ready=0
while ((SECONDS < deadline)); do
  kill -0 "${controller_pid}" 2>/dev/null || { echo "Right-arm controller exited." >&2; exit 1; }
  if timeout 5s ros2 topic echo /right/franka/joint_states --once --no-daemon >/dev/null 2>&1; then
    ready=1
    break
  fi
done
[[ "${ready}" == 1 ]] || { echo "Timed out waiting for right-arm state." >&2; exit 1; }
/usr/bin/python3 -m softhand_grasp.client "$@"
