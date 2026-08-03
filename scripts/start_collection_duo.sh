#!/usr/bin/env bash
#
# Start the five long-running ROS 2 components required for dual-arm data
# collection from one terminal:
#   1. GELLO joint-state publisher
#   2. Franka arm controllers
#   3. Franka gripper clients
#   4. RealSense camera publishers
#   5. LeRobot data bridge
#
# The LeRobot recorder intentionally remains in a separate Conda terminal.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
FRANKA_WORKSPACE_SETUP="${FRANKA_WORKSPACE_SETUP:-${HOME}/franka_ros2_ws/install/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-${REPO_ROOT}/gello_software/ros2/install/setup.bash}"
READY_TIMEOUT="${READY_TIMEOUT:-60}"
COLLECTION_ZMQ_PORT="${COLLECTION_ZMQ_PORT:-5555}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${COLLECTION_LOG_DIR:-${REPO_ROOT}/log/collection/${RUN_ID}}"
LOCK_FILE="${COLLECTION_LOCK_FILE:-/tmp/real-exp-collection-duo.lock}"

declare -a COMPONENT_NAMES=()
declare -a COMPONENT_PIDS=()
SHUTTING_DOWN=0

log() {
  printf '[collection] %s\n' "$*"
}

fail() {
  log "ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  local description="$2"
  [[ -f "${path}" ]] || fail "${description} not found: ${path}"
}

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 || fail "required command is unavailable: ${command_name}"
}

source_ros_environment() {
  require_file "${ROS_SETUP}" "ROS 2 setup file"
  require_file "${FRANKA_WORKSPACE_SETUP}" "Franka workspace setup file"
  require_file "${WORKSPACE_SETUP}" "workspace setup file"

  # ROS 2 Humble's generated setup files probe variables such as
  # AMENT_TRACE_SETUP_FILES before defining them, so they are not compatible
  # with Bash nounset. Restore strict checking immediately after sourcing.
  set +u
  # shellcheck disable=SC1090
  source "${ROS_SETUP}"
  # shellcheck disable=SC1090
  source "${FRANKA_WORKSPACE_SETUP}"
  # shellcheck disable=SC1090
  source "${WORKSPACE_SETUP}"
  set -u

  require_command ros2
}

check_port_available() {
  local port="$1"

  if ! command -v ss >/dev/null 2>&1; then
    log "WARN: ss is unavailable; skipping the ZMQ port ${port} preflight check"
    return
  fi

  if ss -ltnH | awk -v suffix=":${port}" '$4 ~ suffix "$" { found=1 } END { exit !found }'; then
    fail "TCP port ${port} is already in use. An old LeRobot bridge may still be running."
  fi
}

component_pid() {
  local wanted="$1"
  local index

  for index in "${!COMPONENT_NAMES[@]}"; do
    if [[ "${COMPONENT_NAMES[index]}" == "${wanted}" ]]; then
      printf '%s\n' "${COMPONENT_PIDS[index]}"
      return 0
    fi
  done

  return 1
}

assert_component_alive() {
  local name="$1"
  local pid

  pid="$(component_pid "${name}")" || fail "unknown component: ${name}"
  kill -0 "${pid}" 2>/dev/null || fail "${name} exited during startup; inspect ${LOG_DIR}/${name}.log"
}

start_component() {
  local name="$1"
  shift
  local logfile="${LOG_DIR}/${name}.log"
  local pid
  local pgid=""
  local attempt

  log "Starting ${name}"
  (
    # Long-running children must not inherit the collection lock. Run every
    # component in its own session so shutdown can signal the complete process
    # tree instead of only the ros2/launch parent.
    exec 9>&-
    exec setsid stdbuf -oL -eL "$@"
  ) > >(
    exec 9>&-
    sed -u "s/^/[${name}] /" | tee -a "${logfile}"
  ) 2>&1 &

  pid="$!"
  for attempt in {1..40}; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      fail "${name} exited before its process group was ready; inspect ${logfile}"
    fi
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    [[ "${pgid}" == "${pid}" ]] && break
    sleep 0.05
  done

  if [[ "${pgid}" != "${pid}" ]]; then
    kill -TERM "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    fail "could not isolate ${name} in its own process group"
  fi

  COMPONENT_NAMES+=("${name}")
  COMPONENT_PIDS+=("${pid}")
}

wait_for_topic_message() {
  local component="$1"
  local topic="$2"
  local deadline=$((SECONDS + READY_TIMEOUT))

  log "Waiting for a message on ${topic}"
  while (( SECONDS < deadline )); do
    assert_component_alive "${component}"
    # Checking `ros2 topic list` is insufficient because a publisher can
    # advertise a topic without delivering fresh data. A short-lived echo
    # verifies that the upstream hardware path is actually producing samples.
    if timeout 5 ros2 topic echo --once "${topic}" >/dev/null 2>&1; then
      log "Ready: ${topic}"
      return 0
    fi
  done

  fail "timed out after ${READY_TIMEOUT}s waiting for data on ${topic}"
}

wait_for_topic_subscriber() {
  local component="$1"
  local topic="$2"
  local deadline=$((SECONDS + READY_TIMEOUT))

  log "Waiting for ${component} to subscribe to ${topic}"
  while (( SECONDS < deadline )); do
    assert_component_alive "${component}"
    if ros2 topic info "${topic}" 2>/dev/null \
      | grep -Eq '^Subscription count: [1-9][0-9]*$'; then
      log "Ready: ${component} initialized and subscribed"
      return 0
    fi
    sleep 1
  done

  fail "${component} did not finish homing/initialization within ${READY_TIMEOUT}s; inspect ${LOG_DIR}/${component}.log"
}

wait_for_port() {
  local component="$1"
  local port="$2"
  local deadline=$((SECONDS + READY_TIMEOUT))

  if ! command -v ss >/dev/null 2>&1; then
    log "WARN: ss is unavailable; cannot verify bridge port ${port}"
    return 0
  fi

  log "Waiting for TCP port ${port}"
  while (( SECONDS < deadline )); do
    assert_component_alive "${component}"
    if ss -ltnH | awk -v suffix=":${port}" '$4 ~ suffix "$" { found=1 } END { exit !found }'; then
      log "Ready: TCP port ${port}"
      return 0
    fi
    sleep 1
  done

  fail "timed out after ${READY_TIMEOUT}s waiting for TCP port ${port}"
}

stop_components() {
  local index
  local pid
  local deadline

  (( SHUTTING_DOWN == 0 )) || return
  SHUTTING_DOWN=1

  if (( ${#COMPONENT_PIDS[@]} == 0 )); then
    return
  fi

  log "Stopping ROS 2 components in reverse order"
  for (( index=${#COMPONENT_PIDS[@]}-1; index>=0; index-- )); do
    pid="${COMPONENT_PIDS[index]}"
    if kill -0 -- "-${pid}" 2>/dev/null; then
      log "Stopping ${COMPONENT_NAMES[index]} (pid ${pid})"
      kill -INT -- "-${pid}" 2>/dev/null || true
    fi
  done

  deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    local any_alive=0
    for pid in "${COMPONENT_PIDS[@]}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    (( any_alive == 0 )) && break
    sleep 1
  done

  for (( index=${#COMPONENT_PIDS[@]}-1; index>=0; index-- )); do
    pid="${COMPONENT_PIDS[index]}"
    if kill -0 -- "-${pid}" 2>/dev/null; then
      log "WARN: ${COMPONENT_NAMES[index]} did not stop after SIGINT; sending SIGTERM"
      kill -TERM -- "-${pid}" 2>/dev/null || true
    fi
  done

  deadline=$((SECONDS + 5))
  while (( SECONDS < deadline )); do
    local any_alive=0
    for pid in "${COMPONENT_PIDS[@]}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then
        any_alive=1
        break
      fi
    done
    (( any_alive == 0 )) && break
    sleep 1
  done

  for (( index=${#COMPONENT_PIDS[@]}-1; index>=0; index-- )); do
    pid="${COMPONENT_PIDS[index]}"
    if kill -0 -- "-${pid}" 2>/dev/null; then
      log "WARN: ${COMPONENT_NAMES[index]} did not stop after SIGTERM; sending SIGKILL"
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  done
}

on_exit() {
  local status=$?
  # A second Ctrl+C or a terminal hangup must not interrupt cleanup. Otherwise
  # launch children can survive and keep cameras, FCI, ports, or the lock busy.
  trap - EXIT
  trap '' INT TERM HUP
  stop_components
  exit "${status}"
}

monitor_components() {
  local index
  local pid

  log "All collection ROS components are ready"
  log "Logs: ${LOG_DIR}"
  log "Start the recorder in a second terminal:"
  log "  source ~/anaconda3/bin/activate && conda activate lerobot"
  log "  cd ${REPO_ROOT}"
  log "  python data_collection/lerobot_collection.py"
  log "Press Ctrl+C here to stop the complete ROS collection stack."

  while true; do
    for index in "${!COMPONENT_PIDS[@]}"; do
      pid="${COMPONENT_PIDS[index]}"
      if ! kill -0 "${pid}" 2>/dev/null; then
        fail "${COMPONENT_NAMES[index]} exited unexpectedly; inspect ${LOG_DIR}/${COMPONENT_NAMES[index]}.log"
      fi
    done
    sleep 1
  done
}

main() {
  for required_command in flock stdbuf sed tee awk timeout setsid ps tr grep; do
    require_command "${required_command}"
  done

  exec 9>"${LOCK_FILE}"
  flock -n 9 || fail "another dual-arm collection stack is already running"
  mkdir -p "${LOG_DIR}"

  source_ros_environment
  check_port_available "${COLLECTION_ZMQ_PORT}"

  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    log "WARN: a Conda environment is active (${CONDA_PREFIX}). ROS nodes should normally use the system ROS environment."
  fi

  trap on_exit EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP

  start_component gello \
    ros2 launch franka_gello_state_publisher main.launch.py \
    config_file:=gello_duo.yaml
  wait_for_topic_message gello /left/gello/joint_states
  wait_for_topic_message gello /right/gello/joint_states

  # Collection uses the normal joint-impedance controller path. Do not pass
  # deployment_mode:=true here.
  start_component arms \
    ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
    robot_config_file:=example_fr3_duo_config.yaml
  wait_for_topic_message arms /left/franka/joint_states
  wait_for_topic_message arms /right/franka/joint_states

  # Home and initialize the grippers one at a time. Starting both action
  # clients concurrently has occasionally caused one Homing goal response to
  # time out. The command subscription is created only after homing, maximum
  # width discovery, and Move action-server discovery have all succeeded.
  start_component gripper_left \
    ros2 run franka_gripper_manager franka_gripper_client \
    --ros-args -r __node:=franka_gripper_client -r __ns:=/left
  wait_for_topic_subscriber \
    gripper_left /left/gripper/gripper_client/target_gripper_width_percent

  start_component gripper_right \
    ros2 run franka_gripper_manager franka_gripper_client \
    --ros-args -r __node:=franka_gripper_client -r __ns:=/right
  wait_for_topic_subscriber \
    gripper_right /right/gripper/gripper_client/target_gripper_width_percent

  start_component cameras \
    ros2 launch franka_realsense_camera_publisher cameras.launch.py \
    config_file:=example_three_cameras.yaml
  wait_for_topic_message cameras /cameras/cam_left/image_raw
  wait_for_topic_message cameras /cameras/cam_front/image_raw
  wait_for_topic_message cameras /cameras/cam_right/image_raw

  # No deployment config is passed: this starts the normal data-collection
  # bridge, not deployment_duo.yaml.
  start_component bridge \
    ros2 launch franka_lerobot_data_bridge bridge.launch.py \
    config_file:=example_duo.yaml
  wait_for_port bridge "${COLLECTION_ZMQ_PORT}"

  monitor_components
}

main "$@"
