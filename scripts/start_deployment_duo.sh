#!/usr/bin/env bash
#
# Start and supervise the four ROS 2 groups required for dual-arm policy
# deployment from one terminal:
#   1. Franka hardware, state publishers, and inactive deployment controllers
#   2. Left and right Franka gripper clients
#   3. Three RealSense camera publishers
#   4. LeRobot deployment bridge in STANDBY mode
#
# The remote policy server and the ACT/Diffusion executor remain separate.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
FRANKA_WORKSPACE_SETUP="${FRANKA_WORKSPACE_SETUP:-${HOME}/franka_ros2_ws/install/setup.bash}"
WORKSPACE_SETUP="${WORKSPACE_SETUP:-${REPO_ROOT}/gello_software/ros2/install/setup.bash}"
READY_TIMEOUT="${READY_TIMEOUT:-90}"
DEPLOYMENT_OBSERVATION_PORT="${DEPLOYMENT_OBSERVATION_PORT:-5555}"
DEPLOYMENT_COMMAND_PORT="${DEPLOYMENT_COMMAND_PORT:-5556}"
ARM_CONFIG="${DEPLOYMENT_ARM_CONFIG:-example_fr3_duo_config.yaml}"
GRIPPER_CONFIG="${DEPLOYMENT_GRIPPER_CONFIG:-example_fr3_duo_config_franka_hand.yaml}"
CAMERA_CONFIG="${DEPLOYMENT_CAMERA_CONFIG:-example_three_cameras.yaml}"
BRIDGE_CONFIG="${DEPLOYMENT_BRIDGE_CONFIG:-deployment_duo.yaml}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${DEPLOYMENT_LOG_DIR:-${REPO_ROOT}/log/deployment_stack/${RUN_ID}}"
LOCK_FILE="${DEPLOYMENT_LOCK_FILE:-/tmp/real-exp-deployment-duo.lock}"

declare -a COMPONENT_NAMES=()
declare -a COMPONENT_PIDS=()
SHUTTING_DOWN=0
PREFLIGHT_ONLY=0

log() {
  printf '[deployment] %s\n' "$*"
}

fail() {
  log "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/start_deployment_duo.sh
  bash scripts/start_deployment_duo.sh --preflight-only

Options:
  --preflight-only   Validate the ROS overlay, deployment YAML contract,
                     ports, and absence of conflicting ROS stacks, then exit
                     without connecting to either Franka.
  -h, --help         Show this help and exit.

Environment overrides:
  READY_TIMEOUT, DEPLOYMENT_LOG_DIR
  ROS_SETUP, FRANKA_WORKSPACE_SETUP, WORKSPACE_SETUP
  DEPLOYMENT_OBSERVATION_PORT, DEPLOYMENT_COMMAND_PORT
  DEPLOYMENT_ARM_CONFIG, DEPLOYMENT_GRIPPER_CONFIG
  DEPLOYMENT_CAMERA_CONFIG, DEPLOYMENT_BRIDGE_CONFIG
EOF
}

parse_args() {
  while (( $# > 0 )); do
    case "$1" in
      --preflight-only)
        PREFLIGHT_ONLY=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        fail "unknown option: $1"
        ;;
    esac
    shift
  done
}

require_file() {
  local path="$1"
  local description="$2"
  [[ -f "${path}" ]] || fail "${description} not found: ${path}"
}

require_command() {
  local command_name="$1"
  command -v "${command_name}" >/dev/null 2>&1 \
    || fail "required command is unavailable: ${command_name}"
}

source_ros_environment() {
  require_file "${ROS_SETUP}" "ROS 2 setup file"
  require_file "${FRANKA_WORKSPACE_SETUP}" "Franka workspace setup file"
  require_file "${WORKSPACE_SETUP}" "real-exp ROS workspace setup file"

  set +u
  # shellcheck disable=SC1090
  source "${ROS_SETUP}"
  # shellcheck disable=SC1090
  source "${FRANKA_WORKSPACE_SETUP}"
  # Source this overlay last so deployment packages resolve to this checkout.
  # shellcheck disable=SC1090
  source "${WORKSPACE_SETUP}"
  set -u

  require_command ros2
}

package_config_path() {
  local package_name="$1"
  local config_name="$2"
  local prefix
  local expected_install_root
  local resolved_prefix
  local config_path

  prefix="$(ros2 pkg prefix "${package_name}" 2>/dev/null)" \
    || fail "ROS package is unavailable after sourcing the overlay: ${package_name}"
  expected_install_root="$(realpath -m "$(dirname -- "${WORKSPACE_SETUP}")")"
  resolved_prefix="$(realpath -m "${prefix}")"
  case "${resolved_prefix}/" in
    "${expected_install_root}/"*) ;;
    *)
      fail "${package_name} resolves to ${resolved_prefix}, not the expected real-exp overlay ${expected_install_root}. Rebuild and source ${WORKSPACE_SETUP}."
      ;;
  esac

  config_path="${prefix}/share/${package_name}/config/${config_name}"
  require_file "${config_path}" "installed ${package_name} config"
  printf '%s\n' "${config_path}"
}

validate_installed_configs() {
  local arm_config_path
  local gripper_config_path
  local camera_config_path
  local bridge_config_path

  arm_config_path="$(package_config_path franka_fr3_arm_controllers "${ARM_CONFIG}")"
  gripper_config_path="$(package_config_path franka_gripper_manager "${GRIPPER_CONFIG}")"
  camera_config_path="$(package_config_path franka_realsense_camera_publisher "${CAMERA_CONFIG}")"
  bridge_config_path="$(package_config_path franka_lerobot_data_bridge "${BRIDGE_CONFIG}")"

  if ! python3 - \
    "${arm_config_path}" \
    "${gripper_config_path}" \
    "${camera_config_path}" \
    "${bridge_config_path}" \
    "${DEPLOYMENT_OBSERVATION_PORT}" \
    "${DEPLOYMENT_COMMAND_PORT}" <<'PY'
from pathlib import Path
import sys

import yaml


def load(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


arm_path, gripper_path, camera_path, bridge_path, observation_port, command_port = sys.argv[1:]
arm = load(arm_path)
gripper = load(gripper_path)
camera = load(camera_path)
bridge = load(bridge_path)

arm_namespaces = {str(value.get("namespace")) for value in arm.values() if isinstance(value, dict)}
if arm_namespaces != {"left", "right"}:
    raise ValueError(f"arm config must contain left/right namespaces, got {sorted(arm_namespaces)}")
for name, value in arm.items():
    if not isinstance(value, dict):
        raise ValueError(f"arm entry {name!r} is not a mapping")
    if truthy(value.get("use_fake_hardware", False)):
        raise ValueError(f"arm entry {name!r} unexpectedly enables fake hardware")
    if not truthy(value.get("load_gripper", False)):
        raise ValueError(f"arm entry {name!r} must load the Franka gripper state/action server")

gripper_namespaces = {
    str(value.get("namespace")) for value in gripper.values() if isinstance(value, dict)
}
if gripper_namespaces != {"left", "right"}:
    raise ValueError(
        f"gripper config must contain left/right namespaces, got {sorted(gripper_namespaces)}"
    )

camera_params = camera.get("realsense_camera_publisher", {}).get("ros__parameters", {})
camera_topics = {
    str(camera_params.get(f"camera_{index}_topic"))
    for index in range(1, 4)
    if truthy(camera_params.get(f"camera_{index}_enabled", True))
}
expected_camera_topics = {
    "/cameras/cam_left/image_raw",
    "/cameras/cam_front/image_raw",
    "/cameras/cam_right/image_raw",
}
if camera_topics != expected_camera_topics:
    raise ValueError(
        f"camera config topics must be {sorted(expected_camera_topics)}, got {sorted(camera_topics)}"
    )

bridge_params = bridge.get("lerobot_data_bridge", {}).get("ros__parameters", {})
expected_bridge_values = {
    "deployment_mode": True,
    "deployment_start_active": False,
    "deployment_state_source": "topics",
    "include_right_arm": True,
    "include_gripper": True,
    "left_robot_joint_state_topic": "/left/franka/joint_states",
    "right_robot_joint_state_topic": "/right/franka/joint_states",
    "left_deployment_joint_command_topic": "/left/deployment/joint_states",
    "right_deployment_joint_command_topic": "/right/deployment/joint_states",
    "left_deployment_enable_service": (
        "/left/deployment_joint_impedance_controller/set_deployment_enabled"
    ),
    "right_deployment_enable_service": (
        "/right/deployment_joint_impedance_controller/set_deployment_enabled"
    ),
}
for key, expected in expected_bridge_values.items():
    actual = bridge_params.get(key)
    if isinstance(expected, bool):
        actual = truthy(actual)
    if actual != expected:
        raise ValueError(f"bridge config {key} must be {expected!r}, got {actual!r}")

if int(bridge_params.get("publish_port", -1)) != int(observation_port):
    raise ValueError("bridge publish_port does not match DEPLOYMENT_OBSERVATION_PORT")
if int(bridge_params.get("command_port", -1)) != int(command_port):
    raise ValueError("bridge command_port does not match DEPLOYMENT_COMMAND_PORT")

bridge_camera_topics = {
    str(bridge_params.get(f"camera_{index}_topic"))
    for index in range(1, 4)
    if truthy(bridge_params.get(f"camera_{index}_enabled", True))
}
if bridge_camera_topics != expected_camera_topics:
    raise ValueError(
        "deployment bridge camera topics do not match the three-camera publisher config"
    )

print("Validated installed dual-arm deployment YAML contract.")
PY
  then
    fail "installed deployment configuration validation failed"
  fi
}

port_is_listening() {
  local port="$1"
  ss -ltnH | awk -v suffix=":${port}" '$4 ~ suffix "$" { found=1 } END { exit !found }'
}

check_port_available() {
  local port="$1"
  local description="$2"
  if port_is_listening "${port}"; then
    fail "TCP port ${port} (${description}) is already in use. Stop the old bridge before deployment."
  fi
}

service_exists() {
  local service_name="$1"
  timeout 5 ros2 service list 2>/dev/null | grep -Fxq "${service_name}"
}

topic_has_publisher() {
  local topic="$1"
  timeout 5 ros2 topic info "${topic}" 2>/dev/null \
    | grep -Eq '^Publisher count: [1-9][0-9]*$'
}

check_no_conflicting_ros_stack() {
  local service_name
  local topic

  for service_name in \
    /left/controller_manager/list_controllers \
    /right/controller_manager/list_controllers \
    /set_deployment_active; do
    if service_exists "${service_name}"; then
      fail "ROS service ${service_name} already exists. Another arm or bridge stack is running."
    fi
  done

  for topic in /left/gello/joint_states /right/gello/joint_states; do
    if topic_has_publisher "${topic}"; then
      fail "${topic} still has a publisher. Stop the GELLO/data-collection stack before deployment."
    fi
  done
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
  kill -0 "${pid}" 2>/dev/null \
    || fail "${name} exited; inspect ${LOG_DIR}/${name}.log"
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

wait_for_service() {
  local component="$1"
  local service_name="$2"
  local deadline=$((SECONDS + READY_TIMEOUT))

  log "Waiting for service ${service_name}"
  while (( SECONDS < deadline )); do
    assert_component_alive "${component}"
    if service_exists "${service_name}"; then
      log "Ready: ${service_name}"
      return 0
    fi
    sleep 1
  done
  fail "timed out after ${READY_TIMEOUT}s waiting for ${service_name}"
}

wait_for_topic_message() {
  local component="$1"
  local topic="$2"
  local deadline=$((SECONDS + READY_TIMEOUT))

  log "Waiting for a fresh message on ${topic}"
  while (( SECONDS < deadline )); do
    assert_component_alive "${component}"
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
    if timeout 5 ros2 topic info "${topic}" 2>/dev/null \
      | grep -Eq '^Subscription count: [1-9][0-9]*$'; then
      log "Ready: ${component} subscribed to ${topic}"
      return 0
    fi
    sleep 1
  done
  fail "${component} did not initialize within ${READY_TIMEOUT}s; inspect ${LOG_DIR}/${component}.log"
}

wait_for_topic_publisher() {
  local component="$1"
  local topic="$2"
  local deadline=$((SECONDS + READY_TIMEOUT))

  log "Waiting for a publisher on ${topic}"
  while (( SECONDS < deadline )); do
    assert_component_alive "${component}"
    if topic_has_publisher "${topic}"; then
      log "Ready: publisher on ${topic}"
      return 0
    fi
    sleep 1
  done
  fail "timed out after ${READY_TIMEOUT}s waiting for a publisher on ${topic}"
}

controller_has_state() {
  local manager="$1"
  local controller="$2"
  local expected_state="$3"
  timeout 5 ros2 control list_controllers -c "${manager}" 2>/dev/null \
    | awk -v controller="${controller}" -v state="${expected_state}" \
      '$1 == controller && $NF == state { found=1 } END { exit !found }'
}

wait_for_controller_state() {
  local component="$1"
  local manager="$2"
  local controller="$3"
  local expected_state="$4"
  local deadline=$((SECONDS + READY_TIMEOUT))

  log "Waiting for ${manager}/${controller} to be ${expected_state}"
  while (( SECONDS < deadline )); do
    assert_component_alive "${component}"
    if controller_has_state "${manager}" "${controller}" "${expected_state}"; then
      log "Ready: ${manager}/${controller} is ${expected_state}"
      return 0
    fi
    sleep 1
  done
  fail "${manager}/${controller} did not become ${expected_state} within ${READY_TIMEOUT}s"
}

wait_for_parameter_value() {
  local component="$1"
  local node="$2"
  local parameter="$3"
  local expected_line="$4"
  local deadline=$((SECONDS + READY_TIMEOUT))
  local output

  log "Checking ${node} parameter ${parameter}"
  while (( SECONDS < deadline )); do
    assert_component_alive "${component}"
    output="$(timeout 5 ros2 param get "${node}" "${parameter}" 2>/dev/null || true)"
    if grep -Fqx "${expected_line}" <<<"${output}"; then
      log "Ready: ${parameter} matches deployment contract"
      return 0
    fi
    sleep 1
  done
  fail "${node} parameter ${parameter} did not equal ${expected_line@Q}"
}

wait_for_port() {
  local component="$1"
  local port="$2"
  local description="$3"
  local deadline=$((SECONDS + READY_TIMEOUT))

  log "Waiting for TCP port ${port} (${description})"
  while (( SECONDS < deadline )); do
    assert_component_alive "${component}"
    if port_is_listening "${port}"; then
      log "Ready: TCP port ${port} (${description})"
      return 0
    fi
    sleep 1
  done
  fail "timed out after ${READY_TIMEOUT}s waiting for TCP port ${port}"
}

bridge_was_started() {
  component_pid bridge >/dev/null 2>&1
}

request_bridge_standby() {
  local deadline

  bridge_was_started || return 0
  service_exists /set_deployment_active || return 0
  log "Requesting deployment bridge STANDBY before shutdown"
  if ! timeout 8 ros2 service call \
    /set_deployment_active \
    std_srvs/srv/SetBool \
    "{data: false}" >/dev/null 2>&1; then
    log "WARN: bridge did not acknowledge the STANDBY request"
    return 0
  fi

  deadline=$((SECONDS + 8))
  while (( SECONDS < deadline )); do
    if controller_has_state \
      /left/controller_manager \
      deployment_joint_impedance_controller \
      inactive \
      && controller_has_state \
        /right/controller_manager \
        deployment_joint_impedance_controller \
        inactive; then
      log "Deployment controllers are inactive"
      return 0
    fi
    sleep 1
  done
  log "WARN: deployment controllers did not report inactive before shutdown"
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

  request_bridge_standby || true
  log "Stopping deployment ROS 2 components in reverse order"
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
  trap - EXIT
  trap '' INT TERM HUP
  stop_components
  exit "${status}"
}

check_runtime_endpoint() {
  local description="$1"
  shift
  if "$@"; then
    return 0
  fi
  log "WARN: runtime endpoint is unavailable: ${description}"
  return 1
}

runtime_endpoints_are_healthy() {
  local failed=0

  check_runtime_endpoint "left controller manager service" \
    service_exists /left/controller_manager/list_controllers || failed=1
  check_runtime_endpoint "right controller manager service" \
    service_exists /right/controller_manager/list_controllers || failed=1
  check_runtime_endpoint "bridge activation service" \
    service_exists /set_deployment_active || failed=1
  check_runtime_endpoint "observation port ${DEPLOYMENT_OBSERVATION_PORT}" \
    port_is_listening "${DEPLOYMENT_OBSERVATION_PORT}" || failed=1
  check_runtime_endpoint "command port ${DEPLOYMENT_COMMAND_PORT}" \
    port_is_listening "${DEPLOYMENT_COMMAND_PORT}" || failed=1
  check_runtime_endpoint "left arm state publisher" \
    topic_has_publisher /left/franka/joint_states || failed=1
  check_runtime_endpoint "right arm state publisher" \
    topic_has_publisher /right/franka/joint_states || failed=1
  check_runtime_endpoint "left camera publisher" \
    topic_has_publisher /cameras/cam_left/image_raw || failed=1
  check_runtime_endpoint "front camera publisher" \
    topic_has_publisher /cameras/cam_front/image_raw || failed=1
  check_runtime_endpoint "right camera publisher" \
    topic_has_publisher /cameras/cam_right/image_raw || failed=1
  return "${failed}"
}

monitor_components() {
  local index
  local pid
  local next_endpoint_check=$SECONDS
  local endpoint_failure_count=0

  log "All four deployment ROS groups are ready in STANDBY mode"
  log "Logs: ${LOG_DIR}"
  log "Do not start a GELLO publisher during deployment."
  log "Start the ACT/Diffusion executor in a separate Conda terminal."
  log "Press Ctrl+C here to return the bridge to STANDBY and stop the complete stack."

  while true; do
    for index in "${!COMPONENT_PIDS[@]}"; do
      pid="${COMPONENT_PIDS[index]}"
      if ! kill -0 "${pid}" 2>/dev/null; then
        fail "${COMPONENT_NAMES[index]} exited unexpectedly; inspect ${LOG_DIR}/${COMPONENT_NAMES[index]}.log"
      fi
    done

    if (( SECONDS >= next_endpoint_check )); then
      if runtime_endpoints_are_healthy; then
        endpoint_failure_count=0
      else
        endpoint_failure_count=$((endpoint_failure_count + 1))
        log "WARN: runtime health-check failure ${endpoint_failure_count}/3"
        if (( endpoint_failure_count >= 3 )); then
          fail "runtime endpoints failed three consecutive health checks"
        fi
      fi
      next_endpoint_check=$((SECONDS + 5))
    fi
    sleep 1
  done
}

run_preflight() {
  check_port_available "${DEPLOYMENT_OBSERVATION_PORT}" "observation PUB"
  check_port_available "${DEPLOYMENT_COMMAND_PORT}" "command PULL"
  check_no_conflicting_ros_stack
  validate_installed_configs
}

main() {
  parse_args "$@"

  for required_command in \
    flock stdbuf sed tee awk timeout setsid ps tr grep ss realpath python3; do
    require_command "${required_command}"
  done

  exec 9>"${LOCK_FILE}"
  flock -n 9 || fail "another dual-arm deployment stack is already running"

  source_ros_environment
  export RCUTILS_COLORIZED_OUTPUT=0
  run_preflight

  if (( PREFLIGHT_ONLY == 1 )); then
    log "Preflight passed. No Franka connection or ROS deployment process was started."
    return 0
  fi

  mkdir -p "${LOG_DIR}"
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    log "WARN: a Conda environment is active (${CONDA_PREFIX}). ROS nodes should normally use the system ROS environment."
  fi

  trap on_exit EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP

  # Group 1: hardware, robot/joint state publishers, and deployment controllers.
  # deployment_mode=true loads the policy controller inactive; the bridge later
  # activates it only after the executor enables deployment and sends a command.
  start_component arms \
    ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
    robot_config_file:="${ARM_CONFIG}" \
    deployment_mode:=true
  wait_for_service arms /left/controller_manager/list_controllers
  wait_for_service arms /right/controller_manager/list_controllers
  wait_for_service arms /left/controller_manager/switch_controller
  wait_for_service arms /right/controller_manager/switch_controller
  wait_for_topic_message arms /left/franka/joint_states
  wait_for_topic_message arms /right/franka/joint_states
  wait_for_topic_message arms /left/franka_gripper/joint_states
  wait_for_topic_message arms /right/franka_gripper/joint_states
  wait_for_controller_state \
    arms /left/controller_manager deployment_joint_impedance_controller inactive
  wait_for_controller_state \
    arms /right/controller_manager deployment_joint_impedance_controller inactive

  # Group 2: initialize one gripper client at a time. The duo YAML contains the
  # same left/right namespaces, but sequential homing avoids concurrent action
  # startup timeouts observed on this machine.
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

  # Group 3: deployment and collection use the same physical three-camera YAML.
  start_component cameras \
    ros2 launch franka_realsense_camera_publisher cameras.launch.py \
    config_file:="${CAMERA_CONFIG}"
  wait_for_topic_message cameras /cameras/cam_left/image_raw
  wait_for_topic_message cameras /cameras/cam_front/image_raw
  wait_for_topic_message cameras /cameras/cam_right/image_raw

  # Group 4: unlike collection, deployment must use deployment_duo.yaml. It
  # receives executor commands on 5556 and publishes observations on 5555.
  start_component bridge \
    ros2 launch franka_lerobot_data_bridge bridge.launch.py \
    config_file:="${BRIDGE_CONFIG}"
  wait_for_service bridge /set_deployment_active
  wait_for_port bridge "${DEPLOYMENT_OBSERVATION_PORT}" "observation PUB"
  wait_for_port bridge "${DEPLOYMENT_COMMAND_PORT}" "command PULL"
  wait_for_parameter_value \
    bridge /lerobot_data_bridge deployment_mode "Boolean value is: True"
  wait_for_parameter_value \
    bridge /lerobot_data_bridge deployment_start_active "Boolean value is: False"
  wait_for_parameter_value \
    bridge /lerobot_data_bridge deployment_state_source "String value is: topics"
  wait_for_topic_publisher bridge /left/deployment/joint_states
  wait_for_topic_publisher bridge /right/deployment/joint_states
  wait_for_controller_state \
    arms /left/controller_manager deployment_joint_impedance_controller inactive
  wait_for_controller_state \
    arms /right/controller_manager deployment_joint_impedance_controller inactive

  monitor_components
}

main "$@"
