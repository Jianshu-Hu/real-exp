#!/usr/bin/env bash
set -euo pipefail

dataset_root="data/test-ee-pick-and-place-final"
dataset_repo_id="local/test-ee-pick-and-place-final"
common_args=(
  --dataset-root "${dataset_root}"
  --dataset-repo-id "${dataset_repo_id}"
  --val-ratio 0.1
  --disable-wandb
  --batch-size 64
)
log_dir="outputs/train_logs"
run_id="$(date +%Y-%m-%d_%H-%M-%S)_$$"
mkdir -p "${log_dir}"

job_pids=()
job_names=()

start_training() {
  local name="$1"
  local output_dir="$2"
  shift 2
  local log_file="${log_dir}/${run_id}_${name}.log"

  printf 'Starting %s; training output: %s\n' "${name}" "${log_file}"
  python train/train_lerobot_policy.py \
    "${common_args[@]}" \
    "$@" \
    --output-dir "${output_dir}" \
    >"${log_file}" 2>&1 &
  job_pids+=("$!")
  job_names+=("${name}")
}

wait_for_phase() {
  local phase="$1"
  local index
  local phase_status=0

  for index in "${!job_pids[@]}"; do
    if wait "${job_pids[index]}"; then
      printf '%s completed.\n' "${job_names[index]}"
    else
      printf '%s failed; see %s/%s_%s.log\n' \
        "${job_names[index]}" "${log_dir}" "${run_id}" "${job_names[index]}" >&2
      phase_status=1
    fi
  done
  job_pids=()
  job_names=()

  if (( phase_status != 0 )); then
    printf '%s training phase failed; not starting the next phase.\n' "${phase}" >&2
    return 1
  fi
}

# Launch diffusion and ACT together for target-joint state/actions.
start_training \
  joint_diffusion \
  outputs/test-ee-pick-and-place-final_joint_diffusion \
  --policy-type diffusion \
  --state-action-mode joint
start_training \
  joint_act \
  outputs/test-ee-pick-and-place-final_joint_act \
  --policy-type act \
  --state-action-mode joint
wait_for_phase joint

# Start the EE jobs only after both joint-mode jobs complete successfully.
start_training \
  ee_diffusion \
  outputs/test-ee-pick-and-place-final_ee_diffusion \
  --policy-type diffusion \
  --state-action-mode end_effector
start_training \
  ee_act \
  outputs/test-ee-pick-and-place-final_ee_act \
  --policy-type act \
  --state-action-mode end_effector
wait_for_phase end_effector
