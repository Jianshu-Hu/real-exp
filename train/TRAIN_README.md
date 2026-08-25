# Training LeRobot Policies

This directory contains the repo-local training entrypoint for imitation learning with LeRobot:

- `train/train_lerobot_policy.py`
- `train/push_lerobot_policy.py`

Policy inspection, fetching, serving, and robot-side execution are documented in
[`deploy/DEPLOY_README.md`](../deploy/DEPLOY_README.md).

## Environment

```bash
conda activate lerobot
```

## What The Script Does

`train_lerobot_policy.py` wraps LeRobot's training API directly instead of requiring a long `lerobot-train` command.

It currently:

- Uses the local dataset at `data/pick_and_place_test` by default
- Redirects Hugging Face cache writes into `./.hf-cache`
- Supports `act` and `diffusion` policy types
- Writes each run's checkpoints and durable local JSONL logs into a timestamped directory under `./outputs/`
- Saves at most 10 model checkpoints per run by default
- Can split episodes into training and validation subsets
- Can run periodic validation loss evaluation with `--val-freq`
- Disables online evaluation during training, which fits this real-data workflow

## Default Dataset

By default the script trains on:

```text
<repo-root>/data/pick_and_place_test
```

with dataset repo id:

```text
local/pick_and_place_test
```

You can override both with CLI flags if needed.

The training entry point requires `meta/real_exp_action_config.json` with
`arm_action_representation=absolute_joint_position` for joint-mode datasets, or
`delta_end_effector_pose` for end-effector-mode datasets. Current collection also
uses `gripper_action_representation=absolute_width`, which preserves continuous
normalized gripper targets in `[0, 1]`.

It also requires `meta/real_exp_trajectory_config.json`. This metadata is the
authoritative vector layout: it records `arm_mode` (`left`, `right`, or `duo`),
the end-effector type (`arm`, `gripper`, or `hand`), active arms, and the state
and action dimensions. The trainer validates it against `meta/info.json` and
the action metadata before creating a policy. A dataset containing the neutral
joint and EE fields can be trained in either mode with `--state-action-mode joint` or
`--state-action-mode end_effector`; the trainer selects the corresponding
primary fields (`observation.state`/`action` or `observation.ee_pose`/
`action.delta_ee_pose`; joint mode uses the neutral
`observation.joint_state`/`action.target_joint` fields) without rewriting the
parquet data. Every saved `pretrained_model`
also receives the action, trajectory, and complete dataset feature metadata plus
a validated deployment manifest. A checkpoint therefore remains self-describing
when moved to another machine. Deployment reads this embedded metadata from the
server-owned checkpoint; the robot client needs neither dataset nor checkpoint files.

## Recommended First Run

Start with `ACT`, because it works cleanly with the three-camera dataset in this repo.

```bash
python train/train_lerobot_policy.py \
  --policy-type act \
  --steps 50000 \
  --batch-size 8 \
  --disable-wandb
```

## Diffusion Example

```bash
python train/train_lerobot_policy.py \
  --policy-type diffusion \
  --steps 50000 \
  --batch-size 8 \
  --diffusion-horizon 16 \
  --diffusion-n-obs-steps 2 \
  --diffusion-noise-scheduler-type DDIM \
  --diffusion-num-inference-steps 10 \
  --disable-wandb
```

## Useful Flags

```bash
python train/train_lerobot_policy.py --help
```

Important options:

- `--dataset-root`: override the local dataset path
- `--dataset-repo-id`: override the LeRobot dataset repo id
- `--policy-type {act,diffusion}`: choose the imitation-learning policy
- `--output-dir`: choose the parent directory for a new timestamped run; with `--resume`, specify the exact existing run directory
- `--steps`: total number of optimizer steps
- `--batch-size`: training batch size
- `--num-workers`: dataloader worker count
- `--save-freq`: optionally request a checkpoint interval; when omitted, the default is derived to target 10 checkpoints for a standard run
- `--device`: force `cpu`, `cuda`, or `cuda:0`
- `--resume`: resume from an existing output directory
- `--disable-wandb`: fully disable Weights & Biases logging
- `--val-ratio`: reserve a fraction of episodes for validation
- `--val-freq`: run validation every N training steps, defaulting to `--save-freq`
- `--val-batch-size`: validation batch size, defaulting to `--batch-size`
- `--max-val-batches`: optionally cap validation batches per evaluation pass

## Train/Validation Split

The script can split the dataset into a training subset and a validation subset. This is only for validation loss reporting during training; it does not early stop.

The split is controlled by `--val-ratio`. The script randomly assigns episodes into train and validation sets using `--seed`, so the split is reproducible.

Example:

```bash
python train/train_lerobot_policy.py \
  --policy-type act \
  --val-ratio 0.2 \
  --val-freq 500 \
  --steps 50000 \
  --batch-size 8 \
  --disable-wandb
```

For `data/pick_and_place_test`, which currently has 10 episodes, `--val-ratio 0.2` will place 2 episodes into validation and 8 episodes into training. The exact episode ids depend on `--seed`.

## Validation Evaluation

Validation runs only when both conditions are true:

- A validation split exists
- The effective validation frequency is greater than 0

If you do not pass `--val-freq`, the script uses `--save-freq`. Passing `--val-freq 0` disables validation entirely.

At each validation step, the script computes validation loss on the held-out episodes and prints it to stdout. If wandb is enabled, it also logs `val_loss`.

## Local Training Logs

Training metrics are saved even when Weights & Biases is disabled. Each new run
gets a local-time timestamp such as `2026-08-19_14-05-07`. For example, the
files for an ACT run are:

```text
outputs/test-limit-pick-and-place_act/
└── 2026-08-19_14-05-07/
    ├── checkpoints/
    └── logs/
        ├── train_metrics.jsonl
        └── epoch_metrics.jsonl
```

Passing `--output-dir outputs/experiments` for a new run produces
`outputs/experiments/<timestamp>/`. The script prints the concrete run output
directory at startup.

`train_metrics.jsonl` stores the rolling metrics at every `--log-freq` interval,
including loss, learning rate, timing, elapsed time, and ETA. `epoch_metrics.jsonl`
stores the sample-weighted training loss for each dataset-sized epoch. Validation
events are appended to both files with `"record_type": "validation"` and a
`val_loss` field. Training rows use `"record_type": "train"`. The final epoch
record can have `"complete": false` when training stops partway through an epoch.
Both files are newline-delimited JSON and are appended when a run is resumed.

When `--save-freq` is omitted, the checkpoint interval is computed as
`ceil(steps / 10)`. The final step is always saved, producing exactly 10 model
checkpoint directories for the standard 50,000-step run. An explicit
`--save-freq` is honored as provided.

ACT-specific options:

- `--act-chunk-size`
- `--act-kl-weight`

Diffusion-specific options:

- `--diffusion-horizon`
- `--diffusion-n-obs-steps`
- `--diffusion-noise-scheduler-type`
- `--diffusion-num-inference-steps`

Default diffusion scheduler/inference settings in this repo:

- `noise_scheduler_type = DDIM`
- `num_inference_steps = 10`

## Notes About Policy Choice

- `act` is the safest default here because your dataset contains three image streams:
  `observation.images.cam_left`, `observation.images.cam_front`, and `observation.images.cam_right`.
- `diffusion` is also supported by the wrapper.
- `vqbet` is not exposed in this script because the installed LeRobot config expects exactly one image input, while this dataset has three cameras.

## Resume Training

If you want to continue a previous run:

```bash
python train/train_lerobot_policy.py \
  --policy-type act \
  --output-dir outputs/pick_and_place_test_act/2026-08-19_14-05-07 \
  --resume
```

Use the exact timestamped output directory of the prior run. Resume never adds
a second timestamp directory.

## Policy Hub Helpers

Push a saved local policy to Hugging Face:

```bash
python train/push_lerobot_policy.py \
  --policy-path outputs/pick_and_place_test_act/2026-08-19_14-05-07/checkpoints/last \
  --repo-id Jianshu1/pick_and_place_test_act
```

Fetch a policy from Hugging Face:

```bash
python deploy/fetch_lerobot_policy.py \
  --repo-id Jianshu1/pick_and_place_test_act
```

By default:

- `push_lerobot_policy.py` pushes to remote branch `main`
- `deploy/fetch_lerobot_policy.py` fetches from remote branch `main`
- `deploy/fetch_lerobot_policy.py` replaces `outputs/fetched_policies/<repo-name>` so the local copy matches the remote policy

Use `--branch`, `--revision`, or `--no-clean` only when you intentionally want non-default behavior.
