# Training LeRobot Policies

This directory contains the repo-local training entrypoint for imitation learning with LeRobot:

- `train/train_lerobot_policy.py`
- `train/deploy_lerobot_policy.py`
- `train/franka_policy_executor.py`
- `train/push_lerobot_policy.py`
- `train/fetch_lerobot_policy.py`

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
- Writes checkpoints and logs into `./outputs/`
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
- `--output-dir`: choose a custom checkpoint/log directory
- `--steps`: total number of optimizer steps
- `--batch-size`: training batch size
- `--num-workers`: dataloader worker count
- `--save-freq`: save a checkpoint every N optimizer steps
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
  --output-dir outputs/pick_and_place_test_act \
  --resume
```

Use the same output directory as the prior run.

## Policy Hub Helpers

Push a saved local policy to Hugging Face:

```bash
python train/push_lerobot_policy.py \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last \
  --repo-id Jianshu1/pick_and_place_test_act
```

Fetch a policy from Hugging Face:

```bash
python train/fetch_lerobot_policy.py \
  --repo-id Jianshu1/pick_and_place_test_act
```

By default:

- `push_lerobot_policy.py` pushes to remote branch `main`
- `fetch_lerobot_policy.py` fetches from remote branch `main`
- `fetch_lerobot_policy.py` replaces `outputs/fetched_policies/<repo-name>` so the local copy matches the remote policy

Use `--branch`, `--revision`, or `--no-clean` only when you intentionally want non-default behavior.
