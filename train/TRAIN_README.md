# Training LeRobot Policies

This directory contains the repo-local training entrypoint for imitation learning with LeRobot:

- `train/train_lerobot_policy.py`
- `train/deploy_lerobot_policy.py`
- `train/franka_policy_executor.py`

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
  --diffusion-n-action-steps 8 \
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
- `--device`: force `cpu`, `cuda`, or `cuda:0`
- `--resume`: resume from an existing output directory
- `--disable-wandb`: fully disable Weights & Biases logging

ACT-specific options:

- `--act-chunk-size`
- `--act-kl-weight`

Diffusion-specific options:

- `--diffusion-horizon`
- `--diffusion-n-obs-steps`
- `--diffusion-n-action-steps`

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

## Deployment Helper

The deployment helper currently supports two jobs:

- `inspect`: validate a trained checkpoint against the dataset contract
- `server`: start a LeRobot async inference server

Inspect a checkpoint:

```bash
python train/deploy_lerobot_policy.py \
  inspect \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model
```

Start a policy server:

```bash
python train/deploy_lerobot_policy.py \
  server \
  --host 0.0.0.0 \
  --port 8080 \
  --fps 15
```

## Recommended Deployment Topology

For Franka, the safer design is:

- Run the policy model on a server or GPU workstation
- Run the real-time executor on the computer connected to the Franka robot

This split is usually better because:

- The robot computer can stay focused on low-latency IO, safety checks, and command execution
- The server can use a larger GPU and heavier policy models
- You can restart inference without directly disturbing the robot-side process

## What Runs Where

Policy server machine:

- Load the trained LeRobot checkpoint
- Receive observations over the network
- Run policy inference
- Return action chunks

Robot-connected machine:

- Read robot joint state, gripper state, and camera frames
- Build observations in the same format used during training
- Send observations to the policy server
- Receive action chunks
- Convert policy actions into Franka-safe commands
- Enforce velocity limits, interpolation, watchdog timeout, and abort handling locally

## Franka Executor

`franka_policy_executor.py` is the robot-side process for the Franka computer.

It currently:

- Subscribes to the live ZMQ observation stream
- Sends observations to the remote LeRobot policy server
- Receives action chunks back
- Interprets the action layout using the training dataset contract
- Runs in dry-run mode by default
- Can send arm velocity commands to the Franka only when `--execute` is explicitly passed

Dry run example:

```bash
python train/franka_policy_executor.py \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model \
  --server-address 192.168.1.10:8080 \
  --zmq-host 127.0.0.1 \
  --zmq-port 5555 \
  --fps 15
```

Live execution example:

```bash
python train/franka_policy_executor.py \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model \
  --server-address 192.168.1.10:8080 \
  --zmq-host 127.0.0.1 \
  --zmq-port 5555 \
  --fps 15 \
  --execute
```

If your dataset includes gripper actions and you want to apply them:

```bash
python train/franka_policy_executor.py \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model \
  --server-address 192.168.1.10:8080 \
  --execute \
  --enable-gripper
```

## Recommended Bring-Up Order

1. Start the robot/camera observation bridge that publishes the ZMQ packets.
2. Start the policy server on the inference machine.
3. Start `franka_policy_executor.py` in dry-run mode first.
4. Verify that observations are flowing and action predictions look sensible.
5. Only then restart with `--execute`.

## Franka-Specific Note

Your dataset action representation is delta joint position for the arms, with optional gripper commands. That means the robot-side executor should not directly stream raw policy outputs into the robot without a local control layer.

The robot-side executor should interpret actions using the same structure as the dataset:

- 16-dim: `[Left Arm(7), Left Gripper(1), Right Arm(7), Right Gripper(1)]`
- 14-dim: `[Left Arm(7), Right Arm(7)]`

In practice, the executor should:

- Convert delta joint actions into target velocities or target joint positions
- Clamp commands to conservative Franka limits
- Smooth/interpolate commands between network updates
- Stop safely if server responses are delayed or missing
