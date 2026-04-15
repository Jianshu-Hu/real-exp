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
