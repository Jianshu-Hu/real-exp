# Deploying a Trained Policy

The server computer has the cameras and runs policy inference. The robot
computer runs the Franka controllers and the policy executor.

## Network Flow

Both computers use ROS 2 domain `0`. The default server address is
`192.168.50.13`.

| Traffic | Route | Default |
| --- | --- | --- |
| Robot/gripper state and deployment commands | ROS 2 DDS between computers | `ROS_DOMAIN_ID=0` |
| Complete observations | server bridge to robot executor | ZMQ `192.168.50.13:5555` |
| Policy commands | robot executor to server bridge | ZMQ `192.168.50.13:5556` |
| Policy inference | robot executor to policy server | gRPC `192.168.50.13:8080` |

The startup scripts and executors use these defaults. Set
`DEPLOYMENT_SERVER_IP` or pass the corresponding CLI options only if the server
address changes.

## Run Deployment

### 1. Start the robot client

On the robot computer:

```bash
cd /home/pair1/real-exp
./scripts/start_deployment_client.sh
```

This starts the dual-arm deployment controllers, joint-state broadcasters, and
Franka-hand managers. The controllers initially hold the current pose.

### 2. Start the deployment server

On the server computer:

```bash
cd /home/pair1/real-exp
conda activate lerobot
./scripts/start_deployment_server.sh
```

This starts the three RealSense cameras, the server-side deployment bridge,
and the policy server.

### 3. Start an executor in dry-run mode

In a second shell on the robot computer:

```bash
cd /home/pair1/real-exp
conda activate lerobot
source /opt/ros/humble/setup.bash
source gello_software/ros2/install/setup.bash

python deploy/franka_act_policy_executor.py \
  --policy-path /home/pair1/real-exp/outputs/test-limit-pick-and-place_act/checkpoints/last/pretrained_model \
  --dataset-root data/test-limit-pick-and-place \
  --actions-per-chunk 32 \
  --policy-device cuda:0 \
  --fps 15 \
  --task "pick and place"
```

Do not add `--execute` yet. Confirm that the executor reports:

```text
state_dim=16, action_dim=16
cameras=['cam_left', 'cam_front', 'cam_right']
execute: False
```

Also confirm that predictions are finite and stable, inference is continuous,
and there are no camera, dimension, CUDA, ZMQ, or gRPC errors.

Stop the dry run with `Ctrl-C`. After clearing the robot workspace and checking
the emergency stop, repeat the same executor command with:

```text
--execute
```

That is the only step that enables real policy commands. Run only one executor
at a time.

## Diffusion Alternative

Use this instead of ACT, first without `--execute`:

```bash
python deploy/franka_diffusion_policy_executor.py \
  --policy-path /home/pair1/real-exp/outputs/test-limit-pick-and-place_diffusion/checkpoints/last/pretrained_model \
  --dataset-root data/test-limit-pick-and-place \
  --actions-per-chunk 8 \
  --policy-device cuda:0 \
  --fps 15 \
  --task "pick and place" \
  --diffusion-chunk-size-threshold 0.5 \
  --diffusion-aggregate-ratio-old 0.5
```

Diffusion requires two observation frames before its first inference. Add
`--execute` only after the dry run passes the same validation.

## Stop Deployment

Stop processes in this order:

1. Stop the robot-side executor.
2. Stop `start_deployment_server.sh`.
3. Stop `start_deployment_client.sh` last.

The bridge and deployment controller hold position on stale commands or clean
shutdown. If the executor cannot start, first check that robot state reaches
the server over ROS 2 and that server ports `5555`, `5556`, and `8080` are
reachable from the robot computer.
