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

The `server` command in `train/deploy_lerobot_policy.py` includes the deployment-specific runtime fixes locally, so you do not need to edit the `lerobot` submodule to run the Franka deployment flow.

Inspecting the checkpoint before deployment is recommended because it prints:

- the policy type
- the recommended `actions_per_chunk`
- the dataset state/action dimensions
- the expected camera keys
- the recorded action representation

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

ROS 2 bridge on the robot machine:

- Subscribe to robot joint states, gripper state, gripper commands, and RGB cameras
- Publish synchronized live packets over ZMQ for the executor
- Keep the live observation/action dimensions aligned with the training dataset contract

The ROS 2 bridge does not create robot or gripper state by itself.
It only republishes already-running ROS topics into the ZMQ format expected by the policy executor.
For the default Franka duo setup, start the robot and gripper bringup before starting the bridge.

## Deployment Contract

For the default dataset at `data/pick_and_place_test`, the expected runtime contract is:

- dataset FPS: `15`
- observation state dimension: `16`
- action dimension: `16`
- camera keys: `cam_left`, `cam_front`, `cam_right`
- action layout: `[Left Arm(7), Left Gripper(1), Right Arm(7), Right Gripper(1)]`
- arm action representation: `delta_joint_position`
- gripper action representation: `binary_open_close`

These values come from:

- `data/pick_and_place_test/meta/info.json`
- `data/pick_and_place_test/meta/real_exp_action_config.json`

The executor expects live packets from the ROS 2 bridge to match this contract.
It verifies the live `action_dim` against the dataset metadata before starting execution.

## Franka Executor

`franka_policy_executor.py` is the robot-side process for the Franka computer.

It currently:

- Subscribes to the live ZMQ observation stream
- Sends observations to the remote LeRobot policy server
- Receives action chunks back
- Interprets the action layout using the training dataset contract
- Runs in dry-run mode by default
- Can send arm velocity commands to the Franka only when `--execute` is explicitly passed

At runtime it:

- infers the policy type from `<policy-path>/config.json` unless overridden
- connects to the remote LeRobot async inference server over gRPC
- uses the first ZMQ packet to infer live observation feature shapes
- sends only `observation.state`, camera frames, and the task string to the policy
- interprets returned actions using the dataset action contract
- converts arm actions into filtered joint velocity commands before sending them to Franka

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

## Step-By-Step Deployment

This section assumes:

- the ROS 2 bridge runs on the Franka-connected machine
- the policy server runs on a machine that can load the trained checkpoint
- the checkpoint lives at `outputs/pick_and_place_test_act/checkpoints/last/pretrained_model`
- the policy server is reachable from the robot machine at `192.168.1.10:8080`

### 1. Check the trained checkpoint

Run this on the machine that has the checkpoint and the `lerobot` environment:

```bash
python train/deploy_lerobot_policy.py \
  inspect \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model
```

Verify that:

- `dataset_fps` is `15`
- `dataset_state_dim` is `16`
- `dataset_action_dim` is `16`
- `dataset_image_keys` are `observation.images.cam_left`, `observation.images.cam_front`, `observation.images.cam_right`
- `dataset_action_representation` says `arm=delta_joint_position, gripper=binary_open_close`

### 2. Start the ROS 2 bridge on the robot machine

Build and source the ROS 2 workspace:

```bash
cd gello_software/ros2
colcon build
source install/setup.bash
```

For the default dual-arm deployment setup, launch the bridge in direct hardware mode:

```bash
ros2 launch franka_lerobot_data_bridge bridge.launch.py config_file:=deployment_duo.yaml
```

For single-arm deployment, launch:

```bash
ros2 launch franka_lerobot_data_bridge bridge.launch.py config_file:=deployment_single.yaml
```

In deployment mode, the bridge reads robot joint positions and Franka gripper widths directly through `pylibfranka`, so it does not require `franka_fr3_arm_controllers` or `franka_gripper_manager` to be running.

Before launching, check the selected YAML config and make sure these values match the training setup:

- `include_right_arm`
- `include_gripper`
- `left_robot_ip`, `right_robot_ip`
- `camera_1_name`, `camera_2_name`, `camera_3_name`
- camera topic names

The default dual-arm deployment config is:

- `gello_software/ros2/src/franka_lerobot_data_bridge/config/deployment_duo.yaml`

If the bridge logs `waiting for left robot joint states` in deployment mode, it could not read the left robot state from `left_robot_ip`.
If it logs that it is waiting for gripper state, it could not read the Franka gripper width from the configured robot IP.

### 3. Confirm the bridge matches the training contract

For the default dataset, your live bridge should publish:

- 16-dimensional robot state
- 16-dimensional action packets
- three RGB cameras named `cam_left`, `cam_front`, and `cam_right`

Important detail:

- in deployment mode the bridge publishes real `observation.state` from live hardware reads
- it publishes placeholder action vectors only to preserve the trained dataset action dimension and layout
- the action placeholders are not used as policy input by `franka_policy_executor.py`
- the trained policy should still be interpreted using the dataset metadata, which for the default dataset is `delta_joint_position` for arms and `binary_open_close` for grippers

So during deployment, keep using the dataset metadata as the source of truth for action interpretation.

### 4. Start the policy server on the inference machine

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python train/deploy_lerobot_policy.py \
  server \
  --host 0.0.0.0 \
  --port 8080 \
  --fps 15
```

If needed, set:

- `--port` to a different gRPC port
- `--inference-latency` to your target server-side inference budget
- `--obs-queue-timeout` if you need a different observation timeout

### 5. Start the executor in dry-run mode on the robot machine

Run:

```bash
python train/franka_policy_executor.py \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model \
  --server-address 192.168.1.10:8080 \
  --zmq-host 127.0.0.1 \
  --zmq-port 5555 \
  --fps 15 \
  --task "pick and place"
```

What should happen:

- it waits for the first ZMQ packet from the ROS 2 bridge
- it prints the live `state_dim`, `action_dim`, and camera names
- it checks that live `action_dim` matches the dataset action dimension
- it tells the remote server which checkpoint to load
- it starts printing predicted actions instead of moving the robot

If the executor raises an `action_dim` mismatch, fix the ROS 2 bridge configuration so the live stream matches the dataset used for training.

If the checkpoint exists only on the policy server machine and is not present on the robot computer, pass:

- `--policy-path` as the path on the server machine
- `--policy-type` explicitly
- `--actions-per-chunk` explicitly

Example:

```bash
python train/franka_policy_executor.py \
  --policy-path /home/serveruser/real-exp/outputs/pick_and_place_test_diffusion/checkpoints/010000/pretrained_model \
  --policy-type diffusion \
  --actions-per-chunk 8 \
  --policy-device cuda:0 \
  --server-address 192.168.50.6:8080 \
  --zmq-host 127.0.0.1 \
  --zmq-port 5555 \
  --fps 15 \
  --task "pick and place"
```

### 6. Check predictions before moving the robot

In dry-run mode, confirm:

- camera streams are live
- the task string is correct
- predicted actions are finite and stable
- the policy responds continuously without large delays
- no dimension mismatch or missing camera errors appear

Do not move on to real execution until dry-run behavior looks correct.

### 7. Enable live robot execution

Restart the executor with `--execute`:

```bash
python train/franka_policy_executor.py \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model \
  --server-address 192.168.1.10:8080 \
  --zmq-host 127.0.0.1 \
  --zmq-port 5555 \
  --fps 15 \
  --task "pick and place" \
  --execute
```

If gripper actions are part of the dataset, add:

```bash
--enable-gripper
```

The executor will:

- split the policy action into left/right arm and gripper components
- interpret arm actions using the trained dataset layout
- convert arm deltas into velocity commands
- low-pass filter and clamp those commands to conservative limits
- send gripper commands in best-effort mode when enabled

Important live-control caveat:

- the ROS 2 bridge needs upstream robot state topics
- the current `franka_policy_executor.py --execute` path sends commands directly through `pylibfranka`
- do not run two independent controllers that both try to command the same Franka robot at the same time

For live policy execution, choose one control architecture:

- ROS 2 control path: keep `franka_fr3_arm_controllers` running, then adapt the policy executor to publish policy targets to the controller input topics such as `/left/gello/joint_states`, `/right/gello/joint_states`, and gripper command topics
- direct `pylibfranka` path: use `franka_policy_executor.py --execute`, but provide observations from a source that does not also own the robot command interface

The safest next implementation for this repository is usually the ROS 2 control path, because the same launch stack already provides the joint states and gripper state required by the bridge.

### 8. Tune only if needed

Useful executor options:

- `--actions-per-chunk` to override the chunk length requested from the server
- `--policy-type` if automatic inference from `config.json` is not what you want
- `--policy-device` to tell the remote policy server which device to use
- `--velocity-filter-tau` to change arm command smoothing
- `--ip-left` and `--ip-right` to match your robot IPs
- `--gripper-open-width`, `--gripper-closed-width`, and `--gripper-speed` for gripper behavior

## Common Failure Modes

- `action_dim` mismatch:
  - your live ROS 2 bridge config does not match the dataset used for training
- missing camera keys:
  - live camera names differ from the dataset camera feature names
- `Repo id must be in the form ...` on the server:
  - the server cannot find the checkpoint path locally
  - pass `--policy-path` as the path on the server machine, not the robot machine
- no ZMQ packets received:
  - the ROS 2 bridge is not running, not publishing, or is bound to a different host/port
- gRPC connection errors:
  - the executor cannot reach the policy server at `--server-address`
- `failed to connect to all addresses` or `No route to host`:
  - the server IP is wrong or the two machines are not on a reachable network
  - verify with `ping <server-ip>` and `nc -vz <server-ip> 8080`
- `CUDA error: invalid device ordinal` on the server:
  - the checkpoint config and requested runtime device disagree
  - start the server with `CUDA_VISIBLE_DEVICES=0` and use `--policy-device cuda:0`
- `stack expects a non-empty TensorList` on the server:
  - the policy did not receive a valid stacked image batch
  - confirm the live bridge publishes all cameras from the training contract and restart `python train/deploy_lerobot_policy.py server ...`
- `Action receiver RPC error: Channel closed!` on the robot machine:
  - this usually means the server crashed while handling inference
  - check the server log first and fix the upstream error there
- `zmq.error.Again: Resource temporarily unavailable` on the robot machine:
  - the bridge did not publish a fresh packet before the ZMQ receive timeout
  - the patched executor now skips that cycle and waits for the next packet
- robot does not move in dry-run:
  - expected, because dry-run prints predicted actions and does not command Franka
- gripper does not move during execution:
  - start the executor with `--enable-gripper`

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
