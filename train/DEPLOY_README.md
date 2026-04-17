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

Gripper commands are enabled by default. If you want to disable them:

```bash
python train/franka_policy_executor.py \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model \
  --server-address 192.168.1.10:8080 \
  --execute \
  --disable-gripper
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

For the default dual-arm deployment setup, launch the bridge with the ROS 2 control deployment config:

```bash
ros2 launch franka_lerobot_data_bridge bridge.launch.py config_file:=deployment_duo.yaml
```

For single-arm deployment, launch:

```bash
ros2 launch franka_lerobot_data_bridge bridge.launch.py config_file:=deployment_single.yaml
```

For live execution with the ROS 2 controller stack, `deployment_duo.yaml` uses ROS topics as the deployment state source instead of direct `pylibfranka` reads, which avoids conflicts with `ros2_control` during live execution.

Before launching, check the selected YAML config and make sure these values match the training setup:

- `include_right_arm`
- `include_gripper`
- `left_robot_ip`, `right_robot_ip`
- `command_host`, `command_port`
- `left_deployment_joint_command_topic`, `right_deployment_joint_command_topic`
- `left_deployment_gripper_command_topic`, `right_deployment_gripper_command_topic`
- `camera_1_name`, `camera_2_name`, `camera_3_name`
- camera topic names

The default dual-arm deployment config for live ROS 2 execution is:

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

### 4.5. Start the ROS 2 arm and gripper consumers on the robot machine for live execution

If you only want dry-run inference, you can skip this step.

For live execution, start the existing controller stack in deployment-gated mode:

```bash
ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py \
  robot_config_file:=example_fr3_duo_config.yaml \
  deployment_mode:=true
```

and:

```bash
ros2 launch franka_gripper_manager franka_gripper_client.launch.py config_file:=example_fr3_duo_config_franka_hand.yaml
```

In `deployment_mode:=true`, `joint_impedance_controller` holds the current arm pose until the bridge explicitly enables deployment control. The bridge republishes policy actions to the existing topics:

- `/left/gello/joint_states`
- `/right/gello/joint_states`
- `/left/gripper/gripper_client/target_gripper_width_percent`
- `/right/gripper/gripper_client/target_gripper_width_percent`

so the existing ROS 2 consumers can be reused during deployment.

Important ordering for live execution:

1. start the bridge with `deployment_duo.yaml` or `deployment_single.yaml`
2. start the arm controllers with `deployment_mode:=true`
3. start the gripper manager
4. start the remote policy server
5. start the executor with `--execute`
6. the executor auto-activates the bridge before sending the first command

The deployment bridge starts in `STANDBY` mode by default. In standby it can publish hold commands to the ROS 2 controller topics, but it does not publish live ZMQ observation packets until deployment is activated.

If you want to activate the bridge manually instead, pass `--no-auto-activate-bridge` to the executor and then call:

```bash
ros2 service call /set_deployment_active std_srvs/srv/SetBool "{data: true}"
```

The bridge only enables the deployment-gated arm controllers after it receives the first real command packet from the executor. If you start the bridge in direct `pylibfranka` deployment-state mode while `ros2_control` is active, the bridge may stop publishing fresh `/left/right/gello/joint_states`, which causes the joint impedance controllers to time out waiting for valid command input.

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

Important:

- in deployment mode, dry-run does **not** auto-activate the bridge
- if the bridge was launched with `deployment_duo.yaml` or `deployment_single.yaml`, call the activation service once before expecting the first ZMQ packet:

```bash
ros2 service call /set_deployment_active std_srvs/srv/SetBool "{data: true}"
```

- if you skip this step, the executor will wait at `Waiting for the first ZMQ packet...`

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
  --command-zmq-host 127.0.0.1 \
  --command-zmq-port 5556 \
  --fps 15 \
  --task "pick and place" \
  --execute
```

If you want to disable gripper actions during execution, add:

```bash
--disable-gripper
```

The executor will:

- split the policy action into left/right arm and gripper components
- interpret arm actions using the trained dataset layout
- convert `delta_joint_position` actions into absolute joint targets using the live observation state
- send those targets to the bridge command socket
- let the bridge enable the deployment-gated arm controllers and republish targets to the existing ROS 2 arm and gripper topics

With the current executor flow, you do **not** need to call `/set_deployment_active` manually for live execution unless you explicitly pass `--no-auto-activate-bridge`.

Important live-control caveat:

- the ROS 2 bridge needs upstream camera topics and direct Franka state access
- the current deployment execute path is intended to use the ROS 2 controller stack, not direct active `pylibfranka` control
- do not run another node that also publishes conflicting targets to `/left/gello/joint_states` or `/right/gello/joint_states`

### 8. Tune only if needed

Useful executor options:

- `--actions-per-chunk` to override the chunk length requested from the server
- `--policy-type` if automatic inference from `config.json` is not what you want
- `--policy-device` to tell the remote policy server which device to use
- `--execute-backend bridge` to use the ROS 2 deployment bridge for live execution
- `--command-zmq-host` and `--command-zmq-port` to match the bridge command socket
- `--bridge-activation-service` to override the ROS 2 `SetBool` service used for bridge activation
- `--no-auto-activate-bridge` to keep bridge activation manual
- `--ip-left` and `--ip-right` to match your robot IPs when using direct hardware access for observations
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
- executor appears to hang after the first packet when using `--execute`:
  - make sure you are using the ROS 2 bridge execution backend and that `franka_fr3_arm_controllers` is running
  - the executor forwards actions to the bridge command socket instead of using direct active `pylibfranka` control
- executor waits forever at `Waiting for the first ZMQ packet...` in deployment dry-run:
  - expected if the bridge is still in `STANDBY`
  - call `ros2 service call /set_deployment_active std_srvs/srv/SetBool "{data: true}"`
  - live execution with `--execute` auto-activates the bridge unless `--no-auto-activate-bridge` is set
- `Timeout: No valid joint states received from Gello` from `joint_impedance_controller`:
  - the controller is not receiving fresh `/left/right/gello/joint_states`
  - for live ROS 2 deployment, launch the bridge with `deployment_duo.yaml` or `deployment_single.yaml`
  - rebuild and source the ROS 2 workspace before relaunching
- `franka_robot_state_broadcaster` fails with `State interface with key 'fr3/robot_state' does not exist`:
  - this has been observed in the current setup during controller bring-up
  - if `joint_state_broadcaster` and `joint_impedance_controller` still configure successfully, deployment can still proceed
- `zmq.error.Again: Resource temporarily unavailable` on the robot machine:
  - the bridge did not publish a fresh packet before the ZMQ receive timeout
  - the patched executor now skips that cycle and waits for the next packet
- robot does not move in dry-run:
  - expected, because dry-run prints predicted actions and does not command Franka
- gripper does not move during execution:
  - gripper commands are enabled by default; if you passed `--disable-gripper`, remove it

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
