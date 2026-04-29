## Deploying a Trained Policy

This document describes the recommended process for deploying a trained LeRobot policy to a Franka setup.

The intended topology is:

- run policy inference on a server or GPU workstation
- run the real-time executor on the computer connected to the robot
- use the ROS 2 bridge on the robot machine to publish live observations and receive execution commands

This split is safer because:

- the robot machine stays focused on low-latency IO, safety checks, and command execution
- the inference machine can use a larger GPU and heavier models
- server restarts do not directly disturb the robot-side control process

## Deployment Flow

Use this order:

1. inspect the checkpoint and confirm the dataset contract
2. start the camera publisher on the robot machine
3. start the ROS 2 bridge
4. start the policy server on the inference machine
5. for live execution, start the ROS 2 arm and gripper consumers
6. start the executor in dry-run mode on the robot machine
7. verify that observations and predictions look correct
8. restart the executor with `--execute` only after dry-run looks safe

## What Runs Where

Policy server machine:

- load the trained checkpoint
- receive observations over the network
- run policy inference
- return action chunks

Robot-connected machine:

- read robot joint state, gripper state, and camera frames
- build observations in the same format used during training
- send observations to the policy server
- receive action chunks
- convert policy actions into Franka-safe commands
- enforce velocity limits, interpolation, watchdog timeout, and abort handling locally

ROS 2 bridge on the robot machine:

- subscribe to robot joint states, gripper state, gripper commands, and RGB cameras
- publish synchronized live packets over ZMQ for the executor
- receive command packets for deployment execution
- keep the live observation and action dimensions aligned with the training dataset contract

The ROS 2 bridge does not create robot or gripper state by itself.
It republishes already-running ROS topics into the ZMQ format expected by the policy executor.
For the default Franka duo setup, start the robot and gripper bringup before starting the bridge.

## End-to-End Procedure

### 1. Inspect the checkpoint

Run this on the machine that has the checkpoint and the `lerobot` environment:

```bash
python train/deploy_lerobot_policy.py \
  inspect \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model
```

Inspecting before deployment is recommended because it prints:

- the policy type
- the recommended `actions_per_chunk`
- the dataset state and action dimensions
- the expected camera keys
- the recorded action representation

Verify that:

- `dataset_fps` is `15`
- `dataset_state_dim` is `16`
- `dataset_action_dim` is `16`
- `dataset_image_keys` are `observation.images.cam_left`, `observation.images.cam_front`, `observation.images.cam_right`
- `dataset_action_representation` says `arm=absolute_joint_position, gripper=binary_open_close`

### 2. Start the camera publisher

Start the RealSense camera publisher:

```bash
ros2 launch franka_realsense_camera_publisher cameras.launch.py
```

The default camera publisher config is:

- `gello_software/ros2/src/franka_realsense_camera_publisher/config/example_three_cameras.yaml`

If `camera_1_serial`, `camera_2_serial`, or `camera_3_serial` are left empty, the node auto-assigns detected RealSense devices in sorted serial-number order.
For stable camera-to-name mapping, set those serial numbers explicitly in the YAML.

Make sure the publisher exposes the camera topics expected by training before starting the bridge.

### 3. Start the ROS 2 bridge

Before launching the bridge, check the selected bridge config and make sure it matches training:

- `include_right_arm`
- `include_gripper`
- `left_robot_ip`, `right_robot_ip`
- `command_host`, `command_port`
- `left_deployment_joint_command_topic`, `right_deployment_joint_command_topic`
- `left_deployment_gripper_command_topic`, `right_deployment_gripper_command_topic`
- `camera_1_name`, `camera_2_name`, `camera_3_name`
- camera topic names

For the default dual-arm deployment setup:

```bash
ros2 launch franka_lerobot_data_bridge bridge.launch.py config_file:=deployment_duo.yaml
```

For single-arm deployment:

```bash
ros2 launch franka_lerobot_data_bridge bridge.launch.py config_file:=deployment_single.yaml
```

For the default dataset, the live bridge should publish:

- 16-dimensional robot state
- 16-dimensional action packets
- three RGB cameras named `cam_left`, `cam_front`, and `cam_right`

The deployment bridge starts in `STANDBY` mode by default.
In standby it can publish hold commands to the ROS 2 controller topics, but it does not publish live ZMQ observation packets until deployment is activated.

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

### 5. Start the ROS 2 arm and gripper consumers for live execution

If you only want dry-run inference, skip this step until you are ready to move the robot.

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

In `deployment_mode:=true`, `joint_impedance_controller` holds the current arm pose until the bridge explicitly enables deployment control.
The bridge republishes policy actions to these existing topics:

- `/left/gello/joint_states`
- `/right/gello/joint_states`
- `/left/gripper/gripper_client/target_gripper_width_percent`
- `/right/gripper/gripper_client/target_gripper_width_percent`

This lets the existing ROS 2 consumers be reused during deployment.

### 6. Start the executor in dry-run mode

Run this on the robot machine:

```bash
python train/franka_act_policy_executor.py \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model \
  --server-address 192.168.50.6:8080 \
  --zmq-host 127.0.0.1 \
  --zmq-port 5555 \
  --fps 15 \
  --task "pick and place"
```

If you want to control how overlapping action chunks are blended, add:

```bash
--act-aggregate-ratio-old 0.8
```

This ACT-specific flag sets the weight of the already-queued action when the executor receives a new action for the same timestep:

- blended action = `old_ratio * old + (1 - old_ratio) * new`
- valid range is `0.0` to `1.0`
- `0.0` means fully replace the queued action with the new action
- `1.0` means keep the queued action and ignore the new action at overlapping timesteps
- the default is `0.8`

For diffusion deployment, use the diffusion-specific knobs instead:

```bash
python train/franka_diffusion_policy_executor.py \
  --policy-path outputs/pick_and_place_test_diffusion/checkpoints/last/pretrained_model \
  --actions-per-chunk 8 \
  --server-address 192.168.50.6:8080 \
  --zmq-host 127.0.0.1 \
  --zmq-port 5555 \
  --fps 15 \
  --task "pick and place" \
  --diffusion-chunk-size-threshold 0.5 \
  --diffusion-aggregate-ratio-old 0.5
```

If the checkpoint exists only on the policy server machine and not on the robot computer, pass:

- `--policy-path` as the path on the server machine
- `--actions-per-chunk` explicitly

Example:

```bash
python train/franka_diffusion_policy_executor.py \
  --policy-path /home/pair/real-exp/outputs/policy-dir \
  --actions-per-chunk 8 \
  --policy-device cuda:0 \
  --server-address 192.168.50.6:8080 \
  --zmq-host 127.0.0.1 \
  --zmq-port 5555 \
  --fps 15 \
  --task "pick and place" \
  --diffusion-chunk-size-threshold 0.5 \
  --diffusion-aggregate-ratio-old 0.5
```

### 7. Validate dry-run before moving the robot

What should happen:

- the executor waits for the first ZMQ packet from the ROS 2 bridge
- it prints the live `state_dim`, `action_dim`, and camera names
- it checks that live `action_dim` matches the dataset action dimension
- it tells the remote server which checkpoint to load
- it starts printing predicted actions instead of moving the robot

Do not move on to live execution until all of the following look correct:

- camera streams are live
- the task string is correct
- predicted actions are finite and stable
- the policy responds continuously without large delays
- no dimension mismatch or missing camera errors appear

If the executor raises an `action_dim` mismatch, fix the ROS 2 bridge configuration so the live stream matches the dataset used for training.

### 8. Restart the executor with `--execute`

Once dry-run is stable, restart the executor for live execution:

```bash
python train/franka_act_policy_executor.py \
  --policy-path outputs/pick_and_place_test_act/checkpoints/last/pretrained_model \
  --server-address 192.168.50.6:8080 \
  --zmq-host 127.0.0.1 \
  --zmq-port 5555 \
  --command-zmq-host 127.0.0.1 \
  --command-zmq-port 5556 \
  --fps 15 \
  --task "pick and place" \
  --act-aggregate-ratio-old 0.8 \
  --execute
```

Live execution ordering should be:

1. start the bridge with `deployment_duo.yaml` or `deployment_single.yaml`
2. start the arm controllers with `deployment_mode:=true`
3. start the gripper manager
4. start the remote policy server
5. start the executor with `--execute`
6. let the executor auto-activate the bridge before the first command

With the current executor flow, you do not need to call `/set_deployment_active` manually for live execution unless you explicitly pass `--no-auto-activate-bridge`.

The executor will:

- split the policy action into left and right arm and gripper components
- interpret arm actions using the trained dataset layout
- send `absolute_joint_position` arm actions directly as robot joint targets
- send those targets to the bridge command socket
- let the bridge enable the deployment-gated arm controllers and republish targets to the existing ROS 2 arm and gripper topics

Important live-control caveats:

- the ROS 2 bridge needs upstream camera topics and direct Franka state access
- the current deployment execute path is intended to use the ROS 2 controller stack, not direct active `pylibfranka` control
- do not run another node that publishes conflicting targets to `/left/gello/joint_states` or `/right/gello/joint_states`

If you want to keep bridge activation manual during execution, pass `--no-auto-activate-bridge` to the executor and then call:

```bash
ros2 service call /set_deployment_active std_srvs/srv/SetBool "{data: true}"
```

The bridge only enables the deployment-gated arm controllers after it receives the first real command packet from the executor.
If you start the bridge in direct `pylibfranka` deployment-state mode while `ros2_control` is active, the bridge may stop publishing fresh `/left/right/gello/joint_states`, which causes the joint impedance controllers to time out waiting for valid command input.

### 9. Tune only if needed

Useful executor options:

- `--actions-per-chunk` to override the chunk length requested from the server
- `--policy-device` to tell the remote policy server which device to use
- `--act-chunk-size-threshold` and `--act-aggregate-ratio-old` on `franka_act_policy_executor.py` to tune overlapping ACT chunks
- `--diffusion-chunk-size-threshold` and `--diffusion-aggregate-ratio-old` on `franka_diffusion_policy_executor.py` to tune overlapping diffusion chunks
- `--diffusion-noise-scheduler-type` and `--diffusion-num-inference-steps` on the server to override diffusion denoising at load time
- `--command-zmq-host` and `--command-zmq-port` to match the bridge command socket
- `--bridge-activation-service` to override the ROS 2 `SetBool` service used for bridge activation
- `--no-auto-activate-bridge` to keep bridge activation manual

## Common Failure Modes

- `action_dim` mismatch:
  - the live ROS 2 bridge config does not match the dataset used for training
- missing camera keys:
  - live camera names differ from the dataset camera feature names
- `Repo id must be in the form ...` on the server:
  - the server cannot find the checkpoint path locally
  - pass `--policy-path` as the path on the server machine, not the robot machine
- no ZMQ packets received:
  - the ROS 2 bridge is not running, not publishing, or is bound to a different host or port
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
  - the server usually crashed while handling inference
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
  - this has been observed in the current setup during controller bringup
  - if `joint_state_broadcaster` and `joint_impedance_controller` still configure successfully, deployment can still proceed
- `zmq.error.Again: Resource temporarily unavailable` on the robot machine:
  - the bridge did not publish a fresh packet before the ZMQ receive timeout
  - the patched executor now skips that cycle and waits for the next packet
- robot does not move in dry-run:
  - expected because dry-run prints predicted actions and does not command Franka

## Franka Control Note

The dataset action representation is absolute joint position for the arms, with optional gripper commands.
That means the robot-side executor can send policy arm outputs to the deployment bridge as joint targets, while still keeping local safety checks and the ROS 2 controller layer in the loop.

The executor should interpret actions using the same structure as the dataset:

- 16-dim: `[Left Arm(7), Left Gripper(1), Right Arm(7), Right Gripper(1)]`
- 14-dim: `[Left Arm(7), Right Arm(7)]`

In practice, the executor should:

- command absolute joint targets through the deployment bridge
- clamp commands to conservative Franka limits
- smooth or interpolate commands between network updates
- stop safely if server responses are delayed or missing
