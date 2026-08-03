# DATA COLLECTION


## Overview
- `lerobot_collection.py`: Minimal script for recording synchronized RealSense images and robot state/action data into a LeRobot dataset.
- `gello_recording_home.py`: Internal ROS helper used by the recorder's optional GELLO reset-and-hold mode.
- `replay_pylibfranka.py`: Replay a recorded LeRobot episode on the real Franka arms using `pylibfranka`, with optional `--dry-run` inspection before motion.
- `reset_pylibfranka.py`: Reset both Franka arms to the hardcoded initial state copied from `data/pick_and_place_test` episode 0, without reading dataset parquet files at runtime.
- `initialize_franka_for_deployment.py`: Preview and safely initialize both Franka arms from a dataset episode start or a postprocessed absolute policy action before deployment.
- `delete_lerobot_episode.py`: Remove one or more episodes from a local LeRobot dataset while preserving the remaining metadata, videos, and parquet data.
- `validate_dataset.py`: Validate a local LeRobot dataset and print dataset-level and per-episode consistency information.

Quick links:

- [GELLO docs](gello_software/README.md)
- [FR3 ROS 2 docs](gello_software/ros2/README.md)
- [LeRobot docs](lerobot/README.md)

Environment split used in this repo:

- Use `/usr/bin/python3` (Python 3.10 on this machine) for ROS 2 Humble, GELLO helper scripts, `colcon build`, and direct `pylibfranka` replay.
- Use the `lerobot` Conda environment for LeRobot dataset and training. 


## Before launching

- Test offset if the gello connection is unplugged.
```bash
source /opt/ros/humble/setup.sh
source ~/franka_ros2_ws/install/setup.bash
source ~/real-exp/gello_software/ros2/install/setup.bash

ros2 topic echo /left/gello/joint_states
ros2 topic echo /right/gello/joint_states
```
compare the results with the joint angles from ```172.16.0.2/desk/api/robot/robot-state``` and ```172.16.0.3/desk/api/robot/robot-state```

- Set the offset if necessary.
```bash
cd ~/real-exp/gello_software
python3 scripts/setup_offset.py --start-joints 0 0 0 -1.57 0 1.57 0 --joint-signs 1 -1 1 1 1 -1 1 --port /dev/ttyUSB_left
python3 scripts/setup_offset.py --start-joints 0 0 0 -1.57 0 1.57 0 --joint-signs 1 -1 1 1 1 -1 1 --port /dev/ttyUSB_right
```

- Unlock the franka arm and activate FCI.

- Build the ROS 2 workspace

Skip this if nothing under `gello_software/ros2/` changed since the last build.

```bash
cd gello_software/ros2
colcon build
source install/setup.bash
```

If you open a new shell after building, run `source install/setup.bash` again before using `ros2 launch`.

## Robot Reset And Replay

Use the direct `pylibfranka` reset script when you want to return both robots to a dataset start pose before recording, replay, or deployment.

By default, the script preserves the legacy hardcoded target stored inside `data_collection/reset_pylibfranka.py`:

- left arm joint positions
- left gripper width
- right arm joint positions
- right gripper width

Preview the legacy target state without moving the robots:

```bash
python data_collection/reset_pylibfranka.py --dry-run
```

Reset both arms and grippers to the legacy target:

```bash
python data_collection/reset_pylibfranka.py
```

To reset to the actual initial `observation.state` from a dataset episode, pass `--dataset-root`, `--episode`, and optionally `--frame-index`.

Preview dataset episode 0, frame 0:

```bash
python data_collection/reset_pylibfranka.py \
  --dataset-root data/pick_and_place_test \
  --episode 0 \
  --frame-index 0 \
  --dry-run
```

Reset to that dataset frame:

```bash
python data_collection/reset_pylibfranka.py \
  --dataset-root data/pick_and_place_test \
  --episode 0 \
  --frame-index 0
```

## Deployment Initialization

For the normal ROS deployment flow, use `--policy-start` or
`--episode-start [INDEX]` directly on `franka_act_policy_executor.py` or
`franka_diffusion_policy_executor.py`. Those two mutually exclusive options share
the running deployment bridge/controller path, so the four ROS deployment launch
groups stay running.

`initialize_franka_for_deployment.py` is a standalone direct-`pylibfranka`
maintenance tool, not the normal executor startup path. It validates
all arm targets against the FR3 joint limits with a safety margin before opening a robot
connection. Movement is preview-only unless both `--execute` and the confirmation phrase
are supplied. Stop the ROS arm controller and any other Franka command source before using
this direct `pylibfranka` command.

Choose a random dataset episode start and preview it:

```bash
python3 data_collection/initialize_franka_for_deployment.py \
  --episode \
  --dataset-root data/L3_drawer_swap_20260630_trim_start_3s
```

Use a deterministic random selection:

```bash
python3 data_collection/initialize_franka_for_deployment.py \
  --episode \
  --random-seed 7 \
  --dataset-root data/L3_drawer_swap_20260630_trim_start_3s
```

Choose a specific episode:

```bash
python3 data_collection/initialize_franka_for_deployment.py \
  --episode 12 \
  --dataset-root data/L3_drawer_swap_20260630_trim_start_3s
```

After reviewing the preview, add the execution confirmation:

```bash
python3 data_collection/initialize_franka_for_deployment.py \
  --episode 12 \
  --dataset-root data/L3_drawer_swap_20260630_trim_start_3s \
  --execute \
  --confirm INITIALIZE_FRANKA_FOR_DEPLOYMENT
```

Policy mode consumes a policy's already postprocessed absolute action, not a checkpoint
directory. A checkpoint has no single static initial pose because its output depends on the
live images and robot state. Pass 14 arm values directly, or pass 16 values when the action
also contains both gripper commands:

```bash
python3 data_collection/initialize_franka_for_deployment.py \
  --policy LEFT_Q1 LEFT_Q2 LEFT_Q3 LEFT_Q4 LEFT_Q5 LEFT_Q6 LEFT_Q7 \
           LEFT_GRIPPER \
           RIGHT_Q1 RIGHT_Q2 RIGHT_Q3 RIGHT_Q4 RIGHT_Q5 RIGHT_Q6 RIGHT_Q7 \
           RIGHT_GRIPPER
```

The policy target may instead be a JSON array or a JSON file containing `policy_action`,
`action`, `initial_state`, or `target`:

```bash
python3 data_collection/initialize_franka_for_deployment.py \
  --policy outputs/policy_initial_action.json
```

The default policy gripper representation is `binary_open_close`; values below `0.5` close
the gripper and values at or above `0.5` open it to `0.08 m`. For policies trained with
absolute gripper widths, add:

```bash
--policy-gripper-representation absolute_width
```

Add `--skip-grippers` to either mode to leave the grippers unchanged. A 14-D policy target
also leaves both grippers unchanged automatically.

## Automated Dual-Arm Collection Stack

From the repository root, start the five ROS 2 process groups required for
dual-arm collection:

```bash
bash scripts/start_collection_duo.sh
```

The launcher starts and monitors the GELLO publisher, Franka controllers,
gripper clients, camera publishers, and collection bridge in dependency order.
It checks the ROS overlays, ZMQ port, required topics, and process health, then
stops all children in reverse order on Ctrl-C. The LeRobot recorder intentionally
runs in a separate Conda terminal so episode commands remain interactive.

Use `COLLECTION_ZMQ_PORT`, `READY_TIMEOUT`, or the documented setup-path
environment variables at the top of the script when the local installation uses
non-default paths or ports.

## Teleoperation Quick Start

The commands below cover the common FR3 teleoperation workflow from this repository.

### 1. Start the GELLO publisher

Dual-arm example:

```bash
ros2 launch franka_gello_state_publisher main.launch.py config_file:=example_duo.yaml
```

Single left arm example:

```bash
ros2 launch franka_gello_state_publisher main.launch.py config_file:=example_single.yaml
```

### 2. Start teleoperation on the robot side

Dual FR3 setup:

```bash
ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py robot_config_file:=example_fr3_duo_config.yaml
```

Single left FR3 setup:

```bash
ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py robot_config_file:=example_fr3_config.yaml
```

### 3. Start gripper control
Dual arm setup:
```bash
ros2 launch franka_gripper_manager franka_gripper_client.launch.py config_file:=example_fr3_duo_config_franka_hand.yaml
```

Single left arm setup:
```bash
ros2 launch franka_gripper_manager franka_gripper_client.launch.py config_file:=example_fr3_config_franka_hand.yaml
```

### Configuration Notes

- GELLO publisher configs live in `gello_software/ros2/src/franka_gello_state_publisher/config/`.
- FR3 controller configs live in `gello_software/ros2/src/franka_fr3_arm_controllers/config/`.
- `example_duo.yaml` defines the left and right GELLO devices for bimanual control.
- `example_fr3_duo_config.yaml` defines the corresponding left and right FR3 robot IPs and namespaces.
- If you are switching between single-arm and dual-arm setups, make sure the publisher and controller configs match.

## Data Collection

The recording path is split into two pieces:

- A ROS 2 camera publisher in `gello_software/ros2/src/franka_realsense_camera_publisher/` that publishes RGB images from up to three RealSense cameras.
- A ROS 2 bridge node in `gello_software/ros2/src/franka_lerobot_data_bridge/` that subscribes to robot, teleop, gripper, and camera topics and publishes synchronized samples over ZMQ.
- `lerobot_collection.py`, which subscribes to that sample stream and writes a local LeRobot dataset.

The dataset currently records:

- `observation.state`: actual robot joint positions, plus gripper width if enabled
- `action`: absolute arm joint targets for the next sample, plus gripper command if enabled
- `observation.images.cam_left`, `observation.images.cam_front`, `observation.images.cam_right`: RGB video streams

The bridge expects:

- Robot joint states on a topic like `/left/franka/joint_states`
- Arm-controller target joint states on a topic like `/left/franka/commanded_joint_states`
- Robot gripper joint states on a topic like `/left/franka_gripper/joint_states`
- Gripper commands on a topic like `/left/gripper/gripper_client/target_gripper_width_percent`
- RGB image topics for three cameras

By default the bridge publishes current measured robot joint states as `observation.state` and uses the arm-controller target topic (`/left|right/franka/commanded_joint_states`) as the arm action source. The recorder labels each frame with the next packet's absolute arm joint target, so new datasets use `arm_action_representation=absolute_joint_position`.

Launch the camera publisher from the ROS 2 workspace:

```bash
ros2 launch franka_realsense_camera_publisher cameras.launch.py
```

Launch the bridge from the ROS 2 workspace:

```bash
ros2 launch franka_lerobot_data_bridge bridge.launch.py
```

The bridge defaults to the bimanual config. To use single-arm recording (this will only use two cameras):

```bash
ros2 launch franka_lerobot_data_bridge bridge.launch.py config_file:=example_single.yaml
```

Then run the LeRobot recorder from the repo root:

```bash
source ~/anaconda3/bin/activate && conda activate lerobot
python data_collection/lerobot_collection.py
```

### Optional GELLO reset and powered hold

The recorder behaves exactly as before unless `--gello-reset-hold` is present. To
enable the integrated dual-GELLO reset key:

```bash
cd ~/real-exp
source ~/anaconda3/bin/activate && conda activate lerobot
python data_collection/lerobot_collection.py \
  --gello-reset-hold \
  --repo-id local/franka_gello_teleop \
  --local-dir ./lerobot_data
```

The GELLO publishers and their ROS parameter services must already be running from
the rebuilt `gello_software/ros2` workspace. The recorder automatically launches
its homing helper with ROS Humble's system Python, so the LeRobot environment does
not need to provide `rclpy`.

Additional control in this mode:

- `r + Enter`: actively reset both GELLOs to
  `reset_pylibfranka.INITIAL_STATE`, then keep all arm joints powered at the target
- `s + Enter`: start recording normally and arm automatic torque release
- moving either GELLO joint by about `0.75 deg` after `s` restores the original
  gains/current settings and turns off both GELLO arm torques
- `q + Enter` or Ctrl-C: turn off GELLO torque first, then use the recorder's
  existing save/finalize path

Do not press `r` during an episode; the recorder rejects that command until the
episode is saved or discarded. If `s` is followed by `e` or `d` without any GELLO
movement, the hold remains active as requested and can be released by movement
after a later `s`, or by `q`.

This mode uses the settings previously validated with one USB connection per
GELLO: groups of at most three joints per side use `100 mA` while homing, and
other joints use `15 mA`; after homing, every arm joint holds at `15 mA`. It does
not enable the gripper motor. Keep the two GELLO workspaces clear while pressing
`r`, and use Ctrl-C if the motion is not as expected.

## Dataset Validation

After recording or editing a dataset, validate that the metadata, parquet data, and videos still agree.

Run the default validation:

```bash
python3 data_collection/validate_dataset.py \
  --dataset-root data/pick_and_place_test
```

Print one row per episode:

```bash
python3 data_collection/validate_dataset.py \
  --dataset-root data/pick_and_place_test \
  --verbose
```

The validator checks:

- `meta/info.json` totals against actual episode metadata and data rows
- continuous episode indices and global frame indices
- per-episode `length` against state/action row counts
- per-episode `frame_index` and timestamp continuity
- `observation.state` and `action` dimensions against `info.json`
- video timestamp ranges against episode lengths
- physical MP4 frame counts when OpenCV is available

If OpenCV is not available in the active Python environment, either install it or skip physical video checks:

```bash
python3 data_collection/validate_dataset.py \
  --dataset-root data/pick_and_place_test \
  --skip-video-frames
```


## Additional Documentation

- General GELLO docs: [gello_software/README.md](gello_software/README.md)
- Franka FR3 ROS 2 docs: [gello_software/ros2/README.md](gello_software/ros2/README.md)
- LeRobot docs: [lerobot/README.md](lerobot/README.md)

## Dataset Hub Helpers

Two small helpers are available under `data_collection/` for moving LeRobot datasets to and from Hugging Face.

Push a local dataset:

```bash
python data_collection/push_lerobot_dataset.py \
  --dataset-root data/pick_and_place_test \
  --repo-id Jianshu1/pick_and_place_test \
  --private
```

Fetch a dataset from Hugging Face:

```bash
python data_collection/fetch_lerobot_dataset.py \
  --repo-id Jianshu1/pick_and_place_test
```

By default:

- `push_lerobot_dataset.py` pushes to remote branch `main`
- `fetch_lerobot_dataset.py` fetches from remote branch `main`
- `fetch_lerobot_dataset.py` replaces `data/<repo-name>` so the local copy matches the remote dataset

Use `--branch`, `--revision`, `--no-clean`, or `--no-force-cache-sync` only when you intentionally want non-default behavior.
