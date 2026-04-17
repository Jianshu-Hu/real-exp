# DATA COLLECTION


## Overview
- `lerobot_collection.py`: Minimal script for recording synchronized RealSense images and robot state/action data into a LeRobot dataset.
- `replay_pylibfranka.py`: Replay a recorded LeRobot episode on the real Franka arms using `pylibfranka`, with optional `--dry-run` inspection before motion.
- `reset_pylibfranka.py`: Reset both Franka arms to the hardcoded initial state copied from `data/pick_and_place_test` episode 0, without reading dataset parquet files at runtime.
- `delete_lerobot_episode.py`: Remove one or more episodes from a local LeRobot dataset while preserving the remaining metadata, videos, and parquet data.

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

Use the direct `pylibfranka` reset script when you want to return both robots to the fixed initial pose used for the `pick_and_place_test` setup.

The reset target is stored as a constant inside `data_collection/reset_pylibfranka.py`:

- left arm joint positions
- left gripper width
- right arm joint positions
- right gripper width

Preview the target state without moving the robots:

```bash
python data_collection/reset_pylibfranka.py --dry-run
```

Reset both arms and grippers:

```bash
python data_collection/reset_pylibfranka.py
```

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
- `action`: arm state deltas computed as `q[t+1] - q[t]`, plus gripper command if enabled
- `observation.images.cam_left`, `observation.images.cam_front`, `observation.images.cam_right`: RGB video streams

The bridge expects:

- Robot joint states on a topic like `/left/franka/joint_states`
- Commanded Franka joint targets on a topic like `/left/franka/commanded_joint_states`
- Robot gripper joint states on a topic like `/left/franka_gripper/joint_states`
- Gripper commands on a topic like `/left/gripper/gripper_client/target_gripper_width_percent`
- RGB image topics for three cameras

By default the bridge now publishes the current measured robot joint states (`robot_state`). The recorder converts adjacent arm states into dataset actions using `q[t+1] - q[t]`.

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
python lerobot_collection.py
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

By default, `fetch_lerobot_dataset.py` downloads into `data/<repo-name>`.
