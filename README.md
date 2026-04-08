# Real Experiments

Repository for real-world robot experiments centered on GELLO teleoperation, Franka FR3 control, and LeRobot data collection.


## Overview

This repo combines three pieces of the workflow:

- `gello_software/`: GELLO teleoperation code, robot integrations, and ROS 2 support.
- `lerobot/`: LeRobot codebase used for datasets, training, and policy work.
- `lerobot_collection.py`: Minimal script for recording synchronized RealSense images and robot state/action data into a LeRobot dataset.

Quick links:

- [GELLO docs](gello_software/README.md)
- [FR3 ROS 2 docs](gello_software/ros2/README.md)
- [LeRobot docs](lerobot/README.md)


## Teleoperation Quick Start

The commands below cover the common FR3 teleoperation workflow from this repository.

Before launching:

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
source .venv/bin/activate
python3 scripts/setup_offset.py --start-joints 0 0 0 -1.57 0 1.57 0 --joint-signs 1 -1 1 1 1 -1 1 --port /dev/ttyUSB_left
python3 scripts/setup_offset.py --start-joints 0 0 0 -1.57 0 1.57 0 --joint-signs 1 -1 1 1 1 -1 1 --port /dev/ttyUSB_right
```

- Unlock the franka arm and activate FCI.

### 1. Build the ROS 2 workspace

Skip this if nothing under `gello_software/ros2/` changed since the last build.

```bash
source .venv/bin/activate
cd gello_software/ros2
colcon build
source install/setup.bash
```

If you open a new shell after building, run `source install/setup.bash` again before using `ros2 launch`.

### 2. Start the GELLO publisher

Dual-arm example:

```bash
ros2 launch franka_gello_state_publisher main.launch.py config_file:=example_duo.yaml
```

Single-arm example:

```bash
ros2 launch franka_gello_state_publisher main.launch.py config_file:=example_single.yaml
```

### 3. Start teleoperation on the robot side

Dual FR3 setup:

```bash
ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py robot_config_file:=example_fr3_duo_config.yaml
```

Single FR3 setup:

```bash
ros2 launch franka_fr3_arm_controllers franka_fr3_arm_controllers.launch.py robot_config_file:=example_fr3_config.yaml
```

## Configuration Notes

- GELLO publisher configs live in `gello_software/ros2/src/franka_gello_state_publisher/config/`.
- FR3 controller configs live in `gello_software/ros2/src/franka_fr3_arm_controllers/config/`.
- `example_duo.yaml` defines the left and right GELLO devices for bimanual control.
- `example_fr3_duo_config.yaml` defines the corresponding left and right FR3 robot IPs and namespaces.
- If you are switching between single-arm and dual-arm setups, make sure the publisher and controller configs match.

## Data Collection

`lerobot_collection.py` is a lightweight recorder for creating a local LeRobot dataset with:

- RealSense RGB frames
- Robot state vectors
- Teleoperation action vectors

The script currently subscribes to `tcp://127.0.0.1:5555` and saves episodes into `./lerobot_data`.

Run it from the repo root after your teleoperation data stream is available:

```bash
source .venv/bin/activate
python lerobot_collection.py
```

## Additional Documentation

- General GELLO docs: [gello_software/README.md](gello_software/README.md)
- Franka FR3 ROS 2 docs: [gello_software/ros2/README.md](gello_software/ros2/README.md)
- LeRobot docs: [lerobot/README.md](lerobot/README.md)

