# Wuji glove teleoperation

This directory provides a Python teleoperator for a Wuji glove. It publishes
the same ROS 2 topics consumed by the existing FR3 controllers and LeRobot
bridge:

* `/<namespace>/gello/joint_states` (`sensor_msgs/msg/JointState`), seven FR3
  joint targets in radians;
* `/<namespace>/gripper/gripper_client/target_gripper_width_percent`
  (`std_msgs/msg/Float32`), normalized open-width command.

The hardware path imports the Wuji Python package lazily because the package is
provided by Wuji and is not bundled with this repository. By default it probes
the common module names `wuji`, `wuji_glove`, and `wuji_sdk`. Since SDK releases
expose different class and method names, `WujiGloveDevice` probes the common
`WujiGlove`/`Glove` constructors and `read`/`poll`/`get_state` methods. Use
`--device-module` and `--device-class` when your SDK uses another name.

## Install and run

Install the SDK according to the [Wuji glove documentation](https://docs.wuji.tech/docs/en/wuji-glove/latest/),
then source ROS 2 and the FR3 workspace:

```bash
python3 -m pip install <wuji-sdk-package>
source /opt/ros/humble/setup.bash
source ~/franka_ros2_ws/install/setup.bash
source ~/real-exp/gello_software/ros2/install/setup.bash
python3 data_collection/wuji/teleop.py --namespace left
```

The default mapping assumes the same joint sign convention as the calibrated
GELLO setup. Override it at launch when the glove is worn on the opposite hand
or mounted differently, for example:

```bash
python3 data_collection/wuji/teleop.py --namespace right \
  --joint-signs 1 1 1 1 1 1 1 \
  --joint-offsets 0 0 0 0 0 0 0
```

Start the normal FR3 controller for the selected namespace before running this
node (the node intentionally only replaces the GELLO publisher). The existing
`scripts/start_teleoperation.sh` remains the GELLO-specific launcher.

For two gloves, run one process per namespace (`left` and `right`) and set a
different `--device-id` if the SDK supports device selection. Keep the robot
in a safe pose while starting; the node holds the first received target and
applies a per-cycle step limit. `Ctrl-C` closes the SDK device cleanly.

Use `--stdin` to test the mapping without hardware. Each input line is seven
space-separated joint angles followed by a normalized gripper value.
