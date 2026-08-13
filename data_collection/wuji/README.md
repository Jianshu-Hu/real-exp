# Wuji glove teleoperation

This directory provides a Python teleoperator for a Wuji glove. It publishes
the same ROS 2 topics consumed by the existing FR3 controllers and LeRobot
bridge:

* `/<namespace>/gello/joint_states` (`sensor_msgs/msg/JointState`), seven FR3
  joint targets in radians;
* `/<namespace>/gripper/gripper_client/target_gripper_width_percent`
  (`std_msgs/msg/Float32`), normalized open-width command.

The hardware path uses Wuji's official `wuji-sdk` package. It discovers Wuji
Gloves over USB and UDP, connects by serial number, and consumes the glove's
21-DoF `hand_joint_angles` stream. The seven values selected by
`--joint-indices` are mapped to the FR3 command. Legacy SDK-style classes are
also supported through `--device-module` and `--device-class`.

## Install and run

This computer uses ROS 2 Jazzy and Python 3.12. Install the Wuji SDK into the
same Python environment used to launch the node:

```bash
cd ~/real-exp
python3 -m pip install -r data_collection/wuji/requirements.txt
source /opt/ros/jazzy/setup.bash
source ~/franka_ros2_ws/install/setup.bash
source ~/real-exp/gello_software/ros2/install/setup.bash
python3 data_collection/wuji/teleop.py --namespace left
```

The Jazzy setup must be sourced before running so Python can import `rclpy`,
`sensor_msgs`, and `std_msgs`. Do not install `rclpy` with pip. The ROS package
provided under `/opt/ros/jazzy` must match the active Python ABI.

Verify passive glove discovery before starting any FR3 controller:

```bash
python3 - <<'PY'
from wuji_sdk import DeviceType, SdkManager

for device in SdkManager.instance().scan():
    print(device.sn, device.device_type, device.address)
PY
```

If more than one glove is discovered, select one with
`--device-id <serial-number>`. The glove and this computer must be on the same
subnet for UDP discovery, and the firewall must permit the Wuji SDK's discovery
and data traffic. See the [Wuji glove documentation](https://docs.wuji.tech/docs/en/wuji-glove/latest/)
for device network configuration and calibration.

Being connected to the same router or Ethernet switch is not sufficient when
the assigned IPv4 networks differ. Confirm both addresses and the selected
route:

```bash
ip -br -4 address
ip route get <glove-ip>
```

For example, a host at `192.168.50.13/23` and a glove at
`192.168.1.101/24` are in different subnets. Configure the glove with an unused
address in `192.168.50.0/23` (preferred when that is the router LAN), configure
inter-subnet routing on the router, or add an appropriate secondary address to
the Ethernet interface. Also make sure a VPN route does not capture the glove
address; `ip route get` must show the physical Ethernet interface.

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

`hand_joint_angles` describes finger articulation, whereas the FR3 controller
expects seven arm joints. The default indices are only a transport smoke-test
mapping. Before enabling a physical FR3, explicitly calibrate and validate
`--joint-indices`, `--joint-signs`, `--joint-offsets`, `--joint-min`, and
`--joint-max` for the intended glove-to-arm convention. Test the resulting ROS
topic while the robot controller is stopped:

```bash
ros2 topic echo /left/gello/joint_states
```

For two gloves, run one process per namespace (`left` and `right`) and set a
different `--device-id` if the SDK supports device selection. Keep the robot
in a safe pose while starting; the node holds the first received target and
applies a per-cycle step limit. `Ctrl-C` closes the SDK device cleanly.

Use `--stdin` to test the mapping without hardware. Each input line is seven
space-separated joint angles followed by a normalized gripper value.
