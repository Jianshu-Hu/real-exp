# SimToolReal real-robot deployment

This runbook describes the tested two-computer deployment for one right FR3
and one right Wuji Hand 2. Run every command from the indicated computer and
keep it running while starting the next stage.

## Topology

```text
Server / GPU computer (192.168.50.13, ROS 2 Jazzy)
  FoundationPose++ pose PUB :5570
  policy RPC server         :5571
                 |
                 | ZMQ only (no cross-distro ROS/DDS)
                 v
Client / robot computer (ROS 2 Humble, ROS_DOMAIN_ID=73)
  local deployment bridge: state :5555, arm command :5556, hand telemetry :5558
  FR3 deployment controller -> FR3 at 172.16.0.2
  Wuji worker              -> Wuji Hand 2; local command socket :5562
  policy executor          -> reads local state, calls server, dry-runs or commands
```

Keep the ROS bridge on the Humble client. Do not run it on the Jazzy server:
mixing the two ROS distributions/`franka_msgs` builds over DDS has produced
CycloneDDS `serdata.cpp` errors (`invalid data size` and `string data is not
null-terminated`). Only ZMQ traffic should cross the two computers.

## Before every start

- Power the FR3 and release its brakes as required by the normal FR3 procedure.
- Power/plug in the Wuji hand and allow it time to boot.
- Clear the robot workspace and keep the emergency stop ready.
- Confirm that no older bridge, FR3 controller, Wuji worker, or executor is
  still running. Never start two FR3 controllers for the same robot.
- Confirm that the client can reach server `192.168.50.13`.

If a restart reports `Address already in use` on port `5555`, `5556`, `5557`,
`5558`, or `5562`, another client stack (or a recently closed TCP connection)
owns a fixed local endpoint. Stop the old executor first, then stop the old
client launcher/manual bridge/controller/Wuji terminals with `Ctrl-C`. Do not
solve this by changing the ports: all client components are configured to use
this fixed endpoint set. The current launcher holds a per-user single-instance
lock, checks every TCP state, waits for all bridge listeners before contacting
the FR3, and refuses to start when it detects an existing `ros2_control_node`.

The client also needs a dedicated Ethernet path to the right FR3. Before
starting, this command on `landau-Robotics` must report a `172.16.0.x` source
address and the Ethernet device physically connected to the robot:

```bash
ip route get 172.16.0.2
```

If it instead routes through the default `192.168.50.x` network, configure the
FR3-facing adapter with a free `172.16.0.x/24` address (never the robot's own
`172.16.0.2`). The launcher reports the bad route before opening any controller
connection; adapter names and the correct host address are installation-specific.

## Recommended startup: three terminals

### 1. Server terminal: FoundationPose++ and policy server

On the server (`/home/pair1/real-exp`)(conda env: pose):

```bash
cd /home/pair1/real-exp

./simtoolreal/scripts/start_server.sh \
  --config libs/SimToolReal-Franka-Wuji2/pretrained_policy/config.yaml \
  --checkpoint libs/SimToolReal-Franka-Wuji2/pretrained_policy/model.pth \
  --foundationpose-mesh libs/FoundationPose-plus-plus/test/mesh/hammer.stl \
  --device cuda \
  --no-bridge \
  --no-wait-only
```

Select the hammer ROI in the FoundationPose++ window and confirm it. A healthy
server prints that FoundationPose++ is registered and publishing live poses;
the projected 3D box should follow the real object. This one launcher owns both
FoundationPose++ and the policy server. If either child exits, it stops the
other child too.

### 2. Client terminal: local bridge, FR3 controller, and Wuji worker

First copy the current launcher to the client if the client checkout does not
yet contain the `--local-bridge` option:

```bash
scp \
  pair1@192.168.50.13:/home/pair1/real-exp/simtoolreal/scripts/start_client.sh \
  /home/landau/real-exp/simtoolreal/scripts/start_client.sh
```

Then, on the client (`landau-Robotics`):

```bash
cd /home/landau/real-exp

PATH="$HOME/anaconda3/bin:$PATH" \
./simtoolreal/scripts/start_client.sh \
  --server-ip 192.168.50.13 \
  --ros-distro humble \
  --ros-domain-id 73 \
  --local-bridge
```

If SDK discovery cannot find the powered Wuji hand, add its verified SDK
address (do not guess it):

```text
--right-hand-ip IP:PORT
```

Healthy output includes all of the following:

```text
Successfully connected to robot
Configured and activated joint_state_broadcaster
Configured and activated franka_robot_state_broadcaster
Deployment command bridge listening on tcp://0.0.0.0:5556
Publishing LeRobot samples over ZMQ on tcp://0.0.0.0:5555
Received first valid right hand telemetry packet (20 joints)
```

The controller `spawner` processes finishing cleanly is normal: spawners are
one-shot loader programs, while the controller remains inside
`ros2_control_node`.

### 3. Client terminal: policy executor, dry-run first

On the client:

```bash
cd /home/landau/real-exp

"$HOME/anaconda3/bin/conda" run \
  --no-capture-output \
  -n lerobot \
  env PYTHONPATH= LD_LIBRARY_PATH= \
  python simtoolreal/policy_executor.py \
    --server-ip 192.168.50.13 \
    --state-connect tcp://127.0.0.1:5555 \
    --arm-command-connect tcp://127.0.0.1:5556 \
    --robot-urdf /home/landau/real-exp/simtoolreal/assets/fr3v2_wuji_hand2_right_slanted.urdf \
    --goal-pose '1,0,0,0,0,1,0,0,0,0,1,0.6,0,0,0,1' \
    --world-from-camera '1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1'
```

The expected first line is:

```text
SimToolReal executor waiting for right-arm/Wuji state and pose; mode=DRY-RUN
```

Once the complete pipeline is healthy, it continuously prints:

```text
state[27]=[...] object_xyz=[...] target[27]=[...]
```

The explicit local addresses are intentional: state and arm commands connect
to the bridge on the client. Pose `:5570` and policy RPC `:5571` use
`--server-ip 192.168.50.13` and therefore connect to the server.

The transforms above are deterministic placeholders for pipeline testing:
`world_from_camera` is identity and `goal_pose` is identity rotation translated
to `(0, 0, 0.6)` metres. They have no valid physical calibration meaning.
**Never add `--execute` while using these placeholder transforms.**

## Manual client startup (diagnostics/fallback)

The recommended `--local-bridge` launcher replaces the following three client
terminals. Use these separate commands only when individual logs are needed.

### Client terminal A: local bridge

```bash
cd /home/landau/real-exp
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source "$HOME/franka_ros2_ws/install/local_setup.bash"
source "$HOME/real-exp/gello_software/ros2/install/local_setup.bash"
export ROS_DOMAIN_ID=73
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

ros2 launch franka_lerobot_data_bridge bridge.launch.py \
  config_file:="$HOME/real-exp/simtoolreal/config/deployment_right_hand.yaml"
```

### Client terminal B: FR3 controller

```bash
cd /home/landau/real-exp
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source "$HOME/franka_ros2_ws/install/local_setup.bash"
source "$HOME/real-exp/gello_software/ros2/install/local_setup.bash"
export ROS_DOMAIN_ID=73
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

ros2 launch franka_fr3_arm_controllers \
  franka_fr3_arm_controllers.launch.py \
  robot_config_file:=example_fr3_right_config_no_gripper.yaml \
  deployment_mode:=true
```

### Client terminal C: Wuji worker

```bash
cd /home/landau/real-exp

"$HOME/anaconda3/bin/conda" run \
  --no-capture-output \
  -n lerobot \
  env PYTHONPATH= LD_LIBRARY_PATH= \
  python deploy/wuji_hand_command_server.py \
    --side right \
    --hand-ip '' \
    --command-address tcp://127.0.0.1:5562 \
    --telemetry-address tcp://127.0.0.1:5558 \
    --telemetry-rate 60
```

The cleared `PYTHONPATH` and `LD_LIBRARY_PATH` prevent the Conda Python from
loading ROS Humble's Python 3.10 Pinocchio extension.

## Shutdown

1. Stop the executor first with `Ctrl-C`.
2. Stop the client hardware-stack launcher with `Ctrl-C`.
3. Stop the server launcher with `Ctrl-C`.

The executor should remain a separate process: this gives the operator a fast,
unambiguous way to stop policy inference/command generation without first
tearing down robot state and diagnostics.

## Enabling real motion

Do not enable real execution until all of these are true:

- a measured, validated `world_T_camera` replaces the identity placeholder;
- a safe measured goal pose replaces the placeholder goal;
- `world_from_robot` is confirmed for this physical installation;
- the FoundationPose++ projected box remains aligned with the object;
- fresh 27-D state and plausible 27-D targets stream continuously;
- all ZMQ endpoints are stable and there are no ROS/DDS errors;
- the workspace is clear and the emergency stop is immediately accessible.

Only after that validation should the executor be run with `--execute`. The
deployment bridge starts in ACTIVE mode and may publish hold commands as soon
as complete state becomes available, so treat the robot as live even during a
dry-run.
