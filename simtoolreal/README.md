# SimToolReal deployment

Run the following commands from the repository root. The server hosts the ROS
bridge, FoundationPose++, and policy; the client controls the FR3 and Wuji Hand.
Keep each command running while starting the next one.

## 1. Start the server

On the server computer:

```bash
./simtoolreal/scripts/start_server.sh \
  --config libs/SimToolReal-Franka-Wuji2/pretrained_policy/config.yaml \
  --checkpoint libs/SimToolReal-Franka-Wuji2/pretrained_policy/model.pth \
  --foundationpose-mesh ./libs/FoundationPose-plus-plus/test/mesh/hammer.stl \
  # --foundationpose-roi 220 120 160 180 \
  # --foundationpose-no-display \
  --device cuda
```

Important arguments:

- `--config` and `--checkpoint` select the policy configuration and model. If
  omitted, the launcher uses the bundled files under
  `libs/SimToolReal-Franka-Wuji2/pretrained_policy/`.
- `--foundationpose-mesh PATH` starts live FoundationPose++ tracking with the
  specified object mesh. A pose source must be selected.
- `--foundationpose-roi X Y W H` supplies the initial image-space object box.
  This is useful, and normally required, for headless registration.
- `--foundationpose-no-display` disables the tracking display.
- `--device cpu|cuda` selects the device used for policy inference. The default
  is `cpu`.
- `--policy-python PATH` and `--pose-python PATH` select Python executables for
  the policy and FoundationPose++ environments.
- `--no-bridge` prevents this script from starting the ROS bridge when it is
  already running separately.

For a precomputed pose instead of live tracking, replace the FoundationPose++
mesh/ROI arguments with `--foundationpose-pose-file /path/to/pose.json`.

## 2. Start the client

On the client (robot) computer, in a separate terminal:

```bash
./simtoolreal/scripts/start_client.sh \
  --server-ip 192.168.50.13 \
  --right-hand-ip HAND_IP
```

Important arguments:

- `--server-ip SERVER_IP` is the server computer's reachable IP address. It
  defaults to `192.168.50.13`.
- `--right-hand-ip HAND_IP` selects the right Wuji Hand. If omitted, the hand
  SDK attempts device discovery.
- `--robot-config FILE` selects the FR3 controller YAML from
  `gello_software/ros2/src/franka_fr3_arm_controllers/config/`.
- `--ros-domain-id ID` must match the ROS 2 domain used by the server. The
  default is `0`.
- `--ros-distro DISTRO` selects the ROS installation, for example `humble`,
  when it cannot be detected automatically.

Leave this launcher running. It owns the robot controller and hand worker.

## 3. Run the policy executor on the client

In another terminal on the client computer:

```bash
python3 simtoolreal/policy_executor.py \
  --server-ip SERVER_IP \
  --goal-pose /path/to/goal_pose.json \
  --world-from-camera /path/to/world_from_camera.json \
  --execute
```

Important arguments:

- `--server-ip SERVER_IP` selects the server endpoints for robot state,
  FoundationPose++ poses, policy requests, and arm commands.
- `--goal-pose TRANSFORM` is required and gives the desired object pose in the
  policy-world frame.
- `--world-from-camera TRANSFORM` is the calibrated camera-to-policy-world
  transform. It is required for the default `--pose-frame camera` mode.
- `--world-from-robot TRANSFORM` overrides the robot-root calibration. If it is
  omitted, the bundled slanted-training transform is used.
- `--object-scales X,Y,Z` sets the object scale on each axis; the default is
  `1,1,1`.
- `--rate HZ` sets the policy loop rate; the default is `60` Hz.
- `--execute` enables commands to the physical arm and hand. Without it, the
  executor runs in dry-run mode and does not command the hardware.

Each transform argument accepts either a file containing a 4-by-4 transform
(JSON is supported) or 16 comma-separated values. Use measured calibration
transforms; do not infer them from network or ROS settings. Start without
`--execute` to verify state, pose, policy, and calibration inputs before
enabling hardware commands. Stop any of the processes with `Ctrl-C`.
