# Data Collection

## Overview

- `lerobot_collection.py`: Minimal script for recording synchronized RealSense images and robot state/action data into a LeRobot dataset.
- `replay_lerobot_episode.py`: Replay a recorded LeRobot episode through the same ROS 2 arm and gripper controller topics used during collection.
- `reset_pylibfranka.py`: Reset both Franka arms to a selected dataset `observation.state` frame or a measured hardware-specific fallback pose.
- `delete_lerobot_episode.py`: Remove one or more episodes from a local LeRobot dataset while preserving the remaining metadata, videos, and parquet data.
- `process_dataset.py`: Process local LeRobot datasets, including automatic initial-static-segment trimming.
- `validate_dataset.py`: Validate local LeRobot dataset metadata, parquet data, semantics, and videos.

Quick links:

- [GELLO docs](../gello_software/README.md)
- [FR3 ROS 2 docs](../gello_software/ros2/README.md)
- [LeRobot docs](../lerobot/README.md)
- [Wuji glove teleoperation](wuji/README.md)

Environment split used in this repo:

- Use `/usr/bin/python3` (Python 3.10 on this machine) for ROS 2 Humble, GELLO helper scripts, `colcon build`, and ROS 2 episode replay. Install `numpy` and `pyarrow` in this environment because replay reads LeRobot Parquet files.
- Use the `lerobot` Conda environment for collection, dataset processing, and training.


## Before launching

Install the GELLO USB aliases once on the robot-control host:

```bash
sudo ./scripts/setup_usb_rules.sh
```

The aliases depend on the host USB topology. Verify `/dev/ttyUSB_left` and
`/dev/ttyUSB_right` before launching the ROS 2 nodes after reconnecting devices.
The active user must belong to the `dialout` group to access the devices without
root privileges.

Test the offsets after reconnecting a GELLO device:

```bash
source /opt/ros/humble/setup.sh
source ~/franka_ros2_ws/install/setup.bash
source ~/real-exp/gello_software/ros2/install/setup.bash

# start gello publisher
ros2 launch franka_gello_state_publisher main.launch.py \
config_file:=gello_duo.yaml

# Robot-accepted 15 Hz waypoints recorded as dataset actions
ros2 topic echo /left/gello/accepted_joint_states
ros2 topic echo /right/gello/accepted_joint_states

# The controller's generated reference is available for diagnostics here
ros2 topic echo /left/franka/commanded_joint_states
ros2 topic echo /right/franka/commanded_joint_states
```

Compare the results with the joint angles reported by
`172.16.0.2/desk/api/robot/robot-state` and
`172.16.0.3/desk/api/robot/robot-state`.

Set the offset if necessary:

```bash
cd ~/real-exp/gello_software
python3 scripts/setup_offset.py --start-joints 0 0 0 -1.57 0 1.57 0 --joint-signs 1 -1 1 1 1 -1 1 --port /dev/ttyUSB_left
python3 scripts/setup_offset.py --start-joints 0 0 0 -1.57 0 1.57 0 --joint-signs 1 -1 1 1 1 -1 1 --port /dev/ttyUSB_right
```

- Unlock the Franka arms and activate FCI.

Build the ROS 2 workspace.

Skip this if nothing under `gello_software/ros2/` changed since the last build.

```bash
cd gello_software/ros2
colcon build
source install/setup.bash
```

If you open a new shell after building, run `source install/setup.bash` again before using `ros2 launch`.

## Robot Reset And Replay

Use the direct `pylibfranka` reset script when you want to return both robots to a dataset start pose before recording, replay, or deployment.

When a matching dataset is available, prefer selecting its `observation.state` frame. The script validates the expected 16-dimensional layout:

- left arm joint positions
- left gripper width
- right arm joint positions
- right gripper width

If `--dataset-root` is omitted, the script uses the measured `INITIAL_STATE` stored inside `data_collection/reset_pylibfranka.py`. This fallback is specific to the aligned hardware setup and may not be safe for another robot or GELLO calibration.

Preview the fallback target without moving the robots:

```bash
python3 data_collection/reset_pylibfranka.py --dry-run
```

Reset both arms and grippers to the fallback target:

```bash
python3 data_collection/reset_pylibfranka.py
```

To reset to the actual initial `observation.state` from a dataset episode, pass `--dataset-root`, `--episode`, and optionally `--frame-index`.

Dataset gripper values are physical widths in metres. The `[0, 1]` clamp used for normalized continuous gripper commands during collection and replay does not apply to these reset widths.

Preview dataset episode 0, frame 0:

```bash
python3 data_collection/reset_pylibfranka.py \
  --dataset-root data/pick_and_place_test \
  --episode 0 \
  --frame-index 0 \
  --dry-run
```

Reset to that dataset frame:

```bash
python3 data_collection/reset_pylibfranka.py \
  --dataset-root data/pick_and_place_test \
  --episode 0 \
  --frame-index 0
```

For replay, use the replay supervisor from the repository root. It reads
`meta/real_exp_trajectory_config.json`, validates the requested current robot
setting, and refuses a mismatch before starting replay. When no setting flags
are supplied, it uses the recorded setting. It starts the required FR3
controllers and, for gripper or Wuji-hand trajectories, the matching
end-effector processes,
waits for the FR3 controller nodes and arm state topics, starts the gripper
managers, waits for both gripper-client nodes and gripper state topics, and only
waits for each gripper client's command subscription (after homing and client
initialization), and only then runs the episode replay:

```bash
bash scripts/replay.sh \
  --dataset-root data/test-pick-and-place-new \
  --episode 0
```

Select the stored arm representation with `--replay-mode joint` or
`--replay-mode ee`. The replay supervisor passes this option to
`replay_lerobot_episode.py`:

```bash
# Replay the stored joint targets directly.
bash scripts/replay.sh \
  --dataset-root data/test-pick-and-place-new \
  --episode 0 \
  --replay-mode joint

# Apply the stored delta EE poses to the live EE poses, solve IK for the
# resulting targets, and send the solved target joint angles to the robots.
bash scripts/replay.sh \
  --dataset-root data/test-pick-and-place-new \
  --episode 0 \
  --replay-mode ee
```

Joint replay reads `observation.joint_state` and `action.target_joint`. EE
replay reads `observation.ee_pose` and `action.delta_ee_pose`. When
`--replay-mode` is omitted, replay defaults to the dataset's compatibility
`state_action_mode`. A requested mode fails validation if its required fields
are absent, while a dataset containing both representations can be replayed in
either mode.

Explicit settings are useful as a hardware safety check:

```bash
bash scripts/replay.sh \
  --dataset-root data/test-pick-and-place-new --episode 0 \
  --gripper --duo
```

The command exits before starting controllers if the trajectory was recorded
with a different end-effector or arm selection. A gripper trajectory therefore
cannot be replayed with `--arm`, and a Wuji-hand trajectory cannot be replayed
without its hand workers connecting successfully.

After you enter `s`, replay first commands both arms to the first selected
`observation.state` and waits for them to settle. The replay clock and recorded
action sequence start only after both arms are within the required position and
velocity tolerances. The initial move is ramped from the measured pose with
`0.10 rad/s` velocity and `0.20 rad/s^2` acceleration limits by default. Use
`--initial-state-max-velocity`, `--initial-state-max-acceleration`, and
`--initial-state-timeout` to tune the move. The default initial-state position
acceptance tolerance is `0.06 rad` per joint; override it with
`--initial-state-position-tolerance` if needed.

The supervisor forwards replay options such as `--fps`, `--start-frame`,
`--end-frame`, `--max-frames`, `--output`, and `--dry-run`. Use `--no-gripper`
to skip the Franka-hand managers and gripper commands:

```bash
bash scripts/replay.sh \
  --dataset-root data/test-pick-and-place-new \
  --episode 0 \
  --no-gripper
```

With `--dry-run`, the supervisor skips all ROS and robot startup and only
prints the episode summary. During a normal replay, it waits for each gripper
client's command subscription before starting the replay process. The replay
process then advertises `/left/gello/raw_joint_states` and
`/right/gello/raw_joint_states` while it waits for you to press `s`. Replay
actions are sampled absolute targets; the robot-side impedance controller
generates the constrained 1 kHz reference from them.

The supervisor preserves the terminal for this prompt even though the ROS
launches run in separate process groups. Press `s` and Enter to begin, or `q`
and Enter to abort.

Replay waits for all required actual state topics before accepting `s`. If you
intentionally need to begin without those samples, pass
`--allow-missing-state`; the trace will then mark unavailable actual values and
`controller_ready` accordingly.

## Teleoperation Quick Start

Run the teleoperation supervisor from the repository root. It sources the ROS 2
and Franka workspaces, validates the required GELLO USB aliases, starts and
verifies the matching FR3 controller before starting GELLO (the controller owns
the accepted-target topic), and optionally starts gripper control.

Dual-arm teleoperation with Franka gripper control:

```bash
cd ~/real-exp
./scripts/start_teleoperation.sh --duo --gripper
```

Dual-arm teleoperation with Wuji hands:

```bash
cd ~/real-exp
./scripts/start_teleoperation.sh --duo --hand \
  --left-glove-sn <LEFT_SN> --right-glove-sn <RIGHT_SN> \
  --left-hand-ip <LEFT_IP:PORT> --right-hand-ip <RIGHT_IP:PORT>
```

Left-arm teleoperation with Franka gripper control:

```bash
cd ~/real-exp
./scripts/start_teleoperation.sh --left --gripper
```

Right-arm teleoperation with its Wuji glove and hand:

```bash
cd ~/real-exp
./scripts/start_teleoperation.sh --right --hand
```

The argument order is interchangeable. Run the following command for the full
usage summary:

```bash
./scripts/start_teleoperation.sh --help
```

Press `Ctrl-C` once to stop the complete teleoperation stack. If any managed
process fails, the script stops the remaining processes and returns the failing
process's status. `--gripper` selects matching Franka grippers; `--hand` starts
the matching Wuji glove-to-Wuji Hand 2 process instead. Run
Use `start_teleoperation.sh --arm --left` (or `--right`/`--duo`) when no end-effector controller is wanted.

### Configuration Notes

- GELLO publisher configs live in `gello_software/ros2/src/franka_gello_state_publisher/config/`.
- FR3 controller configs live in `gello_software/ros2/src/franka_fr3_arm_controllers/config/`.
- `--duo` selects calibrated `gello_duo.yaml` and `example_fr3_duo_config.yaml`.
- `--left` selects calibrated `gello_single.yaml` and `example_fr3_config.yaml`.
- `--right` selects `gello_right.yaml` and `example_fr3_right_config.yaml`.
- `--gripper` selects the matching `example_fr3*_config_franka_hand.yaml` file.
- `--hand` uses an FR3 config with `load_gripper: false`, so the Franka gripper
  hardware node is not loaded alongside the Wuji Hand 2 process.
- `gello_duo.yaml` defines the calibrated left and right GELLO devices for bimanual control.
- `example_fr3_duo_config.yaml` defines the corresponding left and right FR3 robot IPs and namespaces.
- If you are switching between single-arm and dual-arm setups, make sure the publisher and controller configs match.

## Data Collection

The recording path has three components:

- A ROS 2 camera publisher in `gello_software/ros2/src/franka_realsense_camera_publisher/` that publishes RGB images from up to three RealSense cameras.
- A ROS 2 bridge node in `gello_software/ros2/src/franka_lerobot_data_bridge/` that subscribes to robot, teleop, gripper, and camera topics and publishes synchronized samples over ZMQ.
- `lerobot_collection.py`, which subscribes to that sample stream and writes a local LeRobot dataset.

### Split control/data-server setup

The recommended layout puts the USB cameras and all recording work on the data
server (`192.168.50.13`), while the robot-control computer runs only GELLO,
Franka, and gripper teleoperation:

```text
control host                         data server (192.168.50.13)
GELLO + FR3 + gripper  --ROS 2-->    RealSense publisher + bridge
                                     bridge ZMQ PUB :5555 --TCP--> recorder
```

ROS 2 must be reachable between both computers. Configure the same
`ROS_DOMAIN_ID` and the same RMW implementation on both hosts, and make sure
DDS discovery/data traffic is allowed by the firewall. For Cyclone DDS, use a
network interface reachable from both machines; localhost-only DDS settings
will prevent the server bridge from seeing robot and action topics.

On the control host, start teleoperation:

```bash
./scripts/start_data_collection_client.sh --duo --gripper
```

On `192.168.50.13`, with the cameras connected there, source the ROS workspace
and start the camera publisher and bridge:

```bash
./scripts/start_data_collection_server.sh --duo --gripper
```

The server supervisor binds the bridge to `192.168.50.13:5555`. To use a
different interface or port, pass `--bridge-host` and `--bridge-port`. The
single-arm variant is:

```bash
./scripts/start_data_collection_server.sh --single --gripper
```

Start the recorder on the server (in the `lerobot` environment) and leave its
host set to the server address:

```bash
conda activate lerobot
python data_collection/lerobot_collection.py \
  --host 192.168.50.13 --port 5555 \
  --local-dir ./lerobot_data
```

You can have the server supervisor start it as well:

```bash
./scripts/start_data_collection_server.sh --duo --record \
  --gripper \
  --local-dir ./lerobot_data
```

Do not start `franka_realsense_camera_publisher` or the LeRobot bridge on the
control host; it has no camera USB devices in this layout. The server bridge
still subscribes to `/left/...` and `/right/...` robot/action topics over ROS 2
and to `/cameras/...` image topics locally.

The dataset records all representation-neutral robot fields on every frame:

- `observation.joint_state`: measured robot joints, plus gripper/hand state when enabled
- `action.target_joint`: accepted/commanded joint targets, plus gripper/hand targets when enabled
- `observation.ee_pose`: measured end-effector pose (`x,y,z,roll,pitch,yaw` per arm)
- `action.delta_ee_pose`: target EE pose minus measured EE pose
- `observation.state` and `action`: the bridge's selected compatibility/training view
- `observation.images.<camera_name>`: enabled RGB video streams (`cam_left`, `cam_front`, and `cam_right` for duo; `cam_left` and `cam_front` for single-arm collection)

The bridge expects:

- Robot joint states on a topic like `/left/franka/joint_states`
- Robot-accepted 15 Hz action targets on a topic like `/left/gello/accepted_joint_states`
- Generated controller references on a topic like `/left/franka/commanded_joint_states`
- Robot gripper joint states on a topic like `/left/franka_gripper/joint_states`
- Gripper commands on a topic like `/left/gripper/gripper_client/target_gripper_width_percent`
- RGB image topics for each camera enabled by the selected bridge configuration

By default the bridge publishes current measured robot joint states as `observation.state` and uses the robot-accepted target topic (`/left|right/gello/accepted_joint_states`) as the arm action source. The recorder labels each frame with the next packet's absolute arm joint target, so new datasets use `arm_action_representation=absolute_joint_position`.

The bridge also subscribes to each arm's `franka_robot_state_broadcaster/robot_state`
topic. Every recorded frame contains all four robot representations regardless of
the selected training view. Set the bridge YAML parameter
`state_action_mode: end_effector` to make the policy-facing vectors use
`observation.state = current end-effector pose` and
`action = target pose - current pose`; the default `joint` mode keeps
`observation.state = measured joints` and `action = absolute target joints`.

New datasets use continuous normalized gripper commands with
`gripper_action_representation=absolute_width`. The recorder preserves values
between `0` and `1` and clamps only out-of-range gripper values at serialization.
This normalized command is distinct from the physical gripper widths used by
`reset_pylibfranka.py`.

The recorder writes H.264/yuv420p video by default (`--video-codec h264`). This
is deliberate: the training pipeline uses TorchCodec's random frame access, and
H.264 is the project-supported codec for reliable random seeking. Dataset
trimming and episode deletion also re-encode edited video as H.264.

The control-host supervisor supports the same four arm/gripper combinations as
the teleoperation script:

```bash
./scripts/start_data_collection_client.sh --duo --gripper
./scripts/start_data_collection_client.sh --duo --no-gripper
./scripts/start_data_collection_client.sh --single --gripper
./scripts/start_data_collection_client.sh --single --no-gripper
```

In duo mode, the server bridge uses `example_duo.yaml` and records `cam_left`,
`cam_front`, and `cam_right`. In single mode, it uses `example_single.yaml` and
records `cam_left` and `cam_front`. Press `Ctrl-C` independently on each host to
stop its processes.

## Dataset Validation

After recording a dataset, process it before validation. First preview the initial
static-segment trimming plan:

```bash
python3 data_collection/process_dataset.py trim-initial \
  --dataset-root data/pick_and_place_test \
  --motion-threshold 0.002 \
  --min-static-frames 5 \
  --dry-run
```

Review the proposed trim counts, then remove `--dry-run` to create the processed
dataset:

```bash
python3 data_collection/process_dataset.py trim-initial \
  --dataset-root data/pick_and_place_test \
  --motion-threshold 0.002 \
  --min-static-frames 5
```

Detection uses the 14 arm joints in `observation.state` and ignores gripper
motion. It stops at the first frame whose maximum arm-joint displacement exceeds
the threshold and keeps that first moving frame. Use `--episode-indices 0,1,4-8`
to process selected episodes. Processing renames the original dataset to
`<dataset>_backup` and writes the processed dataset at the original path. The
command stops without changing either directory if that backup already exists.
The processed dataset's `meta/info.json` contains `"processed": true`.

After processing, validate that the processed metadata, parquet data, and videos
still agree.

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

- whether `meta/info.json` marks the dataset as processed
- `meta/info.json` totals against actual episode metadata and data rows
- continuous episode indices and global frame indices
- per-episode `length` against state/action row counts
- per-episode `frame_index` and timestamp continuity
- `observation.state` and `action` dimensions against `info.json`
- measured arm-state and accepted action-target position validity
- approximate measured-state motion from 15 Hz position finite differences
- accepted-waypoint slew as a command-distribution diagnostic
- video timestamp ranges against episode lengths
- strict MP4 decode, frame counts, resolution, and FPS
- TorchCodec random seeking at the first, middle, and final frame of every MP4

For `absolute_joint_position` datasets, consecutive actions are accepted 15 Hz
waypoints, not samples of the constrained controller reference. Their finite
differences therefore do not validate physical velocity or acceleration. The
validator reports them as waypoint-slew diagnostics. The robot-side controller
generates the constrained reference internally at 1 kHz. Measured-state motion
warnings are also approximate because the dataset contains 15 Hz positions,
not the controller's analytic reference derivatives or the robot's full-rate
velocity signal.

If `processed` is missing, false, or not the JSON boolean `true`, validation
prints a warning before the dataset summary and continues with the remaining
checks.

If OpenCV is not available in the active Python environment, either install it or skip physical video checks:

```bash
python3 data_collection/validate_dataset.py \
  --dataset-root data/pick_and_place_test \
  --skip-video-frames
```

Run validation from the `lerobot` Conda environment. The standard video checks
require TorchCodec; do not use `--skip-video-frames` as a pre-training check.

## Repairing Older AV1 Data

Some older AV1 recordings can pass a full sequential FFmpeg decode but fail
when TorchCodec seeks to a later frame during training. Re-encode such a dataset
to H.264 with a retained original backup:

```bash
conda activate lerobot
python data_collection/reencode_dataset_videos.py \
  --dataset-root data/pick_and_place_test
python data_collection/validate_dataset.py \
  --dataset-root data/pick_and_place_test
```

The command moves the original dataset to
`data/pick_and_place_test_av1_backup`, writes the H.264 replacement at the
original path, and restores the original automatically if conversion or the
TorchCodec preflight fails.

## Episode Deletion

Preview deletion of individual episodes and inclusive ranges:

```bash
python data_collection/delete_lerobot_episode.py \
  --dataset-root data/pick_and_place_test \
  --episode-indices 0,3-5,9 \
  --dry-run
```

Without `--in-place`, the command creates a sibling output dataset. In-place
deletion requires the explicit `--in-place` flag and keeps a backup of the
original dataset. The operation reindexes remaining episodes and rebuilds
Parquet data, video metadata, and dataset statistics.


## Additional Documentation

- General GELLO docs: [gello_software/README.md](../gello_software/README.md)
- Franka FR3 ROS 2 docs: [gello_software/ros2/README.md](../gello_software/ros2/README.md)
- LeRobot docs: [lerobot/README.md](../lerobot/README.md)

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
