# Calibration Data Collection

`collect_camera_world_data.py` records the inputs needed for a later
camera-to-world AprilTag pose solve. It does not estimate or apply any
transform.

On the machine with the D435 connected and the RealSense/OpenCV dependencies
installed:

```bash
python calibration/collect_camera_world_data.py \
  --output calibration/runs/table_tag_20260824 \
  --frames 100 \
  --fps 30
```

The output contains:

- `metadata.json`: device identity, stream configuration, color/depth factory
  intrinsics, depth scale, native depth-to-color extrinsics, and timestamps.
- `rgb/######.npy`: captured BGR RGB arrays. NumPy storage keeps this collector
  independent of OpenCV; a later processing script can decode or visualize them.
- `depth/######.npy`: uint16 depth frames aligned to the color stream.

Because depth is aligned to color, the saved depth pixels use the recorded
`color_intrinsics` for later RGB-frame back-projection. The native
`depth_intrinsics` and `depth_to_color` are retained for auditing.

AprilTag detection, tag-corner extraction, and the final `W_T_C` computation are
performed by a separate calibration script. The tabletop Tag center is the
world origin, but their axes differ. World axes are `+x` forward, `+y` left,
`+z` up; Tag axes are `+x` right, `+y` backward, `+z` down. The collector stores
the resulting fixed `world_T_tag` transform in `metadata.json`. It intentionally
does not depend on OpenCV and does not estimate camera transforms.

The stored Tag axes describe the physical printed marker: `+x` points toward
the marker's right edge, `+y` toward its bottom edge, and `+z` into the paper.
For the installed marker, OpenCV's decoded corner order makes the
`SOLVEPNP_IPPE_SQUARE` frame point `+x` toward the physical marker's left edge,
`+y` toward its bottom edge, and `+z` out of the paper. The processing script
therefore applies the required fixed 180 degree rotation about physical Tag
`+y` before composing the PnP pose with `world_T_tag`.

Run the separate processing step after installing OpenCV contrib and SciPy:

```bash
python calibration/calibrate_camera_to_world.py \
  --input calibration/runs/table_tag_20260824 \
  --tag-size-m 0.094 \
  --tag-family tag36h11 \
  --tag-id 0
```

This writes `camera_to_world.json` containing per-frame detections and the
aggregated `world_T_camera` estimate. The processing script reads `world_T_tag`
from the capture metadata. For older captures without that field, it uses the
fixed Tag/world convention above.

## Eye-to-hand data collection across two hosts

For camera-to-robot-base calibration, run one collector on the camera/data
server. The robot-control host only needs the normal teleoperation stack. Both
hosts must use the same ROS 2 domain and a DDS interface reachable from both
hosts.

On the camera/data server, where the D435 is connected:

```bash
python calibration/collect_camera_samples.py \
  --output calibration/runs/eye_to_hand_camera \
  --side right
```

The collector always enables the top `cam_front` D435 from the Gello camera
configuration: serial `401622071701`, at `640x480` RGB-D and 30 FPS. The serial
is hard-coded in the collector so other connected RealSense devices are not
selected accidentally.

Do not start the normal `franka_realsense_camera_publisher` for this same
device at the same time as the collector; two librealsense pipelines cannot
reliably own the same camera concurrently. The mapping comes from
`gello_software/ros2/src/franka_realsense_camera_publisher/config/example_three_cameras.yaml`:
`camera_3_name=cam_front`, `camera_3_serial=401622071701`, and
`camera_3_topic=/cameras/cam_front/image_raw`.

After the robot is held at a static pose, press Enter in the camera-host
terminal. The camera collector reads the latest robot state received over ROS
2 from `/<side>/franka_robot_state_broadcaster/robot_state`, captures one local
RGB-D frame, and writes both to the same local `sample_######` directory. It
also records the latest joint state from `/<side>/franka/joint_states`. No
collector process is needed on the robot-control host beyond
`scripts/start_teleoperation.sh`.

The collector intentionally does not detect AprilTags or solve any transforms.

### AprilTag frame orientation

Tag orientation is part of the calibration contract. For the tabletop Tag, its
center is the world origin. Viewed on the table, Tag `+x` points right, Tag `+y`
points backward, and Tag `+z` points down. These map to world `-y`, `-x`, and
`-z`, respectively. Rotating, flipping, or mirroring the tabletop Tag changes
the resulting `world_T_camera` frame even if PnP reports a small reprojection
error.

For the end-effector Tag, its absolute orientation does not need to match the
robot axes because the hand-eye solve estimates the fixed `ee_T_tag` transform.
It must, however, remain rigidly mounted and use one consistent right-handed
Tag frame for every sample. The processing scripts use the AprilTag corner
order top-left, top-right, bottom-right, bottom-left with Tag-frame corners
`(-s/2,+s/2)`, `(+s/2,+s/2)`, `(+s/2,-s/2)`, `(-s/2,-s/2)` at `z=0`.

### RoboDex nominal EE-to-Wuji transform

The nominal Wuji mounting transform was read from:

`data/githubRepo/RoboDex/task/assets/urdf/panda_wuji_hand_right.urdf`

Its fixed chain is:

```text
panda_link8 --panda_to_wuji_docking_joint--> hand_docking_link
hand_docking_link --wuji_docking_to_palm_joint--> right_palm_link
```

The first joint has `xyz="0 0 0"` and
`rpy="0 0 2.3561944902"`; the second joint is identity. Thus:

```text
panda_link8_T_hand_docking =
[[-0.70710678, -0.70710678, 0.0, 0.0],
 [ 0.70710678, -0.70710678, 0.0, 0.0],
 [ 0.0,         0.0,        1.0, 0.0],
 [ 0.0,         0.0,        0.0, 1.0]]

panda_link8_T_right_palm = panda_link8_T_hand_docking
```

`hand_docking_link` is the root link of the Wuji hand-only URDF used by the
retargeting adapter, and `right_palm_link` is the palm landmark link. This is
only the nominal CAD transform; verify it against the physical mount and the
real FR3 `fr3_link8`/flange/controller EE frame. The end-effector AprilTag
hand-eye calibration remains the source of truth for the installed transform.

For the current grasp experiment, `grasp/ee_to_wuji_nominal.json` deliberately
does not use this CAD rotation. It temporarily sets `ee_T_hand` to the 4x4
identity matrix, treating the Wuji wrist/hand-root frame and controller EE frame
as coincident. Replace that temporary assumption after measuring the physical
mount transform.

## Selecting samples from a LeRobot recording

Recordings made by `scripts/start_data_collection_client.sh` and
`scripts/start_data_collection_server.sh` store synchronized robot poses in
Parquet and `cam_front` images in a video. Use the interactive selector to
choose exactly 20 frames and convert them to the `sample_######` format expected
by `calibrate_camera_to_robot_base.py`.

The selector contains the live D435 Color 640x480 intrinsics for `cam_front`
serial `401622071701`: `fx=606.1522`, `fy=605.6415`, `cx=322.8838`, and
`cy=255.9408`, with zero reported distortion coefficients. These are the same
profile values recorded by the camera-to-world collector and grasp inference.
It checks that the `cam_front` video is exactly 640x480 before exporting and
writes both the serial number and intrinsics into each sample and the selection
manifest. If the camera profile, resolution, crop, or resize changes, update
the intrinsics before exporting samples.

`calibration/runs/D435内参表.txt` is retained as a historical device/profile
dump, but its 640x480 color entry is not the active profile used for these
calibration images and must not be copied into the sample metadata.

Run the selector once for each arm:

```bash
python calibration/select_camera_calibration_samples.py \
  --input calibration/runs/left_arm_camera_calibration \
  --output calibration/runs/left_arm_camera_calibration_samples

python calibration/select_camera_calibration_samples.py \
  --input calibration/runs/right_arm_camera_calibration \
  --output calibration/runs/right_arm_camera_calibration_samples
```

The selector additionally requires `pyarrow`, Pillow, Tkinter, and the
`ffmpeg`/`ffprobe` executables. These are only needed to read the LeRobot
Parquet/video recording and display the selection interface; the calibration
solver's dependencies remain unchanged.

Use the timeline or arrow keys to locate a pose, then click **Add frame** (or
press Space). The interface enables **Export samples** after exactly 20 distinct
frames have been selected. Each exported `sample.json` records the source
episode, frame index, timestamp, camera serial, built-in camera intrinsics, and
the synchronized `B_T_E` converted from `observation.ee_pose`.

After selection, run the existing calibration solver manually:

```bash
python calibration/calibrate_camera_to_robot_base.py \
  --input calibration/runs/left_arm_camera_calibration_samples \
  --tag-size-m 0.037 \
  --tag-family tag36h11 \
  --tag-id 2

python calibration/calibrate_camera_to_robot_base.py \
  --input calibration/runs/right_arm_camera_calibration_samples \
  --tag-size-m 0.037 \
  --tag-family tag36h11 \
  --tag-id 1 \
  --exclude-sample 000014
```

The right-arm command explicitly excludes `sample_000014`, whose planar PnP
solution is an outlier. `--exclude-sample` may be repeated when another dataset
has multiple independently identified outliers; the excluded IDs are stored in
the output JSON.

The processing script reads the measured `B_T_E` stored in each sample,
detects the end-effector Tag, and jointly estimates:

```text
camera_T_base
base_T_camera
ee_T_tag
```

It also writes per-sample Tag reprojection errors and hand-eye translation/
rotation residuals to `camera_to_robot_base.json`. It requires
`opencv-contrib-python`, SciPy, and NumPy, but does not require a live robot or
camera.

## Processing-script dependencies

The two offline computation scripts share the same processing environment:

| Dependency | Purpose | Required by |
| --- | --- | --- |
| Python 3.10+ | Script runtime | Both scripts |
| NumPy | RGB array loading, matrices, point/pose data | Both scripts |
| `opencv-contrib-python` | AprilTag dictionaries, detection, `solvePnP`, Rodrigues, reprojection | Both scripts |
| SciPy | SO(3) rotation averaging and rotation-vector conversions | Both scripts |

Use the contrib OpenCV package, not the minimal `opencv-python` package:

```bash
python -m pip install numpy scipy opencv-contrib-python
```

No ROS 2, `rclpy`, `franka_msgs`, `pyrealsense2`, Gello, or live robot/camera
connection is required by either processing script. Those dependencies are only
needed by `collect_camera_samples.py`, which runs on the camera host.

## Moving the right arm from a world-frame EE pose

`move_right_ee_from_world.sh` converts a world-frame controller-EE target to
the right FR3 base frame with the matrices in `matrix.md`:

```text
W_T_B_R = W_T_C @ C_T_B_R
B_R_T_E = inverse(W_T_B_R) @ W_T_E
```

Both input and output use `x y z roll pitch yaw` in metres/radians, with the
same `Rz(yaw) @ Ry(pitch) @ Rx(roll)` convention as
`scripts/move_to_target_ee.sh`. The launcher performs the NumPy conversion in
the Conda environment selected by `--conda-env`, `CALIBRATION_CONDA_ENV`, or
`LEROBOT_CONDA_ENV` (default `lerobot`). It then delegates ROS control to
`scripts/move_to_target_ee.sh`, which uses system `/usr/bin/python3`; Conda and
ROS Python dependencies therefore run in separate processes.

First inspect a conversion without starting the controller:

```bash
./calibration/move_right_ee_from_world.sh \
  --world-ee-pose 0.20 -0.10 0.30 3.14159 0 0
```

Then request the controller's live-state/IK dry run (no movement):

```bash
./calibration/move_right_ee_from_world.sh \
  --world-ee-pose 0.20 -0.10 0.30 3.14159 0 0 \
  --controller-dry-run
```

Use `--execute` only when the converted target and dry run have been checked.
It calls `scripts/move_to_target_ee.sh --right --arm`; the existing controller
utility still prints current/target state and requires an explicit `y`/`yes`
before commanding real motion.

The input formats differ:

- `calibrate_camera_to_world.py` reads the older sequence collector format:
  one `metadata.json` containing a `frames` list, with each frame pointing to
  an RGB `.npy` file.
- `calibrate_camera_to_robot_base.py` reads the current eye-to-hand format:
  `sample_######/sample.json` plus `sample_######/rgb.npy`, including the
  measured `B_T_E` matrix in each sample. The interactive selector above
  converts LeRobot recordings into this format.
