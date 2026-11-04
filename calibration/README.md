# Camera and robot calibration

Calibrate the front D435i and both wrist D405 cameras against the **same fixed
95 mm tag36h11 AprilTag, ID 0**. Run commands from the repository root.

`A_T_B` maps coordinates from B into A, with translation in metres. World W
is centred on the tag with +X forward, +Y left, +Z up. The physical tag axes
are different; the scripts save their relationship as `world_T_tag`.
Camera coordinates are color optical frames (+X right, +Y down, +Z forward).

## 1. Prepare the camera computer

Keep the tag and camera mounts fixed throughout calibration. Stop other camera
applications, including `start_data_collection_server.sh`; the calibration
scripts open the cameras directly. Both computers need synchronized system
clocks and the same ROS domain.

```bash
source /opt/ros/jazzy/setup.bash
source /home/pair1/franka_ros2_ws/install/local_setup.bash
source gello_software/ros2/install/local_setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET

calib_run="calibration/runs/calibration_$(date +%Y%m%d_%H%M%S)"
```

Use this camera terminal for the remaining steps so `$calib_run` stays set.
The installed `/usr/bin/python3` has the recording and solver dependencies.
Use `/usr/bin/python3` even when a Conda environment is active. In a new
terminal, set `calib_run` again to your chosen run directory before continuing.
All captures and solver JSON files go under the ignored `calibration/runs/`.

## 2. Calibrate the front D435i to world

Make the whole tag visible in the front camera and hold everything still.
On the camera computer:

```bash
/usr/bin/python3 calibration/collect_camera_world_data.py \
  --camera-serial 401622071701 --output "${calib_run:?Set calib_run in step 1}/front" --frames 100

/usr/bin/python3 calibration/calibrate_camera_to_world.py \
  --input "${calib_run:?Set calib_run in step 1}/front" \
  --tag-size-m 0.095 --tag-family tag36h11 --tag-id 0
```

Result: `$calib_run/front/camera_to_world.json`, field `world_T_camera`.
Check that many detections agree and median reprojection RMSE is below about
0.5 pixels. If needed, the camera pose in the physical AprilTag frame is
`tag_T_camera = inverse(world_T_tag) @ world_T_camera`.

## 3. Record the left wrist camera and measured EE poses

On the **robot-control computer**, start left-arm teleoperation:

```bash
./scripts/start_data_collection_client.sh --left --arm
```

On the **camera computer**:

```bash
/usr/bin/python3 calibration/record_wrist_calibration.py \
  --side left --fps 5 --output "${calib_run:?Set calib_run in step 1}/left"
```

Wait for saved frame counts and left-arm `matched` counts to increase. Move
through **15–20 distinct poses**, keeping the entire tag visible to the wrist
camera and pausing **three seconds** at each pose (at least eight images at
5 FPS). Include translations and
roughly 15–30 degree rotations about at least two axes. Press **Ctrl-C** to
finish; there is no per-pose keypress.

The recorder defaults to **1280x720 at 5 FPS**, opening only the selected arm's
wrist camera to reduce USB load. The front camera was calibrated in step 2.
Use `--cameras left front` to add front video, or `--side both` for both wrists;
each selected camera must support the requested profile. A lower recording rate
keeps image resolution but does not itself guarantee sharp, well-exposed images.

It saves color video, camera intrinsics, source timestamps,
and available measured EE states. It pairs each image with interpolated measured
poses; unmatched images are excluded during sample preparation. Keep the
controller's EE definition (`F_T_E`) fixed throughout the recording.
If a camera times out, stop other users of that device and check its USB
connection. Restart with a new output directory; failed recordings are preserved.

## 4. Solve the left arm base and wrist camera mount

On the camera computer:

```bash
/usr/bin/python3 calibration/prepare_wrist_recording.py \
  --input "${calib_run:?Set calib_run in step 1}/left" --side left

/usr/bin/python3 calibration/calibrate_wrist_camera.py \
  --input "${calib_run:?Set calib_run in step 1}/left/samples_left" \
  --front-calibration "${calib_run:?Set calib_run in step 1}/front/camera_to_world.json"
```

Result: `$calib_run/left/samples_left/wrist_calibration.json`, containing
`world_T_base` and the fixed mount `ee_T_camera`. The solver selects clear tag
observations and reserves three poses for validation. Accept results only when
`status` is `validated_on_holdout`; collect more varied, clear poses if it fails.

## 5. Repeat for the right arm

Stop the left teleoperation session and start the client with `--right --arm`.
Repeat steps 3–4, replacing `--side left` with `--side right`,
`$calib_run/left` with `$calib_run/right`, and `samples_left` with `samples_right`.
Keep the same tag and front-camera calibration.

## 6. Store the validated matrices in [matrix.md](matrix.md)

Copy the latest results using these names:

| Matrix in `matrix.md` | Source |
| --- | --- |
| `W_T_D435I` | Front JSON: `world_T_camera` |
| `TAG_T_D435I` (if needed) | `inverse(world_T_tag) @ W_T_D435I` |
| `W_T_BASE_LEFT` | Left wrist JSON: `world_T_base` |
| `W_T_BASE_RIGHT` | Right wrist JSON: `world_T_base` |
| `EE_LEFT_T_D405_LEFT` | Left wrist JSON: `ee_T_camera` |
| `EE_RIGHT_T_D405_RIGHT` | Right wrist JSON: `ee_T_camera` |

The wrist camera pose **in its arm base changes as the arm moves**. Store the
fixed EE-to-camera mount above and use the measured `B_T_E(q)` to compute:

```text
BASE_LEFT_T_D405_LEFT(q)   = B_LEFT_T_EE_LEFT(q)   @ EE_LEFT_T_D405_LEFT
BASE_RIGHT_T_D405_RIGHT(q) = B_RIGHT_T_EE_RIGHT(q) @ EE_RIGHT_T_D405_RIGHT
```

If storing a camera-to-base snapshot, include its measured EE pose and timestamp.
For front-camera points in either robot base, use
`base_T_front = inverse(world_T_base) @ W_T_D435I`.

Keep only the latest calibrated results and a short dated record in `matrix.md`:
calibration time, tag ID/size, camera serials, source JSON paths, and validation
summary. Replace superseded snapshots; scripts save JSON but do not edit
`matrix.md` automatically. Recalibrate after moving a camera mount or robot
base; moving the tag changes the shared world reference for all results.
