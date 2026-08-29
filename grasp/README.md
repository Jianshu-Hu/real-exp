# Real-World Wuji Grasp Pipeline

This directory deploys the generator/retargeting/refinement path from
`execute_generator_grasp.py` across two hosts:

```text
inference host: D435 -> filtered world cloud -> generator -> retarget -> refine
                -> base-frame controller EE target -> ZeroMQ request
control host:   validate request -> scripts/move_to_target_ee.sh -> FR3 + Wuji
```

The transport uses JSON, not Python pickle. Both programs default to a dry run.
A real motion is possible only when the client sends `--execute` and the server
was started with `--allow-execute`; `move_to_target_ee.py` still asks the local
control-host operator for `y/yes` before motion.

## Inference dependencies

Inference host dependencies: `numpy`, `scipy`, `torch`, `pyzmq`, `smplx`, and
`chumpy`. Install the MANO runtime dependencies in the Conda environment used
by the launcher (the default is `wjh_grasp`):

```bash
conda run -n wjh_grasp python -m pip install chumpy smplx
```

The camera-only script keeps `pyrealsense2` in the system Python helper
environment; the Conda `wjh_grasp` environment therefore only needs the model
dependencies.
Nonzero camera distortion coefficients require `opencv-python` or
`opencv-contrib-python`. The required deployment runtime is implemented in
`grasp/runtime`; checkpoints, MANO models, and RoboDex Wuji URDF/mesh assets are
bundled in `grasp/assets` and use fixed script paths.

## Required transforms

Transform notation is `A_T_B`. The inference client uses the calibrated camera
transforms embedded in `grasp/inference_client.py`:

- `world_T_camera` (`W_T_C`) from `calibration/matrix.md`.
- `camera_T_right_base` (`C_T_B_R`) from `calibration/matrix.md`, inverted to
  obtain the `base_T_camera` transform required by the command path.
- `ee_T_hand` comes from the controller-EE-to-Wuji-root mount calibration.

The bundled `grasp/ee_to_wuji_nominal.json` currently uses a temporary identity
transform: the controller EE frame and Wuji wrist/hand-root frame are treated as
coincident. This keeps EE target generation enabled for experiments, but it is
not a physical mount calibration and must be replaced before relying on the
absolute EE pose.

The generated pose is `world_T_hand`. The target sent to the controller is:

```text
base_T_world = base_T_camera @ inverse(world_T_camera)
base_T_ee = base_T_world @ world_T_hand @ inverse(ee_T_hand)
```

The bundled temporary mount file has this structure:

```json
{
  "format": "real_exp_ee_to_wuji_v1",
  "calibration_status": "temporary_identity_assumption",
  "ee_T_hand": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

It temporarily assumes the controller EE and Wuji wrist/hand root are the same
frame, so `ee_T_hand` is the 4x4 identity matrix. Pass a measured replacement
with `--mount-calibration PATH` after physical mount calibration.

## Control host

Install `pyzmq` in the Python environment used to launch the server. Bind to the
control host's inference-network interface (or `*`) and provide the Wuji SDK
address:

```bash
python -m grasp.control_server \
  --bind tcp://*:5570 \
  --side right \
  --hand-ip WUJI_IP:PORT
```

This invokes `move_to_target_ee.sh --dry-run`, including live ROS state and IK
validation, but never moves hardware. After inspecting at least one successful
dry run, enable the final execution gate:

```bash
python -m grasp.control_server \
  --bind tcp://*:5570 \
  --side right \
  --hand-ip WUJI_IP:PORT \
  --allow-execute
```

Restrict TCP port 5570 to the inference host at the OS firewall. The server
rejects stale requests and repeated command IDs, but it is not an encrypted or
authenticated transport.

## Inference host

Run from the `real-exp` repository root in the deployed inference environment.
The world filter bounds are in meters and should tightly cover the object
region while excluding the table and background:

```bash
python -m grasp.inference_client \
  --world-min -0.25 -0.25 0.005 \
  --world-max 0.25 0.25 0.40 \
  --control-address tcp://CONTROL_HOST_IP:5570 \
  --output-dir grasp/runs/trial_0001
```

The first run is a dry run even if the server allows execution. Add `--execute`
only after checking `result.json`, `object_points_world.npy`, the reported
`base_T_ee`, the temporary identity mount assumption, Wuji joint order, and the
control-host IK output.

## Camera-only inference

When the inference machine has the D435 camera but no connection to the control
host, run `camera_inference.py` from the Conda `wjh_grasp` environment. The script
starts `realsense_capture.py` with system `/usr/bin/python3` for camera capture,
then continues in `wjh_grasp` for Torch/model inference. This avoids requiring
`pyrealsense2` inside the conda environment:

```bash
python -m grasp.camera_inference \
  --output-dir grasp/runs/camera_trial_0001 \
  --world-min -0.25 -0.25 0.10 \
  --world-max 0.25 0.25 0.40
```

Override the interpreter when the system camera Python is installed elsewhere:

```bash
python -m grasp.camera_inference \
  --camera-python /usr/bin/python3 \
  --output-dir grasp/runs/camera_trial_0001
```

By default, the helper continuously observes 15 aligned frames (about 0.5 s at
30 FPS) and takes the per-pixel median of nonzero depth values. A pixel is kept
only when it has valid depth in at least half of the frames. Keep the camera and
scene still during this window. The window and support threshold are tunable:

```bash
python -m grasp.camera_inference \
  --output-dir grasp/runs/camera_trial_0001 \
  --observation-frames 21 \
  --min-valid-depth-ratio 0.6
```

The helper writes the fused depth image, the last aligned RGB frame, and
capture metadata to a temporary directory, which the parent process loads
before coordinate conversion and inference. `depth_raw.npy` is therefore in
the camera's original integer depth units, but contains the temporally fused
depth observation rather than one physical frame. `result.json` records the
fusion settings, temporal span, and retained-pixel counts under
`camera.observation`.

The output contains only a `world/` directory with the unfiltered and filtered
scene point clouds, generator input, `grasp_object_points.ply`, `mano.ply`, and
retargeted/refined Wuji meshes. No NPY files are written below `world/`.
`result.json` records the calibration matrices and inference metadata, while
`poses.json` stores the readable `world_T_hand`, `base_T_hand`, `base_T_ee`, and
`base_T_ee_xyz_rpy` values for offline experiments. With the bundled temporary
identity mount, `base_T_ee` equals `base_T_hand`. It also stores the final
refined Wuji target in `hand_joints_rad`, together with the corresponding
canonical `hand_joint_names` ordering. These saved pose fields remain in the
bundled RoboDex/first-generation Wuji model convention because the saved meshes
and contact-refinement result use that model.

## Triggered inference across the two computers

The camera server and robot-control computer can run as a request/response
pair, so pose values no longer need to be copied out of `poses.json` manually:

```text
robot-control computer                    camera server (192.168.50.13)
Enter / g                                 wait on TCP port 5571
    |--- infer_grasp request -----------> capture and fuse D435 frames
    |                                     run grasp inference
    |<-- EE pose + 20 hand joints ------- save a unique trial directory
validate target locally
scripts/move_to_target_ee.sh
```

On the camera server, run the persistent service from the repository root. Its
camera/model options are the same as `camera_inference.py`:

```bash
./grasp/start_grasp_inference_server.sh \
  --world-min -0.25 -0.25 0.10 \
  --world-max 0.25 0.25 0.40 \
  --observation-frames 15
```

It binds to `tcp://192.168.50.13:5571` by default and creates a timestamped
directory such as `grasp/runs/camera_trial_20260828_143052` for every accepted
request. If multiple trials somehow start within the same second, later names
receive `_01`, `_02`, and so on instead of overwriting existing output. Each
directory contains the same inference artifacts as a one-shot
`camera_inference.py` run: `rgb_bgr.npy`, fused `depth_raw.npy`, `poses.json`,
`result.json`, and the point clouds/hand meshes under `world/`. Override the
network and output locations with `GRASP_SERVER_IP`, `GRASP_INFERENCE_PORT`,
and `GRASP_RUNS_DIR`. Use `GRASP_CONDA_ENV` when the inference environment is
not named `wjh_grasp`.

On the robot-control computer, select exactly one control mode. When the Wuji
hand is installed, first run in dry-run arm-with-hand mode:

```bash
./grasp/start_grasp_execution_client.sh --arm-with-hand
```

Before the client enters its existing request loop, the launcher moves the
right-arm EE to the initial `xyzrpy` recorded in `note.txt` (`0.682977,
0.154027, 0.452649, -2.134387, 0.498717, -2.334388`) by invoking
`scripts/move_to_target_ee.sh`. In arm-with-hand mode, this initial move uses
`--right --hand`, commands all 20 Wuji joints to zero, and uses the configured
right-hand SDK endpoint; arm-only mode uses `--right --arm`. The move utility
plans and previews a collision-checked Cartesian path whose EE translation is a
straight line and whose orientation uses shortest-path quaternion
interpolation. It evaluates distinct target IK solutions in short segments,
constrains each segment to a local joint corridor, and selects the lowest-motion
valid path. A final cumulative joint-travel check still rejects unnecessary
redundant-IK rotations before asking the local operator for `y/yes`. Declining
the move, an incomplete Cartesian path, or any planning/execution failure stops
startup instead of continuing with the grasp workflow. Running the launcher
with `-h` or `--help` never starts this initial motion.

Press Enter or type `g` to request one observation and inference. After the
response is validated, the client invokes `scripts/move_to_target_ee.sh` with
`--right --hand`, the returned `base_T_ee_xyz_rpy`, and all 20 returned joint
angles. The server converts the first-generation RoboDex result at this output
boundary to Wuji Hand 2 SDK firmware order. The four non-thumb lateral joints
(`right_finger2_joint2`, `right_finger3_joint2`,
`right_finger4_joint2`, and `right_finger5_joint2`; flat indices 5, 9, 13, and
17) are negated because their positive axes are opposite between the two
models. All other joints are passed without a sign change. Commands identify
the source model, target hand model, joint convention, and conversion version;
the control client rejects a command without the exact expected contract. This
is a temporary compatibility conversion: inference meshes, collision/contact
refinement, thumb geometry, and joint limits still use the first-generation
model and are not equivalent to a native Wuji Hand 2 inference model.

After each successful grasp move, the client invokes the same script again to
return the EE to the configured initial pose. In arm-with-hand mode, the return
uses `--right --hand` and also commands all 20 Wuji joints to zero; arm-only
mode continues to use `--right --arm`. If a grasp move fails before that
automatic return—for example, because its final measured EE pose is just
outside tolerance—the interactive client remains running. Type `r` at the
`grasp>` prompt to manually run the same reset without making another camera
inference request. The hand is also reset to 20 zero joint targets in
arm-with-hand mode. Both grasp and reset moves honor dry-run mode, and each real
move keeps the utility's local `y/yes` confirmation. A failed manual reset
returns to the prompt so it can be retried. Type `q` to stop the client. For a
noninteractive connectivity test, use `--once`; it still defaults to a
hardware-safe dry run.

The right-hand SDK endpoint is fixed near the top of
`grasp/start_grasp_execution_client.sh`; it is not a command-line parameter.
When the hand is not installed, use arm-only mode. The server still computes
and archives the complete grasp, while the control computer passes only the EE
pose to `move_to_target_ee.sh --right --arm` and never starts a hand worker:

```bash
./grasp/start_grasp_execution_client.sh --arm-only
```

Only after a successful dry run and inspection of the saved trial, restart the
control-side process with local execution permission:

```bash
./grasp/start_grasp_execution_client.sh --arm-with-hand --execute
```

The camera server cannot grant execution permission. Even with `--execute`,
the existing move utility reads current robot state, checks IK, prints the
target, and requires `y/yes` from the operator on the robot computer before
moving. The service currently supports the calibrated right arm/hand only.

Both computers need the same updated repository checkout. Allow TCP port 5571
only on the direct control-computer/camera-server link. The protocol checks
request IDs, timestamps, pose/matrix consistency, conservative EE workspace,
and the canonical 20-joint contract, but the direct ZMQ connection is not
encrypted or authenticated.

For a camera-free replay, provide all three offline arguments. The metadata can
be a calibration `metadata.json` and must contain `color_intrinsics` and
`depth_scale_m`:

```bash
python -m grasp.inference_client ... \
  --rgb-npy calibration/runs/RUN/rgb/000000.npy \
  --depth-npy calibration/runs/RUN/depth/000000.npy \
  --camera-metadata calibration/runs/RUN/metadata.json
```

Each trial directory is created atomically and contains raw RGB-D arrays,
valid camera points, the filtered object cloud, sampled generator input, and
`result.json`. A dry run additionally writes `object_points_world.ply`,
`generator_input_world.ply`, `grasp_object_points_world.ply`, `mano_world.ply`,
`wuji_retargeted_world.ply`, and `wuji_refined_world.ply` in the world frame.
A failed control request leaves the inference result on disk.

## Safety sequence

1. Verify both calibration residual reports and a held-out point/pose.
2. Visualize `object_points_world.npy`; confirm z-up and table removal.
3. Run the control server without `--allow-execute` and inspect its live IK dry run.
4. Compare the generated hand root with the physical hand mount orientation.
5. Enable execution with the hand open, an emergency stop available, and an operator at the control host.
