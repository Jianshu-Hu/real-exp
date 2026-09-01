# Real-World Wuji Grasp Pipeline

The supported workflow uses two launchers on two computers:

```text
robot-control computer                 camera/inference computer
start_grasp_execution_client.sh  <-->  start_grasp_inference_server.sh
request + validate + execute           D435i/L515 fusion + inference + archive
```

The ZeroMQ connection carries JSON request/response messages. It is not
encrypted or authenticated, so expose port `5571` only on the direct link
between the two computers.

## 1. Start the inference server

The inference computer needs the calibrated D435i and L515 and a Conda environment named
`wjh_grasp` containing `numpy`, `scipy`, `torch`, `pyzmq`, `smplx`, and
`chumpy`. The system `/usr/bin/python3` must provide `pyrealsense2`.

From the repository root, start the persistent server:

```bash
./grasp/start_grasp_inference_server.sh \
  --secondary-camera-serial f1480539 \
  --world-min -0.25 -0.25 0.10 \
  --world-max 0.25 0.25 0.40 \
  --observation-frames 15
```

The defaults are `tcp://192.168.50.13:5571`, Conda environment `wjh_grasp`,
and output directory `grasp/runs`. They can be changed with:

- `GRASP_SERVER_IP`
- `GRASP_INFERENCE_PORT`
- `GRASP_CONDA_ENV`
- `GRASP_RUNS_DIR`
- `GRASP_SECONDARY_CAMERA_SERIAL`
- `GRASP_L515_CAMERA_PYTHON`
- `GRASP_L515_PYTHONPATH`

The secondary-camera variable is an alternative to the corresponding
command-line option. For example:

```bash
export GRASP_SECONDARY_CAMERA_SERIAL=f1480539
./grasp/start_grasp_inference_server.sh \
  --world-min -0.25 -0.25 0.10 \
  --world-max 0.25 0.25 0.40 \
  --observation-frames 15
```

The dual-camera launcher defaults the last two settings to the `pose` Conda
environment's Python and `.vendor/l515_realsense`, respectively. That local
directory must contain the L515-compatible `pyrealsense2==2.54.2.5684`; see
`calibration/README.md`. D435i-only launch continues to use the system SDK.

Dual-camera mode becomes available only after the measured L515 serial and
`D435I_T_L515` have been recorded in `calibration/matrix.md` and copied into
`CALIBRATED_L515_SERIAL` and `CALIBRATED_D435I_T_L515` in
`grasp/inference_client.py`. A missing calibration, a malformed rigid
transform, or a different L515 serial is rejected before either camera opens.

If the secondary option is omitted, the launcher retains the original
D435i-only mode. The two cameras are captured one after the other and each
camera's configured observation frames are temporally fused, so the scene must
remain still for both captures. The combined cloud is transformed into the
D435i-anchored world frame, cropped, deduplicated on a 3 mm voxel grid, and
sampled for the existing grasp generator. Tune the grid with
`--fusion-voxel-size-m`; use `0` only when you intentionally want no
deduplication.

Use `./grasp/start_grasp_inference_server.sh --help` for camera, filtering,
model, and refinement options. Keep the camera and scene still while the
configured observation frames are captured.

## 2. Start the execution client

On the robot-control computer, select exactly one mode. Start with a dry run:

```bash
# FR3 and Wuji Hand 2
./grasp/start_grasp_execution_client.sh --arm-with-hand

# FR3 only; no hand process or hand command
./grasp/start_grasp_execution_client.sh --arm-only
```

The launcher first moves the right arm to its configured initial pose. In
`--arm-with-hand` mode it also opens the hand by setting all 20 joints to zero.
If this initial move fails or is declined, the grasp client does not start.
Running with `-h` or `--help` never moves hardware.

At the `grasp>` prompt:

- press Enter or type `g` to capture, infer, validate, and run one grasp;
- type `r` to return to the initial pose without requesting new inference;
- type `q` to quit.

After a successful grasp, the client automatically returns to the initial
pose and, in hand mode, opens the hand. Use `--once` for one noninteractive
request. `GRASP_SERVER_IP` and `GRASP_INFERENCE_PORT` select the inference
server. The Wuji SDK endpoint is intentionally fixed in
`start_grasp_execution_client.sh` for this installation.

## Execution safety

Both control modes are dry-run by default. Only after checking the inferred
trial, target pose, IK/path preview, mount calibration, and hand joint order,
enable real execution locally:

```bash
./grasp/start_grasp_execution_client.sh --arm-with-hand --execute
```

The inference server cannot authorize robot motion. Even with `--execute`,
`scripts/move_to_target_ee.sh` reads the live robot state, validates and
previews the motion, and requires the operator to enter `y` or `yes` on the
robot-control computer. Keep an operator and emergency stop available.

The server and client also validate request IDs and ages, pose/matrix
consistency, the conservative EE workspace, and the canonical 20-joint
contract. The current deployment supports only the calibrated right arm and
right hand.

## Calibration and output

Transform notation is `A_T_B`. The original D435i remains the reference for
the calibrated `world_T_d435i` and `d435i_T_right_base` values embedded in the
pipeline. Pair calibration supplies `d435i_T_l515`, and inference computes:

```text
world_T_l515 = world_T_d435i @ d435i_T_l515
```

This means adding the L515 does not alter the validated world-to-robot-base
chain. The installed mount transform remains in `ee_to_wuji_nominal.json`:

```text
base_T_world = base_T_camera @ inverse(world_T_camera)
base_T_ee = base_T_world @ world_T_hand @ inverse(ee_T_hand)
```

The bundled measured `ee_T_hand` has identity rotation and places the Wuji
hand-root origin `0.06 m` along the negative controller-EE z axis. Pass
`--mount-calibration PATH` to the inference launcher if the physical mount
changes.

Every accepted request gets a unique timestamped directory below
`grasp/runs`, containing `cameras/primary_d435i` and (in dual mode)
`cameras/secondary_l515`, their raw/temporally fused RGB-D arrays and individual
world clouds, `poses.json`, `result.json`, and merged world-frame point clouds
and hand meshes. Before executing a grasp, open
`world/scene_points_filtered.ply` and confirm that edges seen by both cameras
overlap without a doubled surface. A systematic double edge means the pair
extrinsic or a camera mount is wrong; recalibrate rather than increasing the
fusion voxel size. The server converts the bundled
first-generation RoboDex hand result to the Wuji Hand 2 firmware convention at
the command boundary; the four non-thumb lateral joints (indices 5, 9, 13,
and 17) have their signs reversed. The archived inference meshes and refined
joints remain in the source RoboDex model convention.
