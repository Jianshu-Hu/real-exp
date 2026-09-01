# Real-World Wuji Grasp Pipeline

The supported workflow uses two launchers on two computers:

```text
robot-control computer                 camera/inference computer
start_grasp_execution_client.sh  <-->  start_grasp_inference_server.sh
request + validate + execute           D435 capture + inference + archive
```

The ZeroMQ connection carries JSON request/response messages. It is not
encrypted or authenticated, so expose port `5571` only on the direct link
between the two computers.

## 1. Start the inference server

The inference computer needs the D435 and a Conda environment named
`wjh_grasp` containing `numpy`, `scipy`, `torch`, `pyzmq`, `smplx`, and
`chumpy`. The system `/usr/bin/python3` must provide `pyrealsense2`.

From the repository root, start the persistent server:

```bash
./grasp/start_grasp_inference_server.sh \
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

Transform notation is `A_T_B`. Inference uses the calibrated
`world_T_camera` and `camera_T_right_base` values embedded in the pipeline plus
the installed mount transform in `ee_to_wuji_nominal.json`:

```text
base_T_world = base_T_camera @ inverse(world_T_camera)
base_T_ee = base_T_world @ world_T_hand @ inverse(ee_T_hand)
```

The bundled measured `ee_T_hand` has identity rotation and places the Wuji
hand-root origin `0.06 m` along the negative controller-EE z axis. Pass
`--mount-calibration PATH` to the inference launcher if the physical mount
changes.

Every accepted request gets a unique timestamped directory below
`grasp/runs`, containing raw/fused RGB-D data, `poses.json`, `result.json`, and
world-frame point clouds and hand meshes. Retargeting, contact refinement,
archived meshes, and commands all use the bundled right Wuji Hand 2 Beta 1
model. Its 20 joints already follow the SDK firmware order, so no model-boundary
sign conversion is applied.
