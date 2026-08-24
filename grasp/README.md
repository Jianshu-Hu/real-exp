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

Inference host dependencies: `numpy`, `scipy`, `torch`, and `pyzmq`. Live D435
capture additionally requires `pyrealsense2`; nonzero camera distortion
coefficients require `opencv-python` or `opencv-contrib-python`. The required
deployment runtime is implemented in `grasp/runtime`; checkpoints, MANO models,
and RoboDex Wuji URDF/mesh assets are bundled in `grasp/assets` and use fixed
script paths.

## Required transforms

Transform notation is `A_T_B`. The inference client reads:

- `world_T_camera` from `calibration/calibrate_camera_to_world.py`.
- `base_T_camera` from `calibration/calibrate_camera_to_robot_base.py`.
- `ee_T_hand` defaults to `grasp/ee_to_wuji_nominal.json`, the temporary
  nominal `panda_link8_T_hand_docking` value from `calibration/README.md`.

The generated pose is `world_T_hand`. The target sent to the controller is:

```text
base_T_world = base_T_camera @ inverse(world_T_camera)
base_T_ee = base_T_world @ world_T_hand @ inverse(ee_T_hand)
```

The bundled nominal mount file has this structure:

```json
{
  "format": "real_exp_ee_to_wuji_v1",
  "description": "Measured transform from the controller EE frame to the generator Wuji root",
  "ee_T_hand": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ]
}
```

It assumes controller EE equals `panda_link8`. This is only a temporary CAD
value; the physical `fr3_link8`, flange, controller EE, and mount must be
measured and checked before hardware use. Override it with
`--mount-calibration PATH` after physical mount calibration.

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
  --camera-to-world calibration/runs/TABLE_RUN/camera_to_world.json \
  --camera-to-robot-base calibration/runs/HANDEYE_RUN/camera_to_robot_base.json \
  --world-min -0.25 -0.25 0.005 \
  --world-max 0.25 0.25 0.40 \
  --control-address tcp://CONTROL_HOST_IP:5570 \
  --output-dir grasp/runs/trial_0001
```

The first run is a dry run even if the server allows execution. Add `--execute`
only after checking `result.json`, `object_points_world.npy`, the reported
`base_T_ee`, mount direction, Wuji joint order, and the control-host IK output.

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
