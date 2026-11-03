# Soft-hand grasp: client/server deployment

The two hosts use the same split as `scripts/start_data_collection_client.sh`
and `scripts/start_data_collection_server.sh`:

| Host | Attached hardware | Work performed |
| --- | --- | --- |
| Client | Right FR3 and pneumatic soft hand | Observation move, execute received grasp/lift poses, publish hand voltages |
| Server | Front D435i and inference GPU | Capture aligned RGB-D, SAM3 text segmentation, depth deprojection, calibration and desired EE pose calculation |

The client moves to its predefined observation pose and verifies arrival before
requesting inference. The server then captures a fresh aligned color/depth pair,
segments the object description, and returns **both grasp and lift EE poses** in
the right-arm base frame. The client validates and executes those poses without
recalculating them. After verified grasp arrival, it publishes the configured
20-channel command for at least two seconds, lifts, and holds until the operator
types `release`. The publisher remains open through the lift.

## Start the server

On the camera/GPU computer, from the repository root:

```bash
SOFTHAND_SERVER_PYTHON=/path/to/server/env/bin/python \
  bash softhand_grasp/start_grasp_server.sh \
  --sam-python /path/to/sam3/env/bin/python \
  --camera-python /usr/bin/python3 \
  --checkpoint /path/to/sam3.pt
```

The server environment needs NumPy and `pyzmq`. The camera interpreter needs
NumPy and `pyrealsense2`. The SAM3 interpreter needs native SAM3, PyTorch, NumPy
and Pillow, with a CUDA GPU and a local `sam3.pt`. Follow
[Meta's installation instructions](https://github.com/facebookresearch/sam3#installation)
for SAM3. No weights are downloaded by this module. The server does not need
ROS, the arm connection, or the pneumatic serial device.

If one environment contains all server dependencies, set
`SOFTHAND_SERVER_PYTHON` to its Python and omit `--sam-python`; the camera Python
can still be selected separately. SAM3 runs in a subprocess for each request.

The default endpoint is `tcp://192.168.50.13:5572`, using the same server IP as
the data-collection scripts. `DATA_COLLECTION_SERVER_IP` or
`SOFTHAND_SERVER_IP` changes the default host on either side. Use `--bind` on
the server and `--server-address` on the client to override the full endpoint.
Port 5572 is separate from data collection and the existing Wuji grasp server.
The JSON ZeroMQ connection is intended for the trusted direct robot network;
it has no authentication or encryption. Synchronize host clocks for freshness
checks. ROS DDS does not carry this module's camera data or grasp requests.

## Start the client

On the robot-control computer, preview without arm or pneumatic commands:

```bash
bash softhand_grasp/start_grasp_client.sh --object 'red cup'
```

The server still captures and infers in preview mode. Preview prints the
observation/grasp/lift poses but does not physically move to the observation
pose or perform IK/collision validation.

After setting the poses, tool offset and installed collision geometry, execute:

```bash
bash softhand_grasp/start_grasp_client.sh --object 'red cup' --execute
```

The client needs NumPy, `pyzmq`, `pyserial`, and the existing ROS Humble,
MoveIt/Pinocchio and FR3 overlays in `/usr/bin/python3`. It requires no camera
SDK, SAM3, GPU, checkpoint, or server filesystem mount. The launcher uses the
same right-arm no-gripper configuration and ROS overlays as teleoperation,
with `motion_controller:=trajectory`. Stop other arm controllers first; this
launcher shares the teleoperation lock and owns its controller's lifecycle.

`--execute` authorizes all three moves, with no intermediate motion prompts.
The existing planner checks IK, joint jumps and the current MoveIt collision
scene, then checks measured arrival (20 mm, 0.08 rad). Include real obstacles
and the installed soft-hand geometry in the planning scene. No camera-derived
collision mesh is created here. Direct Python entry points are
`python -m softhand_grasp.server` and `python -m softhand_grasp.client`; direct
client execution expects a running trajectory controller and sourced ROS.

## Configuration ownership

- **Client: `client_config.json`** — observation EE pose, serial port, 20 channel
  voltages, two-second close duration and 20 Hz command rate. The observation
  pose initially comes from `grasp/start_grasp_execution_client.sh`.
- **Server: `config.json`** — calibrated camera serial/transforms, fixed grasp
  orientation, tool offset, segmentation/depth settings and lift direction/distance.
  These settings are never used by the client to calculate target poses.

Positions use metres and RPY angles use radians. The server defaults to front
D435i `401622071701` at 640x480, using transforms from `calibration/matrix.md`
and live intrinsics/depth scale. Recalibrate after moving the camera or base.
`A_T_B` maps points from B into A.

The server chooses the highest-confidence text-matched instance with sufficient
valid interior depth. `--instance-index N` on the client selects another
confidence rank. Masks from separate objects are never merged. The server
erodes the selected mask, rejects invalid/outlying depths and computes the
median visible-surface point. This is a surface-centroid heuristic, not a
shape-aware grasp optimizer or an object-centre estimate.

`ee_to_grasp_point_m` is the vector from the controller EE origin to the intended
contact point, expressed in EE axes. Its initial zero value makes the EE origin
target the detected surface; set the measured offset for the installed hand.
The default grasp orientation is inherited from the existing grasp launcher and
has not been validated for the soft-hand mount. The server computes the 0.10 m
lift along calibrated world +Z; `lift_frame` also supports `base` or `ee` +Z.
Grasp and lift retain the same orientation.

The requested default value `5` is **5 V on all 20 service channels**, matching
`PneumaticSystem/Service/publisher.py`; it is not a pressure in kPa/bar. Hardware
channels 21–24 are padded with zeros by that publisher. Outputs remain set
through the lift and the subsequent hold. Typing `release`, an exception or
Ctrl-C closes the publisher and switches outputs off; support the object first.

## Failure handling, records and tests

The client checks request ID, description, right-base frame, finite workspace
poses and capture/response age before approach. Server errors and timeouts stop
the sequence; requests are not automatically retried. A failed approach prevents
hand closure, and a failed pressure write prevents lifting. Execution permission
and pneumatic settings come only from the client.

Server records live under `softhand_grasp/runs/server/`: RGB-D capture, selected
mask/overlay, effective calibration, request, calculated plan and response.
Client records under `runs/client/` contain its control settings, received
response and execution status. Paths returned by the server are informational;
the client never tries to read them from its own filesystem.

For camera-free testing, start the server with `--capture-dir /path/to/capture`
using `grasp.realsense_capture` files (`rgb.npy` is BGR, `depth.npy` is raw
aligned depth, plus `metadata.json`). Such replies are marked as replay and the
client rejects them with `--execute`. `--once` makes the server exit after one
request. Run tests in an environment containing NumPy, Pillow, pyzmq and pytest:

```bash
python -m pytest -q softhand_grasp/tests
```

Tests cover geometry, command timing/order, publisher integration, right-arm
planner calls, real TCP request/reply, malformed request recovery, timeouts and
invalid/replay response rejection. Hardware validation still requires the two
hosts with the attached camera, robot and hand.
