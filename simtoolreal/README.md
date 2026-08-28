# SimToolReal FR3 + Wuji Hand 2 deployment

This directory contains the real-inference adapter for the upstream
`libs/SimToolReal-Franka-Wuji2` task and the existing `deploy/` hardware
transport. The server runs policy inference and FoundationPose++; the robot
computer publishes measured joints, runs the FR3 deployment controller, and
hosts the Wuji Hand 2 worker. No teleoperation process is started.

The active upstream contract is the `franka_wuji_right` profile:

- 7 FR3 arm joints followed by 20 Wuji Hand 2 joints, in the canonical order
  defined in `policy_contract.py`;
- 27 normalized actions and a 134-value actor observation;
- 60 Hz policy control (`dt=1/60`), arm velocity deltas, hand absolute targets,
  and the upstream 0.1 moving-average defaults;
- palm/fingertip FK and object keypoints computed from the exact combined URDF
  at `libs/SimToolReal-Franka-Wuji2/assets/urdf/franka_wuji_right/...`.

The server owns the RL checkpoint and waits for the first valid right-arm/Wuji
observation. The policy executor owns observation construction and action
dispatch. Camera-frame FoundationPose++ output requires an explicit calibration
matrix. Hardware commands are opt-in through the executor's `--execute` flag.

## Deployment topology

Use the existing deployment bridge when running the real FR3 and Wuji worker.
The repository includes a matching no-camera profile at
`simtoolreal/config/deployment_right_hand.yaml`. Launch it on the server
computer (after the ROS bridge package has been built):

```bash
ros2 launch franka_lerobot_data_bridge bridge.launch.py \
  config_file=/home/pair1/real-exp/simtoolreal/config/deployment_right_hand.yaml
```

`simtoolreal/scripts/start_server.sh` starts that ROS bridge as a managed child
by default. Pass `--no-bridge` when the bridge is already running, and use
`--bridge-config` to select a different profile. ROS must already be sourced in
the server shell (or selected with `SIMTOOLREAL_ROS_DISTRO`).

The bridge publishes its state on port 5555 and accepts arm targets on 5556.
The robot-side `simtoolreal/scripts/start_client.sh` starts the FR3 deployment
controller and right Wuji worker, then publishes the robot state. It does not
start Gello or teleoperation. The existing ROS bridge combines the seven FR3
positions with the 20-joint Wuji telemetry into the 27-joint state consumed by
the server.

Server computer, checkpoint and live pose:

```bash
./simtoolreal/scripts/start_server.sh \
  --config /path/to/config.yaml \
  --checkpoint /path/to/model.pth \
  --foundationpose-mesh /path/to/object.stl \
  --foundationpose-roi 220 120 160 180 \
  --foundationpose-no-display
```

To have the server start the FoundationPose++ publisher itself, use a supplied
ROI for headless first-frame registration:

```bash
./simtoolreal/scripts/start_server.sh \
  --foundationpose-mesh /path/to/object.stl \
  --foundationpose-roi 220 120 160 180 \
  --foundationpose-no-display \
  --config CONFIG.yaml --checkpoint MODEL.pth
```

The live pose process prints `waiting for first-frame mask/ROI
initialization` and only publishes after `register()` succeeds. It publishes
one `protocol: 1`, `kind: object_pose` JSON packet per tracked frame on
`tcp://127.0.0.1:5570` by default.

The server starts `simtoolreal/policy_server.py` on port `5571`, loads the
checkpoint through the upstream `deployment/rl_player.py`, starts the right
hand bridge on `5555/5556`, and waits until the bridge emits its first valid
27-joint sample. The policy environment must provide PyTorch, `rl_games`,
`gym`, `omegaconf`, and the upstream Python dependencies.

Robot computer:

```bash
./simtoolreal/scripts/start_client.sh \
  --server-ip SERVER_IP \
  --right-hand-ip HAND_IP:PORT
```

Finally, on the robot computer, run the executor with the goal and camera
calibration. It requests actions from the server and, with `--execute`, sends
seven FR3 joints to the bridge and twenty Wuji joints to the local worker:

```bash
python3 simtoolreal/policy_executor.py \
  --server-ip SERVER_IP \
  --goal-pose /path/to/goal_pose.json \
  --world-from-camera /path/to/world_from_camera.json \
  --execute
```

The executor automatically activates the bridge before commanding and returns
it to standby on shutdown. Omit `--execute` for a dry-run that prints received
state, pose, observation, and bounded targets.

The lightweight mock path remains available for transport tests, but it does
not exercise ROS, CUDA, the real checkpoint, or the Wuji SDK.

## Dependency-free transport test

This path is useful without ROS and is separate from the production
three-process deployment. Start the mock policy and pose publishers directly,
then run `policy_executor.py` with synthetic bridge packets. The transport
helpers and mock RPC round trip are covered by `pytest -q simtoolreal/tests`.

```bash
PYTHONPATH=simtoolreal python3 simtoolreal/policy_server.py \
  --mock-policy --bind tcp://127.0.0.1:5571

PYTHONPATH=simtoolreal python3 simtoolreal/foundation_pose_runner.py \
  --mock --connect tcp://127.0.0.1:5570
```

The executor subscribes to the bridge's combined right-arm/Wuji PUB stream on
`5555`; it rejects incomplete unnamed vectors and non-right bridge packets.
`pose_publisher.py` can publish an externally tracked FoundationPose++ JSON or
NumPy transform when a file-driven pose source is preferred.

## Safety and data contract

- Robot state and object pose are rejected when older than their configured
  freshness windows (`--max-state-age`, `--max-pose-age`).
- `--pose-frame camera` requires `--world-from-camera`; all FK, goal, and
  object transforms must end in one policy-world frame.
- Observation normalization uses the upstream training limits. Final command
  targets are additionally intersected with the real FR3 hardware limits.
- `--execute` is opt-in on `policy_executor.py`; dry-run is the default.
- The robot launcher starts deployment controllers and the Wuji worker in real
  mode, but never starts teleoperation.
