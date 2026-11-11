# Real Experiments

This repository groups the code and notes used for real-world Franka FR3 experiments with GELLO and Meta Quest teleoperation, LeRobot dataset collection, and training workflows.

## Repository Layout

- `data_collection/`: scripts and documentation for recording, replaying, and editing local LeRobot datasets.
- `train/`: training-related notes and experiment-specific training assets.
- `deploy/`: policy inspection, serving, fetching, and real-robot execution tools.
- `utils/`: shared dataset-statistics and image-preprocessing helpers.
- `lerobot/`: vendored LeRobot codebase used by the local collection and training workflow.
- `gello_software/`: GELLO and ROS 2 integration code, tracked as a git submodule.
- [`PneumaticSystem/softhand_grasp/`](PneumaticSystem/softhand_grasp/README.md): pneumatic soft-hand grasp client/server and launchers, inside the `PneumaticSystem` submodule.
- `libs/hand-tracking-streamer/`: upstream Meta Quest tracking application, tracked as a git submodule.
- `data_collection/meta_quest/`: standalone, robot-disconnected Quest pose transport and monitor.
- `data/`: local datasets and experiment outputs.

## Submodules

The repo expects its submodules, including `PneumaticSystem/`, `lerobot/`, `gello_software/`, and
`libs/hand-tracking-streamer/`, to be available locally. If a submodule is
empty after cloning, initialize all submodules:

```bash
git submodule update --init --recursive
```

`PneumaticSystem` uses `main` for remote updates. To advance it to the latest
remote revision, run `git submodule update --remote --merge PneumaticSystem`
and commit the updated submodule pointer in this repository. Regular submodule
initialization checks out the revision recorded by this repository.

## Environment Split

This repo uses two different Python environments:

- Use the system ROS 2 / Franka environment for `gello_software`, ROS 2 launch files, and `data_collection/replay_lerobot_episode.py`.
- Use the `lerobot` Conda environment for `data_collection/lerobot_collection.py`, dataset inspection, dataset editing, and training under `lerobot/`.

## Common Workflows

### Data Collection

Data collection and replay instructions live in [data_collection/DATA_COLLECTION_README.md](data_collection/DATA_COLLECTION_README.md).
- [`scripts/start_teleoperation.sh`](scripts/start_teleoperation.sh): start left-, right-, or dual-arm GELLO/FR3 teleoperation with Franka grippers, Wuji hands, or a pneumatic Softhand.
- [`scripts/start_meta_quest_pose_monitor.sh`](scripts/start_meta_quest_pose_monitor.sh): receive and display uncalibrated Quest wrist/controller poses without connecting to a robot. See [`data_collection/meta_quest/README.md`](data_collection/meta_quest/README.md).
- `start_teleoperation.sh` accepts exactly one end-effector mode (`--arm`, `--gripper`, `--hand`, `--softhand-gripper`, or `--softhand-manus`) and one arm selection (`--left`, `--right`, or `--duo`).

For Softhand, `--softhand` is renamed to `--softhand-gripper`. This mode keeps
the GELLO gripper's existing uniform 0–5 V command across all 20 channels.
`--softhand-manus` uses GELLO for the arm and the MANUS glove for the hand:

```bash
./scripts/start_teleoperation.sh --softhand-gripper --right
./scripts/start_teleoperation.sh --softhand-manus --right
```

MANUS mode starts `PneumaticSystem/retargeting/manus/bridge/build/manus_bridge`
(override the executable with `MANUS_BRIDGE_PATH`) and runs
`PneumaticSystem/teleop.py --target-source manus --manus-retarget direct`.
This matches the submodule's native calibrated mapping and cooperative contact
correction. Native `direct` and `shape` modes use the right glove, independent
of which arm is selected. `--manus-retarget none` selects legacy geometric
flexion and also supports `--manus-hand left`.
The bridge sends to `127.0.0.1:9878`; the controller listens on `0.0.0.0:9878`.
Build the native bridge once in this checkout (adjust the SDK path for your
installation; build outputs are ignored by Git):

```bash
cmake -S PneumaticSystem/retargeting/manus/bridge \
  -B PneumaticSystem/retargeting/manus/bridge/build \
  -DMANUS_SDK_ROOT="$HOME/jhu/ManusSDK_v3.1.1/SDKMinimalClient_Linux/ManusSDK" \
  -DMANUS_INTEGRATED_ONLY=ON -DCMAKE_BUILD_TYPE=Release
cmake --build PneumaticSystem/retargeting/manus/bridge/build --parallel 2
PneumaticSystem/retargeting/manus/bridge/build/manus_bridge --help
```

If a bridge already runs in another terminal or on a remote host, add
`--manus-external-bridge` and direct that bridge's UDP stream to this machine's
port 9878. For a bridge on this machine:

```bash
PneumaticSystem/retargeting/manus/bridge/build/manus_bridge --udp 127.0.0.1:9878
# In another terminal:
./scripts/start_teleoperation.sh --softhand-manus --right --manus-external-bridge
```

The glove defaults to the right hand for every arm selection.
Use `--manus-hand left --manus-retarget none` for legacy left-glove control.
Both Softhand modes operate **one pneumatic controller**, even with
`--duo`. Both auto-detect the F407Driver USB device; `--serial-port PATH` or
`SOFTHAND_SERIAL_PORT` overrides detection. MANUS uses system Python by default;
set `SOFTHAND_PYTHON=/path/to/python` to use another interpreter. Install
`PneumaticSystem/retargeting/requirements.txt` and `pyserial` in that interpreter;
native retargeting needs MuJoCo even without a viewer. The launcher validates
the actual arguments, solver, and calibration before starting the arm.

MANUS joint angles use the existing per-channel calibration and channel order
in `PneumaticSystem/Service/voltage_mapping.py` and `Service/hand_joints.py`.
Native retargeting includes signed swing control relative to the measured
zero-voltage pose. Legacy `none` mode keeps side swing at zero. Missing/stale glove data commands
zero volts, and Ctrl-C stops the arm, hand controller, and launcher-owned
bridge together.
Routine joint-target and channel-voltage output is disabled by default; startup,
shutdown, and target-loss messages remain visible. For standalone diagnostics,
append `-- --print-every N` to the launcher to print every N cycles. Other
`teleop.py` settings can follow `--`, for example:

```bash
./scripts/start_teleoperation.sh --softhand-manus --right -- \
  --sim-display --thumb-mode shape --frequency-hz 20
```

Bridge transport, glove selection, and publisher port remain launcher options.

All launchers can be invoked from a bare shell; manually activating Conda is
not required. ROS-only processes use the system ROS Python. LeRobot, policy,
and Wuji processes use the `lerobot` Conda environment automatically without
changing the calling shell. Set `LEROBOT_CONDA_ENV`, `WUJI_CONDA_ENV`, or
`DEPLOYMENT_CONDA_ENV` to select a differently named environment.
- [`scripts/start_data_collection_client.sh`](scripts/start_data_collection_client.sh): start control-host GELLO/Franka teleoperation for split collection.
- [`data_collection/wuji/`](data_collection/wuji/): Python Wuji-glove teleoperation publisher with FR3/GELLO-compatible ROS topics.
- [`scripts/start_data_collection_server.sh`](scripts/start_data_collection_server.sh): start the data-server RealSense publishers, ROS bridge, and optional recorder.
- [`scripts/replay.sh`](scripts/replay.sh): start the dual-arm FR3 replay controllers and Franka-hand managers, then replay a recorded LeRobot episode.

### Train

- Training instructions live in [train/TRAIN_README.md](train/TRAIN_README.md).

### Deploy

- Deployment instructions live in [deploy/DEPLOY_README.md](deploy/DEPLOY_README.md).
- [`scripts/start_deployment_server.sh`](scripts/start_deployment_server.sh): configure cameras and the deployment bridge from checkpoint-embedded metadata, then start the policy server on the inference server.
- [`scripts/start_deployment_client.sh`](scripts/start_deployment_client.sh): query the policy server for its checkpoint contract, then select left/right/duo FR3 hardware plus the matching gripper or Wuji worker. The client does not need the dataset or checkpoint.

## Notes

- Do not change the GELLO USB connections. If the topology changes, reinstall or update the USB rules and verify both aliases before launching ROS 2 nodes.
- Check the GELLO offsets whenever the teleoperator joints do not align with the robot or after a hardware reset. Recompute the offsets with the procedure in [data_collection/DATA_COLLECTION_README.md](data_collection/DATA_COLLECTION_README.md) before collecting data.
