# Deploying a Trained Policy

The server computer has the cameras and runs policy inference. The robot
computer runs the Franka controllers and the policy executor.

## Network Flow

Both computers use ROS 2 domain `0`. The default server address is
`192.168.50.13`.

| Traffic | Route | Default |
| --- | --- | --- |
| Robot/gripper state and deployment commands | ROS 2 DDS between computers | `ROS_DOMAIN_ID=0` |
| State and camera-bundle metadata | server bridge to robot executor | ZMQ `192.168.50.13:5555` |
| Policy commands | robot executor to server bridge | ZMQ `192.168.50.13:5556` |
| Policy inference | robot executor to policy server | gRPC `192.168.50.13:8080` |
| Full synchronized camera bundles | bridge to policy-server cache, server-local only | ZMQ `127.0.0.1:5557` |
| Wuji measured-state telemetry (hand datasets only) | robot hand worker to server bridge | ZMQ `192.168.50.13:5558` |
| Wuji policy targets (hand datasets only) | robot executor to robot hand worker | ZMQ `127.0.0.1:5561` (left), `127.0.0.1:5562` (right) |

The remote observation stream is metadata/state-only during deployment. It contains camera
names, shapes, timestamps, bundle sequence, freshness, and synchronization diagnostics, but
does not send RGB pixels to the robot computer. The executor references the bundle sequence in
its inference request, and the policy server resolves the exact full bundle from the loopback
cache. RGB is always kept on the inference server; the robot-side observation stream never
contains camera pixels.

Deployment freshness gates default to 250 ms for camera, arm, and gripper state and 67 ms for
inter-camera skew. The policy server independently rejects a referenced camera bundle older
than 250 ms, stale robot state, incomplete camera sets, shape mismatches, and excessive
state-to-camera delay. These limits should be relaxed only after inspecting logged camera and
state ages.

Robot joint-state messages are timestamped on the robot computer and checked on the server
computer. Keep both hosts synchronized with Chrony, NTP, or PTP; otherwise fresh state can be
rejected as stale. Do not enable ROS simulated time for deployment.

Overlapping action chunks are aggregated from their original per-chunk proposals using
normalized exponential generation-age weights. Previously aggregated values are not fed back
into later blends, so changing the refresh cadence no longer creates recursive blending bias.
Use `--temporal-proposal-decay`: `0` selects only the newest proposal, `1` averages all proposal
generations equally, and values between them exponentially favor newer chunks. The default `0.5`
is a good starting point: each older chunk generation contributes half as much as the next newer
generation. Gripper values are continuous action dimensions and are aggregated without thresholding.
Deployment requires `gripper_action_representation: absolute_width`; old binary checkpoints are
rejected instead of converting their outputs with a threshold.

Schema-v2 policies emit chunk-anchored deltas. The policy server decodes every complete chunk
against the exact generation observation before sending it to the executor. The executor queue
therefore contains absolute joint targets or absolute 9D EE targets, and overlapping chunks are
never blended in incompatible delta coordinate frames.

The dataset root passed to the client, server, and executor must identify the
same data contract. It does not need to have the same absolute path on both
computers. The startup scripts resolve arm mode, end effector, state/action
dimensions, FPS, camera subset, FR3 configuration, and gripper/hand processes
from the metadata; there is no implicit dual-arm default after this migration.

Set `DEPLOYMENT_SERVER_IP` or pass `--server-ip` only if the inference server
address differs from `192.168.50.13`.

## Run Deployment

### 1. Start the deployment server

Only the server needs the checkpoint. Neither machine needs the source dataset.
On the server computer, preflight and start the checkpoint contract:

```bash
cd /home/pair1/real-exp
POLICY=/path/to/left-gripper-act/checkpoints/last/pretrained_model
./scripts/start_deployment_server.sh --policy-path "$POLICY" --print-config
./scripts/start_deployment_server.sh --policy-path "$POLICY"
```

The server validates `config.json` against the embedded metadata, configures the
camera/bridge layout and FPS, serves inference on port `8080`, and exposes a
read-only deployment contract on HTTP port `8081`.

### 2. Start the robot client

On the robot computer, query the running server without starting ROS/hardware:

```bash
./scripts/start_deployment_client.sh --server-ip 192.168.50.13 --print-config
./scripts/start_deployment_client.sh --server-ip 192.168.50.13
```

A hand checkpoint automatically selects the no-gripper controller and Wuji
worker. Add `--right-hand-ip <WUJI_IP:PORT>` or `--left-hand-ip`; the address can
be omitted when SDK discovery is unambiguous. The client has no dataset or
checkpoint path.

### 3. Start an executor in dry-run mode

In a second shell on the robot computer:

```bash
cd /home/pair1/real-exp
conda activate lerobot
source /opt/ros/humble/setup.bash
source gello_software/ros2/install/setup.bash

python deploy/franka_act_policy_executor.py \
  --server-address 192.168.50.13:8080 \
  --policy-device cuda:0 \
  --task "pick and place" \
  --temporal-proposal-decay 0.5
```

Do not add `--execute` yet. Confirm that the executor reports:

```text
state_dim=8, action_dim=8
cameras=['cam_left', 'cam_front']
execute: False
```

Also confirm that predictions are finite and stable, inference is continuous,
and there are no camera, dimension, CUDA, ZMQ, or gRPC errors.

Stop the dry run with `Ctrl-C`. After clearing the robot workspace and checking
the emergency stop, repeat the same executor command with:

```text
--execute
```

That is the only step that enables real policy commands. Run only one executor
at a time.

## Diffusion Alternative

Use this instead of ACT, first without `--execute`:

```bash
python deploy/franka_diffusion_policy_executor.py \
  --server-address 192.168.50.13:8080 \
  --policy-device cuda:0 \
  --task "pick and place" \
  --diffusion-chunk-size-threshold 0.5 \
  --temporal-proposal-decay 0.5
```

Diffusion requires two observation frames before its first inference. Add
`--execute` only after the dry run passes the same validation.

## Metadata-Driven Hardware Contract

The executor fetches the server-owned checkpoint contract instead of
assuming a 14- or 16-value dual-arm vector. Joint checkpoints use arm current
joint angles and arm target joint angles. EE checkpoints replace only those
seven arm values with current 6D EE pose and 6D delta EE pose; grippers retain
current normalized width in state and target width in action, and hands retain
current and target joint angles. For example, a left-gripper policy uses
`[left arm(7), left gripper(1)]` in joint mode or
`[left EE pose(6), left gripper(1)]` in EE mode. Before inference it rejects any live bridge
whose arm mode, end-effector type, state/action dimensions, or required camera
set differs from the contract. Checkpoint directories must contain
`meta/info.json`, `meta/real_exp_trajectory_config.json`,
`meta/real_exp_action_config.json`, and `meta/real_exp_deployment_config.json`.

The executor sends the fetched trajectory contract during the
policy handshake. The server rejects any executor/checkpoint disagreement
before loading weights, including same-dimensional layouts with different
hardware meaning.

The migrated `test-traj-gen-pick-and-place` checkpoints embed their 16-D
`duo/gripper` contract under `outputs/test-traj-gen-pick-and-place_act` and
`outputs/test-traj-gen-pick-and-place_diffusion`.

## Camera Cache Troubleshooting

The bridge publishes each full bundle to loopback port `5557` before it publishes the matching
metadata packet on port `5555`. A startup-only cache miss can occur because ZMQ PUB/SUB may drop
messages while the subscription connects; the policy request is rejected safely and a later
bundle should recover automatically.

If cache misses continue, confirm that the bridge and policy server use the same cache port and
that no other process owns it. Check bridge warnings for stale camera/state data or camera skew.
RGB rollback through the robot computer is intentionally unsupported; use the server-local cache
path so the network topology remains deterministic.

## Stop Deployment

Stop processes in this order:

1. Stop the robot-side executor.
2. Stop `start_deployment_server.sh`.
3. Stop `start_deployment_client.sh` last.

The bridge and deployment controller hold position on stale commands or clean
shutdown. If the executor cannot start, first check that robot state reaches
the server over ROS 2 and that server ports `5555`, `5556`, `5557`, and `8080`
are reachable from the robot computer. Hand deployment also requires access to
server ports `5558` and `8081`, plus an available robot-local command port
(`5561` or `5562`).
