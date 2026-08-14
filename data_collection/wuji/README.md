# Wuji Glove to Wuji Hand 2 teleoperation

This directory uses only Wuji's Python SDK. `teleop.py` drives one right Wuji
Hand 2 from one right Wuji Glove by reading `hand_skeleton`, passing its 21
MediaPipe-order landmarks through `RetargetSession`, and sending the returned 20
firmware-order joint positions directly through `joint_command()`.

The implementation follows Wuji's official Python example:

- [Retargeting documentation](https://docs.wuji.tech/docs/en/wuji-sdk/latest/retargeting/)
- [`retargeting/1.teleop_real.py`](https://github.com/wuji-technology/wuji-sdk/blob/main/examples/python/retargeting/1.teleop_real.py)

The Wuji Hand ROS 2 documentation is for the separately installed USB
`WujiHand` driver and supported ROS distributions. It is not used for this
Ethernet `WujiHand2` setup.

## Requirements

- Linux x86_64 or aarch64
- Python 3.12 in this repository's `lerobot` environment
- `wuji-sdk==2026.8.3` and NumPy
- right glove and right hand on the same Ethernet LAN as the computer
- computer address `192.168.1.10/24`, right glove `192.168.1.101`, and right
  hand `192.168.1.111` for the current hardware setup

Install the Python dependencies if needed:

```bash
cd /home/pair1/real-exp
python3 -m pip install -r data_collection/wuji/requirements.txt
```

No ROS installation, ROS node, or ROS topic is required.

## 1. Passive glove verification

Start the browser visualizer before enabling the robotic hand:

```bash
python3 data_collection/wuji/visualize.py
```

Open `http://127.0.0.1:8765`. It displays the live 21-landmark glove skeleton
and the 20 retargeted Wuji Hand 2 joint commands. The visualizer connects only
to the glove; it does not connect to, enable, or command the robotic hand. Stop
it with `Ctrl-C` before starting `teleop.py`.

## 2. Passive end-to-end verification

Keep the hand disabled and run:

```bash
python3 data_collection/wuji/teleop.py --dry-run --duration 10
```

This connects to both devices and checks discovery, handedness, all 20 joints,
glove frames, retargeting, tuning, and diagnostics. It never creates a command
publisher and never calls `hand.enable()`. In dry-run mode, diagnostics are
reported for the full duration instead of triggering the live-enable interlock.
The output also separates host Ethernet E2E loss from each hand joint's internal
bus response rate and timeout counters. `command_delta_from_hand_start` is the
intentional difference between the glove target and the disabled physical-hand
pose; use `raw_target_speed_p95` and `raw_target_speed_max` to assess stationary
glove/retargeting jitter.

Do not continue if the output contains `Enc1BitRate`, `BusFrameLossHigh`, an
unknown error, an offline joint, or another hand diagnostic. `Enc1BitRate`
points to encoder/magnet signal quality; `BusFrameLossHigh` points to the hand's
internal bus or wiring. They are not caused by the host-side velocity limiter.

## 3. Stationary actuator test

Only after the passive check is clean, clear the workspace and run a five-second
fixed-pose hold:

```bash
python3 data_collection/wuji/teleop.py --hold-only --duration 5
```

After the typed confirmation, this applies Wuji's official Hand 2 teleoperation
settings (`effort_limit=1.5 A`, `kp=3.0`, `kd=0.05`), enables the hand, and
repeatedly sends only the measured starting pose at 120 Hz. It does not follow
the glove. The process disables the hand if any joint moves more than 0.05 rad
or a diagnostic appears.

If the hand shakes during this fixed-pose test, the glove, retargeting, and speed
limit are excluded from the command path. Disable power and investigate the
reported encoder/internal-bus condition with Wuji support before continuing.

## 4. Live teleoperation

Clear the hand workspace, keep an emergency-stop method within reach, and run:

```bash
python3 data_collection/wuji/teleop.py
```

The program prints both serial numbers and requires the exact confirmation
`ENABLE RIGHT HAND`. It first performs a three-second stationary hold check and
only then follows the glove. Press `Ctrl-C` to stop. On normal exit, input
timeout, invalid tracking, a reported diagnostic, `SIGTERM`, or `SIGHUP`, it
disables the hand, closes every stream, disconnects both devices, and restores
the previously selected SDK user.

For supervised non-interactive launch, confirmation can be bypassed explicitly:

```bash
python3 data_collection/wuji/teleop.py --yes
```

Do not use `--yes` until interactive operation has been validated.

## Safety behavior

- Requires exactly one selected Wuji Glove and one Wuji Hand 2.
- Requires both devices to report right handedness.
- Requires all 20 hand joints online and refuses to enable on active diagnostic
  warnings, unknown errors, or stop-severity faults.
- Rejects malformed, non-finite, degenerate, reordered, or low-confidence
  glove skeleton frames.
- Applies the official Hand 2 settings: 1.5 A effort limit, MIT `kp=3.0`, and
  MIT `kd=0.05`, and verifies the device accepted them before enabling.
- Streams at the official example's 120 Hz rate.
- Holds the measured pose before allowing glove targets, then starts command
  filtering from the latest measured pose to prevent an initial jump.
- Limits each joint to `1.0 rad/s` by default.
- Stops after `0.25 s` without a valid new glove frame.
- Sends zero velocity and zero feed-forward effort with each position command.
- Disables the hand in cleanup after any handled stop or runtime exception.

The safety thresholds can be made more conservative at launch. Raising them
should be treated as a hardware-risk decision, not routine configuration:

```bash
python3 data_collection/wuji/teleop.py \
  --max-velocity-rad-s 0.3 \
  --min-confidence 0.35 \
  --frame-timeout 0.20
```

`--allow-diagnostic-warnings` exists only to isolate a fault under Wuji's
direction. It bypasses the pre-enable interlock and must not be used for normal
teleoperation.

When multiple devices are present, select the intended pair by serial number:

```bash
python3 data_collection/wuji/teleop.py \
  --glove-id WG1KA06260623515 \
  --hand-id WH2KA01260730039
```
