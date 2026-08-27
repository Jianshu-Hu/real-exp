# Standalone Meta Quest Pose Input

This directory is the robot-disconnected foundation for a new Meta Quest
teleoperation stack. It uses the upstream Hand Tracking Streamer repository at
`libs/hand-tracking-streamer` for the headset application and protocol
documentation.

The current code only receives, validates, and displays poses. It does not
import Franka libraries, publish robot targets, or convert Unity coordinates
into a robot frame.

## Current boundary

```text
Hand Tracking Streamer on Quest
  -> TCP over USB/ADB on port 8000
  -> data_collection.meta_quest.receiver
  -> validated latest wrist/controller poses in the Unity tracking frame
  -> 1 Hz terminal report plus live controller-pose visualization
```

The upstream `v1.1.0` application streams left/right wrist poses and hand
landmarks. The lab headset currently has a controller-enabled build of the same
application package; this receiver also accepts its Touch-controller pose
records. The controller extension is not present in upstream `v1.1.0`, so a
reproducible Quest-side controller build remains future work.

Do not commit a locally modified submodule state for that extension: other
clones could not fetch it. The durable options are to contribute controller
support upstream or maintain a reachable project fork/branch and update the
submodule URL and gitlink to that commit.

## Start a safe pose-only test

Connect and authorize the Quest, then run from the repository root:

```bash
./scripts/start_meta_quest_pose_monitor.sh
```

In the headset choose:

```text
Protocol: TCP (Wired / ADB)
IP:       127.0.0.1
Port:     8000
Mode:     Controllers Only (controller-enabled build)
```

Press **Start Streaming**. The terminal should show a live pose for each
tracked side. The configuration panel intentionally disappears while streaming,
so seeing only passthrough in the headset at that point is normal. Press the
flat **Menu** button on the left controller to stop streaming and restore the
panel. Stop the computer-side monitor with `Ctrl+C`.

The launcher initially restores normal physical-proximity behavior. The monitor
requests Quest's mounted/awake state only after it receives the first controller
packet, so starting the monitor while Hand Tracking Streamer is closed does not
prevent normal headset sleep. After two seconds without a controller packet, or
when the monitor exits normally or through `Ctrl+C`, it restores physical
proximity sensing and normal headset sleep. Keep the Quest connected over USB,
and stop the monitor when it is no longer needed to avoid unnecessary display,
tracking, and battery use.

The terminal reports the latest controller poses once per second. A computer-side
window redraws whenever new controller records arrive, displays each controller
as a position plus quaternion-derived XYZ axes, and shows the measured receive
frequency for each side. The plot stays in the raw Unity tracking frame. To run
without a window (for example, over SSH), add `--no-visualization`.

The visualization now uses two distinct axis styles: pastel dashed arrows and a
center marker for the fixed Unity tracking frame, and saturated solid arrows
labelled at each controller for that controller's local orientation. This avoids
mistaking a controller-local axis for a global reference axis.

The controller-enabled APK currently installed on the lab Quest sends each
controller as `position, quaternion_xyzw, trigger, clutch, record`. The monitor
labels the front-trigger value as `grasp`; the final two flags correspond to
the B-button clutch and A-button recording controls used by the controller
build. They are displayed for input verification only and do not command a
robot.

If port 8000 is already occupied, stop the old listener first. ADB reverse
rules are cleared whenever the ADB server or device transport restarts; the
launcher recreates the rule each time.

## Coordinate and safety contract

All reported positions and `xyzw` quaternions remain in the Unity local tracking
frame. The installed controller build obtains them with
`OVRInput.GetLocalControllerPosition()` and
`OVRInput.GetLocalControllerRotation()`, relative to the Quest tracking origin;
they are not global robot-base coordinates and must not be sent directly to a
robot. A robot teleoperation path needs an explicit transform chain:

```text
controller pose in Quest local tracking frame
  -> Quest room/tracking-origin frame (if required)
  -> calibrated Quest-to-robot-base transform
  -> clutch-relative, limited robot target
```

The next layers need explicit implementations and tests for:

1. Quest-to-robot frame calibration and origin management.
2. Clutch-relative motion so tracking origin jumps cannot move the robot.
3. Workspace, velocity, acceleration, and rotation limits.
4. Stale/frozen-pose detection and hold behavior.
5. An explicit dead-man enable and emergency-stop path.
6. A dry-run target visualizer before any FR3 backend is connected.

The calibration primitive is now available in `calibration.py`. At the start of
teleoperation, define a safe target pose, hold the controller at the desired
starting pose, and use one deliberate button rising edge to capture the Quest
anchor. Subsequent poses are mapped as motion relative to that anchor, so the
captured pose maps exactly to the predefined target. The utility does not yet
enable robot output; it is the tested foundation for that later state machine.

The source anchor and target anchor must use the same coordinate convention. The
current Quest packets are Unity left-handed tracking-frame values, while robot
poses are normally right-handed robot-base values. Apply and test the explicit
Unity-to-robot axis conversion before composing those transforms; calibration is
not a substitute for that conversion.

"Global" for teleoperation therefore means global in the robot base frame, not
an absolute coordinate supplied by Quest. Quest's tracking origin can be
recentered or drift over time, so the practical arm mapping should capture a
calibration pose and use clutch-relative deltas, with workspace and velocity
limits, before enabling robot output.

The intended future module boundary is:

```text
validated Quest poses
  -> calibrated, clutch-relative target generator
  -> dry-run target visualizer and safety supervisor
  -> explicitly enabled robot backend
```
