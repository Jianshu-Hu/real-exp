# Real-World Wuji Hand Grasping Experiment Plan

## Objective

Deploy the RGB-D grasp-generation pipeline from
`code/anygrasp_sapien/scripts/debug/execute_generator_grasp.py` to the real
Franka/Wuji platform in `/data/home/wjh/real-exp`. The real-time path is:

`D435 RGB-D -> point-cloud filtering -> camera/world transform -> generator inference -> retargeting -> semantic refinement -> world wrist pose -> Franka-base EE pose -> arm and Wuji joint commands`

The generator and refinement stages operate on the RoboDex z-up world frame.
`data_collection/move_to_target_ee.py` accepts a Franka EE pose and 20 Wuji
joint values; the pose must be expressed in the selected Franka robot-base
frame.

## Coordinate conventions

Use homogeneous transforms with the notation `A_T_B`, meaning that a point in
frame B is transformed into frame A. Keep transform direction in variable names.

- `W`: tabletop/world frame. Origin is the center of the tabletop tag.
- `C`: D435 optical camera frame (`x` right, `y` down, `z` forward).
- `B`: Franka robot-base frame.
- `E`: Franka flange/EE frame used by the ROS controller.
- `T`: AprilTag frame mounted on the calibration target or end effector.
- `H`: Wuji wrist/root frame used by the generator and RoboDex URDF.

The required calibration outputs are `W_T_C`, `C_T_B`, and consequently:

```text
W_T_B = W_T_C @ C_T_B
B_T_W = inverse(W_T_B)
```

Use `B_T_W` to convert generated world-frame wrist poses into the robot base.

## Camera-to-world calibration

Place a rigid AprilTag board at the intended tabletop origin. Define its pose in
the world frame, including the tag-to-table height and the desired world-axis
orientation. Detect the board pose from the RGB image using the calibrated D435
intrinsics and distortion model. If the detector returns `C_T_T`, compute:

```text
W_T_C = W_T_T @ inverse(C_T_T)
```

Verify the detector's pose convention by projecting known tag corners back into
the image. A single planar tag is sensitive to noise and pose ambiguity; prefer
an AprilGrid or several tags with a robust PnP fit. Independently fit the table
plane from depth and check that it is `z=0` in W. Record tag size, physical
offsets, camera serial number, intrinsics, depth scale, and the calibration
matrix in a versioned JSON file.

## Camera-to-robot-base calibration (eye-to-hand)

Mount a second rigid AprilTag close to the Franka flange. At each of many static
robot configurations, record the measured robot pose `B_T_E` and the
time-synchronized camera observation `C_T_T_i`.

For pose pairs define:

```text
A_ij = inverse(B_T_E_i) @ B_T_E_j
B_ij = inverse(C_T_T_i) @ C_T_T_j
```

The hand-eye equation is:

```text
A_ij @ X = X @ B_ij
```

where `X = E_T_T`, the fixed EE-to-tag transform. Thus `AX=XB` does not
directly produce camera-to-base. After estimating `E_T_T`, obtain camera-to-base
from every absolute observation and jointly refine over all samples:

```text
C_T_B_i = C_T_T_i @ inverse(B_T_E_i @ E_T_T)
```

Use a nonlinear SE(3) least-squares refinement of both `E_T_T` and `C_T_B` when
possible. Collect 10--20 or more poses with varied translations and rotations;
avoid motions consisting only of a single-axis rotation or nearly coplanar
configurations. Use measured, settled robot states rather than commanded
targets, and reject samples with blurred or low-confidence tag detections.

Validate on held-out poses by predicting the observed tag pose and reporting
translation and rotation residuals. The tag mount must be rigid and must not
occlude the grasp workspace during normal operation.

## Point-cloud and inference pipeline

1. Acquire synchronized D435 color/depth frames and align depth to color.
2. Back-project depth with the calibrated intrinsics and apply depth-range,
   validity, workspace, and table-plane filters.
3. Transform points with `W_T_C`; keep the generator input in RoboDex's z-up W
   frame. Save diagnostic clouds before and after filtering.
4. Run generator inference, hand retargeting, and semantic contact refinement
   using the same checkpoint, hand adapter, joint limits, and joint ordering as
   the simulation script.
5. Reject results with insufficient object points, invalid values, excessive
   penetration, or a wrist outside the calibrated robot workspace.

## Wuji wrist to Franka EE

Do not send the generated wrist pose directly as a Franka EE pose. Extract the
fixed mount transform from the exact RoboDex `franka/panda with wuji` URDF and
verify its meaning and handedness against the physical assembly. If the URDF
provides `H_T_E`, compute:

```text
W_T_E = W_T_H @ H_T_E
B_T_E_target = B_T_W @ W_T_E
```

Then pass `B_T_E_target` (converted to the script's XYZ + ZYX roll/pitch/yaw
format) and the unchanged 20-element Wuji joint vector to
`move_to_target_ee.py`. Confirm left/right hand mirroring and exact joint-name
ordering before enabling hardware commands.

### Current RoboDex URDF result

The nominal transform was read from:

`data/githubRepo/RoboDex/task/assets/urdf/panda_wuji_hand_right.urdf`

The relevant fixed chain is:

```text
panda_link8 --panda_to_wuji_docking_joint--> hand_docking_link
hand_docking_link --wuji_docking_to_palm_joint--> right_palm_link
```

The first joint has `xyz="0 0 0"` and
`rpy="0 0 2.3561944902"`; the second joint is identity. Therefore the
model nominal transform is:

```text
panda_link8_T_hand_docking =
[[ -0.70710678, -0.70710678, 0.0, 0.0 ],
 [  0.70710678, -0.70710678, 0.0, 0.0 ],
 [  0.0,         0.0,        1.0, 0.0 ],
 [  0.0,         0.0,        0.0, 1.0 ]]

panda_link8_T_right_palm = panda_link8_T_hand_docking
```

Here `hand_docking_link` is the root link of the Wuji hand-only URDF used by
the retargeting adapter, while `right_palm_link` is the palm landmark link.
The inverse matrix must be used when converting a generated palm/root pose
into `panda_link8` coordinates. This is a nominal CAD value: it must be
validated against the physical Wuji mount and the real robot's `fr3_link8`,
flange, and controller EE frame. The end-effector AprilTag calibration remains
the source of truth for the real installed transform and can be used to
estimate the discrepancy from this URDF value.

## Verification and safety gates

- First run the complete pipeline in offline/replay mode and inspect saved RGB-D,
  point clouds, transforms, and hand meshes.
- Test calibration with a known tabletop point and a held-out end-effector tag
  pose; require small residuals before motion is enabled.
- Execute arm-only moves with the hand open, conservative approach offsets,
  bounded velocity/acceleration, and workspace/IK checks.
- Require a human-confirmed dry run before closing the Wuji hand. Keep an
  emergency stop available and abort on stale frames, lost tags, transform
  discontinuities, IK failure, or controller timeout.
- Log timestamps, camera metadata, all calibration matrices, measured robot
  states, generated poses, joint targets, residuals, and execution outcomes for
  every trial.

## Deliverables

- Calibration utility and versioned calibration JSON (`W_T_C`, `C_T_B`,
  `E_T_T`, intrinsics, tag sizes, and residual statistics).
- RealSense RGB-D acquisition and filtering module.
- Real-world inference/retargeting/refinement runner with offline debug output.
- Explicit RoboDex-URDF-derived Wuji mount transform and joint-order mapping.
- Hardware-gated executor that converts world wrist targets to Franka EE targets
  and calls `move_to_target_ee.py` only after all checks pass.
