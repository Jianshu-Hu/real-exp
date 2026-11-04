# Calibration Matrices

`A_T_B` maps coordinates from B to A; translation is in metres.
W is centred on the tabletop AprilTag, with +X forward, +Y left, +Z up.
Camera frames are color optical frames: +X right, +Y down, +Z forward.
EE is the measured controller end-effector frame from Franka `o_t_ee`.

## Calibration record — 2026-11-04

Reference: tag36h11, ID 0, black-square side length 0.095 m.
Times below are China Standard Time (UTC+08:00).

| Calibration | Camera serial | Capture time |
| --- | --- | --- |
| Front D435i to world | `401622071701` | 2026-11-04 10:44:29 |
| Left base to world and left EE-to-D405 mount | `260322270745` | 2026-11-04 13:57:35–13:59:31 and 14:04:07–14:05:21 |
| Right base to world and right EE-to-D405 mount | `260322272504` | 2026-11-04 14:08:04–14:11:21 |

Left result: `validated_on_holdout`; combined two compatible recordings:
9 + 6 distinct settled poses, 156 images, no duplicate poses or rejected samples.
Used the default sample minimum: 12 training poses and 3 held-out poses, with
all held-out poses from the new recording. Held-out median/max residual:
1.80/4.32 mm and 0.245/0.348 degrees. Median selected-pose reprojection error:
0.153 px. Additional leave-one-pose-out checks: maximum prediction residual
3.99 mm / 0.369 degrees; estimated base position varied by up to 4.15 mm and
mount position by 2.69 mm relative to the saved fit (rotation differences below
0.36 degrees). These consistency checks do not measure absolute accuracy
against an independent reference.

Right result: `validated_on_holdout`; 22 accepted settled poses and 243 tag
inlier images from 25 selected poses (286 images). Three poses were rejected
for fewer than five clear tag observations. Used the default sample minimum:
19 training poses and 3 held-out poses. Held-out median/max residual:
3.11/3.17 mm and 0.151/0.214 degrees. Median selected-pose reprojection error:
0.097 px. Additional leave-one-pose-out checks: maximum prediction residual
5.27 mm / 0.541 degrees; estimated base position varied by up to 2.94 mm and
mount position by 1.25 mm relative to the saved fit (rotation differences below
0.24 degrees).

Sources:

- Front: `calibration/runs/tag_world_20261104_104307/d435i_hd/camera_to_world.json`.
- Left: `calibration/runs/calibration_20261104_140402/left/samples_left_combined/wrist_calibration.json`.
- Left provenance: `calibration/runs/calibration_20261104_140402/left/samples_left_combined/merge_manifest.json`.
- Left validation: `calibration/runs/calibration_20261104_140402/left/samples_left_combined/validation_audit.json`.
- Right: `calibration/runs/calibration_20261104_140735/right/samples_right/wrist_calibration.json`.
- Right validation: `calibration/runs/calibration_20261104_140735/right/samples_right/validation_audit.json`.

Both arm-base/world and fixed EE-to-wrist-camera transforms are calibrated
against the same tabletop tag/world convention.

```text
W_T_D435I =
[[ 0.005654385, -0.918375380,  0.395669925, -0.548542816],
 [-0.999980942, -0.006173662, -0.000039079,  0.109195776],
 [ 0.002478622, -0.395662163, -0.918392786,  0.897067971],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

W_T_BASE_LEFT =
[[ 0.999934071, -0.011367786, -0.001620985, -0.622264151],
 [ 0.009222720,  0.710984613,  0.703147084,  0.274698354],
 [-0.006840730, -0.703115676,  0.711042580,  0.099652877],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

EE_LEFT_T_D405_LEFT =
[[-0.047083886,  0.998889961, -0.001398027, -0.059730162],
 [-0.998647703, -0.047041490,  0.022132852,  0.003957835],
 [ 0.022042518,  0.002438237,  0.999754061, -0.077376097],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

F_LEFT_T_EE_LEFT =
[[ 0.707106789,  0.707106769,  0.000000000,  0.000000000],
 [-0.707106769,  0.707106789, -0.000000000,  0.000000000],
 [-0.000000000,  0.000000000,  1.000000000,  0.103399999],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

W_T_BASE_RIGHT =
[[ 0.999848869,  0.000796985,  0.017366748, -0.620139737],
 [ 0.011952964,  0.693865182, -0.720005719, -0.116235540],
 [-0.012624015,  0.720104488,  0.693750791,  0.091266725],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

EE_RIGHT_T_D405_RIGHT =
[[ 0.043820769, -0.999039244, -0.000574219,  0.065736692],
 [ 0.999024164,  0.043823268, -0.005499240, -0.011985249],
 [ 0.005519120, -0.000332678,  0.999984714, -0.076970965],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

F_RIGHT_T_EE_RIGHT =
[[ 0.707106789,  0.707106769,  0.000000000,  0.000000000],
 [-0.707106769,  0.707106789, -0.000000000,  0.000000000],
 [-0.000000000,  0.000000000,  1.000000000,  0.103399999],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]
```

`F_LEFT_T_EE_LEFT` and `F_RIGHT_T_EE_RIGHT` record the configured
flange-to-controller-EE transforms used during calibration. If a controller EE
definition changes, convert its camera mount through the flange before using
it with new measured EE poses. Calibration residuals are consistency checks;
they do not establish independent absolute accuracy.

Each wrist camera pose in its arm base depends on the current measured arm pose:

```text
BASE_LEFT_T_D405_LEFT(q) = B_LEFT_T_EE_LEFT(q) @ EE_LEFT_T_D405_LEFT
BASE_RIGHT_T_D405_RIGHT(q) = B_RIGHT_T_EE_RIGHT(q) @ EE_RIGHT_T_D405_RIGHT
W_T_D405_LEFT(q) = W_T_BASE_LEFT @ BASE_LEFT_T_D405_LEFT(q)
W_T_D405_RIGHT(q) = W_T_BASE_RIGHT @ BASE_RIGHT_T_D405_RIGHT(q)
BASE_LEFT_T_D435I = inverse(W_T_BASE_LEFT) @ W_T_D435I
BASE_RIGHT_T_D435I = inverse(W_T_BASE_RIGHT) @ W_T_D435I
```
