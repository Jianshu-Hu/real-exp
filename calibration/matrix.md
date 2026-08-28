# Calibration Matrices

The robot-base matrices below were recomputed on 2026-08-28 after correcting
the hand-eye sample intrinsics to the active 640x480 `cam_front` profile.

Notation: `A_T_B` transforms coordinates from frame `B` into frame `A`:

```text
p_A = A_T_B @ p_B
```

The translation entries are in metres. Frames used below:

- `C`: `cam_front` D435 optical camera frame
- `B_L`: left Franka base frame
- `B_R`: right Franka base frame
- `W`: tabletop world frame (`+x` forward, `+y` left, `+z` up)

Both robot-base calibrations use the `cam_front` color profile for serial
`401622071701` at 640x480: `fx=606.1522`, `fy=605.6415`, `cx=322.8838`,
`cy=255.9408`, with zero reported distortion coefficients.

## Left Robot Base to Camera

Direction: `B_L -> C`.

Source field: `camera_T_base` in
`calibration/runs/left_arm_camera_calibration_samples/camera_to_robot_base.json`.

```text
C_T_B_L =
[[ -0.094872488, -0.705592508, -0.702238153, -0.194672067],
 [ -0.924629912,  0.323839306, -0.200468526,  0.571976021],
 [  0.368861407,  0.630291454, -0.683135379,  0.898757710],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

p_C = C_T_B_L @ p_B_L
```

Calibration quality:

- Used samples: `20/20`
- Median Tag reprojection RMSE: `0.218 px`
- Median hand-eye translation residual: `4.839 mm`
- Median hand-eye rotation residual: `0.03353 rad` (`1.921 deg`)

## Right Robot Base to Camera

Direction: `B_R -> C`.

Source field: `camera_T_base` in
`calibration/runs/right_arm_camera_calibration_samples/camera_to_robot_base.json`.

```text
C_T_B_R =
[[ 0.061077178, -0.724658647,  0.686395967,  0.162840030],
 [-0.927423222, -0.295425134, -0.229369041,  0.562456279],
 [ 0.368992880, -0.622570346, -0.690108991,  0.901632663],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

p_C = C_T_B_R @ p_B_R
```

Calibration quality (`sample_000014` excluded as a PnP pose outlier):

- Used samples: `19/20` (`sample_000014` excluded)
- Median Tag reprojection RMSE: `0.227 px`
- Median hand-eye translation residual: `4.378 mm`
- Median hand-eye rotation residual: `0.06000 rad` (`3.438 deg`)

## Camera to World

Direction: `C -> W`.

Source field: `world_T_camera` in
`calibration/runs/table_tag_20260824/camera_to_world.json`.

```text
W_T_C =
[[ 0.016116505, -0.947169025,  0.320329670, -0.394891761],
 [-0.998707711,  0.000194370,  0.050821951, -0.041552817],
 [-0.048199240, -0.320734783, -0.945941876,  1.159142768],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

p_W = W_T_C @ p_C
```

Calibration quality:

- Valid Tag detections: `100/100`
- Median Tag reprojection RMSE: `0.063 px`
- Mean Tag reprojection RMSE: `0.068 px`
- Maximum Tag reprojection RMSE: `0.244 px`
- Dominant planar-PnP branch: `97/100` frames within `10 mm` and `1 deg` of the dominant pose
- PnP pose outliers: frames `3`, `14`, and `38`
- Maximum outlier deviation from the dominant pose: `754.7 mm`, `35.61 deg`
- Reported aggregate rotation differs from the dominant pose by `0.654 deg`

To transform a point from either robot base directly into the world frame:

```text
p_W = W_T_C @ C_T_B_L @ p_B_L
p_W = W_T_C @ C_T_B_R @ p_B_R
```

As a cross-calibration check, the two hand-eye matrices imply:

```text
left_base_T_right_base translation = [-0.0241, -0.2535, -0.2511] m
left_base_T_right_base RPY         = [87.96, 6.26, -6.40] deg
```

This derived relation is a useful consistency value, but should be compared
against an independent measurement of the physical base mounting before it is
used as a base-to-base calibration.
