# Calibration Matrices

Notation: `A_T_B` transforms coordinates from frame `B` into frame `A`:

```text
p_A = A_T_B @ p_B
```

The translation entries are in metres. Frames used below:

- `C`: `cam_front` D435 optical camera frame
- `B_L`: left Franka base frame
- `B_R`: right Franka base frame
- `W`: tabletop world frame (`+x` forward, `+y` left, `+z` up)

## Left Robot Base to Camera

Direction: `B_L -> C`.

Source field: `camera_T_base` in
`calibration/runs/left_arm_camera_calibration_samples/camera_to_robot_base.json`.

```text
C_T_B_L =
[[ 0.028105207, -0.794041066, -0.607214034, -0.310235811],
 [-0.934696878,  0.194450200, -0.297541367,  0.543971151],
 [ 0.354332955,  0.575923524, -0.736722643,  0.707608360],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

p_C = C_T_B_L @ p_B_L
```

Calibration quality:

- Valid samples: `20/20`
- Median Tag reprojection RMSE: `0.226 px`
- Median hand-eye translation residual: `9.995 mm`
- Median hand-eye rotation residual: `0.04513 rad` (`2.586 deg`)

## Right Robot Base to Camera

Direction: `B_R -> C`.

Source field: `camera_T_base` in
`calibration/runs/right_arm_camera_calibration_samples/camera_to_robot_base.json`.

```text
C_T_B_R =
[[ 0.064235673, -0.841162479,  0.536953874,  0.249297696],
 [-0.956780317, -0.204838067, -0.206428660,  0.513843229],
 [ 0.283628637, -0.500486814, -0.817965614,  0.782537268],
 [ 0.000000000,  0.000000000,  0.000000000,  1.000000000]]

p_C = C_T_B_R @ p_B_R
```

Calibration quality (`sample_000014` excluded as a PnP pose outlier):

- Valid samples: `19/19`
- Median Tag reprojection RMSE: `0.248 px`
- Median hand-eye translation residual: `11.152 mm`
- Median hand-eye rotation residual: `0.06922 rad` (`3.966 deg`)

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
left_base_T_right_base translation = [0.0704, -0.4070, -0.3860] m
left_base_T_right_base RPY         = [70.23, -2.10, -4.23] deg
```

The approximately `70 deg` relative roll is consistent with the two robot
bases being mounted diagonally in opposite directions.
