# Calibration Matrices

The robot-base matrices below were recomputed on 2026-08-28 after correcting
the hand-eye sample intrinsics to the active 640x480 `cam_front` profile.

Notation: `A_T_B` transforms coordinates from frame `B` into frame `A`:

```text
p_A = A_T_B @ p_B
```

The translation entries are in metres. Frames used below:

- `C`: `cam_front` D435 optical camera frame
- `L`: L515 optical color-camera frame
- `B_L`: left Franka base frame
- `B_R`: right Franka base frame
- `W`: tabletop world frame (`+x` forward, `+y` left, `+z` up)

Both robot-base calibrations use the `cam_front` color profile for serial
`401622071701` at 640x480: `fx=606.1522`, `fy=605.6415`, `cx=322.8838`,
`cy=255.9408`, with zero reported distortion coefficients.

## L515 to D435i

The installed L515 is serial `f1480539`, firmware `1.5.4.1`. The current pair
calibration was recaptured on 2026-09-12 at 10:24 CST with both fixed
cameras observing the same unmoved 95 mm tag36h11 marker, ID 0.

- D435i reference serial: `401622071701`
- L515 serial: `f1480539`
- Required direction: `L515 -> D435i`
- Runtime copies: `CALIBRATED_L515_SERIAL` and
  `CALIBRATED_D435I_T_L515` in `grasp/inference_client.py`

```text
D435I_T_L515 =
[[ -0.289584691,  0.673230171, -0.680368903,  0.493053701],
 [ -0.887169286,  0.078015047,  0.454801397, -0.382073810],
 [  0.359265034,  0.735305916,  0.574677166,  0.755266560],
 [  0.000000000,  0.000000000,  0.000000000,  1.000000000]]

p_D435I = D435I_T_L515 @ p_L515
W_T_L515 = W_T_D435I @ D435I_T_L515
```

Calibration evidence:

- D435i capture: `calibration/runs/d435i_recal_20260912_102413`
- L515 capture: `calibration/runs/l515_recal_20260912_102413`
- Pair result JSON: `calibration/runs/d435i_T_l515_20260912_102413.json`
- Composition: `D435I_T_L515 = inverse(WORLD_T_D435I_PAIR) @
  WORLD_T_L515_PAIR`, using `world_T_camera` from the two capture directories'
  `camera_to_world.json` files
- Tag parameters: `tag36h11`, ID `0`, measured black-square side `0.095 m`
- Saved D435i RGB and aligned depth geometry: `640x480`
- D435i native streams: `640x480@30` color and depth
- L515 native streams: `1280x720@30` color, `640x480@30` depth; depth aligned
  into the color pixel grid before saving
- D435i color/depth timestamp gap: median/max `0.018/0.018 ms`
- L515 color/depth timestamp gap: median/max `5.495/6.348 ms`
- D435i Tag detections and dominant cluster: `100/100`; median/max reprojection
  RMSE `0.272/0.272 px`
- L515 Tag detections and dominant cluster: `100/100`; median/max reprojection
  RMSE `0.061/0.198 px` (mean `0.062 px`)
- Optical-origin baseline: `0.979546 m`
- Independent fused-depth validation capture:
  `calibration/runs/fusion_check_20260912_102550`
- The validation used 15-frame per-pixel nonzero medians, a `0.2--2.0 m`
  depth range, and 3 mm voxels. In the common robust-bounds region, the new
  transform's symmetric nearest-neighbor distance was `7.472 mm` median and
  `18.654 mm` at p90, with `83.497%` of points within `15 mm`.
- On the same validation capture, the previous runtime transform produced
  `18.171 mm` median, `104.458 mm` at p90, and `40.203%` within `15 mm`.
  The new transform therefore materially improves independent depth alignment.
- Relative to the previous runtime transform, this calibration changes the
  camera-to-camera pose by `957.164 mm` and `146.855 deg`, consistent with the
  L515 having been moved to its new installation position.

The solver selects the largest pairwise SE(3) cluster within 10 mm and 1 degree
before averaging, preventing low-reprojection planar-PnP branch flips from
biasing the installed transform. The numerical fused-depth check passed, but
the generated fused cloud should still be inspected visually for doubled edges
before hardware execution.

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
