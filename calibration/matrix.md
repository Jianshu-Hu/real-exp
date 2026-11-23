# Calibration Matrices

`A_T_B` maps coordinates from B to A. Translation is in metres.
World W is centred on the current tabletop AprilTag, with +X forward,
+Y left, and +Z up. Camera transforms use the colour optical frame
(+X right, +Y down, +Z forward). EE is the Franka controller frame from
measured `o_t_ee`.

## Calibration record — 2026-11-23

The 95.5 mm tag36h11 AprilTag (ID 0) defines the world frame. The D435i
fit its pose from the large tag and estimated the 6×6 AprilGrid pose in
the same frame using a 35.2 mm tag edge and a provisional 45.7 mm pitch.
Both wrist fits used synchronized robot poses and corners from both targets.
The refit also used the D435i-to-wrist transforms estimated from six shared
grid placements with the arms parked. The Franka end-effector poses were
read after the capture and paired with the parked views using stationary
image features. The refit used a lower weight for this additional pose
to retain agreement across the moving-arm observations.

| Camera | Serial | Training / held-out observations |
| --- | --- | ---: |
| Front D435i | `401622071701` | 24 / 6 frames |
| Left wrist D405 | `260322270745` | 13 / 3 arm poses |
| Right wrist D405 | `260322272504` | 22 / 3 arm poses |

The D435i held-out reprojection RMSE was 0.130 px for the large tag and
0.417 px for the grid. The table below reports median residuals for the
held-out wrist poses against each target.

| Wrist | Translation, large tag / grid | Rotation, large tag / grid | Reprojection, large tag / grid |
| --- | ---: | ---: | ---: |
| Left | 9.1 / 4.4 mm | 0.65 / 0.66° | 2.19 / 1.88 px |
| Right | 19.6 / 7.9 mm | 1.49 / 1.13° | 5.03 / 2.95 px |

At the parked pose, the refit reduced the position difference from the
shared-grid estimate from 19.8 to 13.2 mm on the left and from 17.6 to
14.1 mm on the right. On the original held-out poses, the right-wrist
reprojection errors increased by 0.12 px for the large tag and 0.07 px for
the grid. The parked pose contributed to the fit, so its residual measures
agreement with the added observation. Absolute accuracy still needs an
independent measurement.

## Transform matrices

```text
W_T_D435I =
[[ 0.007593527589, -0.911088124413,  0.412141684245, -0.561943627096],
 [-0.999935534966, -0.003438900982,  0.010821269473,  0.096236825966],
 [-0.008441815666, -0.412197287125, -0.911055504475,  0.883740800845],
 [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]]

W_T_APRILGRID =
[[ 0.006705429567,  0.999922565252,  0.010483354146, -0.032268938309],
 [-0.999968030549,  0.006659329483,  0.004426195896, -0.339146087775],
 [ 0.004356041045, -0.010512698544,  0.999935251942,  0.018443192340],
 [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]]

W_T_BASE_LEFT =
[[ 0.999974256428, -0.006852320318,  0.002128893295, -0.627575669251],
 [ 0.003313779619,  0.704176757199,  0.710016981124,  0.283945486077],
 [-0.006364380963, -0.709991648068,  0.704181336254,  0.111741496072],
 [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]]

EE_LEFT_T_D405_LEFT =
[[-0.039645861946,  0.999210708857,  0.002482928806, -0.065780451240],
 [-0.999191441461, -0.039661473412,  0.006590207772,  0.015113740767],
 [ 0.006683482794, -0.002219646745,  0.999975201806, -0.076859663827],
 [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]]

F_LEFT_T_EE_LEFT =
[[ 0.707106789136,  0.707106769085,  0.000000000000,  0.000000000000],
 [-0.707106769085,  0.707106789136, -0.000000000000,  0.000000000000],
 [-0.000000000000,  0.000000000000,  1.000000000000,  0.103399999440],
 [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]]

W_T_BASE_RIGHT =
[[ 0.999785786197,  0.019274056508,  0.007542709290, -0.623397374737],
 [-0.008340422414,  0.708707322912, -0.705453306609, -0.097050368875],
 [-0.018942520204,  0.705239279392,  0.708716261794,  0.109403767638],
 [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]]

EE_RIGHT_T_D405_RIGHT =
[[ 0.048297620916, -0.998832869869,  0.000487731654,  0.059700822426],
 [ 0.998814225347,  0.048293714821, -0.006153076783, -0.015688023591],
 [ 0.006122340968,  0.000784332284,  0.999980950701, -0.074630826994],
 [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]]

F_RIGHT_T_EE_RIGHT =
[[ 0.707106789136,  0.707106769085,  0.000000000000,  0.000000000000],
 [-0.707106769085,  0.707106789136, -0.000000000000,  0.000000000000],
 [-0.000000000000,  0.000000000000,  1.000000000000,  0.103399999440],
 [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]]
```

Use the measured EE pose at image capture time for wrist-camera transforms.
If the controller EE frame changes, use the recorded flange-to-EE transform
to express the camera mount in the new EE frame.

```text
BASE_LEFT_T_D405_LEFT(q) = B_LEFT_T_EE_LEFT(q) @ EE_LEFT_T_D405_LEFT
BASE_RIGHT_T_D405_RIGHT(q) = B_RIGHT_T_EE_RIGHT(q) @ EE_RIGHT_T_D405_RIGHT
W_T_D405_LEFT(q) = W_T_BASE_LEFT @ BASE_LEFT_T_D405_LEFT(q)
W_T_D405_RIGHT(q) = W_T_BASE_RIGHT @ BASE_RIGHT_T_D405_RIGHT(q)

D435I_T_BASE_LEFT = inverse(W_T_D435I) @ W_T_BASE_LEFT
D435I_T_BASE_RIGHT = inverse(W_T_D435I) @ W_T_BASE_RIGHT
D435I_T_D405_LEFT(q) = D435I_T_BASE_LEFT @ BASE_LEFT_T_D405_LEFT(q)
D435I_T_D405_RIGHT(q) = D435I_T_BASE_RIGHT @ BASE_RIGHT_T_D405_RIGHT(q)
BASE_LEFT_T_D435I = inverse(D435I_T_BASE_LEFT)
BASE_RIGHT_T_D435I = inverse(D435I_T_BASE_RIGHT)
```

## Source files

- [D435i calibration and grid pose](runs/d435i_dual_target_hbrQbvOa/camera_to_world.json)
- [Left wrist calibration](runs/d435i_dual_target_hbrQbvOa/left/samples_left/dual_target_wrist_calibration.json)
- [Right wrist calibration](runs/d435i_dual_target_hbrQbvOa/right/samples_right/dual_target_wrist_calibration.json)
- [Detected targets](runs/d435i_dual_target_hbrQbvOa/detected_targets.jpg)
- [Shared-grid camera transforms and placement checks](runs/shared_grid_lPLuegFX/shared_grid_calibration.json)
- [Parked Franka end-effector poses](runs/shared_grid_lPLuegFX/parked_robot_states.json)
- [Parked image matching](runs/shared_grid_lPLuegFX/parked_pose_verification.json)
- [Left wrist refit and residuals](runs/shared_grid_lPLuegFX/left_parked_constraint_refit.json)
- [Right wrist refit and residuals](runs/shared_grid_lPLuegFX/right_parked_constraint_refit.json)
