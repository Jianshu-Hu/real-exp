# Wuji Hand Meshes

Mesh files for the Wuji dexterous hand, sourced from
[`wuji-technology/wuji-hand-description`](https://github.com/wuji-technology/wuji-hand-description).

Both **left** and **right** hands are integrated. Used by:
- `task/assets/urdf/panda_wuji_hand_right.urdf` (single arm, right hand)
- `task/assets/urdf/panda_dual_wuji_hand_trapezoid_base.urdf` (dual arm)

## Contents

| File pattern | Origin in upstream repo | Purpose |
|---|---|---|
| `right_palm_link.STL`, `left_palm_link.STL` | `meshes/right/`, `meshes/left/` | Palm — root of each hand's kinematic tree (mirrored, not symmetric) |
| `{left,right}_finger{1..5}_link{1..4}.STL` | `meshes/{left,right}/` | Phalanges of each of the five fingers |
| `{left,right}_finger{1..5}_tip_link.STL` | `meshes/{left,right}/` | Fingertip caps |
| `hand_docking_link.STL` | `docking/meshes/` | **Mounting adapter** between robot-arm flange and palm — shared between left and right (the adapter is geometrically symmetric) |

## ⚠️ Important: the docking adapter

The Wuji hand does **not** mount directly onto an arm flange. The manufacturer
ships a separate **docking adapter** (a small ~50 g plate, ~16 mm thick) that
sits between the arm and the palm. See
`step/adapter/Adapter-Installation-Instructions.md` in the upstream repo for the
real-world assembly procedure.

Our combined URDF (`panda_wuji_hand_right.urdf`) reflects this physical
assembly:

```
panda_link8 (arm flange)
  └── [fixed joint, rpy="0 0 π/4"]
      └── hand_docking_link  (← adapter from docking/meshes/)
          └── [fixed joint]
              └── right_palm_link
                  └── 5 × finger chains
```

### Why include the adapter

- **Sim-to-real fidelity**: the real hand sits ~1–2 cm farther from the flange
  than the bare palm would. Skipping the adapter introduces a constant offset
  error that surfaces during calibration, IK, and policy transfer.
- **Consistency with other hands**: Shadow ships with `forearm` + `wrist`
  meshes, Sharpa Wave ships with a `wrist` link in its `_with_wrist` URDF.
  Wuji's `hand_docking_link` is the equivalent piece.

### If you don't want the adapter

You can short-circuit the chain by making `panda_link8` the direct parent of
`right_palm_link`. The hand will still work in simulation; just be aware of the
sim/real geometry mismatch noted above.

## How these files were generated

```bash
git clone https://github.com/wuji-technology/wuji-hand-description.git
cp wuji-hand-description/meshes/right/*.STL          task/assets/wuji_hand_meshes/
cp wuji-hand-description/meshes/left/*.STL           task/assets/wuji_hand_meshes/
cp wuji-hand-description/docking/meshes/*.STL        task/assets/wuji_hand_meshes/
```

In the dual-arm URDF, the docking adapter is instantiated twice (once per side)
with side-prefixed link names (`left_hand_docking_link`, `right_hand_docking_link`)
both pointing to the same `hand_docking_link.STL` file.

URDF mesh paths were rewritten from the upstream `../meshes/right/...` /
`../meshes/...` conventions to `../wuji_hand_meshes/...` so everything resolves
relative to `task/assets/urdf/`.

See `docs/howto_add_new_hand.md` for the full integration procedure.

## Upstream license

Mesh files are redistributed under the original license of
`wuji-technology/wuji-hand-description`. See the upstream repository for the
authoritative LICENSE file.
