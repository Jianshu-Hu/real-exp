# Real Experiments

This repository groups the code and notes used for real-world Franka FR3 experiments with GELLO teleoperation, LeRobot dataset collection, and training workflows.

## Repository Layout

- `data_collection/`: scripts and documentation for recording, replaying, and editing local LeRobot datasets.
- `train/`: training-related notes and experiment-specific training assets.
- `lerobot/`: vendored LeRobot codebase used by the local collection and training workflow.
- `gello_software/`: GELLO and ROS 2 integration code, tracked as a git submodule.
- `data/`: local datasets and experiment outputs.

## Submodules

The repo expects `lerobot/` and `gello_software/` to be available locally. If either directory is empty after cloning, initialize the submodules:

```bash
git submodule update --init --recursive
```

## Environment Split

This repo uses two different Python environments:

- Use the system ROS 2 / Franka environment for `gello_software`, ROS 2 launch files, and `data_collection/replay_pylibfranka.py`.
- Use the `lerobot` Conda environment for `data_collection/lerobot_collection.py`, dataset inspection, dataset editing, and training under `lerobot/`.

## Common Workflows

For data collection and replay:

```bash
source ~/anaconda3/bin/activate
conda activate lerobot
python data_collection/lerobot_collection.py --help
python data_collection/delete_lerobot_episode.py --help
```

For replay with `pylibfranka`:

```bash
python3 data_collection/replay_pylibfranka.py --help
```

The detailed data collection and replay instructions now live in [data_collection/DATA_COLLECTION_README.md](data_collection/DATA_COLLECTION_README.md).

## Notes

* Do not change GELLO USB connection. If the device mapping changes, verify the configured serial port or by-id path before launching the ROS 2 nodes again.
* Start the arm controllers in the correct mode for the task you are running. Use the normal teleoperation/data-collection launch flow for recording, and use `deployment_mode:=true` only for live policy execution as described in [train/DEPLOY_README.md](train/DEPLOY_README.md).
* Check the GELLO offsets whenever the teleoperator joints do not align with the robot or after any hardware reset. Recompute the offsets with the procedure in [data_collection/DATA_COLLECTION_README.md](data_collection/DATA_COLLECTION_README.md) and update the selected GELLO config before collecting data.
