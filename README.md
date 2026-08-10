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

- Use the system ROS 2 / Franka environment for `gello_software`, ROS 2 launch files, and `data_collection/replay_lerobot_episode.py`.
- Use the `lerobot` Conda environment for `data_collection/lerobot_collection.py`, dataset inspection, dataset editing, and training under `lerobot/`.

## Common Workflows

For data collection and local dataset processing:

```bash
source ~/anaconda3/bin/activate
conda activate lerobot
python data_collection/lerobot_collection.py --help
python data_collection/delete_lerobot_episode.py --help
python data_collection/process_dataset.py --help
```

For replay through the ROS 2 collection controller:

```bash
python3 data_collection/replay_lerobot_episode.py --help
```

The detailed data collection and replay instructions now live in [data_collection/DATA_COLLECTION_README.md](data_collection/DATA_COLLECTION_README.md).
Training and deployment instructions live in [train/TRAIN_README.md](train/TRAIN_README.md) and [train/DEPLOY_README.md](train/DEPLOY_README.md).

## Notes

- Do not change the GELLO USB connections. If the topology changes, reinstall or update the USB rules and verify both aliases before launching ROS 2 nodes.
- Start the arm controllers in the correct mode for the task. Collection and episode replay use the normal collection controller; live policy execution uses `deployment_mode:=true` as described in [train/DEPLOY_README.md](train/DEPLOY_README.md).
- Check the GELLO offsets whenever the teleoperator joints do not align with the robot or after a hardware reset. Recompute the offsets with the procedure in [data_collection/DATA_COLLECTION_README.md](data_collection/DATA_COLLECTION_README.md) before collecting data.
