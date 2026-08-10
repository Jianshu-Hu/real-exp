# Real Experiments

This repository groups the code and notes used for real-world Franka FR3 experiments with GELLO teleoperation, LeRobot dataset collection, and training workflows.

## Repository Layout

- `data_collection/`: scripts and documentation for recording, replaying, and editing local LeRobot datasets.
- `train/`: training-related notes and experiment-specific training assets.
- `deploy/`: policy inspection, serving, fetching, and real-robot execution tools.
- `utils/`: shared dataset-statistics and image-preprocessing helpers.
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

## Common Workflows (Data collection-Train-Deploy)
- Data collection and replay instructions live in [data_collection/DATA_COLLECTION_README.md](data_collection/DATA_COLLECTION_README.md).
- Training instructions live in [train/TRAIN_README.md](train/TRAIN_README.md).
- Deployment instructions live in [deploy/DEPLOY_README.md](deploy/DEPLOY_README.md).

## Notes

- Do not change the GELLO USB connections. If the topology changes, reinstall or update the USB rules and verify both aliases before launching ROS 2 nodes.
- Check the GELLO offsets whenever the teleoperator joints do not align with the robot or after a hardware reset. Recompute the offsets with the procedure in [data_collection/DATA_COLLECTION_README.md](data_collection/DATA_COLLECTION_README.md) before collecting data.
