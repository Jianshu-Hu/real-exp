# Dataset Quality Audit and Repair

`dataset_quality_report.py` audits locally stored LeRobot datasets for structural,
video, numeric, and robot-motion quality problems. It can also create a repaired
dataset after the user reviews an explicit repair plan.

The source dataset is always read-only. Reports, repair plans, and post-repair
reports are stored outside every dataset directory so LeRobot training cannot
mistake them for dataset metadata or samples.

## Requirements

Run the tool from the same Conda environment used for LeRobot collection or
training. The environment must provide `numpy` and `pyarrow`. Physical MP4 frame
checks additionally require OpenCV.

```bash
cd <REAL_EXP_REPO>
source <CONDA_INSTALL>/bin/activate
conda activate <LEROBOT_ENV>
```

## Scan One Dataset

```bash
python data_collection/dataset_quality_report.py scan \
  --dataset-root <DATASET_ROOT>
```

By default, reports are written under:

```text
~/.local/share/real-exp/dataset-quality/<DATASET_NAME>/<UTC_TIMESTAMP>/
```

This path is deliberately separate from `<DATASET_ROOT>`. The tool rejects a
`--report-root` located inside the source dataset.

Override the external report root when needed:

```bash
python data_collection/dataset_quality_report.py scan \
  --dataset-root <DATASET_ROOT> \
  --report-root <EXTERNAL_REPORT_ROOT>
```

Do not set `<EXTERNAL_REPORT_ROOT>` to the dataset itself or one of its
subdirectories.

## Scan Multiple Datasets

The batch command checks immediate child directories that match a glob and skips
directories that are not readable LeRobot datasets:

```bash
python data_collection/dataset_quality_report.py scan \
  --datasets-root <DATASETS_PARENT_DIR> \
  --name-pattern '<TASK_NAME_PATTERN>'
```

Use `--check-video-frames` to open the MP4 containers and verify physical frame
counts. Without this option, the default scan still checks camera metadata,
timestamp ranges, and cross-camera consistency without decoding video files.

```bash
python data_collection/dataset_quality_report.py scan \
  --dataset-root <DATASET_ROOT> \
  --check-video-frames
```

## Report Files

Each report directory contains only external audit artifacts:

```text
summary.md        Human-readable summary and highest-risk samples
report.json       Complete machine-readable report
events.csv        One row per detected event
repair_plan.json  Per-episode repair choices; no action is selected initially
```

These files are never copied into the source dataset or a repaired dataset.

Findings use four severity levels:

| Severity | Meaning |
|---|---|
| `ERROR` | Corruption, invalid values, hard-limit violations, or broken synchronization |
| `UNSAFE` | A saved action is outside the configured deployment safety envelope |
| `REVIEW` | A diagnostic signal needs human review; 15 Hz state acceleration is included here |
| `INFO` | Statistical outliers such as unusual initial configurations or duration |

The motion checks cover:

- FR3 hard joint position ranges
- the configurable position safety margin
- measured-state and saved-action velocity
- measured-state and saved-action acceleration
- absolute action-to-state tracking error
- episode initial-configuration distance from the task median
- gripper range, finite values, frame/timestamp continuity, and exact duplicates

Measured gripper state uses a `0.002 m` tolerance for small closed-width sensor
noise. Normalized gripper actions retain a strict `1e-5` tolerance.

State velocity and acceleration are reconstructed from the dataset FPS. In
particular, 15 Hz state acceleration is diagnostic and is not treated as proof of
a 1 kHz hardware acceleration violation.

## Review Repair Choices

The scan never selects or applies a repair automatically. Review the generated
plan interactively:

```bash
python data_collection/dataset_quality_report.py review \
  --plan <EXTERNAL_REPORT_DIR>/repair_plan.json
```

The same JSON file can be edited manually by setting `selected_action` to one of
the listed `allowed_actions`.

Available choices are:

| Choice | Result |
|---|---|
| `keep` | Keep the episode unchanged in the derived dataset |
| `exclude_episode` | Remove and reindex the complete episode and its video ranges |
| `constrained_action_repair` | Replace arm action targets with a stateful position/velocity/acceleration-limited sequence |
| `rerecord` | Record a human decision; automatic apply refuses until the episode is replaced or excluded |

`constrained_action_repair` is intended only for action-target spikes while the
measured robot state remains plausible. It does not edit images or measured
state. The tool aborts when the required correction exceeds `0.05 rad` by
default, because a large label-only change would no longer describe the recorded
behavior reliably.

## Apply a Reviewed Plan

Applying a plan always requires a new, nonexistent output directory:

```bash
python data_collection/dataset_quality_report.py apply \
  --plan <EXTERNAL_REPORT_DIR>/repair_plan.json \
  --output-dir <NEW_DERIVED_DATASET_ROOT>
```

Safety behavior:

- there is no in-place mode
- `<NEW_DERIVED_DATASET_ROOT>` cannot be inside the source dataset
- the source fingerprint must still match the scan
- unresolved `ERROR` or `UNSAFE` episodes block apply
- `rerecord` choices block apply
- failed repairs remove their incomplete output and leave the source untouched
- dataset statistics are recomputed after an applied repair
- a second external quality report is generated after repair

For a manually reviewed large action correction:

```bash
python data_collection/dataset_quality_report.py apply \
  --plan <EXTERNAL_REPORT_DIR>/repair_plan.json \
  --output-dir <NEW_DERIVED_DATASET_ROOT> \
  --allow-large-action-repair
```

Use this override only after checking the affected video and state trajectory.
It is normally safer to exclude or re-record the episode.

## Threshold Overrides

The default motion envelope matches the current policy-action limiter:

- FR3 joint position limits with `0.02 rad` margin
- `80%` of the configured FR3 joint velocity limits
- `80%` of the configured `10 rad/s^2` acceleration limit
- `0.25 rad` maximum action-state joint gap before review
- `0.50 rad` initial-configuration L2 distance before an informational flag

Scan thresholds are configurable:

```bash
python data_collection/dataset_quality_report.py scan \
  --dataset-root <DATASET_ROOT> \
  --safety-factor <VALUE_IN_0_TO_1> \
  --position-margin-rad <RADIANS> \
  --max-action-state-gap-rad <RADIANS> \
  --initial-configuration-l2-rad <RADIANS>
```

The selected values are embedded in `report.json` and `repair_plan.json`, and the
same values are reused by `apply`.

## Scope

The dataset does not contain 1 kHz controller velocity, acceleration, joint
torque, Cartesian velocity, collision, or reflex state. The reporter therefore
cannot prove that those hardware signals remained within limits. Logging those
signals during future collection is a separate control/data-path feature.

Collection-time safe q-goal filtering is intentionally separate from this
offline tool. This reporter never changes the GELLO-to-Franka control path or
writes audit artifacts inside a training dataset.
