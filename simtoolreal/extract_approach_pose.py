#!/usr/bin/env python3
"""Extract an intermediate approaching pose from a rollout for real robot initialization."""
import argparse
import json
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True, help="Frame index to extract joint configuration from")
    parser.add_argument("--out", type=Path, required=True, help="Output snapshot JSON file")
    args = parser.parse_args()

    rollout = args.rollout.resolve()
    data = np.load(rollout / "rollout.npz")
    snapshot = json.loads((rollout / "initial_snapshot.json").read_text(encoding="utf-8"))

    if args.frame < 0 or args.frame >= len(data["joint_position"]):
        parser.error(f"frame {args.frame} out of range [0, {len(data['joint_position'])})")

    # Extract joint configuration from specified frame
    q_approach = data["joint_position"][args.frame].tolist()
    q_approach_vel = data["joint_velocity"][args.frame].tolist()

    # Update snapshot with the extracted pose
    snapshot["state"]["joint_position_27"] = q_approach
    snapshot["state"]["joint_velocity_27"] = q_approach_vel

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    report = {
        "format": "simtoolreal_real_snapshot_v1",
        "source_rollout": str(rollout),
        "extracted_frame": args.frame,
        "time_s": float(data["time_s"][args.frame]),
        "joint_position_27": q_approach,
        "output_snapshot": str(args.out.resolve()),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
