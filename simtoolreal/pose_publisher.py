"""Publish FoundationPose++ object poses from a file or deterministic mock."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from transport import make_object_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", default="tcp://127.0.0.1:5565")
    parser.add_argument("--pose-file", type=Path, help="JSON or NumPy file containing a 4x4 pose")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--object-id", default="object")
    parser.add_argument("--frame-id", default="camera")
    return parser.parse_args()


def read_pose(path: Path) -> list[float]:
    if path.suffix.lower() == ".json":
        value: Any = json.loads(path.read_text())
    else:
        value = np.load(path, allow_pickle=False)
        if value.ndim == 3:
            value = value[-1]
    return np.asarray(value, dtype=float).reshape(-1).tolist()


def main() -> int:
    args = parse_args()
    if not args.mock and args.pose_file is None:
        raise SystemExit("provide --pose-file or --mock")
    if args.rate <= 0:
        raise SystemExit("--rate must be positive")
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 2)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.connect)
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    print(f"simtoolreal pose publisher connected to {args.connect}", flush=True)
    last_signature: tuple[int, int] | None = None
    sequence = 0
    try:
        while not stop[0]:
            if args.mock:
                pose = [1.0, 0.0, 0.0, 0.01 * sequence, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0]
                sequence += 1
                socket.send_json(make_object_pose(pose, object_id=args.object_id, frame_id=args.frame_id, source="mock-foundationpose++"))
            else:
                assert args.pose_file is not None
                try:
                    stat = args.pose_file.stat()
                except FileNotFoundError:
                    time.sleep(1.0 / args.rate)
                    continue
                signature = (stat.st_mtime_ns, stat.st_size)
                if signature != last_signature:
                    socket.send_json(make_object_pose(read_pose(args.pose_file), object_id=args.object_id, frame_id=args.frame_id))
                    last_signature = signature
            time.sleep(1.0 / args.rate)
    finally:
        socket.close(0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
