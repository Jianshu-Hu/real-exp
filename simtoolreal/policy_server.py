"""Minimal server-computer RPC host for a SimToolReal rl_games checkpoint."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path

import zmq

from rl_policy import MockPolicy, RlGamesPolicy
from transport import make_error, make_policy_action, validate_packet


ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="tcp://0.0.0.0:5571")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upstream-root", type=Path, default=ROOT / "libs/SimToolReal-Franka-Wuji2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mock-policy", action="store_true")
    args = parser.parse_args(argv)
    if not args.mock_policy and (args.config is None or args.checkpoint is None):
        parser.error("--config and --checkpoint are required unless --mock-policy is used")
    if args.mock_policy:
        policy = MockPolicy(27)
        print("Using deterministic mock policy (27 actions)", flush=True)
    else:
        print(
            f"Loading SimToolReal checkpoint: config={args.config} checkpoint={args.checkpoint} "
            f"device={args.device}",
            flush=True,
        )
        policy = RlGamesPolicy(
            args.upstream_root, args.config, args.checkpoint, 134, 27, args.device
        )
        print("Loaded SimToolReal checkpoint (134 observations -> 27 actions)", flush=True)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    print(f"SimToolReal policy server waiting for observations on {args.bind}", flush=True)
    try:
        while not stop[0]:
            if not socket.poll(100, zmq.POLLIN):
                continue
            try:
                packet = validate_packet(socket.recv_json())
                if packet["kind"] != "policy_observation":
                    raise ValueError("expected policy_observation")
                import numpy as np
                observation = np.asarray(packet["observation"], dtype=np.float32)
                action = policy.act(observation)
                socket.send_json(make_policy_action(action.tolist()))
            except Exception as exc:
                socket.send_json(make_error(str(exc)))
                print(f"rejected policy request: {exc}", flush=True)
    finally:
        socket.close(0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
