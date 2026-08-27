"""Print and visualize live Meta Quest poses without commanding a robot."""

from __future__ import annotations

import argparse
import time

from .power import QuestKeepAwake
from .receiver import PoseReceiver, ReceiverSnapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor Hand Tracking Streamer poses; this never connects to a robot."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--print-hz",
        type=float,
        default=1.0,
        help="terminal controller-pose report frequency (default: 1 Hz)",
    )
    parser.add_argument("--stale-after-s", type=float, default=0.5)
    parser.add_argument(
        "--keep-awake-grace-s",
        type=float,
        default=2.0,
        help="release Quest off-head keep-awake after this long without a controller packet",
    )
    parser.add_argument(
        "--no-keep-awake",
        action="store_true",
        help="never change Quest proximity power behavior",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="disable the local controller-pose window",
    )
    args = parser.parse_args()
    if not 0 < args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.print_hz <= 0.0 or args.stale_after_s <= 0.0 or args.keep_awake_grace_s <= 0.0:
        parser.error("--print-hz, --stale-after-s, and --keep-awake-grace-s must be positive")
    return args


def _format_pose(
    source: str,
    side: str,
    pose,
    receive_hz: float,
    now: float,
    stale_after_s: float,
) -> str:
    sample = pose.sample
    age = max(0.0, now - pose.received_monotonic_s)
    status = "STALE" if age > stale_after_s else "live"
    position = ", ".join(f"{value:+.4f}" for value in sample.position)
    quaternion = ", ".join(f"{value:+.4f}" for value in sample.quaternion_xyzw)
    grasp = "" if sample.grasp is None else f" grasp={sample.grasp:.3f}"
    controls = ""
    if sample.clutch is not None:
        controls += f" clutch={int(sample.clutch)}"
    if sample.record is not None:
        controls += f" record={int(sample.record)}"
    return (
        f"{source:10s} {side:5s} {status:5s} rx={receive_hz:5.1f}Hz age={age:5.3f}s "
        f"p=[{position}] q_xyzw=[{quaternion}]"
        f"{grasp}{controls} layout={sample.layout}"
    )


def _print_snapshot(
    snapshot: ReceiverSnapshot,
    now: float,
    stale_after_s: float,
) -> None:
    print(
        f"\nconnections={snapshot.accepted_connections} "
        f"poses={snapshot.pose_records} rejected={snapshot.rejected_pose_records}",
        flush=True,
    )
    controllers = [
        (key, pose)
        for key, pose in sorted(snapshot.poses.items())
        if key[0] == "controller"
    ]
    if not controllers:
        print("waiting for controller pose records...", flush=True)
        return
    for (source, side), pose in controllers:
        print(
            _format_pose(
                source,
                side,
                pose,
                snapshot.receive_hz.get((source, side), 0.0),
                now,
                stale_after_s,
            ),
            flush=True,
        )


def _run_terminal_only(
    receiver: PoseReceiver,
    print_period: float,
    stale_after_s: float,
    keep_awake_grace_s: float,
    power: QuestKeepAwake,
) -> None:
    try:
        while True:
            started = time.monotonic()
            snapshot = receiver.snapshot()
            _update_power(power, snapshot, started, keep_awake_grace_s)
            _print_snapshot(snapshot, started, stale_after_s)
            time.sleep(max(0.0, print_period - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass


def _run_visualizer(
    receiver: PoseReceiver,
    print_period: float,
    stale_after_s: float,
    keep_awake_grace_s: float,
    power: QuestKeepAwake,
) -> None:
    from .visualizer import ControllerPoseVisualizer

    visualizer = ControllerPoseVisualizer()
    state = {
        "last_pose_records": -1,
        "last_draw_s": 0.0,
        "next_print_s": time.monotonic(),
    }

    def update() -> None:
        now = time.monotonic()
        snapshot = receiver.snapshot()
        _update_power(power, snapshot, now, keep_awake_grace_s)
        if now >= state["next_print_s"]:
            _print_snapshot(snapshot, now, stale_after_s)
            state["next_print_s"] = now + print_period
        if (
            snapshot.pose_records != state["last_pose_records"]
            or now - state["last_draw_s"] >= 0.1
        ):
            visualizer.update(snapshot)
            state["last_pose_records"] = snapshot.pose_records
            state["last_draw_s"] = now
        visualizer.root.after(5, update)

    visualizer.root.after(0, update)
    try:
        visualizer.root.mainloop()
    except KeyboardInterrupt:
        visualizer.root.destroy()


def _update_power(
    power: QuestKeepAwake,
    snapshot: ReceiverSnapshot,
    now: float,
    keep_awake_grace_s: float,
) -> None:
    stream_active = any(
        source == "controller"
        and now - pose.received_monotonic_s <= keep_awake_grace_s
        for (source, _side), pose in snapshot.poses.items()
    )
    power.update(stream_active)


def main() -> None:
    args = parse_args()
    print_period = 1.0 / args.print_hz
    power = QuestKeepAwake(enabled=not args.no_keep_awake)
    with PoseReceiver(args.host, args.port) as receiver:
        print(f"Listening for Hand Tracking Streamer TCP poses on {args.host}:{args.port}")
        print("This monitor is read-only: values remain in the Unity tracking frame.")
        try:
            if args.no_visualization:
                _run_terminal_only(
                    receiver,
                    print_period,
                    args.stale_after_s,
                    args.keep_awake_grace_s,
                    power,
                )
            else:
                _run_visualizer(
                    receiver,
                    print_period,
                    args.stale_after_s,
                    args.keep_awake_grace_s,
                    power,
                )
        finally:
            power.close()
    print("Pose monitor stopped.")


if __name__ == "__main__":
    main()
