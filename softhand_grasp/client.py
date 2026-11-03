"""Control-host client: request server poses, move the arm and close the hand."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("client_config.json")


def load_config(path=DEFAULT_CONFIG):
    import numpy as np
    from softhand_grasp.geometry import validate_pose, vector

    config = json.loads(Path(path).read_text())
    validate_pose(config["pregrasp_pose_xyzrpy"])
    volts = vector(config["channel_voltages"], 20, "channel_voltages")
    if np.any((volts < 0) | (volts > 5)):
        raise ValueError("channel_voltages must be in [0, 5] V")
    for name in ("close_duration_s", "publish_hz"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not config["serial_port"]:
        raise ValueError("serial_port is required")
    return config


def request_plan(args):
    import zmq
    from softhand_grasp.protocol import request_message, validate_response

    request = request_message(args.object, args.instance_index)
    with zmq.Context() as context:
        with context.socket(zmq.REQ) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SNDTIMEO, min(10000, int(args.request_timeout_s * 1000)))
            socket.setsockopt(zmq.RCVTIMEO, int(args.request_timeout_s * 1000))
            socket.connect(args.server_address)
            try:
                socket.send_json(request)
                response = socket.recv_json()
            except zmq.Again as exc:
                raise TimeoutError(f"No grasp response from {args.server_address}; no automatic retry") from exc
    validate_response(response, request, execute=args.execute, max_age_s=args.max_plan_age_s)
    return response


def publisher_factory(config):
    # publisher.py imports the sibling Service package using this root.
    sys.path.insert(0, str(ROOT / "PneumaticSystem"))
    from Service.publisher import Publisher, CHANNEL_COUNT
    if CHANNEL_COUNT != len(config["channel_voltages"]):
        raise RuntimeError("Pneumatic publisher channel count differs from configuration")
    return Publisher(port=config["serial_port"])


def publish_for(publisher, config, *, clock=time.monotonic, sleep=time.sleep):
    """Keep sending until a full close_duration_s has passed after first ACK."""
    publisher.publish_all(config["channel_voltages"])
    deadline = clock() + config["close_duration_s"]
    interval = 1.0 / config["publish_hz"]
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            return
        sleep(min(interval, remaining))
        publisher.publish_all(config["channel_voltages"])


def wait_for_release():
    print("Lift complete. Outputs remain at grasp voltage. Support the object before release.", flush=True)
    while input("Type release to turn all pneumatic outputs off: ").strip().lower() != "release":
        pass


def run_sequence(config, arm, perceive, *, execute=False, make_publisher=publisher_factory,
                 hold=wait_for_release, pressurize=publish_for):
    """No close before verified approach, and no lift before pressure dwell."""
    from softhand_grasp.geometry import validate_pose

    arm.move_to(config["pregrasp_pose_xyzrpy"])
    plan = perceive()
    # The server owns pose calculation; the client only validates commands.
    poses = {key: plan[key] for key in ("grasp_pose_xyzrpy", "lift_pose_xyzrpy")}
    for key in ("grasp_pose_xyzrpy", "lift_pose_xyzrpy"):
        validate_pose(poses[key])
    print(json.dumps(poses, indent=2), flush=True)
    arm.move_to(poses["grasp_pose_xyzrpy"])
    if not execute:
        print(f"Preview: all 20 channels = {config['channel_voltages']} V for "
              f"{config['close_duration_s']} s; then lift (no hardware commands).")
        arm.move_to(poses["lift_pose_xyzrpy"])
        return poses
    # Publisher.close() turns all outputs off. Keep this context open through
    # the lift and operator-controlled release; outputs latch between writes.
    with make_publisher(config) as publisher:
        pressurize(publisher, config)
        arm.move_to(poses["lift_pose_xyzrpy"])
        hold()
    return poses


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, help="SAM3 object description, e.g. 'red cup'")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    host = os.environ.get("SOFTHAND_SERVER_IP", os.environ.get("DATA_COLLECTION_SERVER_IP", "192.168.50.13"))
    parser.add_argument("--server-address", default=f"tcp://{host}:5572")
    parser.add_argument("--request-timeout-s", type=float, default=600)
    parser.add_argument("--max-plan-age-s", type=float, default=300)
    parser.add_argument("--instance-index", type=int, help="Rank among depth-valid matches, highest score first")
    parser.add_argument("--output-dir", type=Path, help="New directory for received plan and execution status")
    parser.add_argument("--execute", action="store_true", help="Execute the entire sequence on hardware")
    parser.add_argument("--validate-only", action="store_true", help="Check arguments/config without opening hardware")
    return parser


def checked_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.object.strip():
        parser.error("--object must be nonempty")
    if args.execute and not sys.stdin.isatty():
        parser.error("--execute requires a terminal for releasing the object after lifting")
    if args.instance_index is not None and args.instance_index < 0:
        parser.error("--instance-index must be nonnegative")
    import math
    for value in (args.request_timeout_s, args.max_plan_age_s):
        if not math.isfinite(value) or value <= 0:
            parser.error("Timeout and maximum plan age must be positive and finite")
    return args, load_config(args.config)


def main():
    args, config = checked_args()
    import zmq  # Check transport availability before starting robot motion.

    if args.validate_only:
        print("Configuration and arguments valid; hardware was not opened.")
        return
    from softhand_grasp.arm import PreviewArm, RightArm

    output = (args.output_dir or ROOT / "softhand_grasp/runs/client" /
              datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    def perceive():
        response = request_plan(args)
        (output / "response.json").write_text(json.dumps(response, indent=2) + "\n")
        print(f"Server artifacts: {response.get('server_output_dir')}", flush=True)
        return response["plan"]

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    status = {"execute": args.execute, "status": "started"}
    try:
        with RightArm() if args.execute else PreviewArm() as arm:
            run_sequence(config, arm, perceive, execute=args.execute)
        status["status"] = "completed" if args.execute else "previewed"
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(f"Run artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
