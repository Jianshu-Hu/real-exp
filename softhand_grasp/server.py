"""Camera-host server: capture D435i, run SAM3 and calculate both EE poses."""

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from softhand_grasp.protocol import FORMAT, validate_request, validate_response

ROOT = Path(__file__).resolve().parents[1]


def infer_request(request, args, config):
    output = args.runs_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    (output / "request.json").write_text(json.dumps(request, indent=2) + "\n")
    env = os.environ.copy()
    for key in ("AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH", "LD_LIBRARY_PATH", "PYTHONHOME"):
        env.pop(key, None)
    env["PYTHONPATH"] = str(ROOT)
    status = {"status": "started", "live_capture": args.capture_dir is None}
    try:
        capture = args.capture_dir.resolve() if args.capture_dir else output / "capture"
        capture_started = time.time()
        if args.capture_dir is None:
            subprocess.run([args.camera_python, "-m", "grasp.realsense_capture",
                            "--output-dir", str(capture), "--camera-serial", config["camera_serial"],
                            "--observation-frames", "1"], cwd=ROOT, env=env, check=True, timeout=60)
        command = [args.sam_python, "-m", "softhand_grasp.perception",
                   "--capture-dir", str(capture), "--output-dir", str(output),
                   "--config", str(config_path), "--object", request["description"],
                   "--checkpoint", str(args.checkpoint), "--device", args.device]
        if request.get("instance_index") is not None:
            command += ["--instance-index", str(request["instance_index"])]
        subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=600)
        plan = json.loads((output / "plan.json").read_text())
        result = {"side": "right", "pose_frame": "right_base",
                  "live_capture": args.capture_dir is None,
                  "capture_started_unix_s": capture_started,
                  "completed_unix_s": time.time(), "plan": plan,
                  "server_output_dir": str(output)}
        status["status"] = "completed"
        return result
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")


def handle_request(request, infer, seen):
    request_id = request.get("request_id") if isinstance(request, dict) else None
    response = {"format": FORMAT, "request_id": request_id, "ok": False}
    try:
        validate_request(request)
        if request_id in seen:
            raise ValueError("Duplicate request_id; request a fresh grasp")
        seen.append(request_id)
        response.update(infer(request), ok=True)
        validate_response(response, request, max_age_s=660)
    except Exception as exc:
        response.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return response


def serve_once(socket, infer, seen):
    # Consume even malformed JSON, then send a reply so REP can accept another request.
    raw = socket.recv()
    try:
        request = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        request = None
    response = handle_request(request, infer, seen)
    socket.send_json(response)
    if response.get("server_output_dir"):
        (Path(response["server_output_dir"]) / "response.json").write_text(
            json.dumps(response, indent=2) + "\n")
    print(f"Request {response['request_id']}: {'completed' if response['ok'] else response['error']}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    host = os.environ.get("SOFTHAND_SERVER_IP", os.environ.get("DATA_COLLECTION_SERVER_IP", "192.168.50.13"))
    parser.add_argument("--bind", default=f"tcp://{host}:5572")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sam-python", default=sys.executable)
    parser.add_argument("--camera-python", default="/usr/bin/python3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).parent / "runs/server")
    parser.add_argument("--capture-dir", type=Path, help="Replay RGB-D for testing; client execution is rejected")
    parser.add_argument("--once", action="store_true", help="Handle one request and exit")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    from softhand_grasp.geometry import load_config
    config = load_config(args.config)
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"SAM3 checkpoint does not exist: {args.checkpoint}")
    if args.validate_only:
        print("Server configuration valid; camera was not opened.")
        return
    import zmq
    with zmq.Context() as context:
        with context.socket(zmq.REP) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(args.bind)
            print(f"Soft-hand perception server listening at {args.bind}", flush=True)
            seen = deque(maxlen=256)
            try:
                while True:
                    serve_once(socket, lambda request: infer_request(request, args, config), seen)
                    if args.once:
                        break
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
