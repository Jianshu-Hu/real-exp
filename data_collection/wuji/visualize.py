#!/usr/bin/env python3
"""Serve a passive live visualization of Wuji Glove skeleton and retargeting."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .teleop import drain_latest, select_device, skeleton_keypoints
except ImportError:  # pragma: no cover - direct script execution
    from teleop import drain_latest, select_device, skeleton_keypoints


class FrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: dict[str, Any] = {"ready": False, "error": None}

    def update(self, frame: dict[str, Any]) -> None:
        with self._lock:
            self._frame = frame

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._frame)


class GloveSampler(threading.Thread):
    def __init__(self, store: FrameStore, stop: threading.Event, glove_id: str | None) -> None:
        super().__init__(name="wuji-glove-visualizer", daemon=True)
        self.store = store
        self.stop = stop
        self.glove_id = glove_id
        self.ready = threading.Event()

    def run(self) -> None:
        import wuji_sdk

        wuji_sdk.set_log_level("warn")
        manager = wuji_sdk.SdkManager.instance()
        prior_user = manager.current_user()
        subscription = None
        try:
            manager.switch_to_default_user()
            device = select_device(
                manager.scan(), wuji_sdk.DeviceType.WujiGlove, self.glove_id, "Wuji Glove"
            )
            glove = manager.connect(sn=device.sn, device_name="visualizer_glove")
            side = glove.hand_side().get().lower()
            if side != "right":
                raise RuntimeError(f"visualizer expects the right glove, found {side}")
            retarget = wuji_sdk.RetargetSession.for_hand(
                wuji_sdk.HandModel.WujiHand2, wuji_sdk.Handedness.Right
            )
            subscription = glove.hand_skeleton().subscribe()
            frame_times: deque[float] = deque(maxlen=100)
            sequence = 0
            self.ready.set()
            while not self.stop.is_set():
                frame = drain_latest(subscription)
                if frame is None:
                    time.sleep(0.002)
                    continue
                try:
                    keypoints = skeleton_keypoints(frame, min_confidence=0.0)
                    confidence = [float(joint.confidence) for joint in frame.joints]
                    qpos = np.asarray(retarget.step(keypoints), dtype=np.float64)
                    if qpos.shape != (20,) or not np.all(np.isfinite(qpos)):
                        raise ValueError("retargeter returned an invalid command")
                    now = time.monotonic()
                    frame_times.append(now)
                    hz = 0.0
                    if len(frame_times) > 1:
                        hz = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                    sequence += 1
                    self.store.update(
                        {
                            "ready": True,
                            "error": None,
                            "sequence": sequence,
                            "serial": device.sn,
                            "address": device.address,
                            "side": side,
                            "hz": hz,
                            "keypoints": keypoints.tolist(),
                            "confidence": confidence,
                            "qpos": qpos.tolist(),
                        }
                    )
                except Exception as exc:
                    self.store.update({"ready": False, "error": str(exc)})
        except Exception as exc:
            self.store.update({"ready": False, "error": str(exc)})
            self.ready.set()
        finally:
            if subscription is not None:
                try:
                    subscription.close()
                except Exception:
                    pass
            try:
                manager.disconnect_all()
            finally:
                user_id = prior_user.get("user_id", "")
                if user_id:
                    manager.switch_user(user_id)
                else:
                    manager.switch_to_default_user()


def make_handler(store: FrameStore, html: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - HTTP method name
            if self.path in ("/", "/index.html"):
                body = html
                content_type = "text/html; charset=utf-8"
            elif self.path.startswith("/api/frame"):
                body = json.dumps(store.snapshot(), separators=(",", ":")).encode()
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-id", help="glove serial number or discovered IP:port")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.port <= 65535:
        raise ValueError("--port must be in [0, 65535]")

    # Bind the HTTP endpoint before connecting to the glove. If another
    # visualizer is already serving this port, fail without opening a second
    # SDK device connection and without emitting an SDK crash traceback.
    try:
        server = ThreadingHTTPServer(
            (args.host, args.port), make_handler(FrameStore(), b"")
        )
    except OSError as exc:
        if exc.errno == 98:
            alternative = args.port + 1 if args.port < 65535 else args.port - 1
            print(
                f"Port {args.port} is already in use. Open "
                f"http://{args.host}:{args.port} to use the existing visualizer, "
                f"or start another one with --port {alternative}.",
                file=sys.stderr,
            )
            return
        raise

    stop = threading.Event()
    store = FrameStore()
    sampler = GloveSampler(store, stop, args.glove_id)
    sampler.start()
    sampler.ready.wait(timeout=10.0)
    initial = store.snapshot()
    if initial.get("error"):
        stop.set()
        sampler.join(timeout=5.0)
        server.server_close()
        raise RuntimeError(initial["error"])

    html = Path(__file__).with_name("visualizer.html").read_bytes()
    server.RequestHandlerClass = make_handler(store, html)

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    old_handlers = {}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    print(f"Wuji glove visualizer: http://{args.host}:{server.server_port}", flush=True)
    print("The robotic hand is not connected or enabled by this process.", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        stop.set()
        server.server_close()
        sampler.join(timeout=5.0)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
