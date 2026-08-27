"""Threaded TCP receiver for Hand Tracking Streamer pose records."""

from __future__ import annotations

import socket
import threading
import time
from collections import deque
from dataclasses import dataclass

from .protocol import PoseSample, ProtocolError, parse_pose_line


@dataclass(frozen=True, slots=True)
class ReceivedPose:
    sample: PoseSample
    received_monotonic_s: float


@dataclass(frozen=True, slots=True)
class ReceiverSnapshot:
    poses: dict[tuple[str, str], ReceivedPose]
    receive_hz: dict[tuple[str, str], float]
    accepted_connections: int
    pose_records: int
    rejected_pose_records: int


class PoseReceiver:
    """Accept HTS's concurrent TCP streams and retain each latest pose."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        frequency_window_s: float = 1.0,
    ):
        self.host = host
        self.port = int(port)
        if frequency_window_s <= 0.0:
            raise ValueError("frequency_window_s must be positive")
        self.frequency_window_s = float(frequency_window_s)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._poses: dict[tuple[str, str], ReceivedPose] = {}
        self._receive_times: dict[tuple[str, str], deque[float]] = {}
        self._accepted_connections = 0
        self._pose_records = 0
        self._rejected_pose_records = 0
        self._server: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._connection_threads: set[threading.Thread] = set()

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("pose receiver is already running")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(8)
        server.settimeout(0.2)
        self._server = server
        self.port = int(server.getsockname()[1])
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="meta-quest-pose-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        assert self._server is not None
        while not self._stop_event.is_set():
            try:
                connection, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            connection.settimeout(0.5)
            thread = threading.Thread(
                target=self._connection_loop,
                args=(connection,),
                name="meta-quest-pose-connection",
                daemon=True,
            )
            with self._lock:
                self._accepted_connections += 1
                self._connection_threads.add(thread)
            thread.start()

    def _connection_loop(self, connection: socket.socket) -> None:
        buffer = ""
        try:
            with connection:
                while not self._stop_event.is_set():
                    try:
                        chunk = connection.recv(8192)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        self._record_line(line.rstrip("\r"))
        finally:
            if buffer.strip():
                self._record_line(buffer.strip())
            with self._lock:
                self._connection_threads.discard(threading.current_thread())

    def _record_line(self, line: str) -> None:
        try:
            sample = parse_pose_line(line)
        except ProtocolError:
            with self._lock:
                self._rejected_pose_records += 1
            return
        if sample is None:
            return
        received_at = time.monotonic()
        received = ReceivedPose(sample=sample, received_monotonic_s=received_at)
        key = (sample.source, sample.side)
        with self._lock:
            self._poses[key] = received
            times = self._receive_times.setdefault(key, deque())
            times.append(received_at)
            self._trim_receive_times(times, received_at)
            self._pose_records += 1

    def snapshot(self) -> ReceiverSnapshot:
        now = time.monotonic()
        with self._lock:
            receive_hz: dict[tuple[str, str], float] = {}
            for key, times in self._receive_times.items():
                self._trim_receive_times(times, now)
                if len(times) < 2:
                    receive_hz[key] = 0.0
                else:
                    receive_hz[key] = (len(times) - 1) / (times[-1] - times[0])
            return ReceiverSnapshot(
                poses=dict(self._poses),
                receive_hz=receive_hz,
                accepted_connections=self._accepted_connections,
                pose_records=self._pose_records,
                rejected_pose_records=self._rejected_pose_records,
            )

    def _trim_receive_times(self, times: deque[float], now: float) -> None:
        cutoff = now - self.frequency_window_s
        while times and times[0] < cutoff:
            times.popleft()

    def close(self) -> None:
        self._stop_event.set()
        server, self._server = self._server, None
        if server is not None:
            server.close()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
            self._accept_thread = None
        with self._lock:
            threads = list(self._connection_threads)
        for thread in threads:
            thread.join(timeout=1.0)

    def __enter__(self) -> "PoseReceiver":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
