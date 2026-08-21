from __future__ import annotations

import threading
import time

import numpy as np

from utils.wuji_hand_control import (
    HAND_COMMAND_RATE_HZ,
    HAND_JOINT_COUNT,
    HandTrajectoryGenerator,
    make_smoothed_backend_class,
)


def test_hand_generator_clips_position_targets_and_replans_smoothly() -> None:
    lower = np.full(HAND_JOINT_COUNT, -1.0)
    upper = np.full(HAND_JOINT_COUNT, 1.0)
    generator = HandTrajectoryGenerator(lower, upper, transition_duration_s=0.2)
    generator.reset(np.zeros(HAND_JOINT_COUNT))
    target = np.full(HAND_JOINT_COUNT, 2.0)
    generator.set_target(target)

    samples = [generator.advance(0.004) for _ in range(60)]
    values = np.asarray(samples)
    assert np.all(values <= 1.0)
    assert np.all(values >= -1.0)
    np.testing.assert_allclose(generator.position, np.ones(HAND_JOINT_COUNT))

    previous = values[0]
    for sample in values[1:]:
        assert np.max(np.abs(sample - previous)) < 0.1
        previous = sample


def test_smoothed_backend_publishes_from_measured_position() -> None:
    class FakeHand:
        def read_joint_state(self) -> object:
            return type("State", (), {"position": [0.25] * HAND_JOINT_COUNT})()

        def get_soft_limits(self) -> tuple[list[float], list[float]]:
            return [1.0] * HAND_JOINT_COUNT, [-1.0] * HAND_JOINT_COUNT

    class FakeBackend:
        sent: list[np.ndarray] = []
        lock = threading.Lock()

        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self._hand = FakeHand()

        @staticmethod
        def send(backend: "FakeBackend", qpos: np.ndarray) -> None:
            del backend
            with FakeBackend.lock:
                FakeBackend.sent.append(np.asarray(qpos, dtype=float).copy())

        def close(self) -> None:
            return

    FakeBackend.sent = []
    SmoothedBackend = make_smoothed_backend_class(FakeBackend)
    backend = SmoothedBackend(command_rate_hz=HAND_COMMAND_RATE_HZ, ip="", kp=1.0, kd=0.1, current_limit=1.0)
    try:
        np.testing.assert_allclose(backend.target_position, np.full(HAND_JOINT_COUNT, 0.25))
        backend.send(np.full(HAND_JOINT_COUNT, 0.75))
        time.sleep(0.03)
        with FakeBackend.lock:
            samples = list(FakeBackend.sent)
        assert len(samples) >= 4
        assert all(np.all(sample >= -1.0) and np.all(sample <= 1.0) for sample in samples)
        assert np.max(samples[-1]) > 0.25
    finally:
        backend.close()


def test_smoothed_backend_falls_back_to_joint_state_stream() -> None:
    class FakeSubscription:
        def __init__(self) -> None:
            self.closed = False

        def recv(self) -> object:
            return type("Frame", (), {
                "joints": [
                    type("Joint", (), {"nid": index, "position": 0.1})()
                    for index in range(HAND_JOINT_COUNT)
                ]
            })()

        def close(self) -> None:
            self.closed = True

    class FakeHand:
        def __init__(self) -> None:
            self.subscription = FakeSubscription()

        def joint_states(self) -> object:
            return type("Resource", (), {"subscribe": lambda resource: self.subscription})()

        def get_soft_limits(self) -> tuple[list[float], list[float]]:
            return [1.0] * HAND_JOINT_COUNT, [-1.0] * HAND_JOINT_COUNT

    class FakeBackend:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self._hand = FakeHand()

        @staticmethod
        def send(backend: "FakeBackend", qpos: np.ndarray) -> None:
            del backend, qpos

        def close(self) -> None:
            return

    SmoothedBackend = make_smoothed_backend_class(FakeBackend)
    backend = SmoothedBackend(command_rate_hz=HAND_COMMAND_RATE_HZ, ip="", kp=1.0, kd=0.1, current_limit=1.0)
    try:
        np.testing.assert_allclose(backend.target_position, np.full(HAND_JOINT_COUNT, 0.1))
    finally:
        backend.close()
    assert backend._backend._hand.subscription.closed
