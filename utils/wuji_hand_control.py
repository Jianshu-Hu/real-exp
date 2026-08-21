"""Position-limited, fixed-rate command generation for Wuji Hand 2.

The Wuji SDK accepts whole-hand joint commands at up to 1 kHz, but replay and
teleoperation provide targets at much lower and variable rates. This module
keeps the SDK backend unchanged and adds a repository-owned trajectory layer
around it.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

HAND_JOINT_COUNT = 20
HAND_COMMAND_RATE_HZ = 250.0
HAND_COMMAND_PERIOD_S = 1.0 / HAND_COMMAND_RATE_HZ
HAND_TARGET_TRANSITION_DURATION_S = 0.10
HAND_INITIAL_POSITION_TOLERANCE_RAD = 0.05


def normalize_hand_positions(state: Any) -> np.ndarray | None:
    """Normalize SDK state variants into the device's 20-joint order."""
    positions = getattr(state, "position", None)
    if positions is not None:
        values = [float(value) for value in positions]
    else:
        joints = getattr(state, "joints", None)
        if joints is None:
            return None
        ordered_joints = sorted(joints, key=lambda joint: int(joint.nid))
        values = [float(joint.position) for joint in ordered_joints]
    if len(values) != HAND_JOINT_COUNT or not np.all(np.isfinite(values)):
        return None
    return np.asarray(values, dtype=float)


class HandTrajectoryGenerator:
    """Generate a smooth position path while enforcing firmware position limits."""

    def __init__(
        self,
        lower_limits: np.ndarray,
        upper_limits: np.ndarray,
        transition_duration_s: float = HAND_TARGET_TRANSITION_DURATION_S,
    ) -> None:
        self.lower_limits = self._limits(lower_limits)
        self.upper_limits = self._limits(upper_limits)
        if np.any(self.lower_limits > self.upper_limits):
            raise ValueError("Hand lower position limits must not exceed upper limits.")
        if not np.isfinite(transition_duration_s) or transition_duration_s <= 0.0:
            raise ValueError("Hand trajectory transition duration must be positive and finite.")
        self.transition_duration_s = float(transition_duration_s)
        self.position = np.zeros(HAND_JOINT_COUNT, dtype=float)
        self.velocity = np.zeros(HAND_JOINT_COUNT, dtype=float)
        self.acceleration = np.zeros(HAND_JOINT_COUNT, dtype=float)
        self.target = self.position.copy()
        self._coefficients = np.zeros((HAND_JOINT_COUNT, 6), dtype=float)
        self._elapsed_s = 0.0
        self._duration_s = 0.0
        self._target_pending = False
        self._initialized = False

    @staticmethod
    def _limits(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != (HAND_JOINT_COUNT,):
            raise ValueError(f"Hand position limits must have shape ({HAND_JOINT_COUNT},).")
        if not np.all(np.isfinite(values)):
            raise ValueError("Hand position limits must be finite.")
        return values.copy()

    def _clip(self, values: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(values, dtype=float), self.lower_limits, self.upper_limits)

    def reset(self, position: np.ndarray) -> None:
        position = np.asarray(position, dtype=float)
        if position.shape != (HAND_JOINT_COUNT,) or not np.all(np.isfinite(position)):
            raise ValueError("Hand initial position must be a finite 20-joint vector.")
        self.position = self._clip(position)
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)
        self.target = self.position.copy()
        self._coefficients.fill(0.0)
        self._elapsed_s = 0.0
        self._duration_s = 0.0
        self._target_pending = False
        self._initialized = True

    def set_target(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=float)
        if target.shape != (HAND_JOINT_COUNT,) or not np.all(np.isfinite(target)):
            raise ValueError("Hand target must be a finite 20-joint vector.")
        safe_target = self._clip(target)
        if not self._initialized:
            self.reset(safe_target)
            return
        if np.array_equal(safe_target, self.target):
            return
        self.target = safe_target
        self._target_pending = True

    @property
    def safe_target(self) -> np.ndarray:
        return self.target.copy()

    def _replan(self) -> None:
        duration = self.transition_duration_s
        start = self.position
        delta = self.target - start
        t2 = duration * duration
        t3 = t2 * duration
        t4 = t3 * duration
        t5 = t4 * duration
        self._coefficients[:, 0] = start
        self._coefficients[:, 1] = self.velocity
        self._coefficients[:, 2] = 0.5 * self.acceleration
        self._coefficients[:, 3] = (
            20.0 * delta - 12.0 * self.velocity * duration - 3.0 * self.acceleration * t2
        ) / (2.0 * t3)
        self._coefficients[:, 4] = (
            -30.0 * delta + 16.0 * self.velocity * duration + 3.0 * self.acceleration * t2
        ) / (2.0 * t4)
        self._coefficients[:, 5] = (
            12.0 * delta - 6.0 * self.velocity * duration - self.acceleration * t2
        ) / (2.0 * t5)
        self._elapsed_s = 0.0
        self._duration_s = duration
        self._target_pending = False

    def advance(self, dt_s: float) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("Hand trajectory generator must be reset before advance().")
        dt_s = max(0.0, min(float(dt_s), 0.1))
        if self._target_pending:
            self._replan()
        if self._duration_s <= 0.0 or dt_s <= 0.0:
            return self.position.copy()

        self._elapsed_s = min(self._elapsed_s + dt_s, self._duration_s)
        t = self._elapsed_s
        c = self._coefficients
        self.position = self._clip(
            (((((c[:, 5] * t) + c[:, 4]) * t + c[:, 3]) * t + c[:, 2]) * t + c[:, 1]) * t
            + c[:, 0]
        )
        self.velocity = ((((5.0 * c[:, 5] * t) + 4.0 * c[:, 4]) * t + 3.0 * c[:, 3]) * t + 2.0 * c[:, 2]) * t + c[:, 1]
        self.acceleration = (((20.0 * c[:, 5] * t) + 12.0 * c[:, 4]) * t + 6.0 * c[:, 3]) * t + 2.0 * c[:, 2]
        if self._elapsed_s >= self._duration_s:
            self.position = self.target.copy()
            self.velocity.fill(0.0)
            self.acceleration.fill(0.0)
            self._elapsed_s = 0.0
            self._duration_s = 0.0
        return self.position.copy()


class SmoothedWujiHand2Backend:
    """Wrap the submodule backend with a 250 Hz position trajectory loop."""

    def __init__(
        self,
        original_backend_cls: type,
        *,
        ip: str,
        kp: float,
        kd: float,
        current_limit: float,
        handedness: str | None = None,
        command_rate_hz: float = HAND_COMMAND_RATE_HZ,
    ) -> None:
        if not np.isfinite(command_rate_hz) or command_rate_hz <= 0.0:
            raise ValueError("Hand command rate must be positive and finite.")
        self._original_backend_cls = original_backend_cls
        self._backend = original_backend_cls(
            ip=ip,
            kp=kp,
            kd=kd,
            current_limit=current_limit,
            handedness=handedness,
        )
        self._state_subscription = None
        self._state_stop_event = threading.Event()
        self._actual_lock = threading.Lock()
        self._actual: np.ndarray | None = None
        initial_position = self._read_initial_position()
        if initial_position is None:
            self._backend.close()
            raise RuntimeError("Wuji Hand 2 did not return a valid 20-joint initial state.")
        try:
            upper, lower = self._backend._hand.get_soft_limits()
            upper_limits = np.asarray(upper, dtype=float)
            lower_limits = np.asarray(lower, dtype=float)
        except Exception:
            self._backend.close()
            raise RuntimeError("Could not read Wuji Hand 2 firmware position limits.")
        self._generator = HandTrajectoryGenerator(lower_limits, upper_limits)
        self._generator.reset(initial_position)
        self._target_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._target = initial_position.copy()
        self._actual = initial_position.copy()
        self._command_period_s = 1.0 / float(command_rate_hz)
        self._state_thread = threading.Thread(
            target=self._state_loop, name="wuji-hand-state-reader", daemon=True
        )
        self._state_thread.start()
        self._thread = threading.Thread(target=self._command_loop, name="wuji-hand-trajectory", daemon=True)
        self._thread.start()
        print(
            f"Wuji Hand 2 smoothed position control enabled at {command_rate_hz:g} Hz",
            flush=True,
        )

    @property
    def _hand(self) -> Any:
        return self._backend._hand

    def send(self, qpos: np.ndarray) -> None:
        target = np.asarray(qpos, dtype=float)
        if target.shape != (HAND_JOINT_COUNT,) or not np.all(np.isfinite(target)):
            raise ValueError("Wuji Hand 2 target must be a finite 20-joint vector.")
        with self._target_lock:
            self._target = target.copy()

    def actual_position(self) -> np.ndarray | None:
        with self._actual_lock:
            return None if self._actual is None else self._actual.copy()

    @property
    def target_position(self) -> np.ndarray:
        return self._generator.safe_target

    def _command_loop(self) -> None:
        last_time = time.monotonic()
        next_time = last_time
        while not self._stop_event.is_set():
            now = time.monotonic()
            dt_s = now - last_time
            last_time = now
            with self._target_lock:
                target = self._target.copy()
            self._generator.set_target(target)
            command = self._generator.advance(dt_s)
            self._original_backend_cls.send(self._backend, command)
            next_time += self._command_period_s
            wait_s = max(0.0, next_time - time.monotonic())
            self._stop_event.wait(wait_s)

    def _read_initial_position(self) -> np.ndarray | None:
        """Read one state from either SDK generation's synchronous or stream API."""
        hand = self._backend._hand
        read_joint_state = getattr(hand, "read_joint_state", None)
        if callable(read_joint_state):
            return normalize_hand_positions(read_joint_state())
        joint_states = getattr(hand, "joint_states", None)
        if not callable(joint_states):
            return None
        subscription = joint_states().subscribe()
        self._state_subscription = subscription
        deadline = time.monotonic() + 10.0
        initial_position: np.ndarray | None = None
        try:
            while time.monotonic() < deadline:
                time.sleep(0.02)
                try:
                    position = normalize_hand_positions(subscription.recv())
                except Exception:
                    position = None
                if position is not None:
                    initial_position = position
                    break
        finally:
            if initial_position is None:
                subscription.close()
                self._state_subscription = None
        return initial_position

    def _state_loop(self) -> None:
        subscription = self._state_subscription
        if subscription is None:
            return
        try:
            while not self._state_stop_event.is_set():
                position = normalize_hand_positions(subscription.recv())
                if position is not None:
                    with self._actual_lock:
                        self._actual = position
        except Exception:
            # The command loop remains usable if state streaming is interrupted;
            # status requests will report the last valid measured position.
            return

    def close(self) -> None:
        self._stop_event.set()
        self._state_stop_event.set()
        if self._state_subscription is not None:
            try:
                self._state_subscription.close()
            except Exception:
                pass
        if hasattr(self, "_state_thread"):
            self._state_thread.join(timeout=1.0)
        self._thread.join(timeout=max(1.0, 4.0 * self._command_period_s))
        self._backend.close()


def make_smoothed_backend_class(original_backend_cls: type) -> type:
    """Create a drop-in class without importing or editing the submodule."""

    class _SmoothedBackend(SmoothedWujiHand2Backend):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(original_backend_cls, **kwargs)

    _SmoothedBackend.__name__ = "WujiHand2Backend"
    return _SmoothedBackend
