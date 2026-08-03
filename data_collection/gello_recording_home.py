from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from typing import Any

import numpy as np


ACTIVE_CURRENT_MA = 100
HOLD_CURRENT_MA = 15
ACTIVE_KP_P = 700
HOLD_KP_P = 300
KP_D = 120
ACTIVE_GROUP_SEC = 0.25
HOMING_TIMEOUT_SEC = 25.0
DONE_TOLERANCE_RAD = math.radians(0.75)
READY_TOLERANCE_RAD = math.radians(2.0)
RELEASE_MOTION_RAD = math.radians(0.75)
CROSSING_DEADBAND_RAD = math.radians(1.0)
MAX_TARGET_CROSSINGS = 2
MAX_JOINT_MOVE_RAD = 1.30
ACTIVE_GROUPS = ((0, 3, 6), (1, 4), (2, 5))
HELPER_START_TIMEOUT_SEC = HOMING_TIMEOUT_SEC + 10.0


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _parent_is_alive(parent_pid: int) -> bool:
    try:
        os.kill(parent_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_with_spin(
    session: Any,
    duration_sec: float,
    *,
    parent_pid: int,
    stop_requested: list[bool],
) -> None:
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        if stop_requested[0] or not _parent_is_alive(parent_pid):
            raise InterruptedError("GELLO homing interrupted.")
        session.rclpy.spin_once(session.node, timeout_sec=0.01)


def _set_dual_control(
    homing: Any,
    session: Any,
    *,
    torque_enabled: bool,
    active_by_side: dict[str, tuple[int, ...]] | None = None,
) -> None:
    active_by_side = active_by_side or {}
    updates: dict[str, dict[str, Any]] = {}
    for side in homing.SIDES:
        saved = session.parameters[side.name]
        current = [HOLD_CURRENT_MA] * 7 + [
            int(saved["_original_goal_current"][7])
        ]
        kp_p = [HOLD_KP_P] * 7 + [int(saved["_original_kp_p"][7])]
        kp_i = [0] * 7 + [int(saved["_original_kp_i"][7])]
        kp_d = [KP_D] * 7 + [int(saved["_original_kp_d"][7])]
        for joint_index in active_by_side.get(side.name, ()):
            current[joint_index] = ACTIVE_CURRENT_MA
            kp_p[joint_index] = ACTIVE_KP_P
        updates[side.name] = {
            "dynamixel_goal_current": current,
            "dynamixel_kp_p": kp_p,
            "dynamixel_kp_i": kp_i,
            "dynamixel_kp_d": kp_d,
            "dynamixel_torque_enable": (
                [1] * 7 + [0] if torque_enabled else [0] * 8
            ),
        }
    session.set_parameters(updates, "updating recording GELLO homing control")


def _run_helper(ready_file: Path, parent_pid: int) -> int:
    import home_gello_and_franka as homing

    stop_requested = [False]
    release_requested = [False]

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested[0] = True

    def arm_release(_signum: int, _frame: Any) -> None:
        release_requested[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGUSR1, arm_release)

    ros = homing.load_ros_types()
    session = homing.RosGelloSession(ros)
    targets = homing.initial_targets()
    original_parameters_saved = False
    try:
        session.wait_ready()
        session.preserve_original_gains()
        original_parameters_saved = True
        session.set_arm_torque(False)

        start_q = session.snapshot_q()
        largest_move = max(
            float(np.max(np.abs(targets[side.name] - start_q[side.name])))
            for side in homing.SIDES
        )
        if largest_move > MAX_JOINT_MOVE_RAD:
            raise RuntimeError(
                f"Required GELLO movement {largest_move:.3f} rad exceeds the "
                f"{MAX_JOINT_MOVE_RAD:.2f} rad safety bound."
            )

        print()
        print("RECORDING GELLO RESET")
        print("---------------------")
        print(
            "Two USB supplies assumed: up to three joints per side at "
            f"{ACTIVE_CURRENT_MA} mA; other joints hold at {HOLD_CURRENT_MA} mA."
        )
        for side in homing.SIDES:
            homing.print_vector(
                f"{side.name} start error deg",
                np.degrees(targets[side.name] - start_q[side.name]),
            )

        session.set_goal_positions(targets)
        previous_error = {
            side.name: targets[side.name] - start_q[side.name]
            for side in homing.SIDES
        }
        crossings = {
            side.name: np.zeros(7, dtype=int) for side in homing.SIDES
        }
        frozen = {
            side.name: np.zeros(7, dtype=bool) for side in homing.SIDES
        }
        deadline = time.monotonic() + HOMING_TIMEOUT_SEC
        cycle = 0

        while time.monotonic() < deadline:
            if stop_requested[0] or not _parent_is_alive(parent_pid):
                raise InterruptedError("GELLO homing interrupted.")
            current_q = session.snapshot_q()
            active_by_side: dict[str, tuple[int, ...]] = {}
            all_done = True
            active_group = ACTIVE_GROUPS[cycle % len(ACTIVE_GROUPS)]

            for side in homing.SIDES:
                error = targets[side.name] - current_q[side.name]
                done = np.abs(error) <= DONE_TOLERANCE_RAD
                previous_sign = np.sign(previous_error[side.name])
                current_sign = np.sign(error)
                crossed = (
                    (previous_sign != 0)
                    & (current_sign != 0)
                    & (previous_sign != current_sign)
                    & (np.abs(previous_error[side.name]) > CROSSING_DEADBAND_RAD)
                    & (np.abs(error) > CROSSING_DEADBAND_RAD)
                )
                crossings[side.name][crossed] += 1
                frozen[side.name] |= (
                    crossings[side.name] >= MAX_TARGET_CROSSINGS
                )
                previous_error[side.name] = error
                active_by_side[side.name] = tuple(
                    index
                    for index in active_group
                    if not done[index] and not frozen[side.name][index]
                )
                all_done &= bool(np.all(done | frozen[side.name]))

            if all_done:
                break

            _set_dual_control(
                homing,
                session,
                torque_enabled=True,
                active_by_side=active_by_side,
            )
            _wait_with_spin(
                session,
                ACTIVE_GROUP_SEC,
                parent_pid=parent_pid,
                stop_requested=stop_requested,
            )
            cycle += 1
            if cycle % 6 == 0:
                current_q = session.snapshot_q()
                status = []
                for side in homing.SIDES:
                    max_error_deg = math.degrees(
                        float(
                            np.max(
                                np.abs(
                                    targets[side.name] - current_q[side.name]
                                )
                            )
                        )
                    )
                    status.append(f"{side.name}={max_error_deg:.2f}deg")
                print("GELLO reset: " + " ".join(status), flush=True)

        _set_dual_control(homing, session, torque_enabled=True)
        _wait_with_spin(
            session,
            0.5,
            parent_pid=parent_pid,
            stop_requested=stop_requested,
        )
        hold_q = session.snapshot_q()
        final_error = max(
            float(np.max(np.abs(targets[side.name] - hold_q[side.name])))
            for side in homing.SIDES
        )
        if final_error > READY_TOLERANCE_RAD:
            raise RuntimeError(
                "GELLO reset did not reach the hold tolerance: "
                f"max error {math.degrees(final_error):.2f} deg exceeds "
                f"{math.degrees(READY_TOLERANCE_RAD):.2f} deg."
            )

        _write_json_atomic(
            ready_file,
            {
                "status": "holding",
                "helper_pid": os.getpid(),
                "parent_pid": parent_pid,
                "max_error_deg": math.degrees(final_error),
                "release_motion_deg": math.degrees(RELEASE_MOTION_RAD),
                "left_hold_q": hold_q["left"].tolist(),
                "right_hold_q": hold_q["right"].tolist(),
            },
        )
        print(
            "GELLO reset complete. Holding at INITIAL_STATE; "
            "press s + Enter in the recorder, then move GELLO to release.",
            flush=True,
        )

        release_baseline: dict[str, np.ndarray] | None = None
        while not stop_requested[0] and _parent_is_alive(parent_pid):
            session.rclpy.spin_once(session.node, timeout_sec=0.02)
            if release_requested[0] and release_baseline is None:
                release_baseline = session.snapshot_q()
                print(
                    "Recording started. GELLO hold will release after real joint motion.",
                    flush=True,
                )
            if release_baseline is None:
                continue

            current_q = session.snapshot_q()
            max_motion = max(
                float(
                    np.max(
                        np.abs(
                            current_q[side.name] - release_baseline[side.name]
                        )
                    )
                )
                for side in homing.SIDES
            )
            if max_motion >= RELEASE_MOTION_RAD:
                print(
                    f"GELLO moved {math.degrees(max_motion):.2f} deg; "
                    "disabling hold torque now.",
                    flush=True,
                )
                break
        return 0
    except InterruptedError:
        print("GELLO hold stop requested; disabling torque.", flush=True)
        return 0
    finally:
        try:
            if original_parameters_saved:
                session.restore_gains_torque_off()
            else:
                session.set_arm_torque(False)
        except Exception as exc:
            print(
                f"WARNING: GELLO torque-off confirmation failed: {exc}",
                flush=True,
            )
        finally:
            session.close()


class GelloRecordingHomeController:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.helper_path = Path(__file__).resolve()
        self.ready_file = Path(
            f"/tmp/real_exp_gello_recording_home_{os.getpid()}.json"
        )
        self.process: subprocess.Popen[bytes] | None = None
        self.release_armed = False

    @property
    def is_holding(self) -> bool:
        return (
            self.process is not None
            and self.process.poll() is None
            and self.ready_file.exists()
        )

    def _helper_command(self) -> list[str]:
        ros_setup = Path("/opt/ros/humble/setup.bash")
        workspace_setup = (
            self.repo_root / "gello_software" / "ros2" / "install" / "setup.bash"
        )
        for setup_file in (ros_setup, workspace_setup):
            if not setup_file.exists():
                raise FileNotFoundError(f"Missing ROS setup file: {setup_file}")

        command = " && ".join(
            (
                f"source {shlex.quote(str(ros_setup))}",
                f"source {shlex.quote(str(workspace_setup))}",
                "exec /usr/bin/python3 "
                f"{shlex.quote(str(self.helper_path))} "
                f"--helper --ready-file {shlex.quote(str(self.ready_file))} "
                f"--parent-pid {os.getpid()}",
            )
        )
        return ["bash", "-lc", command]

    def start_and_hold(self) -> dict[str, Any]:
        self.poll()
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("GELLO reset/hold is already active.")
        self.ready_file.unlink(missing_ok=True)
        self.release_armed = False
        self.process = subprocess.Popen(self._helper_command())  # nosec B603

        deadline = time.monotonic() + HELPER_START_TIMEOUT_SEC
        try:
            while time.monotonic() < deadline:
                return_code = self.process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        f"GELLO reset helper exited before hold, code {return_code}."
                    )
                if self.ready_file.exists():
                    status = json.loads(self.ready_file.read_text())
                    if status.get("status") == "holding":
                        return status
                time.sleep(0.1)
            raise TimeoutError(
                f"GELLO reset did not enter hold within {HELPER_START_TIMEOUT_SEC:.0f}s."
            )
        except Exception:
            self.stop("reset startup failure")
            raise

    def arm_release_after_motion(self) -> bool:
        self.poll()
        if not self.is_holding or self.process is None:
            return False
        if not self.release_armed:
            self.process.send_signal(signal.SIGUSR1)
            self.release_armed = True
        return True

    def poll(self) -> None:
        if self.process is None:
            return
        return_code = self.process.poll()
        if return_code is None:
            return
        if self.release_armed and return_code == 0:
            print("GELLO hold released; torque is off.")
        elif return_code != 0:
            print(f"WARNING: GELLO reset helper exited with code {return_code}.")
        self.process = None
        self.release_armed = False
        self.ready_file.unlink(missing_ok=True)

    def stop(self, reason: str) -> None:
        if self.process is None:
            self.ready_file.unlink(missing_ok=True)
            return
        if self.process.poll() is None:
            print(f"Stopping GELLO hold ({reason})...")
            self.process.terminate()
            try:
                self.process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                self.process.send_signal(signal.SIGINT)
                self.process.wait(timeout=5.0)
        self.process = None
        self.release_armed = False
        self.ready_file.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Internal ROS helper for lerobot_collection.py GELLO reset/hold."
    )
    parser.add_argument("--helper", action="store_true")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--parent-pid", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.helper or args.ready_file is None or args.parent_pid is None:
        raise SystemExit(
            "This helper is managed by lerobot_collection.py --gello-reset-hold."
        )
    raise SystemExit(_run_helper(args.ready_file, args.parent_pid))


if __name__ == "__main__":
    main()
