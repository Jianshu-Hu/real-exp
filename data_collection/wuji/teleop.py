#!/usr/bin/env python3
"""Teleoperate one Wuji Hand 2 directly from one same-side Wuji Glove."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from collections.abc import Sequence
from typing import Any

import numpy as np

try:  # Works as a script and as ``python -m data_collection.wuji``.
    from .mapping import WujiHandCommandLimiter
except ImportError:  # pragma: no cover - direct script execution
    from mapping import WujiHandCommandLimiter


LANDMARK_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

# Wuji Hand 2 joint-state frames identify the 20 firmware-order joints by NID.
HAND2_JOINT_NIDS = (
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    21,
    22,
    23,
    24,
)


class TeleopStop(Exception):
    """Raised for an operator or watchdog stop that should shut down cleanly."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-id", help="glove serial number or discovered IP:port")
    parser.add_argument("--hand-id", help="hand serial number or discovered IP:port")
    parser.add_argument(
        "--rate",
        type=float,
        default=120.0,
        help="maximum command rate in Hz (official Hand 2 example: 120)",
    )
    parser.add_argument(
        "--max-velocity-rad-s",
        type=float,
        default=1.0,
        help="per-joint command velocity limit (default: 1.0 rad/s)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.25,
        help="minimum confidence required for every glove landmark",
    )
    parser.add_argument(
        "--frame-timeout",
        type=float,
        default=0.25,
        help="disable the hand after this many seconds without a valid glove frame",
    )
    parser.add_argument(
        "--effort-limit",
        type=float,
        default=1.5,
        help="Hand 2 current limit in A (official example: 1.5)",
    )
    parser.add_argument(
        "--mit-kp",
        type=float,
        default=3.0,
        help="Hand 2 MIT position gain (official example: 3.0)",
    )
    parser.add_argument(
        "--mit-kd",
        type=float,
        default=0.05,
        help="Hand 2 MIT damping gain (official example: 0.05)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=3.0,
        help="hold the measured pose for this long before following the glove",
    )
    parser.add_argument(
        "--hold-only",
        action="store_true",
        help="enable and hold the measured pose without following the glove",
    )
    parser.add_argument(
        "--max-hold-delta-rad",
        type=float,
        default=0.05,
        help="abort if a joint moves this far during the stationary hold check",
    )
    parser.add_argument(
        "--allow-diagnostic-warnings",
        action="store_true",
        help="permit enable despite active hand warnings (unsafe diagnostic override)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="retarget and print commands without enabling or commanding the hand",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="stop after this many seconds; zero runs until interrupted",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="bypass the live-mode typed confirmation (for supervised automation only)",
    )
    parser.add_argument(
        "--sdk-log-level",
        choices=("trace", "debug", "info", "warn", "error", "off"),
        default="warn",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.max_velocity_rad_s <= 0:
        raise ValueError("--max-velocity-rad-s must be positive")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("--min-confidence must be in [0, 1]")
    if args.frame_timeout <= 0:
        raise ValueError("--frame-timeout must be positive")
    if args.effort_limit <= 0 or args.mit_kp <= 0 or args.mit_kd < 0:
        raise ValueError("--effort-limit/--mit-kp must be positive and --mit-kd non-negative")
    if args.hold_seconds < 0:
        raise ValueError("--hold-seconds cannot be negative")
    if args.max_hold_delta_rad <= 0:
        raise ValueError("--max-hold-delta-rad must be positive")
    if args.duration < 0:
        raise ValueError("--duration cannot be negative")


def select_device(devices: Sequence[Any], device_type: Any, selector: str | None, label: str) -> Any:
    matches = [device for device in devices if device.device_type == device_type]
    if selector is not None:
        matches = [
            device
            for device in matches
            if device.sn == selector
            or device.address == selector
            or device.address.split(":", 1)[0] == selector
        ]
    if not matches:
        suffix = f" matching {selector!r}" if selector else ""
        raise RuntimeError(f"No {label} discovered{suffix}")
    if len(matches) > 1:
        found = ", ".join(f"{device.sn} ({device.address})" for device in matches)
        raise RuntimeError(f"Multiple {label}s discovered: {found}; select one explicitly")
    return matches[0]


def skeleton_keypoints(frame: Any, min_confidence: float) -> np.ndarray:
    joints = list(frame.joints)
    names = tuple(joint.name for joint in joints)
    if names != LANDMARK_NAMES:
        raise ValueError(f"unexpected glove landmark order: {names!r}")
    confidence = np.asarray([joint.confidence for joint in joints], dtype=np.float64)
    if confidence.shape != (21,) or not np.all(np.isfinite(confidence)):
        raise ValueError("glove landmark confidence is malformed or non-finite")
    if float(np.min(confidence)) < min_confidence:
        index = int(np.argmin(confidence))
        raise ValueError(
            f"low confidence for {LANDMARK_NAMES[index]}: {confidence[index]:.3f} "
            f"< {min_confidence:.3f}"
        )
    keypoints = np.asarray([joint.pose.position for joint in joints], dtype=np.float32)
    if keypoints.shape != (21, 3) or not np.all(np.isfinite(keypoints)):
        raise ValueError("glove skeleton positions must be a finite (21, 3) array")
    if float(np.max(np.linalg.norm(keypoints - keypoints[0], axis=1))) < 0.03:
        raise ValueError("glove skeleton is degenerate")
    return keypoints


def hand_joint_positions(frame: Any) -> np.ndarray:
    by_nid = {int(joint.nid): float(joint.position) for joint in frame.joints}
    missing = [nid for nid in HAND2_JOINT_NIDS if nid not in by_nid]
    if missing:
        raise ValueError(f"hand state is missing joint NIDs {missing}")
    positions = np.asarray([by_nid[nid] for nid in HAND2_JOINT_NIDS], dtype=np.float64)
    if not np.all(np.isfinite(positions)):
        raise ValueError("hand state contains non-finite positions")
    return positions


def validate_diagnostics(
    frame: Any, describe_error: Any | None = None
) -> list[tuple[int, int, str]]:
    by_nid = {int(joint.nid): int(joint.error_code_current) for joint in frame.joints}
    missing = [nid for nid in HAND2_JOINT_NIDS if nid not in by_nid]
    if missing:
        raise RuntimeError(f"hand diagnostics are missing joint NIDs {missing}")
    faults: list[tuple[int, int, str]] = []
    warnings: list[tuple[int, int, str]] = []
    for nid in HAND2_JOINT_NIDS:
        code = by_nid[nid]
        if code == 0:
            continue
        description = describe_error(code) if describe_error is not None else None
        name = "Unknown" if description is None else str(description.get("name", "Unknown"))
        severity = None if description is None else description.get("severity")
        item = (nid, code, name)
        if severity == "Warning":
            warnings.append(item)
        else:
            faults.append(item)
    if faults:
        details = ", ".join(
            f"NID {nid}: 0x{code:04x} {name}" for nid, code, name in faults
        )
        raise RuntimeError(f"hand reports active joint faults: {details}")
    return warnings


def diagnostic_comm_summary(frame: Any) -> tuple[str, list[str]]:
    """Format host/device and per-joint communication quality for diagnosis."""
    comm = frame.comm
    e2e_received = int(comm.e2e_received)
    e2e_lost = int(comm.e2e_lost)
    e2e_total = e2e_received + e2e_lost
    e2e_loss_pct = 0.0 if e2e_total == 0 else 100.0 * e2e_lost / e2e_total
    overall = (
        f"host_link received={e2e_received} lost={e2e_lost} "
        f"loss={e2e_loss_pct:.4f}% reordered={int(comm.e2e_reordered)} "
        f"duplicates={int(comm.e2e_duplicates)} sdk_dropped={int(comm.sdk_dropped)} "
        f"rpc_retries={int(comm.rpc_retries)}/{int(comm.rpc_total)} "
        f"rpc_timeouts={int(comm.rpc_timeouts)} "
        f"comm_get_failures={int(comm.comm_get_failures)}"
    )
    joint_details = []
    for joint in sorted(frame.joints, key=lambda item: int(item.nid)):
        response = int(joint.comm_response_rate_pct)
        timeouts = int(joint.comm_timeout_total)
        if response < 100 or timeouts > 0 or int(joint.error_code_current) != 0:
            joint_details.append(
                f"NID {int(joint.nid)}: response={response}% "
                f"timeouts={timeouts} error=0x{int(joint.error_code_current):04x}"
            )
    return overall, joint_details


def drain_latest(subscription: Any) -> Any | None:
    latest = None
    while True:
        frame = subscription.recv()
        if frame is None:
            return latest
        latest = frame


def wait_for_frame(subscription: Any, timeout: float, label: str) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = drain_latest(subscription)
        if frame is not None:
            return frame
        time.sleep(0.002)
    raise TimeoutError(f"timed out waiting for {label} after {timeout:.1f}s")


def read_hand_tuning(hand: Any) -> tuple[list[float], list[float], list[float]]:
    effort_values = hand.effort_limit().get()
    if len(effort_values) != 20 or any(value is None for value in effort_values):
        raise RuntimeError("could not read all 20 hand effort limits")
    efforts = [float(value) for value in effort_values]
    if not np.all(np.isfinite(efforts)) or min(efforts) < 0:
        raise RuntimeError("hand effort limits are invalid")
    mit_values = hand.mit_params().get()
    if len(mit_values) != 20 or any(value is None for value in mit_values):
        raise RuntimeError("could not read all 20 hand MIT parameter sets")
    kp_values = [float(value.kp) for value in mit_values]
    kd_values = [float(value.kd) for value in mit_values]
    if (
        not np.all(np.isfinite(kp_values))
        or not np.all(np.isfinite(kd_values))
        or min(kp_values) < 0
        or min(kd_values) < 0
    ):
        raise RuntimeError("hand MIT parameters are invalid")
    return efforts, kp_values, kd_values


def require_clean_diagnostics(
    warnings: set[tuple[int, int, str]], *, allow_warnings: bool
) -> None:
    if not warnings or allow_warnings:
        return
    grouped: dict[tuple[int, str], list[int]] = {}
    for nid, code, name in sorted(warnings):
        grouped.setdefault((code, name), []).append(nid)
    details = "; ".join(
        f"0x{code:04x} {name} on NIDs {nids}"
        for (code, name), nids in grouped.items()
    )
    raise RuntimeError(
        "refusing to enable while hand diagnostic warnings are active: "
        f"{details}. Resolve the encoder/internal-bus problem first; "
        "--allow-diagnostic-warnings is an unsafe diagnostic override"
    )


def configure_hand_tuning(hand: Any, effort_limit: float, kp: float, kd: float) -> None:
    """Apply the Wuji Hand 2 settings used by the official teleop example."""
    hand.effort_limit().set(float(effort_limit))
    hand.mit_params().set((float(kp), float(kd)))


def send_position(publisher: Any, joint_command_type: Any, position: np.ndarray) -> None:
    publisher.send(
        [joint_command_type(float(value), 0.0, 0.0) for value in position]
    )


def require_confirmation(args: argparse.Namespace, glove: Any, hand: Any) -> None:
    if args.dry_run or args.yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("live mode needs an interactive terminal or explicit --yes")
    print("\nLIVE CONTROL IS READY")
    print(f"  Glove: {glove.serial_number} ({glove.hand_side().get()})")
    print(f"  Hand:  {hand.serial_number} ({hand.handedness().get()})")
    print("Clear the hand workspace and keep an emergency-stop method within reach.")
    answer = input("Type ENABLE RIGHT HAND to start motion: ").strip()
    if answer != "ENABLE RIGHT HAND":
        raise TeleopStop("operator declined live enable")


def _restore_user(manager: Any, prior_user: dict[str, Any]) -> None:
    user_id = prior_user.get("user_id", "")
    if user_id:
        manager.switch_user(user_id)
    else:
        manager.switch_to_default_user()


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    try:
        import wuji_sdk
    except ModuleNotFoundError as exc:  # pragma: no cover - environment failure
        raise RuntimeError(
            "wuji-sdk is not installed; run: python3 -m pip install "
            "-r data_collection/wuji/requirements.txt"
        ) from exc

    wuji_sdk.set_log_level(args.sdk_log_level)
    manager = wuji_sdk.SdkManager.instance()
    prior_user = manager.current_user()
    subscriptions: list[Any] = []
    publisher = None
    hand = None
    enabled = False
    old_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        raise TeleopStop(f"received {signal.Signals(signum).name}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    try:
        # Wuji recommends the built-in default URDF for live glove retargeting.
        manager.switch_to_default_user()
        discovered = manager.scan()
        glove_info = select_device(
            discovered, wuji_sdk.DeviceType.WujiGlove, args.glove_id, "Wuji Glove"
        )
        hand_info = select_device(
            discovered, wuji_sdk.DeviceType.WujiHand2, args.hand_id, "Wuji Hand 2"
        )
        glove = manager.connect(sn=glove_info.sn, device_name="direct_teleop_glove")
        hand = manager.connect(sn=hand_info.sn, device_name="direct_teleop_hand")

        glove_side = glove.hand_side().get().lower()
        hand_side = hand.handedness().get().lower()
        if glove_side != "right" or hand_side != "right" or glove_side != hand_side:
            raise RuntimeError(
                f"this entry point requires a right glove and right hand; got {glove_side}/{hand_side}"
            )
        online_joints = int(hand.online_joints_count().get())
        if online_joints != 20:
            raise RuntimeError(f"expected 20 online hand joints, found {online_joints}")
        efforts, kp_values, kd_values = read_hand_tuning(hand)

        mode = "DRY RUN" if args.dry_run else "LIVE"
        print(f"Mode: {mode}")
        print(f"Glove: {glove_info.sn} at {glove_info.address} ({glove_side})")
        print(f"Hand:  {hand_info.sn} at {hand_info.address} ({hand_side})")
        print(f"Online joints: {online_joints}/20")
        print(
            f"Stored hand tuning: effort={min(efforts):.3f}..{max(efforts):.3f} A, "
            f"kp={min(kp_values):.3f}..{max(kp_values):.3f}, "
            f"kd={min(kd_values):.3f}..{max(kd_values):.3f}"
        )
        print(
            f"Live tuning to apply: effort={args.effort_limit:.3f} A, "
            f"kp={args.mit_kp:.3f}, kd={args.mit_kd:.3f}"
        )
        print(
            f"Watchdog: {args.frame_timeout:.3f}s; command limit: "
            f"{args.max_velocity_rad_s:.3f} rad/s per joint"
        )
        require_confirmation(args, glove, hand)

        # Create high-rate streams only after confirmation so an operator pause
        # cannot accumulate thousands of stale frames before live control begins.
        skeleton_sub = glove.hand_skeleton().subscribe()
        state_sub = hand.joint_states().subscribe()
        diagnostics_sub = hand.joint_diagnostics().subscribe()
        subscriptions.extend((skeleton_sub, state_sub, diagnostics_sub))

        first_skeleton = wait_for_frame(skeleton_sub, 8.0, "a valid glove skeleton")
        keypoints = skeleton_keypoints(first_skeleton, args.min_confidence)
        state = wait_for_frame(state_sub, 3.0, "the hand joint state")
        initial_position = hand_joint_positions(state)
        diagnostics = wait_for_frame(diagnostics_sub, 3.0, "the hand diagnostics")
        diagnostic_warnings = set(validate_diagnostics(diagnostics, hand.describe_error))
        comm_overall, comm_joints = diagnostic_comm_summary(diagnostics)
        print(f"Communication: {comm_overall}")
        for detail in comm_joints:
            print(f"  {detail}")
        if diagnostic_warnings:
            grouped: dict[tuple[int, str], list[int]] = {}
            for nid, code, name in sorted(diagnostic_warnings):
                grouped.setdefault((code, name), []).append(nid)
            for (code, name), nids in grouped.items():
                print(
                    f"WARNING: hand diagnostic 0x{code:04x} {name} on NIDs {nids}",
                    file=sys.stderr,
                )
        if not args.dry_run:
            require_clean_diagnostics(
                diagnostic_warnings,
                allow_warnings=args.allow_diagnostic_warnings,
            )

        period = 1.0 / args.rate
        started_at = time.monotonic()
        next_tick = started_at
        last_valid_frame_at = started_at
        next_status_at = started_at
        next_comm_status_at = started_at + 2.0
        latest_skeleton = first_skeleton
        sent = 0
        actual_position = initial_position.copy()

        if not args.dry_run:
            configure_hand_tuning(
                hand,
                effort_limit=args.effort_limit,
                kp=args.mit_kp,
                kd=args.mit_kd,
            )
            applied_efforts, applied_kps, applied_kds = read_hand_tuning(hand)
            if not (
                np.allclose(applied_efforts, args.effort_limit, atol=1e-6)
                and np.allclose(applied_kps, args.mit_kp, atol=1e-6)
                and np.allclose(applied_kds, args.mit_kd, atol=1e-6)
            ):
                raise RuntimeError("Hand 2 did not accept the requested effort/MIT tuning")

            # Match Wuji's Hand 2 example: configure, enable, then open the MIT
            # joint-command stream before initializing the retargeting session.
            enabled = True
            hand.enable()
            publisher = hand.joint_command().publish()
            send_position(publisher, wuji_sdk.JointCommand, initial_position)
            print(
                "Hand enabled with official MIT tuning. Holding measured pose "
                "before glove control."
            )

            hold_started_at = time.monotonic()
            next_hold_status_at = hold_started_at
            while args.hold_only or time.monotonic() - hold_started_at < args.hold_seconds:
                now = time.monotonic()
                if args.duration and now - started_at >= args.duration:
                    raise TeleopStop(f"completed requested {args.duration:.1f}s duration")
                if now < next_tick:
                    time.sleep(next_tick - now)
                now = time.monotonic()
                next_tick = max(next_tick + period, now)

                send_position(publisher, wuji_sdk.JointCommand, initial_position)
                diagnostics = drain_latest(diagnostics_sub)
                if diagnostics is not None:
                    diagnostic_warnings = set(
                        validate_diagnostics(diagnostics, hand.describe_error)
                    )
                    require_clean_diagnostics(
                        diagnostic_warnings,
                        allow_warnings=args.allow_diagnostic_warnings,
                    )
                state = drain_latest(state_sub)
                if state is not None:
                    actual_position = hand_joint_positions(state)
                    max_hold_delta = float(
                        np.max(np.abs(actual_position - initial_position))
                    )
                    if max_hold_delta > args.max_hold_delta_rad:
                        raise TeleopStop(
                            "stationary hold check failed: actual joint moved "
                            f"{max_hold_delta:.3f} rad (limit "
                            f"{args.max_hold_delta_rad:.3f} rad)"
                        )
                if now >= next_hold_status_at:
                    max_hold_delta = float(
                        np.max(np.abs(actual_position - initial_position))
                    )
                    print(
                        f"hold max_delta={max_hold_delta:.4f} rad",
                        flush=True,
                    )
                    next_hold_status_at = now + 0.5
        else:
            print("Hand remains disabled. Press Ctrl-C to stop the dry run.")

        # Initialize after the live hold so no old warm-start/filter state is
        # carried across the operator pause or actuator preflight.
        latest_skeleton = wait_for_frame(skeleton_sub, 3.0, "a current glove skeleton")
        keypoints = skeleton_keypoints(latest_skeleton, args.min_confidence)
        retarget = wuji_sdk.RetargetSession.for_hand(
            wuji_sdk.HandModel.WujiHand2, side=wuji_sdk.Handedness.Right
        )
        retarget.reset()
        target = np.asarray(retarget.step(keypoints), dtype=np.float64)
        if target.shape != (20,) or not np.all(np.isfinite(target)):
            raise RuntimeError("retargeter returned an invalid initial command")
        limiter = WujiHandCommandLimiter(
            actual_position,
            max_velocity_rad_s=args.max_velocity_rad_s,
            rate_hz=args.rate,
        )
        last_valid_frame_at = time.monotonic()
        previous_target = target.copy()
        previous_target_at = last_valid_frame_at
        target_speed_samples: list[float] = []
        if not args.dry_run:
            print("Stationary hold passed; following glove. Press Ctrl-C to stop.")

        while True:
            now = time.monotonic()
            if args.duration and now - started_at >= args.duration:
                raise TeleopStop(f"completed requested {args.duration:.1f}s duration")
            if now < next_tick:
                time.sleep(next_tick - now)
            now = time.monotonic()
            next_tick = max(next_tick + period, now)

            new_skeleton = drain_latest(skeleton_sub)
            if new_skeleton is not None:
                latest_skeleton = new_skeleton
                try:
                    keypoints = skeleton_keypoints(latest_skeleton, args.min_confidence)
                    new_target = np.asarray(retarget.step(keypoints), dtype=np.float64)
                    if new_target.shape != (20,) or not np.all(np.isfinite(new_target)):
                        raise ValueError("retargeter returned a non-finite 20-joint command")
                except (ValueError, RuntimeError) as exc:
                    if now - last_valid_frame_at >= args.frame_timeout:
                        raise TeleopStop(f"glove input stayed invalid: {exc}") from exc
                    continue
                target_dt = now - previous_target_at
                if target_dt > 0:
                    target_speed_samples.append(
                        float(np.max(np.abs(new_target - previous_target))) / target_dt
                    )
                target = new_target
                previous_target = target.copy()
                previous_target_at = now
                last_valid_frame_at = now
            elif now - last_valid_frame_at >= args.frame_timeout:
                raise TeleopStop(
                    f"glove frame watchdog expired after {now - last_valid_frame_at:.3f}s"
                )
            else:
                continue

            diagnostics = drain_latest(diagnostics_sub)
            if diagnostics is not None:
                warnings = set(validate_diagnostics(diagnostics, hand.describe_error))
                # Dry-run is a passive diagnostic mode: report warnings but do
                # not apply the live-enable interlock or raise because of them.
                if not args.dry_run:
                    require_clean_diagnostics(
                        warnings,
                        allow_warnings=args.allow_diagnostic_warnings,
                    )
                new_warnings = warnings - diagnostic_warnings
                for nid, code, name in sorted(new_warnings):
                    print(
                        f"WARNING: hand diagnostic 0x{code:04x} {name} on NID {nid}",
                        file=sys.stderr,
                    )
                diagnostic_warnings = warnings
                if args.dry_run and now >= next_comm_status_at:
                    comm_overall, comm_joints = diagnostic_comm_summary(diagnostics)
                    print(f"Communication: {comm_overall}", flush=True)
                    for detail in comm_joints:
                        print(f"  {detail}", flush=True)
                    next_comm_status_at = now + 2.0
            state = drain_latest(state_sub)
            if state is not None:
                hand_joint_positions(state)

            command = limiter.limit(target)
            if publisher is not None:
                send_position(publisher, wuji_sdk.JointCommand, command)
            sent += 1
            if now >= next_status_at:
                max_delta = float(np.max(np.abs(command - initial_position)))
                raw_speed_p95 = (
                    float(np.percentile(target_speed_samples, 95))
                    if target_speed_samples
                    else 0.0
                )
                raw_speed_max = max(target_speed_samples, default=0.0)
                print(
                    f"frames={sent} min={command.min():+.3f} max={command.max():+.3f} "
                    f"command_delta_from_hand_start={max_delta:.3f} "
                    f"raw_target_speed_p95={raw_speed_p95:.4f}rad/s "
                    f"raw_target_speed_max={raw_speed_max:.4f}rad/s",
                    flush=True,
                )
                target_speed_samples.clear()
                next_status_at = now + 1.0
    except (KeyboardInterrupt, TeleopStop) as exc:
        print(f"Stopping: {exc}")
    except (RuntimeError, TimeoutError, ValueError) as exc:
        # Hardware/stream validation failures must still run the cleanup path,
        # but should be presented as an operator-readable stop rather than a
        # Python traceback during a supervised robot test.
        print(f"ABORTED: {exc}", file=sys.stderr)
    finally:
        if enabled and hand is not None:
            try:
                hand.disable()
                print("Hand disabled.")
            except Exception as exc:  # pragma: no cover - hardware failure
                print(f"WARNING: hand disable failed: {exc}", file=sys.stderr)
        if publisher is not None:
            try:
                publisher.close()
            except Exception:
                pass
        for subscription in reversed(subscriptions):
            try:
                subscription.close()
            except Exception:
                pass
        try:
            manager.disconnect_all()
        finally:
            _restore_user(manager, prior_user)
            for signum, old_handler in old_handlers.items():
                signal.signal(signum, old_handler)


if __name__ == "__main__":
    run(parse_args())
