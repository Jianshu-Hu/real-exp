"""Parse pose records emitted by Hand Tracking Streamer.

The upstream application publishes wrist poses.  Controller-enabled builds
additionally publish Touch-controller poses in several layouts.  This module
only validates and names those raw Unity-frame values; coordinate calibration
belongs in a later teleoperation mapping layer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


_FRAME_RE = re.compile(r"(?:^|\|)\s*f\s*=\s*(\d+)", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"(?:^|\|)\s*t\s*=\s*(\d+)", re.IGNORECASE)
_SIDES = ("left", "right")


class ProtocolError(ValueError):
    """A pose-labelled HTS record has an invalid or unsupported payload."""


@dataclass(frozen=True, slots=True)
class PoseSample:
    """One validated pose in the uncalibrated Unity tracking frame."""

    side: str
    source: str
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    tracked: bool
    grasp: float | None = None
    clutch: bool | None = None
    record: bool | None = None
    frame_id: int | None = None
    device_timestamp_ns: int | None = None
    layout: str = ""


def _header_and_values(line: str) -> tuple[str, list[float]]:
    header, separator, payload = line.strip().rpartition(":")
    if not separator:
        raise ProtocolError("missing ':' separator")

    values: list[float] = []
    for field in payload.split(","):
        field = field.strip()
        if not field:
            continue
        try:
            value = float(field)
        except ValueError as exc:
            raise ProtocolError(f"non-numeric pose field {field!r}") from exc
        if not math.isfinite(value):
            raise ProtocolError("pose fields must be finite")
        values.append(value)
    return header.strip(), values


def _side(header: str) -> str | None:
    lowered = header.lower()
    return next((side for side in _SIDES if side in lowered), None)


def _normalized_quaternion(values: list[float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1e-6:
        raise ProtocolError("pose quaternion has zero length")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _metadata(header: str, pattern: re.Pattern[str]) -> int | None:
    match = pattern.search(header)
    return int(match.group(1)) if match else None


def parse_pose_line(line: str) -> PoseSample | None:
    """Parse one HTS line, returning ``None`` for non-pose telemetry.

    Supported layouts are:

    - upstream wrist/head: ``px,py,pz,qx,qy,qz,qw``;
    - lab controller build: ``pose,grasp,clutch,record`` (10 floats);
    - legacy controller: ``tracked,pose,basis-vectors,grasp`` (18 floats);
    - padded controller extension: ``pose,padding,grasp`` (17 floats).

    Receipt of a position-first controller record implies tracking because the
    Quest-side component does not send a record when its controller is absent.
    """

    stripped = line.strip()
    if not stripped:
        return None

    raw_header, separator, _ = stripped.rpartition(":")
    if not separator:
        return None
    raw_header_lower = raw_header.lower()
    if not any(
        pose_label in raw_header_lower
        for pose_label in ("controller", "wrist", "head")
    ):
        return None

    header, values = _header_and_values(stripped)
    lowered = header.lower()
    side = _side(header)
    frame_id = _metadata(header, _FRAME_RE)
    timestamp_ns = _metadata(header, _TIMESTAMP_RE)

    if "controller" in lowered:
        if side is None:
            raise ProtocolError("controller record has no left/right side")
        clutch: bool | None = None
        record: bool | None = None
        if len(values) == 10:
            if values[8] not in (0.0, 1.0) or values[9] not in (0.0, 1.0):
                raise ProtocolError("controller clutch/record flags must be 0 or 1")
            tracked = True
            position = values[0:3]
            quaternion = values[3:7]
            grasp = values[7]
            clutch = bool(values[8])
            record = bool(values[9])
            layout = "controller_compact_controls_v1"
        elif len(values) == 18:
            if values[0] not in (0.0, 1.0):
                raise ProtocolError("tracked-first controller flag must be 0 or 1")
            tracked = bool(values[0])
            position = values[1:4]
            quaternion = values[4:8]
            grasp = values[17]
            layout = "controller_tracked_first_v1"
        elif len(values) == 17:
            tracked = True
            position = values[0:3]
            quaternion = values[3:7]
            grasp = values[16]
            layout = "controller_position_first_v1"
        else:
            raise ProtocolError(
                f"controller record has {len(values)} floats; expected 10, 17, or 18"
            )
        return PoseSample(
            side=side,
            source="controller",
            position=tuple(position),  # type: ignore[arg-type]
            quaternion_xyzw=_normalized_quaternion(quaternion),
            tracked=tracked,
            grasp=grasp,
            clutch=clutch,
            record=record,
            frame_id=frame_id,
            device_timestamp_ns=timestamp_ns,
            layout=layout,
        )

    if "wrist" in lowered or "head" in lowered:
        if len(values) != 7:
            raise ProtocolError(
                f"wrist/head record has {len(values)} floats; expected 7"
            )
        source = "head" if "head" in lowered else "wrist"
        if source == "wrist" and side is None:
            raise ProtocolError("wrist record has no left/right side")
        return PoseSample(
            side=side or "head",
            source=source,
            position=tuple(values[0:3]),  # type: ignore[arg-type]
            quaternion_xyzw=_normalized_quaternion(values[3:7]),
            tracked=True,
            frame_id=frame_id,
            device_timestamp_ns=timestamp_ns,
            layout=f"{source}_pose_v1",
        )

    return None
