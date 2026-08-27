"""Dependency-free 3D canvas for uncalibrated Quest controller poses."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Sequence

from .receiver import ReceiverSnapshot


Vector3 = tuple[float, float, float]


def rotate_vector_xyzw(quaternion: Sequence[float], vector: Vector3) -> Vector3:
    """Rotate a vector by a normalized xyzw quaternion."""

    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


class ControllerPoseVisualizer:
    """Render latest controller poses in an isometric Unity-frame view."""

    _SIDE_COLORS = {"left": "#38bdf8", "right": "#fb923c"}
    _AXIS_COLORS = {"X": "#ef4444", "Y": "#22c55e", "Z": "#3b82f6"}
    _WORLD_AXIS_COLORS = {"X": "#fca5a5", "Y": "#86efac", "Z": "#93c5fd"}

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Meta Quest controller poses — raw Unity frame")
        self.root.minsize(760, 560)
        self.canvas = tk.Canvas(self.root, background="#0f172a", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._snapshot: ReceiverSnapshot | None = None
        self.canvas.bind("<Configure>", lambda _event: self.redraw())

    def update(self, snapshot: ReceiverSnapshot) -> None:
        self._snapshot = snapshot
        self.redraw()

    def _project(self, point: Vector3) -> tuple[float, float]:
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        scale = min(width / 3.2, height / 2.4)
        x, y, z = point
        return (
            width * 0.5 + scale * (x - 0.55 * z),
            height * 0.88 - scale * (y - 0.30 * z),
        )

    def _line(self, start: Vector3, end: Vector3, **options: object) -> None:
        self.canvas.create_line(*self._project(start), *self._project(end), **options)

    def _draw_grid(self) -> None:
        for step in range(-5, 6):
            coordinate = step * 0.2
            color = "#334155" if step else "#64748b"
            self._line((-1.0, 0.0, coordinate), (1.0, 0.0, coordinate), fill=color)
            self._line((coordinate, 0.0, -1.0), (coordinate, 0.0, 1.0), fill=color)
        for label, endpoint in (
            ("+X", (1.08, 0.0, 0.0)),
            ("+Y", (0.0, 1.8, 0.0)),
            ("+Z", (0.0, 0.0, 1.08)),
        ):
            axis = label[-1]
            self._line(
                (0.0, 0.0, 0.0),
                endpoint,
                fill=self._WORLD_AXIS_COLORS[axis],
                dash=(4, 4),
                width=2,
                arrow=tk.LAST,
            )
            x, y = self._project(endpoint)
            self.canvas.create_text(
                x + 7,
                y,
                text=label,
                anchor="w",
                fill=self._WORLD_AXIS_COLORS[axis],
                font=("TkDefaultFont", 10, "bold"),
            )

        origin_x, origin_y = self._project((0.0, 0.0, 0.0))
        self.canvas.create_oval(
            origin_x - 3,
            origin_y - 3,
            origin_x + 3,
            origin_y + 3,
            fill="#e2e8f0",
            outline="",
        )

    def _draw_controller(self, side: str, pose, receive_hz: float, row: int) -> None:
        sample = pose.sample
        position = sample.position
        color = self._SIDE_COLORS.get(side, "white")
        screen_position = self._project(position)

        self._line((position[0], 0.0, position[2]), position, fill=color, dash=(3, 4), width=2)
        radius = 7
        self.canvas.create_oval(
            screen_position[0] - radius,
            screen_position[1] - radius,
            screen_position[0] + radius,
            screen_position[1] + radius,
            fill=color,
            outline="white",
            width=1,
        )

        axis_length = 0.16
        for label, local_axis in (
            ("X", (1.0, 0.0, 0.0)),
            ("Y", (0.0, 1.0, 0.0)),
            ("Z", (0.0, 0.0, 1.0)),
        ):
            direction = rotate_vector_xyzw(sample.quaternion_xyzw, local_axis)
            endpoint = tuple(
                position[index] + axis_length * direction[index] for index in range(3)
            )
            self._line(
                position,
                endpoint,
                fill=self._AXIS_COLORS[label],
                width=3,
                arrow=tk.LAST,
                arrowshape=(10, 12, 4),
            )
            endpoint_x, endpoint_y = self._project(endpoint)
            self.canvas.create_text(
                endpoint_x + 5,
                endpoint_y,
                text=label,
                anchor="w",
                fill=self._AXIS_COLORS[label],
                font=("TkDefaultFont", 9, "bold"),
            )

        position_text = ", ".join(f"{value:+.3f}" for value in position)
        grasp = 0.0 if sample.grasp is None else sample.grasp
        controls = f"trigger={grasp:.2f}"
        if sample.clutch is not None:
            controls += f"  clutch={int(sample.clutch)}  record={int(bool(sample.record))}"
        self.canvas.create_text(
            18,
            70 + row * 48,
            anchor="nw",
            fill=color,
            font=("TkFixedFont", 11, "bold"),
            text=(
                f"{side.upper():5s}  {receive_hz:5.1f} Hz  p=[{position_text}]\n"
                f"       {controls}"
            ),
        )

    def redraw(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self.canvas.create_text(
            18,
            16,
            anchor="nw",
            fill="white",
            font=("TkDefaultFont", 15, "bold"),
            text="Live controller pose",
        )
        self.canvas.create_text(
            18,
            43,
            anchor="nw",
            fill="#94a3b8",
            text="Tracking-frame meters • fixed axes: X/Y/Z pastel • controller axes: X/Y/Z saturated",
        )
        self.canvas.create_text(
            self.canvas.winfo_width() - 18,
            18,
            anchor="ne",
            fill="#cbd5e1",
            font=("TkDefaultFont", 10, "bold"),
            text="FIXED UNITY TRACKING FRAME",
        )
        if self._snapshot is None:
            return

        controllers = [
            (key, pose)
            for key, pose in sorted(self._snapshot.poses.items())
            if key[0] == "controller"
        ]
        if not controllers:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="Waiting for controller poses…",
                fill="#cbd5e1",
                font=("TkDefaultFont", 16),
            )
            return
        for row, ((source, side), pose) in enumerate(controllers):
            self._draw_controller(
                side,
                pose,
                self._snapshot.receive_hz.get((source, side), 0.0),
                row,
            )
