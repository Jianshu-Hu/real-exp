#!/usr/bin/env python3
"""Interactively select synchronized LeRobot frames for hand-eye calibration."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


SAMPLE_COUNT = 20
VIDEO_KEY = "observation.images.cam_front"
COLOR_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 394.361236572266,
    "fy": 393.302062988281,
    "cx": 318.513824462891,
    "cy": 230.271636962891,
    "model": "distortion.inverse_brown_conrady",
    "coeffs": [
        -0.0534335076808929,
        0.0580953061580658,
        0.000541441899258643,
        -0.000415135698858649,
        -0.019531236961484,
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="LeRobot dataset root")
    parser.add_argument("--output", type=Path, required=True, help="New sample directory")
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    if args.episode < 0:
        parser.error("episode must be non-negative")
    return args


def require_dependencies() -> tuple[Any, Any, Any]:
    try:
        import pyarrow.parquet as pq
        from PIL import Image, ImageTk
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise SystemExit(
            "this selector requires pyarrow, Pillow, scipy, numpy, tkinter, and ffmpeg; "
            f"missing {exc.name}"
        ) from exc
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg was not found on PATH")
    return pq, (Image, ImageTk), Rotation


def read_parquet_rows(pq: Any, directory: Path, columns: list[str] | None = None) -> list[dict[str, Any]]:
    files = sorted(directory.glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found under {directory}")
    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows


def video_dimensions(path: Path) -> tuple[int, int]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(path),
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not inspect video {path}: {exc}") from exc
    stream = json.loads(output)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def decode_frame(video: Path, frame_index: int, Image: Any) -> Any:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1",
        "-f", "image2pipe", "-vcodec", "png", "-",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"failed to decode video frame {frame_index}: {message}") from exc
    if not result.stdout:
        raise RuntimeError(f"video did not produce frame {frame_index}")
    image = Image.open(io.BytesIO(result.stdout))
    image.load()
    return image.convert("RGB")


def pose_to_matrix(pose: Any, Rotation: Any) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("observation.ee_pose must be [x, y, z, roll, pitch, yaw]")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
    transform[:3, 3] = values[:3]
    return transform


def load_episode(pq: Any, root: Path, episode_index: int) -> tuple[list[dict[str, Any]], Path]:
    episodes = read_parquet_rows(pq, root / "meta/episodes")
    matches = [row for row in episodes if int(row["episode_index"]) == episode_index]
    if len(matches) != 1:
        raise ValueError(f"episode {episode_index} was not found exactly once in {root}")
    episode = matches[0]
    columns = ["episode_index", "frame_index", "timestamp", "observation.ee_pose"]
    rows = read_parquet_rows(pq, root / "data", columns)
    rows = sorted(
        (row for row in rows if int(row["episode_index"]) == episode_index),
        key=lambda row: int(row["frame_index"]),
    )
    if not rows or [int(row["frame_index"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"episode {episode_index} frame_index values are not continuous from zero")
    chunk = int(episode[f"videos/{VIDEO_KEY}/chunk_index"])
    file_index = int(episode[f"videos/{VIDEO_KEY}/file_index"])
    video = root / "videos" / VIDEO_KEY / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    if not video.is_file():
        raise FileNotFoundError(f"camera video is missing: {video}")
    return rows, video


def export_samples(
    output: Path,
    dataset: Path,
    episode: int,
    rows: list[dict[str, Any]],
    selected: list[int],
    video: Path,
    intrinsics: dict[str, Any],
    Image: Any,
    Rotation: Any,
) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        manifest_samples = []
        for sample_number, frame_index in enumerate(sorted(selected)):
            row = rows[frame_index]
            image = np.asarray(decode_frame(video, frame_index, Image))[:, :, ::-1].copy()
            sample_id = f"{sample_number:06d}"
            sample_dir = temporary / f"sample_{sample_id}"
            sample_dir.mkdir()
            np.save(sample_dir / "rgb.npy", image)
            metadata = {
                "sample_id": sample_id,
                "source": {
                    "dataset": str(dataset),
                    "episode_index": episode,
                    "frame_index": frame_index,
                    "timestamp_s": float(row["timestamp"]),
                    "video": str(video),
                },
                "color_intrinsics": intrinsics,
                "B_T_E": pose_to_matrix(row["observation.ee_pose"], Rotation).tolist(),
            }
            (sample_dir / "sample.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            manifest_samples.append(metadata["source"])
        manifest = {
            "format": "real_exp_camera_calibration_selection_v1",
            "source_dataset": str(dataset),
            "episode_index": episode,
            "video_key": VIDEO_KEY,
            "sample_count": len(selected),
            "samples": manifest_samples,
        }
        (temporary / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class Selector:
    def __init__(
        self,
        root: Any,
        rows: list[dict[str, Any]],
        video: Path,
        sample_count: int,
        Image: Any,
        ImageTk: Any,
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.rows = rows
        self.video = video
        self.sample_count = sample_count
        self.Image = Image
        self.ImageTk = ImageTk
        self.selected: set[int] = set()
        self.result: list[int] | None = None
        self.pending: str | None = None
        self.photo: Any | None = None

        root.title("Camera calibration sample selection")
        root.geometry("980x720")
        root.minsize(720, 560)
        root.protocol("WM_DELETE_WINDOW", self.cancel)

        self.image_label = ttk.Label(root, anchor="center")
        self.image_label.pack(fill="both", expand=True, padx=12, pady=(12, 4))
        self.status = ttk.Label(root, anchor="center")
        self.status.pack(fill="x", padx=12, pady=4)
        self.scale = ttk.Scale(root, from_=0, to=len(rows) - 1, command=self.seek)
        self.scale.pack(fill="x", padx=16, pady=6)

        controls = ttk.Frame(root)
        controls.pack(pady=6)
        ttk.Button(controls, text="Previous frame", command=lambda: self.step(-1)).grid(row=0, column=0, padx=4)
        ttk.Button(controls, text="Next frame", command=lambda: self.step(1)).grid(row=0, column=1, padx=4)
        self.select_button = ttk.Button(controls, text="Add frame", command=self.toggle)
        self.select_button.grid(row=0, column=2, padx=12)
        self.export_button = ttk.Button(controls, text="Export samples", command=self.finish, state="disabled")
        self.export_button.grid(row=0, column=3, padx=4)
        ttk.Button(controls, text="Cancel", command=self.cancel).grid(row=0, column=4, padx=4)

        root.bind("<Left>", lambda _event: self.step(-1))
        root.bind("<Right>", lambda _event: self.step(1))
        root.bind("<space>", lambda _event: self.toggle())
        self.show(0)

    def current(self) -> int:
        return int(round(float(self.scale.get())))

    def seek(self, _value: str) -> None:
        if self.pending is not None:
            self.root.after_cancel(self.pending)
        self.pending = self.root.after(100, lambda: self.show(self.current()))

    def step(self, amount: int) -> None:
        index = min(max(self.current() + amount, 0), len(self.rows) - 1)
        self.scale.set(index)
        self.show(index)

    def show(self, index: int) -> None:
        self.pending = None
        try:
            image = decode_frame(self.video, index, self.Image)
            image.thumbnail((940, 570))
            self.photo = self.ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo)
            row = self.rows[index]
            selected = " | selected" if index in self.selected else ""
            self.status.configure(
                text=f"Frame {index}/{len(self.rows) - 1} | time {float(row['timestamp']):.3f} s{selected} | "
                f"{len(self.selected)}/{self.sample_count} samples"
            )
            self.select_button.configure(text="Remove frame" if index in self.selected else "Add frame")
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror("Frame decoding failed", str(exc), parent=self.root)

    def toggle(self) -> None:
        from tkinter import messagebox

        index = self.current()
        if index in self.selected:
            self.selected.remove(index)
        elif len(self.selected) >= self.sample_count:
            messagebox.showinfo("Selection full", f"Remove a frame before adding another; exactly {self.sample_count} are required.", parent=self.root)
            return
        else:
            self.selected.add(index)
        self.export_button.configure(state="normal" if len(self.selected) == self.sample_count else "disabled")
        self.show(index)

    def finish(self) -> None:
        if len(self.selected) == self.sample_count:
            self.result = sorted(self.selected)
            self.root.destroy()

    def cancel(self) -> None:
        self.result = None
        self.root.destroy()


def main() -> int:
    args = parse_args()
    pq, (Image, ImageTk), Rotation = require_dependencies()
    try:
        rows, video = load_episode(pq, args.input, args.episode)
        width, height = video_dimensions(video)
        expected_size = (COLOR_INTRINSICS["width"], COLOR_INTRINSICS["height"])
        if (width, height) != expected_size:
            raise ValueError(
                f"cam_front video is {width}x{height}, but the built-in D435 color intrinsics "
                f"are for {expected_size[0]}x{expected_size[1]}"
            )
        if SAMPLE_COUNT > len(rows):
            raise ValueError(f"cannot select {SAMPLE_COUNT} samples from an episode with only {len(rows)} frames")
        if args.output.exists():
            raise FileExistsError(f"output already exists: {args.output.resolve()}")
    except (OSError, KeyError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc

    try:
        import tkinter as tk
    except ImportError as exc:
        raise SystemExit("tkinter is required for the selection interface") from exc
    root = tk.Tk()
    selector = Selector(root, rows, video, SAMPLE_COUNT, Image, ImageTk)
    root.mainloop()
    if selector.result is None:
        print("selection cancelled; no samples were written")
        return 1
    try:
        export_samples(
            args.output, args.input.resolve(), args.episode, rows, selector.result,
            video.resolve(), dict(COLOR_INTRINSICS), Image, Rotation,
        )
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"exported {len(selector.result)} samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
