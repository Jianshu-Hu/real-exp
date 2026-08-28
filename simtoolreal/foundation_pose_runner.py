"""Publish FoundationPose++ poses in real time.

``--pose-file`` and ``--mock`` are dependency-free transport modes used for
bring-up.  Live mode reuses the upstream RealSense script's estimator helpers
and keeps its first-frame ROI/mask registration behavior, while publishing
each successful estimate immediately over a versioned ZMQ packet.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq

from pose_publisher import read_pose
from transport import make_object_pose


def _make_mask(frame_shape: tuple[int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = roi
    h, w = frame_shape
    mask = np.zeros((h, w), dtype=bool)
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + width), min(h, y + height)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _select_and_register(estimator: Any, K: np.ndarray, rgb: np.ndarray, depth: np.ndarray,
                         refine_iter: int, roi: tuple[int, int, int, int] | None) -> tuple[Any, np.ndarray | None]:
    import cv2
    if roi is None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        x, y, width, height = cv2.selectROI("FoundationPose++", bgr, showCrosshair=True)
    else:
        x, y, width, height = roi
    if width < 2 or height < 2:
        return None, None
    mask = _make_mask(rgb.shape[:2], (int(x), int(y), int(width), int(height)))
    if np.count_nonzero(mask & (depth > 1e-4)) < 20:
        logging.warning("The selected ROI has too few valid depth pixels; select it again.")
        return None, None
    return estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=refine_iter), mask


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connect", default="tcp://127.0.0.1:5570")
    parser.add_argument("--pose-file", type=Path, help="Poll a JSON/NumPy pose (useful with an external tracker).")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--mesh-scale", type=float, default=0.001)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--est-refine-iter", type=int, default=10)
    parser.add_argument("--track-refine-iter", type=int, default=3)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--object-id", default="object")
    return parser.parse_args(argv)


def _publish(socket: Any, pose: Any, source: str, object_id: str) -> None:
    matrix = np.asarray(pose, dtype=np.float64)
    if matrix.ndim == 3:
        matrix = matrix[-1]
    socket.send_json(make_object_pose(matrix.reshape(4, 4).tolist(), object_id=object_id, frame_id="camera", source=source))


def _run_file(socket: Any, args: argparse.Namespace, stop: list[bool]) -> None:
    signature = None
    while not stop[0]:
        if args.mock:
            pose = np.eye(4); pose[:3, 3] = (0.0, 0.0, 0.5)
            _publish(socket, pose, "mock-foundationpose++", args.object_id)
        elif args.pose_file is not None:
            try:
                stat = args.pose_file.stat()
            except FileNotFoundError:
                time.sleep(1.0 / args.rate); continue
            current = (stat.st_mtime_ns, stat.st_size)
            if current != signature:
                _publish(socket, read_pose(args.pose_file), "foundationpose++-file", args.object_id)
                signature = current
        time.sleep(1.0 / args.rate)


def _run_realsense(socket: Any, args: argparse.Namespace, stop: list[bool]) -> None:
    if args.mesh is None:
        raise SystemExit("live FoundationPose++ mode requires --mesh")
    if args.no_display and args.roi is None:
        raise SystemExit("--no-display requires --roi X Y W H for first-frame initialization")
    # Import only in live mode so transport tests work on machines without CUDA.
    import cv2
    import pyrealsense2 as rs
    import torch
    import trimesh
    repo = Path(__file__).resolve().parents[1] / "libs/FoundationPose-plus-plus"
    import sys
    sys.path.insert(0, str(repo)); sys.path.insert(0, str(repo / "FoundationPose"))
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
    from Utils import dr, trimesh_add_pure_colored_texture
    if not torch.cuda.is_available():
        raise SystemExit("FoundationPose++ live mode requires a CUDA-enabled PyTorch")
    mesh = trimesh.load(str(args.mesh), force="mesh")
    mesh.apply_scale(args.mesh_scale)
    # FoundationPose's crop/projection path expects float32 vertices.  Trimesh
    # commonly loads STL vertices as float64 and can promote them again through
    # its public setter, so update the tracked data buffer directly.
    mesh._data["vertices"] = np.asarray(mesh.vertices, dtype=np.float32)
    mesh._cache.clear()
    try: mesh = trimesh_add_pure_colored_texture(mesh, color=np.asarray((0, 159, 237), dtype=np.uint8), resolution=10)
    except Exception as exc: logging.warning("mesh texture fallback failed: %s", exc)
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                scorer=scorer, refiner=refiner, glctx=dr.RasterizeCudaContext(), debug=0)
    if estimator.mesh is not None:
        estimator.mesh._data["vertices"] = np.asarray(estimator.mesh.vertices, dtype=np.float32)
        estimator.mesh._cache.clear()
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config); align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.asarray(((intrinsics.fx, 0, intrinsics.ppx), (0, intrinsics.fy, intrinsics.ppy), (0, 0, 1)), dtype=np.float32)
    initialized = False; pose = None
    print("FoundationPose++ waiting for first-frame mask/ROI initialization", flush=True)
    try:
        while not stop[0]:
            frames = align.process(pipeline.wait_for_frames()); color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth: continue
            bgr = np.asanyarray(color.get_data()); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * scale
            if not initialized:
                pose, _ = _select_and_register(estimator, K, rgb, depth_m, args.est_refine_iter, roi=args.roi)
                initialized = pose is not None
                if initialized: print("FoundationPose++ registered; publishing live poses", flush=True)
            else:
                pose = estimator.track_one(rgb=rgb, depth=depth_m, K=K, iteration=args.track_refine_iter)
            if pose is not None: _publish(socket, pose, "foundationpose++", args.object_id)
            if not args.no_display:
                cv2.imshow("FoundationPose++", bgr)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
    finally:
        pipeline.stop(); cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rate <= 0: raise SystemExit("--rate must be positive")
    if not args.mock and args.pose_file is None and args.mesh is None:
        raise SystemExit("provide --mesh for live mode, or --pose-file/--mock for transport mode")
    context = zmq.Context(); socket = context.socket(zmq.PUB); socket.setsockopt(zmq.SNDHWM, 2); socket.setsockopt(zmq.LINGER, 0); socket.bind(args.connect)
    stop = [False]; signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True)); signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    print(f"FoundationPose++ pose publisher on {args.connect}", flush=True)
    try:
        if args.mock or args.pose_file is not None: _run_file(socket, args, stop)
        else: _run_realsense(socket, args, stop)
    finally: socket.close(0); context.term()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raise SystemExit(main())
