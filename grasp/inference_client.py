#!/usr/bin/env python3
"""Capture a top-D435 frame, generate a Wuji grasp, and send it to the robot host."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from grasp.common import (
    COMMAND_FORMAT,
    WUJI_COMMAND_CONVERSION,
    WUJI_COMMAND_HAND_MODEL,
    WUJI_COMMAND_JOINT_CONVENTION,
    WUJI_COMMAND_SOURCE_MODEL,
    WUJI_RIGHT_JOINT_NAMES,
    hand_pose_to_ee_pose,
    invert_transform,
    matrix_to_xyz_rpy,
    read_json,
    read_transform,
    reorder_wuji_joints,
    transform_points,
)


DEFAULT_CAMERA_SERIAL = "401622071701"
CALIBRATED_L515_SERIAL: str | None = "f1480539"
CALIBRATED_D435I_T_L515: np.ndarray | None = np.asarray(
    [
        [-0.997594459, 0.068617076, 0.009848427, 0.023147317],
        [-0.000304589, -0.146409000, 0.989224096, -0.728169935],
        [0.069319564, 0.986841477, 0.146077708, 0.980355134],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
ASSETS_ROOT = Path(__file__).resolve().parent / "assets"
DEFAULT_GENERATOR_CHECKPOINT = ASSETS_ROOT / "checkpoints" / "generator_best.pt"
DEFAULT_MANO_ROOT = ASSETS_ROOT / "mano"
DEFAULT_WUJI_HAND2_ROOT = ASSETS_ROOT / "Wuji_hand2"
DEFAULT_MOUNT_CALIBRATION = Path(__file__).resolve().parent / "ee_to_wuji_nominal.json"

# Calibrated transforms from calibration/matrix.md. The right-arm matrix is
# recorded as C_T_B_R (right robot base -> camera), while the command path
# needs B_R_T_C; it is inverted when the transform bundle is built below. The
# L515 matrix is D435I_T_L515 and is composed with WORLD_T_D435I by dual-camera
# inference.
CALIBRATED_WORLD_T_CAMERA = np.asarray(
    [
        [0.016116505, -0.947169025, 0.320329670, -0.394891761],
        [-0.998707711, 0.000194370, 0.050821951, -0.041552817],
        [-0.048199240, -0.320734783, -0.945941876, 1.159142768],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
CALIBRATED_CAMERA_T_RIGHT_BASE = np.asarray(
    [
        [0.061077178, -0.724658647, 0.686395967, 0.162840030],
        [-0.927423222, -0.295425134, -0.229369041, 0.562456279],
        [0.368992880, -0.622570346, -0.690108991, 0.901632663],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def load_calibration_transforms(mount_path: Path) -> dict[str, np.ndarray]:
    """Build the fixed camera/world/right-base transform bundle."""
    world_t_camera = read_transform(CALIBRATED_WORLD_T_CAMERA, "world_T_camera")
    camera_t_right_base = read_transform(
        CALIBRATED_CAMERA_T_RIGHT_BASE, "camera_T_right_base"
    )
    mount_data = read_json(mount_path)
    mount_value = mount_data.get("ee_T_hand", mount_data.get("flange_T_hand"))
    if mount_value is None:
        raise ValueError(f"{mount_path} must define ee_T_hand (or flange_T_hand)")
    ee_t_hand = read_transform(mount_value, "ee_T_hand")
    base_t_camera = invert_transform(camera_t_right_base)
    base_t_world = base_t_camera @ invert_transform(world_t_camera)
    return {
        "world_T_camera": world_t_camera,
        "base_T_camera": base_t_camera,
        "base_T_world": read_transform(base_t_world, "base_T_world"),
        "ee_T_hand": ee_t_hand,
    }


def load_l515_calibration(l515_serial: str) -> np.ndarray:
    """Return the documented D435i_T_L515 transform for the installed L515."""
    if CALIBRATED_L515_SERIAL is None or CALIBRATED_D435I_T_L515 is None:
        raise ValueError(
            "L515 extrinsic calibration is not installed: add the measured "
            "D435I_T_L515 and L515 serial to calibration/matrix.md, then copy "
            "them into CALIBRATED_D435I_T_L515 and CALIBRATED_L515_SERIAL in "
            "grasp/inference_client.py"
        )
    if str(l515_serial) != CALIBRATED_L515_SERIAL:
        raise ValueError(
            f"requested L515 serial {l515_serial!r} does not match calibrated "
            f"serial {CALIBRATED_L515_SERIAL!r}"
        )
    return read_transform(CALIBRATED_D435I_T_L515, "D435I_T_L515")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator-checkpoint", type=Path, default=DEFAULT_GENERATOR_CHECKPOINT)
    parser.add_argument(
        "--mount-calibration",
        type=Path,
        default=DEFAULT_MOUNT_CALIBRATION,
        help="JSON containing ee_T_hand (default: bundled measured mount calibration).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-address", default="tcp://127.0.0.1:5570")
    parser.add_argument("--execute", action="store_true", help="Request execution on the control host")
    parser.add_argument("--side", choices=("right",), default="right")
    parser.add_argument("--camera-serial", default=DEFAULT_CAMERA_SERIAL)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    offline = parser.add_argument_group("offline RGB-D input")
    offline.add_argument("--rgb-npy", type=Path, default=None)
    offline.add_argument("--depth-npy", type=Path, default=None)
    offline.add_argument("--camera-metadata", type=Path, default=None)
    parser.add_argument("--min-depth-m", type=float, default=0.15)
    parser.add_argument("--max-depth-m", type=float, default=1.50)
    parser.add_argument("--world-min", type=float, nargs=3, default=(-0.50, -0.50, 0.005))
    parser.add_argument("--world-max", type=float, nargs=3, default=(0.50, 0.50, 0.50))
    parser.add_argument("--min-filtered-points", type=int, default=300)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--world-z-segmentation-min-m", type=float, default=0.002)
    parser.add_argument("--generator-weights", choices=("ema", "model"), default="ema")
    parser.add_argument(
        "--posterior-conditioning",
        choices=("auto", "target_film", "full_feature_only"),
        default="auto",
    )
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--retarget-landmark-fit-steps", type=int, default=75)
    parser.add_argument("--semantic-refine-steps", type=int, default=40)
    parser.add_argument("--semantic-learning-rate", type=float, default=1e-2)
    parser.add_argument("--max-penetration-m", type=float, default=0.02)
    parser.add_argument("--contact-sigma-m", type=float, default=0.01)
    parser.add_argument("--contact-d-max-m", type=float, default=0.03)
    parser.add_argument("--contact-binary-threshold-m", type=float, default=0.010)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--retarget-device",
        default="cpu",
        help="Device for MANO/Wuji retargeting (default: cpu).",
    )
    parser.add_argument(
        "--refinement-device",
        default="cpu",
        help="Device for semantic contact refinement (default: cpu).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    offline_values = (args.rgb_npy, args.depth_npy, args.camera_metadata)
    if any(value is not None for value in offline_values) and not all(
        value is not None for value in offline_values
    ):
        parser.error("offline mode requires --rgb-npy, --depth-npy, and --camera-metadata together")
    if args.min_depth_m < 0 or args.max_depth_m <= args.min_depth_m:
        parser.error("depth bounds must satisfy 0 <= min < max")
    if np.any(np.asarray(args.world_max) <= np.asarray(args.world_min)):
        parser.error("every --world-max value must exceed --world-min")
    if min(args.num_points, args.min_filtered_points) <= 0:
        parser.error("point counts must be positive")
    if args.semantic_refine_steps < 0 or args.max_penetration_m < 0:
        parser.error("refinement steps and penetration limit must be non-negative")
    for path, name in (
        (args.generator_checkpoint, "generator checkpoint"),
        (DEFAULT_MANO_ROOT / "models" / "MANO_RIGHT.pkl", "MANO right-hand model"),
        (
            DEFAULT_WUJI_HAND2_ROOT / "hand2_beta1/body/urdf/right.urdf",
            "Wuji Hand 2 Beta 1 URDF",
        ),
    ):
        if not path.is_file():
            parser.error(f"bundled {name} is missing: {path}")


def _intrinsics_dict(profile: Any) -> dict[str, Any]:
    intrinsics = profile.get_intrinsics()
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def capture_rgbd(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.rgb_npy is not None:
        rgb = np.load(args.rgb_npy)
        depth = np.load(args.depth_npy)
        metadata = json.loads(args.camera_metadata.read_text(encoding="utf-8"))
        intrinsics = metadata.get("color_intrinsics")
        if not isinstance(intrinsics, dict):
            raise ValueError("offline camera metadata must contain color_intrinsics")
        return rgb, depth, {
            "intrinsics": intrinsics,
            "depth_scale_m": float(metadata["depth_scale_m"]),
            "camera_serial": str(metadata.get("camera_serial", metadata.get("device", {}).get("serial", "unknown"))),
            "source": "offline_npy",
        }
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(f"live capture requires pyrealsense2: missing {exc.name}") from exc
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.camera_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        for _ in range(args.warmup_frames):
            pipeline.wait_for_frames()
        aligned = align.process(pipeline.wait_for_frames())
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("D435 returned an incomplete aligned RGB-D frame")
        return (
            np.asanyarray(color_frame.get_data()).copy(),
            np.asanyarray(depth_frame.get_data()).copy(),
            {
                "intrinsics": _intrinsics_dict(color_frame.profile.as_video_stream_profile()),
                "depth_scale_m": float(profile.get_device().first_depth_sensor().get_depth_scale()),
                "camera_serial": str(profile.get_device().get_info(rs.camera_info.serial_number)),
                "color_timestamp_ms": float(color_frame.get_timestamp()),
                "depth_timestamp_ms": float(depth_frame.get_timestamp()),
                "source": "live_d435",
            },
        )
    finally:
        pipeline.stop()


def backproject_depth(
    depth: np.ndarray, intrinsics: dict[str, Any], depth_scale_m: float
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth image must be 2-D, got {depth.shape}")
    height, width = depth.shape
    if int(intrinsics["width"]) != width or int(intrinsics["height"]) != height:
        raise ValueError("depth dimensions do not match aligned color intrinsics")
    rows, columns = np.indices((height, width), dtype=np.float64)
    pixels = np.stack((columns.reshape(-1), rows.reshape(-1)), axis=1)
    camera_matrix = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    coefficients = np.asarray(intrinsics.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64)
    if np.any(np.abs(coefficients) > 1e-12):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required to undistort nonzero D435 coefficients") from exc
        normalized = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2), camera_matrix, coefficients
        ).reshape(-1, 2)
    else:
        normalized = np.column_stack(
            (
                (pixels[:, 0] - float(intrinsics["cx"])) / float(intrinsics["fx"]),
                (pixels[:, 1] - float(intrinsics["cy"])) / float(intrinsics["fy"]),
            )
        )
    z = depth.reshape(-1).astype(np.float64) * float(depth_scale_m)
    points = np.column_stack((normalized[:, 0] * z, normalized[:, 1] * z, z))
    return points.astype(np.float32), z.astype(np.float32)


def filter_and_sample_points(
    camera_points: np.ndarray,
    depth_m: np.ndarray,
    world_t_camera: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_depth = (
        np.isfinite(depth_m)
        & (depth_m >= args.min_depth_m)
        & (depth_m <= args.max_depth_m)
    )
    valid_camera = camera_points[valid_depth]
    world = transform_points(world_t_camera, valid_camera)
    lower = np.asarray(args.world_min, dtype=np.float32)
    upper = np.asarray(args.world_max, dtype=np.float32)
    keep = np.all((world >= lower) & (world <= upper), axis=1)
    filtered = world[keep]
    if filtered.shape[0] < args.min_filtered_points:
        raise ValueError(
            f"only {filtered.shape[0]} points remain after filtering; "
            f"need at least {args.min_filtered_points}"
        )
    rng = np.random.default_rng(args.seed)
    if filtered.shape[0] >= args.num_points:
        indices = rng.choice(filtered.shape[0], size=args.num_points, replace=False)
        sampled = filtered[indices]
    else:
        sampled = _interpolate_local_points(filtered, args.num_points, rng)
    return valid_camera, filtered, sampled.astype(np.float32)


def _interpolate_local_points(
    points: np.ndarray, target_count: int, rng: np.random.Generator
) -> np.ndarray:
    """Resize a cloud using the same local-edge interpolation as simulation."""
    if points.shape[0] >= target_count:
        return points
    if points.shape[0] == 1:
        return np.repeat(points, target_count, axis=0)

    from scipy.spatial import cKDTree

    interpolation_count = target_count - points.shape[0]
    anchor_order = rng.permutation(points.shape[0])
    anchor_indices = np.tile(
        anchor_order,
        (interpolation_count + points.shape[0] - 1) // points.shape[0],
    )[:interpolation_count]
    neighbour_count = min(8, points.shape[0] - 1)
    _, neighbour_indices = cKDTree(points).query(
        points[anchor_indices], k=neighbour_count + 1
    )
    if neighbour_count == 1:
        selected_neighbours = neighbour_indices[:, 1]
    else:
        neighbour_columns = rng.integers(
            1, neighbour_count + 1, size=interpolation_count
        )
        selected_neighbours = neighbour_indices[
            np.arange(interpolation_count), neighbour_columns
        ]
    interpolation_weight = rng.random(interpolation_count, dtype=np.float32)[:, None]
    interpolated = (
        (1.0 - interpolation_weight) * points[anchor_indices]
        + interpolation_weight * points[selected_neighbours]
    )
    return np.concatenate((points, interpolated), axis=0)


def geometric_semantic_contacts(
    object_points: np.ndarray,
    mano_vertices: np.ndarray,
    finger_indices: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    from scipy.spatial import cKDTree

    distances = np.empty((object_points.shape[0], len(FINGER_NAMES)), dtype=np.float32)
    for channel, finger in enumerate(FINGER_NAMES):
        indices = np.asarray(finger_indices[finger], dtype=np.int64)
        distances[:, channel] = cKDTree(mano_vertices[indices]).query(object_points, workers=1)[0]
    channels = np.where(
        distances <= args.contact_d_max_m,
        np.exp(-np.square(distances / args.contact_sigma_m)),
        0.0,
    ).astype(np.float32)
    nearest = distances.argmin(axis=1)
    rows = np.arange(object_points.shape[0])
    binary = distances[rows, nearest] <= args.contact_binary_threshold_m
    return {
        "scores": channels[rows, nearest].astype(np.float32),
        "binary": binary.astype(np.float32),
        "labels": np.where(binary, nearest + 1, 0).astype(np.int64),
    }


def run_model(scene_points_world: np.ndarray, args: argparse.Namespace) -> tuple[Any, Any]:
    import torch
    from grasp.runtime.generator_runtime import GeneratorRuntime, GeneratorRuntimeConfig
    from grasp.runtime.refinement.semantic_contact_refiner import (
        SemanticContactRefinementConfig,
        SemanticContactRefiner,
    )
    from grasp.runtime.retargeting.wuji_hand2 import create_wuji_hand2_beta1_right_spec
    runtime = GeneratorRuntime(
        config=GeneratorRuntimeConfig(
            generator_checkpoint=args.generator_checkpoint,
            contact_checkpoint=None,
            generator_weights=args.generator_weights,
            posterior_conditioning=args.posterior_conditioning,
            world_z_segmentation_min_m=args.world_z_segmentation_min_m,
            mano_root=DEFAULT_MANO_ROOT,
            hand_assets_root=DEFAULT_WUJI_HAND2_ROOT,
            diffusion_steps=args.diffusion_steps,
            retarget_landmark_fit_steps=args.retarget_landmark_fit_steps,
            device=args.device,
            retarget_device=getattr(args, "retarget_device", "cpu"),
            seed=args.seed,
        ),
    )
    generated = runtime.run(scene_points_world)
    contacts = geometric_semantic_contacts(
        generated.object_points,
        generated.mano_vertices,
        runtime.mano_model.finger_vertex_indices(("proximal", "middle", "distal", "fingertip")),
        args,
    )
    robot_spec = create_wuji_hand2_beta1_right_spec(hand_root=DEFAULT_WUJI_HAND2_ROOT)
    refiner = SemanticContactRefiner(
        robot_spec=robot_spec,
        config=SemanticContactRefinementConfig(
            steps=args.semantic_refine_steps,
            learning_rate=args.semantic_learning_rate,
            device=getattr(args, "refinement_device", "cpu"),
            record_history=True,
        ),
    )
    with torch.enable_grad():
        refined = refiner.refine(
            object_points=generated.object_points,
            contact_scores=contacts["scores"],
            contact_binary=contacts["binary"],
            contact_finger_labels=contacts["labels"],
            sdf_object_points=generated.object_points,
            seed_trans=generated.robot_trans,
            seed_global_orient=generated.robot_global_orient,
            seed_joints=generated.robot_joints,
        )
    return generated, refined


def build_command(
    generated: Any,
    refined: Any,
    transforms: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    from scipy.spatial.transform import Rotation

    final_penetration = float(
        refined.metadata["final_penetration"]["max_penetration_depth_m"]
    )
    if final_penetration > args.max_penetration_m:
        raise ValueError(
            f"refined grasp penetration {final_penetration:.6f} m exceeds "
            f"limit {args.max_penetration_m:.6f} m"
        )
    world_t_hand = np.eye(4, dtype=np.float64)
    world_t_hand[:3, :3] = Rotation.from_rotvec(refined.robot_global_orient).as_matrix()
    world_t_hand[:3, 3] = refined.robot_trans
    base_t_ee = hand_pose_to_ee_pose(
        world_t_hand, transforms["base_T_world"], transforms["ee_T_hand"]
    )
    model_joints = reorder_wuji_joints(
        refined.robot_joints, generated.robot_joint_names
    )
    command_id = str(uuid.uuid4())
    return {
        "format": COMMAND_FORMAT,
        "command_id": command_id,
        "created_unix_s": time.time(),
        "side": args.side,
        "execute": bool(args.execute),
        "base_T_ee": base_t_ee.tolist(),
        "ee_pose_xyz_rpy": matrix_to_xyz_rpy(base_t_ee).tolist(),
        "hand_joints": model_joints.tolist(),
        "hand_joint_names": list(WUJI_RIGHT_JOINT_NAMES),
        "hand_model": WUJI_COMMAND_HAND_MODEL,
        "hand_joint_convention": WUJI_COMMAND_JOINT_CONVENTION,
        "hand_joint_source_model": WUJI_COMMAND_SOURCE_MODEL,
        "hand_joint_conversion": WUJI_COMMAND_CONVERSION,
        "world_T_hand": world_t_hand.tolist(),
        "base_T_world": transforms["base_T_world"].tolist(),
        "ee_T_hand": transforms["ee_T_hand"].tolist(),
        "inference": {
            "seed": args.seed,
            "generator_checkpoint": str(args.generator_checkpoint.resolve()),
            "object_extraction": "world_z_threshold",
            "retarget_fit_error": generated.retarget_fit_error,
            "refinement": refined.metadata,
        },
    }


def send_command(command: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    try:
        import zmq
    except ImportError as exc:
        raise RuntimeError(f"sending a grasp requires pyzmq: missing {exc.name}") from exc
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    timeout_ms = int(args.request_timeout_s * 1000)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, min(timeout_ms, 10000))
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.control_address)
    try:
        socket.send_json(command)
        response = socket.recv_json()
    finally:
        socket.close(0)
        context.term()
    if not isinstance(response, dict) or not response.get("ok", False):
        raise RuntimeError(f"control host rejected or failed the command: {response!r}")
    return response


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_ply_points(path: Path, points: np.ndarray) -> None:
    vertices = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with path.open("w", encoding="ascii") as handle:
        handle.write(header)
        for vertex in vertices:
            handle.write(f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")


def _write_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {vertices.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        f"element face {faces.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with path.open("w", encoding="ascii") as handle:
        handle.write(header)
        for vertex in vertices:
            handle.write(f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            handle.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def _posed_wuji_mesh(
    robot_trans: np.ndarray,
    robot_global_orient: np.ndarray,
    robot_joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from grasp.runtime.retargeting import (
        RobotHandModel,
        create_wuji_hand2_beta1_right_spec,
    )

    model = RobotHandModel(
        create_wuji_hand2_beta1_right_spec(hand_root=DEFAULT_WUJI_HAND2_ROOT)
    )
    meshes = model.collision_meshes(
        trans=robot_trans,
        global_orient=robot_global_orient,
        joints=robot_joints,
    )
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for _link_name, link_vertices, link_faces in meshes:
        vertices.append(link_vertices)
        faces.append(link_faces + offset)
        offset += link_vertices.shape[0]
    return np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0)


def write_dry_run_visualizations(
    output_dir: Path,
    filtered_world: np.ndarray,
    sampled_world: np.ndarray,
    generated: Any,
    refined: Any,
) -> None:
    _write_ply_points(output_dir / "object_points_world.ply", filtered_world)
    _write_ply_points(output_dir / "generator_input_world.ply", sampled_world)
    _write_ply_points(output_dir / "grasp_object_points_world.ply", generated.object_points)
    _write_ply_mesh(output_dir / "mano_world.ply", generated.mano_vertices, generated.mano_faces)
    for name, trans, orient, joints in (
        (
            "wuji_retargeted_world.ply",
            generated.robot_trans,
            generated.robot_global_orient,
            generated.robot_joints,
        ),
        (
            "wuji_refined_world.ply",
            refined.robot_trans,
            refined.robot_global_orient,
            refined.robot_joints,
        ),
    ):
        vertices, faces = _posed_wuji_mesh(trans, orient, joints)
        _write_ply_mesh(output_dir / name, vertices, faces)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    transforms = load_calibration_transforms(args.mount_calibration)
    rgb, depth, camera = capture_rgbd(args)
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and aligned depth image dimensions differ")
    camera_points, depth_m = backproject_depth(
        depth, camera["intrinsics"], camera["depth_scale_m"]
    )
    valid_camera, filtered_world, sampled_world = filter_and_sample_points(
        camera_points, depth_m, transforms["world_T_camera"], args
    )
    np.save(args.output_dir / "rgb_bgr.npy", rgb)
    np.save(args.output_dir / "depth_raw.npy", depth)
    np.save(args.output_dir / "depth_valid_camera_points.npy", valid_camera)
    np.save(args.output_dir / "object_points_world.npy", filtered_world)
    np.save(args.output_dir / "generator_input_world.npy", sampled_world)
    generated, refined = run_model(sampled_world, args)
    if not args.execute:
        write_dry_run_visualizations(
            args.output_dir, filtered_world, sampled_world, generated, refined
        )
    command = build_command(generated, refined, transforms, args)
    record = {
        "camera": camera,
        "filter": {
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "world_min": args.world_min,
            "world_max": args.world_max,
            "valid_depth_points": int(valid_camera.shape[0]),
            "filtered_world_points": int(filtered_world.shape[0]),
            "generator_input_points": int(sampled_world.shape[0]),
        },
        "calibration_files": {
            "camera_to_world": "hardcoded: calibration/matrix.md (W_T_C)",
            "camera_to_robot_base": "hardcoded: calibration/matrix.md (C_T_B_R; inverted to B_T_C)",
            "mount": str(args.mount_calibration.resolve()),
        },
        "command": command,
        "response": None,
    }
    result_path = args.output_dir / "result.json"
    result_path.write_text(json.dumps(record, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"Generated command {command['command_id']}; debug output saved to {args.output_dir}")
    response = send_command(command, args)
    record["response"] = response
    result_path.write_text(json.dumps(record, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"Control host completed in {response['mode']} mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
