from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from grasp.runtime.retargeting.mano_landmarks import MANO_LANDMARK_NAMES


SHARPA_FINGER_SEGMENT_LINKS: tuple[tuple[str, str, str], ...] = (
    ("thumb", "proximal", "right_thumb_PP"),
    ("thumb", "distal", "right_thumb_DP"),
    ("thumb", "fingertip", "right_thumb_elastomer"),
    ("index", "proximal", "right_index_PP"),
    ("index", "middle", "right_index_MP"),
    ("index", "distal", "right_index_DP"),
    ("index", "fingertip", "right_index_elastomer"),
    ("middle", "proximal", "right_middle_PP"),
    ("middle", "middle", "right_middle_MP"),
    ("middle", "distal", "right_middle_DP"),
    ("middle", "fingertip", "right_middle_elastomer"),
    ("ring", "proximal", "right_ring_PP"),
    ("ring", "middle", "right_ring_MP"),
    ("ring", "distal", "right_ring_DP"),
    ("ring", "fingertip", "right_ring_elastomer"),
    ("pinky", "proximal", "right_pinky_PP"),
    ("pinky", "middle", "right_pinky_MP"),
    ("pinky", "distal", "right_pinky_DP"),
    ("pinky", "fingertip", "right_pinky_elastomer"),
)

WUJI_FINGER_SEGMENT_LINKS: tuple[tuple[str, str, str], ...] = tuple(
    (finger, segment, f"right_finger{number}_{suffix}")
    for number, finger in enumerate(
        ("thumb", "index", "middle", "ring", "pinky"), start=1
    )
    for segment, suffix in (
        ("proximal", "link2"),
        ("middle", "link3"),
        ("distal", "link4"),
        ("fingertip", "tip_link"),
    )
)

WUJI_HAND2_BETA1_FINGER_SEGMENT_LINKS: tuple[tuple[str, str, str], ...] = tuple(
    (finger, segment, f"r_{stem}_{suffix}")
    for finger, stem in (
        ("thumb", "thumb"),
        ("index", "index_finger"),
        ("middle", "middle_finger"),
        ("ring", "ring_finger"),
        ("pinky", "pinky"),
    )
    for segment, suffix in (
        ("proximal", "proximal_abd"),
        ("middle", "middle"),
        ("distal", "distal"),
    )
)

SHADOW_FINGER_SEGMENT_LINKS: tuple[tuple[str, str, str], ...] = tuple(
    (finger, segment, f"{prefix}{suffix}")
    for finger, prefix in (
        ("thumb", "th"),
        ("index", "ff"),
        ("middle", "mf"),
        ("ring", "rf"),
        ("pinky", "lf"),
    )
    for segment, suffix in (
        ("proximal", "proximal"),
        ("middle", "middle"),
        ("distal", "distal"),
    )
)

ALLEGRO_FINGER_SEGMENT_LINKS = tuple(
    (finger, segment, link)
    for finger, links in (
        ("thumb", ("link_13.0", "link_14.0", "link_15.0", "link_15.0_tip")),
        ("index", ("link_1.0", "link_2.0", "link_3.0", "link_3.0_tip")),
        ("middle", ("link_5.0", "link_6.0", "link_7.0", "link_7.0_tip")),
        ("pinky", ("link_9.0", "link_10.0", "link_11.0", "link_11.0_tip")),
    )
    for segment, link in zip(("proximal", "middle", "distal", "fingertip"), links)
)

LEAP_FINGER_SEGMENT_LINKS = (
    ("thumb", "proximal", "thumb_pip"),
    ("thumb", "middle", "thumb_dip"),
    ("thumb", "distal", "thumb_fingertip"),
    ("index", "proximal", "pip"),
    ("index", "middle", "dip"),
    ("index", "distal", "fingertip"),
    ("middle", "proximal", "pip_2"),
    ("middle", "middle", "dip_2"),
    ("middle", "distal", "fingertip_2"),
    ("pinky", "proximal", "pip_3"),
    ("pinky", "middle", "dip_3"),
    ("pinky", "distal", "fingertip_3"),
)

FINGER_SEGMENT_LINKS_BY_HAND = {
    "right_sharpa_wave": SHARPA_FINGER_SEGMENT_LINKS,
    "right_wuji_hand": WUJI_FINGER_SEGMENT_LINKS,
    "right_wuji_hand2_beta1": WUJI_HAND2_BETA1_FINGER_SEGMENT_LINKS,
    "right_shadow_hand": SHADOW_FINGER_SEGMENT_LINKS,
    "right_allegro_hand": ALLEGRO_FINGER_SEGMENT_LINKS,
    "right_leap_hand": LEAP_FINGER_SEGMENT_LINKS,
}


@dataclass(frozen=True)
class FingerSurfacePatch:
    finger: str
    segment: str
    link_name: str
    points: np.ndarray
    normals: np.ndarray
    points_local: np.ndarray
    normals_local: np.ndarray
    inner_mask: np.ndarray
    inner_direction_local: np.ndarray
    inner_direction_world: np.ndarray


@dataclass(frozen=True)
class FingerSurfaceRegions:
    patches: tuple[FingerSurfacePatch, ...]
    joint_names: tuple[str, ...]
    reference_joints: np.ndarray
    grasp_center: np.ndarray
    inner_max_angle_deg: float


def pose_finger_surface_regions(
    robot_model,
    regions: FingerSurfaceRegions,
    *,
    trans: torch.Tensor,
    global_orient: torch.Tensor,
    joints: torch.Tensor,
) -> list[tuple[FingerSurfacePatch, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Pose calibrated link-local finger patches with differentiable FK."""

    root_rotation = robot_model._axis_angle_to_matrix(global_orient)[0]
    fk_results = robot_model.chain.forward_kinematics(joints)
    posed = []
    for patch in regions.patches:
        local_points = torch.as_tensor(
            patch.points_local, dtype=trans.dtype, device=trans.device
        )
        local_normals = torch.as_tensor(
            patch.normals_local, dtype=trans.dtype, device=trans.device
        )
        link_transform = fk_results[patch.link_name]
        link_points = link_transform.transform_points(local_points.unsqueeze(0))[0]
        link_rotation = link_transform.get_matrix()[0, :3, :3]
        link_normals = local_normals @ link_rotation.T
        world_points = link_points @ root_rotation.T + trans[0]
        world_normals = torch.nn.functional.normalize(
            link_normals @ root_rotation.T, dim=1
        )
        inner_mask = torch.as_tensor(
            patch.inner_mask, dtype=torch.bool, device=trans.device
        )
        posed.append((patch, world_points, world_normals, inner_mask))
    return posed


def build_finger_surface_regions(
    robot_model,
    *,
    samples_per_geometry: int = 2048,
    reference_flexion: float = 0.55,
    inner_max_angle_deg: float = 60.0,
) -> FingerSurfaceRegions:
    """Classify finger inner surfaces in a deterministic reference grasp pose.

    Each link's inward direction is calibrated toward the reference grasp cavity,
    then stored in link-local coordinates so the same semantic patch can later be
    transformed to arbitrary hand poses.
    """

    segment_links = FINGER_SEGMENT_LINKS_BY_HAND.get(robot_model.spec.name)
    if segment_links is None:
        raise ValueError(
            "finger surface semantics are not configured for "
            f"{robot_model.spec.name!r}"
        )
    if samples_per_geometry <= 0:
        raise ValueError("samples_per_geometry must be positive")
    if not 0.0 <= reference_flexion <= 1.0:
        raise ValueError("reference_flexion must be between 0 and 1")
    if not 0.0 < inner_max_angle_deg < 90.0:
        raise ValueError("inner_max_angle_deg must be between 0 and 90")

    joints = _reference_joints(robot_model, reference_flexion)
    device = robot_model.joint_limits.lower.device
    dtype = robot_model.joint_limits.lower.dtype
    joints_t = torch.as_tensor(joints, dtype=dtype, device=device).reshape(1, -1)
    zeros = torch.zeros((1, 3), dtype=dtype, device=device)
    landmarks = robot_model.landmarks(
        trans=zeros,
        global_orient=zeros,
        joints=joints_t,
    )[0]
    grasp_center = _reference_grasp_center(landmarks, robot_model.spec.landmark_names)
    fk_results = robot_model.chain.forward_kinematics(joints_t)
    cosine_threshold = math.cos(math.radians(inner_max_angle_deg))
    patches = []

    for finger, segment, link_name in segment_links:
        points_t, normals_t = robot_model.collision_surface_samples(
            trans=zeros,
            global_orient=zeros,
            joints=joints_t,
            samples_per_geometry=samples_per_geometry,
            link_names=(link_name,),
        )
        points_t = points_t[0]
        normals_t = normals_t[0]
        link_center = points_t.mean(dim=0)
        direction_world = torch.nn.functional.normalize(
            grasp_center - link_center,
            dim=0,
        )
        link_transform = fk_results[link_name].get_matrix()[0]
        link_rotation = link_transform[:3, :3]
        link_translation = link_transform[:3, 3]
        direction_local = direction_world @ link_rotation
        points_local = (points_t - link_translation) @ link_rotation
        normals_local = normals_t @ link_rotation
        alignment = torch.sum(normals_t * direction_world, dim=1)
        inner_mask = alignment >= cosine_threshold
        if not bool(inner_mask.any()):
            inner_mask[alignment.argmax()] = True
        patches.append(
            FingerSurfacePatch(
                finger=finger,
                segment=segment,
                link_name=link_name,
                points=points_t.detach().cpu().numpy().astype(np.float32),
                normals=normals_t.detach().cpu().numpy().astype(np.float32),
                points_local=points_local.detach().cpu().numpy().astype(np.float32),
                normals_local=normals_local.detach().cpu().numpy().astype(np.float32),
                inner_mask=inner_mask.detach().cpu().numpy().astype(bool),
                inner_direction_local=(
                    direction_local.detach().cpu().numpy().astype(np.float32)
                ),
                inner_direction_world=(
                    direction_world.detach().cpu().numpy().astype(np.float32)
                ),
            )
        )

    return FingerSurfaceRegions(
        patches=tuple(patches),
        joint_names=tuple(robot_model.joint_names),
        reference_joints=joints,
        grasp_center=grasp_center.detach().cpu().numpy().astype(np.float32),
        inner_max_angle_deg=float(inner_max_angle_deg),
    )


def build_sharpa_finger_surface_regions(
    robot_model,
    **kwargs,
) -> FingerSurfaceRegions:
    """Backward-compatible alias for the former Sharpa-only builder."""
    return build_finger_surface_regions(robot_model, **kwargs)


def _reference_joints(robot_model, flexion: float) -> np.ndarray:
    lower = robot_model.joint_limits.lower.detach().cpu().numpy()
    upper = robot_model.joint_limits.upper.detach().cpu().numpy()
    joints = lower + float(flexion) * (upper - lower)
    if robot_model.spec.name in ("right_wuji_hand", "right_wuji_hand2_beta1"):
        for index, name in enumerate(robot_model.joint_names):
            is_abduction = (
                int(name.rsplit("joint", 1)[1]) == 2
                if "joint" in name
                else name.endswith(("cmc_abd", "mcp_abd"))
            )
            if is_abduction:
                joints[index] = np.clip(0.0, lower[index], upper[index])
        return joints.astype(np.float32)
    for index, name in enumerate(robot_model.joint_names):
        if name.endswith("_AA"):
            joints[index] = np.clip(0.0, lower[index], upper[index])
        elif name == "right_pinky_CMC":
            joints[index] = lower[index] + 0.5 * (upper[index] - lower[index])
    return joints.astype(np.float32)


def _reference_grasp_center(
    landmarks: torch.Tensor, landmark_names: tuple[str, ...]
) -> torch.Tensor:
    names = tuple(landmark_names)
    index_by_name = {name: index for index, name in enumerate(names)}
    palm = landmarks[index_by_name["palm"]]
    fingertips = torch.stack(
        [
            landmarks[index_by_name[name]]
            for name in index_by_name if name.endswith("_tip")
        ]
    )
    # Bias toward the fingertips so proximal links still face into the grasp cavity.
    return 0.35 * palm + 0.65 * fingertips.mean(dim=0)
