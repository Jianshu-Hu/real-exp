from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from grasp.runtime.retargeting.mano_landmarks import MANO_LANDMARK_NAMES


FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class RobotHandSpec:
    name: str
    urdf_path: Path
    landmark_links: dict[str, str]
    finger_contact_links: dict[str, tuple[str, ...]]
    active_fingers: tuple[str, ...] | None = None
    mano_to_joint_seed: Callable[[np.ndarray, list[str]], np.ndarray] | None = None
    mano_to_global_orient_seed: Callable[[np.ndarray], np.ndarray] | None = None
    # Optional source-name remapping for robots whose semantic labels differ
    # from MANO's anatomical finger names (for example Allegro's fourth
    # finger is physically the ring finger but is exposed as ``pinky`` by the
    # runtime contact schema).
    mano_landmark_name_map: dict[str, str] | None = None

    def __post_init__(self) -> None:
        finger_names = self.finger_names
        if not finger_names or len(set(finger_names)) != len(finger_names):
            raise ValueError("active_fingers must be a non-empty tuple of unique fingers")
        unknown_fingers = sorted(set(finger_names).difference(FINGER_NAMES))
        if unknown_fingers:
            raise ValueError(f"active_fingers contains unknown fingers: {unknown_fingers}")
        required_landmarks = {"palm"}
        for finger in finger_names:
            required_landmarks.update(
                f"{finger}_{part}" for part in ("middle", "distal", "tip")
            )
        missing = sorted(required_landmarks.difference(self.landmark_links))
        extra = sorted(set(self.landmark_links).difference(MANO_LANDMARK_NAMES))
        if missing or extra:
            raise ValueError(
                "landmark_links must define palm and every active-finger landmark. "
                f"Missing: {missing}, extra: {extra}"
            )
        missing_fingers = sorted(set(finger_names).difference(self.finger_contact_links))
        extra_fingers = sorted(set(self.finger_contact_links).difference(finger_names))
        empty_fingers = sorted(
            finger for finger, links in self.finger_contact_links.items() if not links
        )
        if missing_fingers or extra_fingers or empty_fingers:
            raise ValueError(
                "finger_contact_links must define every active finger with at least one link. "
                f"Missing: {missing_fingers}, extra: {extra_fingers}, "
                f"empty: {empty_fingers}"
            )

    @property
    def finger_names(self) -> tuple[str, ...]:
        return FINGER_NAMES if self.active_fingers is None else self.active_fingers

    @property
    def landmark_names(self) -> tuple[str, ...]:
        return tuple(name for name in MANO_LANDMARK_NAMES if name in self.landmark_links)

    @property
    def contact_links(self) -> tuple[str, ...]:
        return tuple(
            link
            for finger in self.finger_names
            for link in self.finger_contact_links[finger]
        )


@dataclass(frozen=True)
class RobotJointLimits:
    joint_names: list[str]
    lower: torch.Tensor
    upper: torch.Tensor


class RobotHandModel:
    """Differentiable URDF hand wrapper for landmark retargeting."""

    def __init__(self, spec: RobotHandSpec) -> None:
        self.spec = spec
        self.urdf_path = Path(spec.urdf_path)
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"URDF path does not exist: {self.urdf_path}")

        import pytorch_kinematics as pk
        from pytorch_kinematics.transforms.rotation_conversions import (
            axis_angle_to_matrix,
        )

        self._pk = pk
        self._axis_angle_to_matrix = axis_angle_to_matrix
        self.chain = pk.build_chain_from_urdf(self.urdf_path.read_bytes())
        self.joint_names = list(self.chain.get_joint_parameter_names())
        if not self.joint_names:
            raise RuntimeError(f"No actuated joints found in URDF: {self.urdf_path}")
        self.link_names = tuple(self.chain.get_link_names())
        unknown_links = sorted(set(spec.landmark_links.values()).difference(self.link_names))
        if unknown_links:
            raise ValueError(
                f"Robot hand spec references unknown URDF links: {unknown_links}"
            )
        unknown_contact_links = sorted(set(spec.contact_links).difference(self.link_names))
        if unknown_contact_links:
            raise ValueError(
                "Robot hand spec references unknown contact links: "
                f"{unknown_contact_links}"
            )
        self.joint_limits = self._load_joint_limits()
        self._collision_sample_cache: dict[
            int,
            list[tuple[str, torch.Tensor, torch.Tensor]],
        ] = {}
        self._collision_mesh_cache: list[tuple[str, np.ndarray, np.ndarray]] | None = None
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def to(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> RobotHandModel:
        target_device = device or self._device
        target_dtype = dtype or self._dtype
        self.chain = self.chain.to(device=target_device, dtype=target_dtype)
        # pytorch_kinematics traverses these tensors from Python and repeatedly
        # calls .item() on them.  Keeping static topology on CUDA therefore
        # introduces a device synchronization for every visited frame/joint.
        # They are indices/control-flow data, not differentiable model state.
        self.chain.parents_indices = [
            indices.cpu() for indices in self.chain.parents_indices
        ]
        self.chain.joint_indices = self.chain.joint_indices.cpu()
        self.chain.joint_type_indices = self.chain.joint_type_indices.cpu()
        self.joint_limits = RobotJointLimits(
            joint_names=self.joint_limits.joint_names,
            lower=self.joint_limits.lower.to(device=target_device, dtype=target_dtype),
            upper=self.joint_limits.upper.to(device=target_device, dtype=target_dtype),
        )
        self._collision_sample_cache = {
            count: [
                (
                    link_name,
                    points.to(device=target_device, dtype=target_dtype),
                    normals.to(device=target_device, dtype=target_dtype),
                )
                for link_name, points, normals in samples
            ]
            for count, samples in self._collision_sample_cache.items()
        }
        self._device = target_device
        self._dtype = target_dtype
        return self

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    @property
    def collision_link_names(self) -> tuple[str, ...]:
        """URDF links that provide collision mesh geometry, in URDF order."""

        return tuple(
            dict.fromkeys(
                link_name
                for link_name, _, _ in self._collision_surface_samples(1)
            )
        )

    def clamp_joints(self, joints: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            joints,
            min=self.joint_limits.lower.unsqueeze(0),
            max=self.joint_limits.upper.unsqueeze(0),
        )

    def landmarks(
        self,
        *,
        trans: torch.Tensor,
        global_orient: torch.Tensor,
        joints: torch.Tensor,
    ) -> torch.Tensor:
        if trans.ndim != 2 or trans.shape[-1] != 3:
            raise ValueError(f"trans must have shape (B, 3), got {tuple(trans.shape)}")
        if global_orient.ndim != 2 or global_orient.shape[-1] != 3:
            raise ValueError(
                f"global_orient must have shape (B, 3), got {tuple(global_orient.shape)}"
            )
        if joints.ndim != 2 or joints.shape[-1] != self.num_joints:
            raise ValueError(
                f"joints must have shape (B, {self.num_joints}), got {tuple(joints.shape)}"
            )
        if trans.shape[0] != global_orient.shape[0] or trans.shape[0] != joints.shape[0]:
            raise ValueError("trans, global_orient, and joints must share batch size")

        rotation = self._axis_angle_to_matrix(global_orient)
        fk_results = self.chain.forward_kinematics(joints)
        zero = torch.zeros(
            (trans.shape[0], 1, 3),
            dtype=trans.dtype,
            device=trans.device,
        )
        points = []
        for landmark_name in self.spec.landmark_names:
            link_name = self.spec.landmark_links[landmark_name]
            link_point = fk_results[link_name].transform_points(zero)[:, 0, :]
            points.append(torch.matmul(link_point.unsqueeze(1), rotation.transpose(1, 2))[:, 0, :] + trans)
        return torch.stack(points, dim=1)

    def collision_surface_points(
        self,
        *,
        trans: torch.Tensor,
        global_orient: torch.Tensor,
        joints: torch.Tensor,
        samples_per_geometry: int,
        link_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        points, _ = self.collision_surface_samples(
            trans=trans,
            global_orient=global_orient,
            joints=joints,
            samples_per_geometry=samples_per_geometry,
            link_names=link_names,
        )
        return points

    def collision_surface_samples(
        self,
        *,
        trans: torch.Tensor,
        global_orient: torch.Tensor,
        joints: torch.Tensor,
        samples_per_geometry: int,
        link_names: tuple[str, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform fixed collision-surface points and normals into the world frame."""
        if samples_per_geometry <= 0:
            raise ValueError("samples_per_geometry must be positive")
        if trans.ndim != 2 or trans.shape[-1] != 3:
            raise ValueError(f"trans must have shape (B, 3), got {tuple(trans.shape)}")
        if global_orient.ndim != 2 or global_orient.shape[-1] != 3:
            raise ValueError(
                f"global_orient must have shape (B, 3), got {tuple(global_orient.shape)}"
            )
        if joints.ndim != 2 or joints.shape[-1] != self.num_joints:
            raise ValueError(
                f"joints must have shape (B, {self.num_joints}), got {tuple(joints.shape)}"
            )
        if (
            trans.shape[0] != global_orient.shape[0]
            or trans.shape[0] != joints.shape[0]
        ):
            raise ValueError("trans, global_orient, and joints must share batch size")

        samples = self._collision_surface_samples(samples_per_geometry)
        if link_names is not None:
            selected_links = set(link_names)
            samples = [sample for sample in samples if sample[0] in selected_links]
            sampled_links = {link_name for link_name, _, _ in samples}
            missing_links = sorted(selected_links.difference(sampled_links))
            if missing_links:
                raise RuntimeError(
                    "Contact links do not provide collision mesh geometry: "
                    f"{missing_links}"
                )
        rotation = self._axis_angle_to_matrix(global_orient)
        fk_results = self.chain.forward_kinematics(joints)
        points = []
        normals = []
        for link_name, local_points, local_normals in samples:
            batch_points = local_points.unsqueeze(0).expand(trans.shape[0], -1, -1)
            link_points = fk_results[link_name].transform_points(batch_points)
            link_rotation = fk_results[link_name].get_matrix()[:, :3, :3]
            batch_normals = local_normals.unsqueeze(0).expand(trans.shape[0], -1, -1)
            link_normals = torch.matmul(batch_normals, link_rotation.transpose(1, 2))
            points.append(
                torch.matmul(link_points, rotation.transpose(1, 2))
                + trans[:, None, :]
            )
            normals.append(torch.matmul(link_normals, rotation.transpose(1, 2)))
        return torch.cat(points, dim=1), F.normalize(torch.cat(normals, dim=1), dim=2)

    def collision_meshes(
        self,
        *,
        trans: np.ndarray,
        global_orient: np.ndarray,
        joints: np.ndarray,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Return posed collision triangle meshes in the root-pose frame."""
        trans_t = torch.as_tensor(
            trans, dtype=self._dtype, device=self._device
        ).reshape(1, 3)
        orient_t = torch.as_tensor(
            global_orient, dtype=self._dtype, device=self._device
        ).reshape(1, 3)
        joints_t = torch.as_tensor(
            joints, dtype=self._dtype, device=self._device
        ).reshape(1, self.num_joints)
        root_rotation = self._axis_angle_to_matrix(orient_t)
        fk_results = self.chain.forward_kinematics(joints_t)
        posed = []
        with torch.no_grad():
            for link_name, vertices, faces in self._collision_meshes():
                local = torch.as_tensor(
                    vertices, dtype=self._dtype, device=self._device
                ).unsqueeze(0)
                link_points = fk_results[link_name].transform_points(local)
                world_points = (
                    torch.matmul(link_points, root_rotation.transpose(1, 2))
                    + trans_t[:, None, :]
                )
                posed.append(
                    (
                        link_name,
                        world_points[0].cpu().numpy().astype(np.float32),
                        faces.copy(),
                    )
                )
        return posed

    def _collision_surface_samples(
        self,
        samples_per_geometry: int,
    ) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        cached = self._collision_sample_cache.get(samples_per_geometry)
        if cached is not None:
            return cached

        samples: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for geometry_index, (link_name, vertices, faces) in enumerate(
            self._collision_meshes()
        ):
            local_points, local_normals = _sample_mesh_surface(
                vertices,
                faces,
                count=samples_per_geometry,
                seed=geometry_index,
            )
            samples.append(
                (
                    link_name,
                    torch.as_tensor(
                        local_points,
                        dtype=self._dtype,
                        device=self._device,
                    ),
                    torch.as_tensor(
                        local_normals,
                        dtype=self._dtype,
                        device=self._device,
                    ),
                )
            )
        if not samples:
            raise RuntimeError(
                f"No collision mesh geometry found in URDF: {self.urdf_path}"
            )
        self._collision_sample_cache[samples_per_geometry] = samples
        return samples

    def _collision_meshes(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        if self._collision_mesh_cache is not None:
            return self._collision_mesh_cache
        import trimesh

        meshes = []
        root = ET.parse(self.urdf_path).getroot()
        for link_element in root.findall("link"):
            link_name = str(link_element.attrib["name"])
            for collision_element in link_element.findall("collision"):
                geometry_element = collision_element.find("geometry")
                if geometry_element is None:
                    continue
                mesh_element = geometry_element.find("mesh")
                if mesh_element is not None:
                    filename = mesh_element.attrib.get("filename")
                    if not filename:
                        continue
                    mesh = trimesh.load_mesh(
                        _resolve_urdf_mesh_path(self.urdf_path, filename),
                        process=False,
                    )
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                    vertices = np.asarray(mesh.vertices, dtype=np.float32)
                    scale_text = mesh_element.attrib.get("scale")
                    if scale_text:
                        vertices = vertices * _parse_vector(
                            scale_text, size=3
                        ).reshape(1, 3)
                else:
                    mesh = _primitive_collision_mesh(geometry_element, trimesh)
                    if mesh is None:
                        continue
                    vertices = np.asarray(mesh.vertices, dtype=np.float32)
                vertices = _transform_points(
                    _origin_matrix(collision_element.find("origin")), vertices
                )
                meshes.append(
                    (
                        link_name,
                        vertices,
                        np.asarray(mesh.faces, dtype=np.int64),
                    )
                )
        if not meshes:
            raise RuntimeError(
                f"No collision mesh geometry found in URDF: {self.urdf_path}"
            )
        self._collision_mesh_cache = meshes
        return meshes
    def _load_joint_limits(self) -> RobotJointLimits:
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        limit_by_name: dict[str, tuple[float, float]] = {}
        for joint_element in root.findall("joint"):
            joint_name = joint_element.attrib["name"]
            limit_element = joint_element.find("limit")
            if limit_element is None:
                continue
            lower_text = limit_element.attrib.get("lower")
            upper_text = limit_element.attrib.get("upper")
            if lower_text is None or upper_text is None:
                continue
            limit_by_name[joint_name] = (float(lower_text), float(upper_text))
        missing = [name for name in self.joint_names if name not in limit_by_name]
        if missing:
            raise RuntimeError(f"Missing joint limits for actuated joints: {missing}")
        lower = torch.tensor(
            [limit_by_name[name][0] for name in self.joint_names],
            dtype=torch.float32,
        )
        upper = torch.tensor(
            [limit_by_name[name][1] for name in self.joint_names],
            dtype=torch.float32,
        )
        return RobotJointLimits(joint_names=list(self.joint_names), lower=lower, upper=upper)


def _primitive_collision_mesh(geometry_element: ET.Element, trimesh):
    box = geometry_element.find("box")
    if box is not None:
        return trimesh.creation.box(
            extents=_parse_vector(box.attrib["size"], size=3)
        )
    sphere = geometry_element.find("sphere")
    if sphere is not None:
        return trimesh.creation.icosphere(
            subdivisions=2,
            radius=float(sphere.attrib["radius"]),
        )
    cylinder = geometry_element.find("cylinder")
    if cylinder is not None:
        return trimesh.creation.cylinder(
            radius=float(cylinder.attrib["radius"]),
            height=float(cylinder.attrib["length"]),
            sections=32,
        )
    capsule = geometry_element.find("capsule")
    if capsule is not None:
        return trimesh.creation.capsule(
            radius=float(capsule.attrib["radius"]),
            height=float(capsule.attrib["length"]),
            count=[16, 16],
        )
    return None


def _sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            "collision mesh vertices must have shape (N, 3), "
            f"got {vertices.shape}"
        )
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(
            f"collision mesh faces must have shape (F, 3), got {faces.shape}"
        )
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid = np.isfinite(areas) & (areas > 0.0)
    triangles = triangles[valid]
    areas = areas[valid]
    if triangles.shape[0] == 0:
        raise ValueError("collision mesh contains no non-degenerate triangles")
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(triangles.shape[0], size=count, p=areas / areas.sum())
    selected = triangles[face_indices]
    selected_normals = cross[valid][face_indices]
    selected_normals /= np.linalg.norm(selected_normals, axis=1, keepdims=True)
    uv = rng.random((count, 2), dtype=np.float32)
    flip = uv.sum(axis=1) > 1.0
    uv[flip] = 1.0 - uv[flip]
    points = (
        selected[:, 0]
        + uv[:, :1] * (selected[:, 1] - selected[:, 0])
        + uv[:, 1:] * (selected[:, 2] - selected[:, 0])
    ).astype(np.float32)
    outward = points - vertices.mean(axis=0, keepdims=True)
    flip_normals = np.sum(selected_normals * outward, axis=1) < 0.0
    selected_normals[flip_normals] *= -1.0
    return points, selected_normals.astype(np.float32)


def _resolve_urdf_mesh_path(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        relative = filename.removeprefix("package://")
        parts = Path(relative).parts
        if len(parts) >= 2 and parts[0] == urdf_path.parent.name:
            relative = str(Path(*parts[1:]))
        return (urdf_path.parent / relative).resolve()
    path = Path(filename)
    return path if path.is_absolute() else (urdf_path.parent / path).resolve()


def _parse_vector(text: str, *, size: int) -> np.ndarray:
    values = np.asarray([float(value) for value in text.split()], dtype=np.float32)
    if values.shape != (size,):
        raise ValueError(f"expected {size} values, got {text!r}")
    return values


def _origin_matrix(origin_element: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    if origin_element is None:
        return matrix
    xyz = _parse_vector(origin_element.attrib.get("xyz", "0 0 0"), size=3)
    roll, pitch, yaw = _parse_vector(origin_element.attrib.get("rpy", "0 0 0"), size=3)
    cx, cy, cz = np.cos([roll, pitch, yaw])
    sx, sy, sz = np.sin([roll, pitch, yaw])
    matrix[:3, :3] = np.asarray(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ],
        dtype=np.float32,
    )
    matrix[:3, 3] = xyz
    return matrix


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _transform_vectors(transform: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    transformed = vectors @ transform[:3, :3].T
    return transformed / np.clip(np.linalg.norm(transformed, axis=1, keepdims=True), 1e-8, None)


@dataclass(frozen=True)
class RobotHandLandmarkFitConfig:
    device: str = "auto"
    steps: int = 300
    learning_rate: float = 2e-2
    landmark_weight: float = 100.0
    direction_weight: float = 0.5
    joint_limit_weight: float = 1.0
    joint_reg_weight: float = 5e-3
    trans_reg_weight: float = 1e-4
    orient_reg_weight: float = 1e-4
    record_history: bool = False


@dataclass(frozen=True)
class RobotHandLandmarkFitResult:
    trans: np.ndarray
    global_orient: np.ndarray
    joints: np.ndarray
    loss: np.ndarray
    metadata: list[dict[str, Any]]


class RobotHandLandmarkFitter:
    def __init__(
        self,
        *,
        hand_model: RobotHandModel,
        config: RobotHandLandmarkFitConfig | None = None,
    ) -> None:
        self.config = RobotHandLandmarkFitConfig() if config is None else config
        if self.config.steps < 0:
            raise ValueError("steps must be non-negative")
        self.device = _resolve_device(self.config.device)
        self.hand_model = hand_model.to(device=self.device, dtype=torch.float32)

    def fit_batch(
        self,
        *,
        target_landmarks: np.ndarray,
        init_trans: np.ndarray,
        init_global_orient: np.ndarray,
        init_joints: np.ndarray | None = None,
    ) -> RobotHandLandmarkFitResult:
        target_landmarks = np.asarray(target_landmarks, dtype=np.float32)
        landmark_count = len(self.hand_model.spec.landmark_names)
        if target_landmarks.ndim != 3 or target_landmarks.shape[1:] != (landmark_count, 3):
            raise ValueError(
                "target_landmarks must have shape "
                f"(B, {landmark_count}, 3), got {target_landmarks.shape}"
            )
        batch_size = int(target_landmarks.shape[0])
        target_t = torch.as_tensor(target_landmarks, dtype=torch.float32, device=self.device)
        trans0 = torch.as_tensor(
            np.asarray(init_trans, dtype=np.float32).reshape(batch_size, 3),
            dtype=torch.float32,
            device=self.device,
        )
        orient0 = torch.as_tensor(
            np.asarray(init_global_orient, dtype=np.float32).reshape(batch_size, 3),
            dtype=torch.float32,
            device=self.device,
        )
        if init_joints is None:
            raise ValueError("init_joints is required; initialize robot joints from MANO pose")
        joints0 = torch.as_tensor(
            np.asarray(init_joints, dtype=np.float32).reshape(
                batch_size,
                self.hand_model.num_joints,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        joints0 = self.hand_model.clamp_joints(joints0)

        trans = torch.nn.Parameter(trans0.clone())
        global_orient = torch.nn.Parameter(orient0.clone())
        joints = torch.nn.Parameter(joints0.clone())
        optimizer = torch.optim.Adam(
            [trans, global_orient, joints],
            lr=float(self.config.learning_rate),
        )
        landmark_weights = torch.as_tensor(
            _landmark_weights(self.hand_model.spec.landmark_names),
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1, 1)
        history: list[dict[str, float]] = []
        for step in range(int(self.config.steps)):
            optimizer.zero_grad(set_to_none=True)
            pred = self.hand_model.landmarks(
                trans=trans,
                global_orient=global_orient,
                joints=joints,
            )
            diff = pred - target_t
            landmark_loss = torch.mean(landmark_weights * diff.square())
            direction_loss = _finger_direction_loss(
                pred, target_t, self.hand_model.spec.landmark_names
            )
            joint_limit_loss = _joint_limit_loss(
                joints,
                self.hand_model.joint_limits.lower,
                self.hand_model.joint_limits.upper,
            )
            joint_reg = torch.mean((joints - joints0).square())
            trans_reg = torch.mean((trans - trans0).square())
            orient_reg = torch.mean((global_orient - orient0).square())
            total = (
                float(self.config.landmark_weight) * landmark_loss
                + float(self.config.direction_weight) * direction_loss
                + float(self.config.joint_limit_weight) * joint_limit_loss
                + float(self.config.joint_reg_weight) * joint_reg
                + float(self.config.trans_reg_weight) * trans_reg
                + float(self.config.orient_reg_weight) * orient_reg
            )
            total.backward()
            optimizer.step()
            with torch.no_grad():
                joints.copy_(self.hand_model.clamp_joints(joints))
            if self.config.record_history and (
                step == 0 or step == self.config.steps - 1 or (step + 1) % 50 == 0
            ):
                history.append(
                    {
                        "step": float(step + 1),
                        "total": float(total.detach().cpu()),
                        "weighted_landmark_rmse_m": float(
                            torch.sqrt(landmark_loss).detach().cpu()
                        ),
                        "joint_limit_loss": float(joint_limit_loss.detach().cpu()),
                    }
                )

        pred = self.hand_model.landmarks(
            trans=trans,
            global_orient=global_orient,
            joints=joints,
        )
        diff = pred - target_t
        rmse = torch.sqrt(torch.mean(diff.square(), dim=(1, 2)))
        weighted_rmse = torch.sqrt(torch.mean(landmark_weights * diff.square(), dim=(1, 2)))
        per_sample_loss = float(self.config.landmark_weight) * torch.mean(
            landmark_weights * diff.square(),
            dim=(1, 2),
        )
        metadata = [
            {
                "target_landmark_names": list(self.hand_model.spec.landmark_names),
                "landmark_rmse_m": float(rmse[index].detach().cpu()),
                "weighted_landmark_rmse_m": float(weighted_rmse[index].detach().cpu()),
                "history": history,
            }
            for index in range(batch_size)
        ]
        return RobotHandLandmarkFitResult(
            trans=trans.detach().cpu().numpy().astype(np.float32),
            global_orient=global_orient.detach().cpu().numpy().astype(np.float32),
            joints=joints.detach().cpu().numpy().astype(np.float32),
            loss=per_sample_loss.detach().cpu().numpy().astype(np.float32),
            metadata=metadata,
        )


def _joint_limit_loss(
    joints: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    lower_violation = F.relu(lower.unsqueeze(0) - joints)
    upper_violation = F.relu(joints - upper.unsqueeze(0))
    return torch.mean(lower_violation.square() + upper_violation.square())


def _landmark_weights(landmark_names: tuple[str, ...]) -> np.ndarray:
    """Prioritize the wrist and finger bases over absolute fingertip position."""
    weights = np.ones(len(landmark_names), dtype=np.float32)
    for index, name in enumerate(landmark_names):
        if name == "palm":
            weights[index] = 0.05
        elif name.endswith("_distal"):
            weights[index] = 1.5
        elif name.endswith("_tip"):
            weights[index] = 10
    return (weights / weights.mean()).astype(np.float32)


def _finger_direction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    landmark_names: tuple[str, ...],
) -> torch.Tensor:
    """Match finger directions while allowing their lengths to differ."""
    index_by_name = {name: index for index, name in enumerate(landmark_names)}
    finger_indices = [
        (index_by_name[f"{finger}_middle"], index_by_name[f"{finger}_distal"], index_by_name[f"{finger}_tip"])
        for finger in FINGER_NAMES
        if all(f"{finger}_{part}" in index_by_name for part in ("middle", "distal", "tip"))
    ]
    losses = []
    for middle, _, tip in finger_indices:
        pred_vec = F.normalize(predicted[:, tip] - predicted[:, middle], dim=-1)
        target_vec = F.normalize(target[:, tip] - target[:, middle], dim=-1)
        losses.append((pred_vec - target_vec).square().mean())
    return torch.stack(losses).mean()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
