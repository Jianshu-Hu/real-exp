from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grasp.runtime.normalization import GraspDataNormalizer, GraspTargetNormalizer
from grasp.runtime.models import (
    ContactMapHead,
    DDPMSchedule,
    GraspGeneratorModel,
    grasp_generator_ablation_group,
    posterior_points_source_from_checkpoint,
)
from grasp.runtime.retargeting import ManoToRobotRetargetConfig, ManoToRobotRetargeter
from grasp.runtime.retargeting.mano_model import (
    ManoModel,
    ManoModelConfig,
    flat_hand_mean_from_contract,
)
from grasp.runtime.retargeting.wuji import create_wuji_hand_right_spec, wuji_hand_right_global_orient_from_mano


@dataclass(frozen=True)
class GeneratorRuntimeConfig:
    generator_checkpoint: Path
    contact_checkpoint: Path | None
    generator_weights: str = "ema"
    posterior_conditioning: str = "auto"
    world_z_segmentation_min_m: float = 0.002
    mano_root: Path = Path("data/HandsAndObejects/mano")
    mano_side: str = "right"
    robodex_root: Path = Path("grasp/assets/RoboDex")
    diffusion_steps: int = 100
    retarget_landmark_fit_steps: int = 50
    device: str = "auto"
    retarget_device: str = "cpu"
    seed: int = 0


@dataclass(frozen=True)
class GeneratorRuntimeOutput:
    scene_points: np.ndarray
    object_points: np.ndarray
    contact_points: np.ndarray | None
    contact_scores: np.ndarray | None
    contact_binary: np.ndarray | None
    contact_finger_labels: np.ndarray | None
    object_center: np.ndarray
    condition_feature: np.ndarray
    prior_mu: np.ndarray
    prior_logvar: np.ndarray
    normalized_mano_grasp: np.ndarray
    wrist_translation: np.ndarray
    mano_orient: np.ndarray
    mano_hand_pose: np.ndarray
    mano_translation: np.ndarray
    mano_vertices: np.ndarray
    mano_faces: np.ndarray
    robot_trans: np.ndarray
    robot_global_orient: np.ndarray
    robot_joints: np.ndarray
    robot_joint_names: list[str]
    retarget_fit_error: dict[str, float]
    metadata: dict[str, Any]

    @property
    def object_bbox_center_world(self) -> np.ndarray:
        return self.object_center


class GeneratorRuntime:
    """Online prior inference following the packed-dataset validation contracts."""

    def __init__(self, *, config: GeneratorRuntimeConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        torch.manual_seed(int(config.seed))

        generator_checkpoint = _load_checkpoint(config.generator_checkpoint, self.device)
        _require_world_contract(generator_checkpoint, "generator")

        self.generator_normalizer = _data_normalizer_from_checkpoint(generator_checkpoint)
        self.target_normalizer = _target_normalizer_from_checkpoint(generator_checkpoint)
        self.mano_flat_hand_mean = flat_hand_mean_from_contract(generator_checkpoint)
        self.generator_source = _partial_points_source(generator_checkpoint)
        self.generator_translation = _point_translation_normalization(generator_checkpoint)
        self.contact_source: str | None = None
        self.contact_translation: str | None = None
        contact_backbone_path: Path | None = None

        generator_conditioning = _resolve_posterior_conditioning(
            generator_checkpoint,
            requested=config.posterior_conditioning,
        )
        generator_ablation_group = grasp_generator_ablation_group(generator_checkpoint)
        generator_posterior_points = posterior_points_source_from_checkpoint(
            generator_checkpoint
        )
        self.generator = GraspGeneratorModel(
            posterior_conditioning=generator_conditioning,
            posterior_points_source=generator_posterior_points,
        ).to(self.device)
        _load_model_weights(
            self.generator,
            generator_checkpoint,
            weights=config.generator_weights,
        )
        self.generator.eval()

        self.contact_head = None
        if config.contact_checkpoint is not None:
            contact_checkpoint = _load_checkpoint(config.contact_checkpoint, self.device)
            contact_backbone_path = _resolve_backbone_checkpoint_path(contact_checkpoint)
            _require_world_contract(contact_checkpoint, "contact")
            _validate_shared_backbone_contract(
                generator_checkpoint_path=config.generator_checkpoint,
                generator_checkpoint=generator_checkpoint,
                generator_weights=config.generator_weights,
                contact_checkpoint=contact_checkpoint,
                contact_backbone_path=contact_backbone_path,
            )
            self.contact_source = _checkpoint_contract(
                contact_checkpoint, "partial_points_source"
            )
            self.contact_translation = _checkpoint_contract(
                contact_checkpoint, "point_translation_normalization"
            )
            _validate_contact_points_contract(
                generator_source=self.generator_source,
                contact_source=self.contact_source,
            )
            self.contact_head = _make_contact_head(contact_checkpoint).to(self.device)
            self.contact_head.load_state_dict(contact_checkpoint["contact_head_state_dict"])
            self.contact_head.eval()

        self.diffusion_schedule = DDPMSchedule.create(
            num_steps=int(config.diffusion_steps), device=self.device
        )
        self.mano_model = ManoModel(
            ManoModelConfig(
                mano_root=Path(config.mano_root),
                mano_side=str(config.mano_side),
                flat_hand_mean=self.mano_flat_hand_mean,
            )
        )
        robot_spec = create_wuji_hand_right_spec(robodex_root=config.robodex_root)
        self.retargeter = ManoToRobotRetargeter(
            robot_spec=robot_spec,
            config=ManoToRobotRetargetConfig(
                mano_root=Path(config.mano_root),
                mano_side=str(config.mano_side),
                device=str(config.retarget_device),
                landmark_fit_steps=int(config.retarget_landmark_fit_steps),
                flat_hand_mean=self.mano_flat_hand_mean,
            ),
        )
        self.metadata = {
            "generator_checkpoint": Path(config.generator_checkpoint).as_posix(),
            "generator_weights": config.generator_weights,
            "generator_partial_points_source": self.generator_source,
            "generator_point_translation_normalization": self.generator_translation,
            "contact_checkpoint": (
                None
                if config.contact_checkpoint is None
                else Path(config.contact_checkpoint).as_posix()
            ),
            "contact_backbone_checkpoint": (
                None if contact_backbone_path is None else contact_backbone_path.as_posix()
            ),
            "contact_backbone_weights": (
                None if config.contact_checkpoint is None else config.generator_weights
            ),
            "contact_query_points_source": self.contact_source,
            "contact_point_translation_normalization": self.contact_translation,
            "shared_backbone_forward": config.contact_checkpoint is not None,
            "object_extraction": "world_z_threshold",
            "world_z_segmentation_min_m": float(
                config.world_z_segmentation_min_m
            ),
            "generator_normalization": self.generator_normalizer.config(),
            "target_normalization": self.target_normalizer.config(),
            "mano_flat_hand_mean": self.mano_flat_hand_mean,
            "generator_posterior_conditioning": generator_conditioning,
            "generator_ablation_group": generator_ablation_group,
            "generator_posterior_points_source": generator_posterior_points,
            "diffusion_steps": int(config.diffusion_steps),
            "sample_latent": False,
            "seed": int(config.seed),
            "hand_type": "wuji_hand_right",
            "device": str(self.device),
            "retarget_device": str(config.retarget_device),
            "frame": "robodex_world_z_up",
        }

    @property
    def robot_model(self):
        """Differentiable robot geometry shared by retargeting and virtual RL checks."""
        return self.retargeter._robot_model

    @torch.no_grad()
    def run(self, scene_points_world: np.ndarray) -> GeneratorRuntimeOutput:
        scene_raw = _validate_points(scene_points_world, "scene_points_world")
        object_raw = _select_points_above_world_z(
            scene_raw,
            min_world_z_m=float(self.config.world_z_segmentation_min_m),
        )
        object_center = _bbox_center(object_raw)

        generator_points = _prepare_backbone_points(
            scene_points=scene_raw,
            object_points=object_raw,
            normalizer=self.generator_normalizer,
            source=self.generator_source,
            translation=self.generator_translation,
            object_center=object_center,
        )
        generator_points_t = torch.as_tensor(
            generator_points, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        prediction = self.generator.sample(
            generator_points_t,
            diffusion_schedule=self.diffusion_schedule,
            sample_latent=False,
        )
        wrist, mano_orient, mano_hand_pose = _denormalize_prediction(
            prediction, target_normalizer=self.target_normalizer
        )
        if self.generator_translation == "object_partial_bbox":
            wrist = wrist + object_center

        contact_raw = None
        contact_scores = None
        contact_binary = None
        contact_finger_labels = None
        if self.contact_head is not None:
            assert self.contact_source is not None
            assert self.contact_translation is not None
            if self.contact_source == "segmentation_cache":
                # Offline cached points are online segmentation outputs at runtime.
                contact_raw = object_raw
            else:
                contact_raw = _select_runtime_points(
                    scene_points=scene_raw,
                    object_points=object_raw,
                    source=self.contact_source,
                    role="contact query",
                )
            contact_query_points = self.generator_normalizer.normalize_points(contact_raw)
            if self.contact_translation == "object_partial_bbox":
                contact_query_points = (
                    contact_query_points
                    - self.generator_normalizer.normalize_points(object_center)[None, :]
                )
            contact_query_t = torch.as_tensor(
                contact_query_points, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            contact_output = self.contact_head(
                points=contact_query_t,
                condition_feature=prediction.condition_feature,
                prior_mu=prediction.prior_mu,
            )
            contact_scores = (
                contact_output.scores[0].detach().cpu().numpy().astype(np.float32)
            )
            contact_binary = (
                contact_output.binary[0].detach().cpu().numpy().astype(np.float32)
            )
            contact_finger_labels = (
                contact_output.finger_labels[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
            )

        init_orient = wuji_hand_right_global_orient_from_mano(mano_orient)
        mano_transl = _mano_translation_for_wrist(
            mano_model=self.mano_model,
            wrist_translation=wrist,
            global_orient=mano_orient,
            hand_pose=mano_hand_pose,
        )
        mano_output = self.mano_model.forward(
            global_orient=mano_orient,
            hand_pose=mano_hand_pose,
            betas=np.zeros(10, dtype=np.float32),
            transl=mano_transl,
        )
        with torch.enable_grad():
            robot = self.retargeter.retarget_sample(
                mano_global_orient=mano_orient,
                mano_hand_pose=mano_hand_pose,
                mano_transl=mano_transl,
                init_robot_global_orient=init_orient,
            )
        return GeneratorRuntimeOutput(
            scene_points=scene_raw,
            object_points=object_raw,
            contact_points=contact_raw,
            contact_scores=contact_scores,
            contact_binary=contact_binary,
            contact_finger_labels=contact_finger_labels,
            object_center=object_center,
            condition_feature=prediction.condition_feature[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            prior_mu=prediction.prior_mu[0].detach().cpu().numpy().astype(np.float32),
            prior_logvar=prediction.prior_logvar[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            normalized_mano_grasp=prediction.grasp_target[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            wrist_translation=wrist.astype(np.float32),
            mano_orient=mano_orient,
            mano_hand_pose=mano_hand_pose,
            mano_translation=mano_transl,
            mano_vertices=mano_output["vertices"][0],
            mano_faces=self.mano_model.faces,
            robot_trans=robot.robot_trans,
            robot_global_orient=robot.robot_global_orient,
            robot_joints=robot.robot_joints,
            robot_joint_names=robot.robot_joint_names,
            retarget_fit_error=dict(robot.fit_error),
            metadata={
                **self.metadata,
                "world_z_input_point_count": int(scene_raw.shape[0]),
                "world_z_selected_point_count": int(object_raw.shape[0]),
                "object_output_unique_point_count": int(
                    np.unique(object_raw, axis=0).shape[0]
                ),
            },
        )


def _select_points_above_world_z(
    points_world: np.ndarray,
    *,
    min_world_z_m: float,
) -> np.ndarray:
    """Select world-frame points above the tabletop with a noise margin."""

    if min_world_z_m < 0.0:
        raise ValueError("min_world_z_m must be non-negative")
    selected = points_world[points_world[:, 2] > min_world_z_m]
    if selected.shape[0] == 0:
        raise ValueError(
            "world-z segmentation found no scene points with "
            f"z > {min_world_z_m:g} m"
        )
    return selected.astype(np.float32, copy=False)


def _prepare_backbone_points(
    *,
    scene_points: np.ndarray,
    object_points: np.ndarray,
    normalizer: GraspDataNormalizer,
    source: str,
    translation: str,
    object_center: np.ndarray,
) -> np.ndarray:
    raw = _select_runtime_points(
        scene_points=scene_points,
        object_points=object_points,
        source=source,
        role="generator backbone",
    )
    points = normalizer.normalize_points(raw)
    if translation == "object_partial_bbox":
        points = points - normalizer.normalize_points(object_center)[None, :]
    elif translation != "none":
        raise ValueError(f"unsupported point translation normalization: {translation}")
    return points.astype(np.float32)


def _select_runtime_points(
    *,
    scene_points: np.ndarray,
    object_points: np.ndarray,
    source: str,
    role: str,
) -> np.ndarray:
    if source == "scene":
        return scene_points
    if source in {"object", "segmentation", "segmentation_cache"}:
        return object_points
    raise ValueError(f"unsupported {role} points source: {source}")


def _validate_contact_points_contract(*, generator_source: str, contact_source: str) -> None:
    if contact_source == "scene":
        raise ValueError(
            "contact-centric RL requires object query points; scene contact points "
            "mix background geometry into the contact-error contract"
        )
    if contact_source == "segmentation_cache" and generator_source not in {
        "object",
        "segmentation",
        "segmentation_cache",
    }:
        raise ValueError(
            "a segmentation_cache contact head requires an object-point generator "
            f"backbone, got partial_points_source={generator_source}"
        )
    if contact_source not in {"object", "segmentation", "segmentation_cache"}:
        raise ValueError(f"unsupported contact query points source: {contact_source}")


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains non-finite values")
    return points


def _bbox_center(points: np.ndarray) -> np.ndarray:
    points = _validate_points(points, "object_points")
    return ((points.min(axis=0) + points.max(axis=0)) * np.float32(0.5)).astype(np.float32)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _load_model_weights(
    model: GraspGeneratorModel, checkpoint: dict[str, Any], *, weights: str
) -> None:
    if weights == "ema" and "ema_state_dict" in checkpoint:
        state = dict(checkpoint["model_state_dict"])
        state.update(dict(checkpoint["ema_state_dict"]["shadow"]))
        model.load_state_dict(state)
        return
    if weights not in {"model", "ema"}:
        raise ValueError("weights must be 'ema' or 'model'")
    model.load_state_dict(checkpoint["model_state_dict"])


def _resolve_backbone_checkpoint_path(contact_checkpoint: dict[str, Any]) -> Path:
    stored = contact_checkpoint.get("backbone_checkpoint")
    if not stored and isinstance(contact_checkpoint.get("args"), dict):
        stored = contact_checkpoint["args"].get("backbone_checkpoint")
    if not stored:
        raise ValueError("contact checkpoint does not store backbone_checkpoint")
    path = Path(str(stored))
    if not path.is_file():
        raise FileNotFoundError(f"contact backbone checkpoint does not exist: {path}")
    return path


def _validate_shared_backbone_contract(
    *,
    generator_checkpoint_path: Path,
    generator_checkpoint: dict[str, Any],
    generator_weights: str,
    contact_checkpoint: dict[str, Any],
    contact_backbone_path: Path,
) -> None:
    generator_path = Path(generator_checkpoint_path).expanduser().resolve()
    contact_path = Path(contact_backbone_path).expanduser().resolve()
    if generator_path != contact_path:
        raise ValueError(
            "contact head must be trained with the current generator checkpoint: "
            f"{contact_path} != {generator_path}"
        )
    args = contact_checkpoint.get("args", {})
    contact_weights = args.get("weights") if isinstance(args, dict) else None
    if contact_weights not in {"ema", "model"}:
        raise KeyError("contact checkpoint does not record backbone weights")
    if str(contact_weights) != str(generator_weights):
        raise ValueError(
            "contact head backbone weights must match --generator-weights: "
            f"{contact_weights} != {generator_weights}"
        )
    generator_translation = _point_translation_normalization(generator_checkpoint)
    contact_translation = _checkpoint_contract(
        contact_checkpoint, "point_translation_normalization"
    )
    if contact_translation != generator_translation:
        raise ValueError(
            "contact head point translation must match the generator backbone: "
            f"{contact_translation} != {generator_translation}"
        )


def _resolve_posterior_conditioning(
    checkpoint: dict[str, Any],
    *,
    requested: str,
    contract_checkpoint: dict[str, Any] | None = None,
) -> str:
    if requested != "auto":
        if requested not in {"target_film", "full_feature_only"}:
            raise ValueError("invalid posterior_conditioning")
        return requested
    for candidate in (contract_checkpoint, checkpoint):
        if not candidate:
            continue
        value = candidate.get("posterior_conditioning")
        if value in {"target_film", "full_feature_only"}:
            return str(value)
        args = candidate.get("args", {})
        if isinstance(args, dict) and args.get("posterior_conditioning") in {
            "target_film", "full_feature_only"
        }:
            return str(args["posterior_conditioning"])
    return "target_film"


def _make_contact_head(checkpoint: dict[str, Any]) -> ContactMapHead:
    config = dict(checkpoint.get("contact_head_config", {}))
    allowed = {
        "point_dim",
        "condition_dim",
        "latent_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
        "num_fingers",
        "binary_threshold",
    }
    return ContactMapHead(**{key: value for key, value in config.items() if key in allowed})


def _denormalize_prediction(
    prediction: Any, *, target_normalizer: GraspTargetNormalizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrist = prediction.wrist_translation[0].detach().cpu().numpy().astype(np.float32)
    orient = prediction.mano_orient[0].detach().cpu().numpy().astype(np.float32)
    hand_pose = prediction.hand_pose[0].detach().cpu().numpy().astype(np.float32)
    return (
        target_normalizer.denormalize_wrist_translation(wrist),
        target_normalizer.denormalize_mano_orient(orient),
        target_normalizer.denormalize_hand_pose(hand_pose),
    )


def _target_normalizer_from_checkpoint(checkpoint: dict[str, Any]) -> GraspTargetNormalizer:
    config = checkpoint.get("target_normalization")
    if isinstance(config, dict):
        return GraspTargetNormalizer.from_config(config)
    return GraspTargetNormalizer.from_data_normalizer(_data_normalizer_from_checkpoint(checkpoint))


def _data_normalizer_from_checkpoint(checkpoint: dict[str, Any]) -> GraspDataNormalizer:
    config = checkpoint.get("normalization")
    if not isinstance(config, dict):
        return GraspDataNormalizer()
    allowed = {"point_scale_m", "wrist_translation_scale_m", "mano_pose_scale_rad", "hand_joint_scale_rad"}
    return GraspDataNormalizer(**{key: value for key, value in config.items() if key in allowed})


def _checkpoint_contract(checkpoint: dict[str, Any], key: str) -> str:
    value = checkpoint.get(key)
    args = checkpoint.get("args", {})
    if value is None and isinstance(args, dict):
        value = args.get(key)
    if value is None:
        raise KeyError(f"checkpoint does not store {key}")
    return str(value)


def _partial_points_source(checkpoint: dict[str, Any]) -> str:
    return _checkpoint_contract(checkpoint, "partial_points_source")


def _point_translation_normalization(checkpoint: dict[str, Any]) -> str:
    return _checkpoint_contract(checkpoint, "point_translation_normalization")


def _require_world_contract(checkpoint: dict[str, Any], role: str) -> None:
    contract = checkpoint.get("coordinate_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{role} checkpoint has no coordinate_contract")
    frame = contract.get("physical_frame")
    if frame != "robodex_world_z_up":
        raise ValueError(f"{role} checkpoint is not world-frame compatible: {frame!r}")
    view = checkpoint.get("pointcloud_view")
    if view != "fused":
        raise ValueError(f"{role} checkpoint is not fused-view compatible: {view!r}")


def _mano_translation_for_wrist(
    *,
    mano_model: ManoModel,
    wrist_translation: np.ndarray,
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
) -> np.ndarray:
    zero_output = mano_model.forward(
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=np.zeros(10, dtype=np.float32),
        transl=np.zeros(3, dtype=np.float32),
    )
    wrist_at_zero = np.asarray(zero_output["joints"][0, 0], dtype=np.float32).reshape(3)
    return (np.asarray(wrist_translation, dtype=np.float32).reshape(3) - wrist_at_zero).astype(
        np.float32
    )


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
