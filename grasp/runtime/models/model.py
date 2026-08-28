from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .decoder import GraspDiffusionDecoder
from .diffusion import DDPMSchedule
from .encoder import GraspEncoder
from .latent import PosteriorEncoder, PriorEncoder


ABLATION_GROUPS = (1, 2, 3)
POSTERIOR_POINTS_SOURCES = ("partial", "full")


def grasp_generator_ablation_group(checkpoint: dict) -> int:
    """Resolve the module-ablation group, including legacy checkpoints."""
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        args = {}
    stored = checkpoint.get("ablation_group", args.get("ablation_group"))
    if stored is not None:
        group = int(stored)
        if group not in ABLATION_GROUPS:
            raise ValueError(f"invalid grasp-generator ablation group: {group}")
        return group
    source = checkpoint.get(
        "posterior_points_source",
        args.get("posterior_points_source"),
    )
    if source == "partial" or args.get("use_full_point_cloud") is False:
        return 1
    alignment_weight = float(args.get("latent_alignment_weight", 0.0))
    logvar_alignment_weight = float(
        args.get("latent_logvar_alignment_weight", 0.0)
    )
    return 3 if alignment_weight > 0.0 or logvar_alignment_weight > 0.0 else 2


def posterior_points_source_from_checkpoint(checkpoint: dict) -> str:
    """Return the posterior point-cloud branch recorded by a checkpoint."""
    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        args = {}
    source = checkpoint.get(
        "posterior_points_source",
        args.get("posterior_points_source"),
    )
    if source is None:
        source = "partial" if grasp_generator_ablation_group(checkpoint) == 1 else "full"
    if source not in POSTERIOR_POINTS_SOURCES:
        raise ValueError(f"invalid posterior_points_source: {source!r}")
    return str(source)


@dataclass(frozen=True)
class GraspGeneratorOutput:
    latent: torch.Tensor
    condition_feature: torch.Tensor
    partial_sa2_points: torch.Tensor
    full_feature: torch.Tensor
    posterior_mu: torch.Tensor
    posterior_logvar: torch.Tensor
    prior_mu: torch.Tensor
    prior_logvar: torch.Tensor
    grasp_target: torch.Tensor
    grasp_pose: torch.Tensor
    hand_pose: torch.Tensor
    wrist_translation: torch.Tensor
    mano_orient: torch.Tensor
    noise_prediction: torch.Tensor
    grasp_pose_noise_prediction: torch.Tensor
    hand_pose_noise_prediction: torch.Tensor
    prior_noise_prediction: torch.Tensor
    prior_grasp_pose_noise_prediction: torch.Tensor
    prior_hand_pose_noise_prediction: torch.Tensor


class GraspGeneratorModel(nn.Module):
    def __init__(
        self,
        *,
        point_dim: int = 3,
        feature_dim: int = 384,
        latent_dim: int = 128,
        grasp_pose_dim: int = 6,
        hand_pose_dim: int = 45,
        posterior_conditioning: str = "target_film",
        posterior_points_source: str = "full",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        if posterior_conditioning not in ("target_film", "full_feature_only"):
            raise ValueError(
                "posterior_conditioning must be 'target_film' or 'full_feature_only'"
            )
        if posterior_points_source not in POSTERIOR_POINTS_SOURCES:
            raise ValueError(
                "posterior_points_source must be 'partial' or 'full'"
            )
        self.encoder = GraspEncoder(
            point_dim=point_dim,
            latent_input_dim=feature_dim,
        )
        grasp_target_dim = grasp_pose_dim + hand_pose_dim
        self.posterior_encoder = PosteriorEncoder(
            feature_dim=feature_dim,
            grasp_target_dim=grasp_target_dim,
            latent_dim=latent_dim,
            conditioning=posterior_conditioning,
        )
        self.prior_encoder = PriorEncoder(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
        )
        self.decoder = GraspDiffusionDecoder(
            condition_dim=feature_dim,
            latent_dim=latent_dim,
            grasp_pose_dim=grasp_pose_dim,
            hand_pose_dim=hand_pose_dim,
        )
        self.latent_dim = latent_dim
        self.posterior_conditioning = posterior_conditioning
        self.posterior_points_source = posterior_points_source
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for module in (self.encoder, self.decoder):
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        partial_points: torch.Tensor,
        full_points: torch.Tensor,
        grasp_target: torch.Tensor,
        timesteps: torch.Tensor,
        noisy_grasp_target: torch.Tensor | None = None,
        sample_latent: bool = True,
    ) -> GraspGeneratorOutput:
        partial_encoded = self.encoder(partial_points)
        posterior_encoded = (
            partial_encoded
            if self.posterior_points_source == "partial"
            else self.encoder(full_points)
        )
        prior = self.prior_encoder(
            partial_feature=partial_encoded.latent_input,
            sample_latent=sample_latent,
        )
        posterior = self.posterior_encoder(
            full_feature=posterior_encoded.latent_input,
            grasp_target=self._posterior_grasp_target(grasp_target),
            sample_latent=sample_latent,
        )
        posterior_decoded = self.decoder(
            partial_encoded.latent_input,
            posterior.latent,
            timesteps,
            noisy_grasp_target=noisy_grasp_target,
        )
        prior_decoded = self.decoder(
            partial_encoded.latent_input,
            prior.latent,
            timesteps,
            noisy_grasp_target=noisy_grasp_target,
        )
        target_parts = self._split_grasp_target(grasp_target)
        return GraspGeneratorOutput(
            latent=posterior.latent,
            condition_feature=partial_encoded.latent_input,
            partial_sa2_points=partial_encoded.sa2_points,
            full_feature=posterior_encoded.latent_input,
            posterior_mu=posterior.mu,
            posterior_logvar=posterior.logvar,
            prior_mu=prior.mu,
            prior_logvar=prior.logvar,
            grasp_target=grasp_target,
            grasp_pose=target_parts["grasp_pose"],
            hand_pose=target_parts["hand_pose"],
            wrist_translation=target_parts["wrist_translation"],
            mano_orient=target_parts["mano_orient"],
            noise_prediction=posterior_decoded.noise_prediction,
            grasp_pose_noise_prediction=posterior_decoded.grasp_pose_noise_prediction,
            hand_pose_noise_prediction=posterior_decoded.hand_pose_noise_prediction,
            prior_noise_prediction=prior_decoded.noise_prediction,
            prior_grasp_pose_noise_prediction=prior_decoded.grasp_pose_noise_prediction,
            prior_hand_pose_noise_prediction=prior_decoded.hand_pose_noise_prediction,
        )

    @torch.no_grad()
    def sample(
        self,
        partial_points: torch.Tensor,
        *,
        diffusion_schedule: DDPMSchedule | None = None,
        sample_latent: bool = True,
    ) -> GraspGeneratorOutput:
        partial_encoded = self.encoder(partial_points)
        if diffusion_schedule is None:
            diffusion_schedule = DDPMSchedule.create(device=partial_points.device)
        prior = self.prior_encoder(
            partial_feature=partial_encoded.latent_input,
            sample_latent=sample_latent,
        )
        latent = prior.latent
        grasp, final_decoded = self._sample_grasp_from_latent(
            condition_feature=partial_encoded.latent_input,
            latent=latent,
            diffusion_schedule=diffusion_schedule,
        )
        target_parts = self._split_grasp_target(grasp)
        empty = torch.empty(
            0,
            dtype=partial_points.dtype,
            device=partial_points.device,
        )
        return GraspGeneratorOutput(
            latent=latent,
            condition_feature=partial_encoded.latent_input,
            partial_sa2_points=partial_encoded.sa2_points,
            full_feature=empty,
            posterior_mu=empty,
            posterior_logvar=empty,
            prior_mu=prior.mu,
            prior_logvar=prior.logvar,
            grasp_target=grasp,
            grasp_pose=target_parts["grasp_pose"],
            hand_pose=target_parts["hand_pose"],
            wrist_translation=target_parts["wrist_translation"],
            mano_orient=target_parts["mano_orient"],
            noise_prediction=final_decoded.noise_prediction,
            grasp_pose_noise_prediction=final_decoded.grasp_pose_noise_prediction,
            hand_pose_noise_prediction=final_decoded.hand_pose_noise_prediction,
            prior_noise_prediction=final_decoded.noise_prediction,
            prior_grasp_pose_noise_prediction=final_decoded.grasp_pose_noise_prediction,
            prior_hand_pose_noise_prediction=final_decoded.hand_pose_noise_prediction,
        )

    @torch.no_grad()
    def sample_posterior(
        self,
        partial_points: torch.Tensor,
        full_points: torch.Tensor,
        grasp_target: torch.Tensor,
        *,
        diffusion_schedule: DDPMSchedule | None = None,
        sample_latent: bool = False,
    ) -> GraspGeneratorOutput:
        partial_encoded = self.encoder(partial_points)
        posterior_encoded = (
            partial_encoded
            if self.posterior_points_source == "partial"
            else self.encoder(full_points)
        )
        if diffusion_schedule is None:
            diffusion_schedule = DDPMSchedule.create(device=partial_points.device)
        prior = self.prior_encoder(
            partial_feature=partial_encoded.latent_input,
            sample_latent=sample_latent,
        )
        posterior = self.posterior_encoder(
            full_feature=posterior_encoded.latent_input,
            grasp_target=self._posterior_grasp_target(grasp_target),
            sample_latent=sample_latent,
        )
        grasp, final_decoded = self._sample_grasp_from_latent(
            condition_feature=partial_encoded.latent_input,
            latent=posterior.latent,
            diffusion_schedule=diffusion_schedule,
        )
        target_parts = self._split_grasp_target(grasp)
        return GraspGeneratorOutput(
            latent=posterior.latent,
            condition_feature=partial_encoded.latent_input,
            partial_sa2_points=partial_encoded.sa2_points,
            full_feature=posterior_encoded.latent_input,
            posterior_mu=posterior.mu,
            posterior_logvar=posterior.logvar,
            prior_mu=prior.mu,
            prior_logvar=prior.logvar,
            grasp_target=grasp,
            grasp_pose=target_parts["grasp_pose"],
            hand_pose=target_parts["hand_pose"],
            wrist_translation=target_parts["wrist_translation"],
            mano_orient=target_parts["mano_orient"],
            noise_prediction=final_decoded.noise_prediction,
            grasp_pose_noise_prediction=final_decoded.grasp_pose_noise_prediction,
            hand_pose_noise_prediction=final_decoded.hand_pose_noise_prediction,
            prior_noise_prediction=final_decoded.noise_prediction,
            prior_grasp_pose_noise_prediction=final_decoded.grasp_pose_noise_prediction,
            prior_hand_pose_noise_prediction=final_decoded.hand_pose_noise_prediction,
        )

    def _split_grasp_target(self, grasp_target: torch.Tensor) -> dict[str, torch.Tensor]:
        grasp_pose, hand_pose = torch.split(
            grasp_target,
            [self.decoder.grasp_pose_dim, self.decoder.hand_pose_dim],
            dim=-1,
        )
        wrist_translation, mano_orient = torch.split(grasp_pose, [3, 3], dim=-1)
        return {
            "grasp_pose": grasp_pose,
            "hand_pose": hand_pose,
            "wrist_translation": wrist_translation,
            "mano_orient": mano_orient,
        }

    def _posterior_grasp_target(self, grasp_target: torch.Tensor) -> torch.Tensor | None:
        if self.posterior_conditioning == "target_film":
            return grasp_target
        return None

    def _sample_grasp_from_latent(
        self,
        *,
        condition_feature: torch.Tensor,
        latent: torch.Tensor,
        diffusion_schedule: DDPMSchedule,
    ) -> tuple[torch.Tensor, object]:
        grasp = torch.randn(
            condition_feature.shape[0],
            self.decoder.grasp_target_dim,
            dtype=condition_feature.dtype,
            device=condition_feature.device,
        )
        decoded = None
        for step in reversed(range(diffusion_schedule.num_steps)):
            timesteps = torch.zeros(
                (condition_feature.shape[0],),
                dtype=torch.long,
                device=condition_feature.device,
            )
            timesteps.fill_(step)
            decoded = self.decoder(
                condition_feature,
                latent,
                timesteps,
                noisy_grasp_target=grasp,
            )
            grasp = diffusion_schedule.denoise_step(
                noisy=grasp,
                noise_prediction=decoded.noise_prediction,
                timesteps=timesteps,
            )
        if decoded is None:
            raise RuntimeError("diffusion schedule has no denoising steps")
        final_decoded = self.decoder(
            condition_feature,
            latent,
            torch.zeros(
                (condition_feature.shape[0],),
                dtype=torch.long,
                device=condition_feature.device,
            ),
            noisy_grasp_target=grasp,
        )
        return grasp, final_decoded
