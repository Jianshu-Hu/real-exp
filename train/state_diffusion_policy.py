from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("state_diffusion")
@dataclass
class StateDiffusionConfig(DiffusionConfig):
    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("You must provide 'observation.state' among the inputs.")
        if self.action_feature is None:
            raise ValueError("You must provide 'action' among the outputs.")
        if self.image_features:
            raise ValueError("StateDiffusionConfig does not support image inputs.")
        if self.env_state_feature is not None:
            raise ValueError("StateDiffusionConfig does not support environment-state inputs.")

    def get_optimizer_preset(self) -> AdamConfig:
        return super().get_optimizer_preset()

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return super().get_scheduler_preset()


class StateDiffusionModel(DiffusionModel):
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch[ACTION]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
        if self.config.do_mask_loss_for_padding:
            loss = loss * (~batch["action_is_pad"]).unsqueeze(-1)
        return loss.mean()


class StateDiffusionPolicy(PreTrainedPolicy):
    config_class = StateDiffusionConfig
    name = "state_diffusion"

    def __init__(self, config: StateDiffusionConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = StateDiffusionModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.diffusion.generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        loss = self.diffusion.compute_loss(batch)
        return loss, None
