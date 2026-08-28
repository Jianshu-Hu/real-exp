"""Lazy checkpoint adapter for upstream rl_games policies plus a test policy."""

from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import yaml

import numpy as np


class MockPolicy:
    """Deterministic dependency-free policy used for transport/safety tests."""

    def __init__(self, action_dim: int = 27, value: float = 0.0) -> None:
        self.action_dim = int(action_dim)
        self.value = float(value)

    def act(self, observation: np.ndarray) -> np.ndarray:
        if np.asarray(observation).ndim != 1:
            raise ValueError("mock policy expects one flat observation")
        return np.full(self.action_dim, self.value, dtype=np.float64)

    def reset(self) -> None:
        return None


class RlGamesPolicy:
    """Wrap upstream ``deployment.RlPlayer`` without importing Isaac Lab."""

    def __init__(
        self,
        upstream_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        observation_dim: int,
        action_dim: int,
        device: str,
    ) -> None:
        for path, label in ((upstream_root, "upstream repository"), (config_path, "config"), (checkpoint_path, "checkpoint")):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        self._validate_checkpoint(checkpoint_path, observation_dim, action_dim)
        sys.path.insert(0, str(upstream_root.resolve()))
        # Newer Isaac Sim exports store the rl_games agent under ``agent``;
        # deployment/rl_player.py still reads the legacy ``train`` key. Keep
        # the user's bundle untouched and materialize a compatibility YAML.
        config_for_player = config_path
        temporary_config: Path | None = None
        raw_config = yaml.safe_load(config_path.read_text())
        if isinstance(raw_config, dict) and "train" not in raw_config and "agent" in raw_config:
            compatibility = dict(raw_config)
            compatibility["train"] = compatibility["agent"]
            compatibility.pop("agent", None)
            temp = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="simtoolreal-", delete=False)
            yaml.safe_dump(compatibility, temp, sort_keys=False)
            temp.close()
            config_for_player = Path(temp.name)
            temporary_config = config_for_player
        try:
            import torch
            from deployment.rl_player import RlPlayer
        except ImportError as exc:
            raise RuntimeError(
                "policy inference requires torch, gym, omegaconf, rl_games, and the upstream SimToolReal dependencies"
            ) from exc
        self.torch = torch
        self.device = device
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        try:
            self.player = RlPlayer(
                num_observations=self.observation_dim,
                num_actions=self.action_dim,
                config_path=str(config_for_player),
                checkpoint_path=str(checkpoint_path),
                device=device,
            )
        finally:
            if temporary_config is not None:
                temporary_config.unlink(missing_ok=True)

    @staticmethod
    def _validate_checkpoint(path: Path, observation_dim: int, action_dim: int) -> None:
        """Fail early with a useful message when a checkpoint is incompatible."""
        try:
            import torch
            state = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"could not read rl_games checkpoint {path}: {exc}") from exc
        if isinstance(state, dict) and 0 in state:
            state = state[0]
        model = state.get("model", state) if isinstance(state, dict) else state
        if not isinstance(model, dict):
            raise ValueError(f"checkpoint {path} does not contain an rl_games model state")
        obs_shape = model.get("running_mean_std.running_mean")
        action_bias = model.get("a2c_network.mu.bias")
        if obs_shape is None or action_bias is None:
            raise ValueError(f"checkpoint {path} is missing running_mean_std or a2c_network.mu tensors")
        checkpoint_obs = int(obs_shape.shape[0])
        checkpoint_actions = int(action_bias.shape[0])
        if (checkpoint_obs, checkpoint_actions) != (int(observation_dim), int(action_dim)):
            raise ValueError(
                f"checkpoint dimensions are ({checkpoint_obs}, {checkpoint_actions}); "
                f"SimToolReal expects ({observation_dim}, {action_dim})"
            )

    def act(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (self.observation_dim,):
            raise ValueError(f"checkpoint expects observation shape ({self.observation_dim},), got {values.shape}")
        tensor = self.torch.as_tensor(values, device=self.device).unsqueeze(0)
        with self.torch.no_grad():
            action = self.player.get_normalized_action(tensor, deterministic_actions=True)
        result = action.detach().cpu().numpy().reshape(-1)
        if result.shape != (self.action_dim,) or not np.all(np.isfinite(result)):
            raise ValueError(f"checkpoint returned invalid action {result.shape}")
        return result

    def reset(self) -> None:
        self.player.reset()
