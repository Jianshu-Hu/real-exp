"""Lazy checkpoint adapter for upstream rl_games policies plus a test policy."""

from __future__ import annotations

import sys
from pathlib import Path

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
        sys.path.insert(0, str(upstream_root.resolve()))
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
        self.player = RlPlayer(
            num_observations=self.observation_dim,
            num_actions=self.action_dim,
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            device=device,
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

