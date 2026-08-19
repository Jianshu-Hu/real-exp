"""Temporal aggregation for overlapping absolute-timestep action chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TemporalProposalAggregator:
    """Keep raw proposals and recompute normalized generation-age weighting.

    ``decay`` is the multiplicative weight retained by a proposal for every newer
    chunk generation.  The newest proposal has weight 1, the preceding proposal
    ``decay``, then ``decay**2``, and so on.  Because raw proposals are retained,
    a previously aggregated value is never fed back as a new proposal.
    """

    decay: float
    proposals: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    next_generation: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"decay must be between 0 and 1, got {self.decay}")

    def begin_chunk(self) -> int:
        generation = self.next_generation
        self.next_generation += 1
        return generation

    def add(self, timestep: int, generation: int, action: np.ndarray) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32)
        timestep_proposals = self.proposals.setdefault(int(timestep), {})
        timestep_proposals[int(generation)] = action_array.copy()
        generations = sorted(timestep_proposals)
        newest = generations[-1]
        weights = np.asarray(
            [self.decay ** (newest - item) for item in generations], dtype=np.float64
        )
        weights /= weights.sum()
        return np.sum(
            [weights[index] * timestep_proposals[item] for index, item in enumerate(generations)],
            axis=0,
        ).astype(np.float32)

    def proposal_count(self, timestep: int) -> int:
        return len(self.proposals.get(int(timestep), {}))

    def diagnostics(self, timestep: int) -> dict[str, float | int | list[float]]:
        """Describe the normalized weights currently used for one action timestep."""
        generations = sorted(self.proposals.get(int(timestep), {}))
        if not generations:
            return {"proposal_count": 0, "max_generation_age": 0, "normalized_weights": []}

        newest = generations[-1]
        weights = np.asarray(
            [self.decay ** (newest - generation) for generation in generations],
            dtype=np.float64,
        )
        weights /= weights.sum()
        return {
            "proposal_count": len(generations),
            "max_generation_age": newest - generations[0],
            "normalized_weights": weights.tolist(),
        }

    def discard(self, timestep: int) -> None:
        self.proposals.pop(int(timestep), None)
