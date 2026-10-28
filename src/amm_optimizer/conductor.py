"""Resource conductor: a mass-and-spring system that moves samples between samplers.

Each of the ``n`` sub-algorithms is a mass held near its resting allocation by a
spring. A normalised "change vector" (how much each sub-algorithm improved the
cost) pushes resources toward the winners through a resource-conserving
coupling matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConductorConfig:
    """How the samples of an iteration are shared between the samplers.

    Each sampler is a mass held near an equal share by a spring; the samplers that
    improved the cost most pull samples toward themselves, and the total stays
    fixed. The defaults are the original's.

    Attributes
    ----------
    total_resources : int, default 60
        Samples per iteration, shared between the samplers. ``pool_rule="budget"``
        sets it from the budget.
    n_algorithms : int, default 4
        The number of samplers.
    mass : float, default 10.0
        The inertia of each sampler's share.
    self_spring : float, default 10.0
        The pull of each share back toward its equal share.
    interaction_spring : float, default 40.0
        How strongly an improvement moves samples between samplers.
    min_samples : int, default 1
        Every sampler keeps at least this many samples.
    """

    total_resources: int = 60
    n_algorithms: int = 4
    mass: float = 10.0
    self_spring: float = 10.0
    interaction_spring: float = 40.0
    # Every sampler keeps at least this many samples (it needs one to exist).
    min_samples: int = 1


class ResourceConductor:
    def __init__(self, config: ConductorConfig | None = None) -> None:
        config = config or ConductorConfig()
        n = config.n_algorithms
        rest = int(config.total_resources / n) * np.ones(n)
        rest[0] = rest[0] + (config.total_resources - np.sum(rest))  # rounding
        self.rest: np.ndarray = rest
        self.mass = config.mass * np.ones(n)
        self.self_spring = config.self_spring * np.ones(n)
        self.coupling = config.interaction_spring * (
            -np.ones((n, n)) + n * np.diag(np.ones(n))
        )
        self._min_samples = config.min_samples
        self._total = int(np.sum(rest))

    def step(self, current: np.ndarray, change_vector: np.ndarray) -> np.ndarray:
        """Return the next integer allocation (same total as the resting one)."""
        norm = np.sqrt(np.matmul(change_vector, change_vector))
        if norm != 0:
            change_vector = change_vector / norm

        restoring = -(self.self_spring / self.mass) * (current - self.rest)
        coupling = np.matmul(self.coupling, (change_vector / self.mass).T)
        proposed = np.round(current + restoring + coupling.T)

        # Conserve the total by giving the rounding error to the largest holder.
        largest = np.argmax(proposed)
        proposed[largest] = proposed[largest] + (np.sum(self.rest) - np.sum(proposed))

        if not np.isfinite(proposed).all():
            # No usable signal (NaN or infinite push): leave the allocation alone.
            # The original reset it to 16 per sampler, which broke conservation.
            return np.asarray(current).astype(np.int64)

        allocation: np.ndarray = proposed.astype(np.int64)
        if allocation.min() < self._min_samples:
            allocation = self._enforce_minimum(allocation)
        return allocation

    def _enforce_minimum(self, allocation: np.ndarray) -> np.ndarray:
        """Raise starved samplers to the minimum, paid for by the largest holders."""
        allocation = np.asarray(np.maximum(allocation, self._min_samples))
        for _ in range(int(allocation.sum()) - self._total):
            allocation[np.argmax(allocation)] -= 1
        return allocation
