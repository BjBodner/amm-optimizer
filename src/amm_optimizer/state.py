"""Mutable state of a search and of its four samplers."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

from . import legacy_stream
from .sampler import ModeParams


class SamplerState:
    """Hyperparameters, sample count and growth factors of one sampler."""

    def __init__(self, n_params: int, rng: np.random.RandomState) -> None:
        self.adaptive_vector = legacy_stream.sampler_init_draws(n_params, rng)
        self.n_samples = 10
        self.random_growth: float = 1.0
        self.targeted_growth: float = 1.0
        self.params = ModeParams.from_vector(np.zeros(legacy_stream.N_ROWS))

    def set_hyperparameters(self, vector: np.ndarray) -> None:
        self.params = ModeParams.from_vector(vector)


@dataclass
class SearchState:
    """What all four samplers share: the incumbent and its history."""

    best: np.ndarray
    second_best: np.ndarray
    previous_cost: float
    change_vector: np.ndarray
    current_cost: float
    cost_history: list[float] = field(default_factory=list)
    best_eighth: np.ndarray | None = None
    # Covariance shape and its square-root factor (variant ``covariance``).
    cov: np.ndarray | None = None
    cov_factor: np.ndarray | None = None
    force_stop: bool = False
    restart: bool = False

    @property
    def multi_parent(self) -> bool:
        """Whether the next draw crosses over the best eighth (else second best)."""
        # Legacy: the multi-parent path only starts once two costs exist.
        return len(self.cost_history) > 1

    @property
    def parents(self) -> np.ndarray:
        """Crossover partners: the second best at first, then the best eighth."""
        if self.multi_parent:
            assert self.best_eighth is not None
            return self.best_eighth
        return self.second_best


# Legacy runner-up trick: inflate the winner's cost and take argmin again.
_LEGACY_INFLATION = 10000


def runner_up(cost: np.ndarray, winner: int, mode: str) -> int:
    """Index of the second best candidate.

    ``"legacy"`` multiplies the winner's cost by 10000, which only works for
    costs that are positive and within a factor of 10000 of each other; for a
    negative or zero best cost, or a wide spread, it returns the winner itself
    (audit B21). ``"exclude_winner"`` picks the best of the others.
    """
    cost = cost.copy()
    if mode == "legacy":
        cost[winner] = _LEGACY_INFLATION * cost[winner]
    else:
        cost[winner] = np.inf
    return int(np.argmin(cost))


def _median_of_sorted(values: list[float], count: int) -> float:
    """``np.median(values[:count])`` for sorted ``values``, bit for bit."""
    half = count >> 1
    if count & 1:
        return values[half]
    return (values[half - 1] + values[half]) / 2


def _best_eighth_by_medians(cost: np.ndarray) -> np.ndarray:
    first = cost[cost <= np.median(cost)]
    second = cost[cost <= np.median(first)]
    return np.flatnonzero(cost <= np.median(second))


def best_eighth(cost: np.ndarray) -> np.ndarray:
    """Indices of costs at or below the median of the median of the median.

    The three medians are taken on prefixes of one sorted copy (the elements at
    or below a median always form a prefix), which gives the same numbers as
    three ``np.median`` calls at a fraction of the cost. The sorted values are
    plain Python floats: for scalars the arithmetic is the same IEEE double
    arithmetic, and bisection and indexing are cheaper than NumPy calls on 60
    numbers.
    """
    ordered: list[float] = np.sort(cost).tolist()
    if not ordered or ordered[-1] != ordered[-1]:
        # NaN or empty input: keep NumPy's semantics (and its warnings).
        return _best_eighth_by_medians(cost)
    threshold = _median_of_sorted(ordered, len(ordered))
    for _ in range(2):
        threshold = _median_of_sorted(ordered, bisect_right(ordered, threshold))
    return np.asarray((cost <= threshold).nonzero()[0])
