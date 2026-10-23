"""Online hyperparameter search.

The eight sampler hyperparameters are tuned by running the same two-mode
sampler one level up. Each candidate hyperparameter vector is scored by how much
the corresponding sampler improved the cost during the last section.
"""

from __future__ import annotations

import numpy as np

from . import hyperparams, legacy_stream
from .config import N_SAMPLERS
from .sampler import ModeParams, two_mode_samples

# Offset used to push the winner out of the argmin when finding the runner-up.
_SECOND_BEST_OFFSET = 10**10
# Denominator constants of the finite-difference surrogate gradient.
_GRADIENT_FLOOR = -0.0001
_GRADIENT_OFFSET = 0.001


class HyperparameterSearch:
    def __init__(self, initial: np.ndarray) -> None:
        """``initial`` has shape ``(N_ROWS, N_SAMPLERS)``, already clipped."""
        self._bounds = hyperparams.bounds()
        self._params = ModeParams.from_vector(hyperparams.meta_hyperparameters())
        self._best = initial[:, 0]
        self._second_best = initial[:, 0]
        self._candidates = initial
        self._change = np.zeros(legacy_stream.N_ROWS)
        self._random_growth: float = 0
        self._targeted_growth: float = 0
        self._adaptive = np.ones((legacy_stream.N_ROWS, N_SAMPLERS))
        self._squared_gradient: float | np.ndarray = 0

    def propose(self, rng: np.random.RandomState) -> np.ndarray:
        """Next hyperparameters, one column per sampler, clipped to the bounds."""
        candidates, self._random_growth, self._targeted_growth = two_mode_samples(
            self._best,
            self._second_best,
            N_SAMPLERS,
            self._change,
            self._random_growth,
            self._targeted_growth,
            self._params,
            self._adaptive,
            rng,
        )
        self._candidates = hyperparams.clip_to_bounds(candidates, self._bounds)
        return self._candidates

    def update(self, gains: np.ndarray) -> None:
        """Learn from the section just run with the proposed candidates.

        ``gains`` has shape ``(N_SAMPLERS, iterations)``.
        """
        candidates = self._candidates
        # Legacy (audit B12): mean/min here, mean/max for the (dead) first score.
        score = (np.mean(gains, 1) + np.min(gains, 1)) / 2
        winner = int(np.argmin(score))
        self._change = candidates[:, winner] - self._best

        probe = score - score[0]
        steps = candidates - np.array([self._best]).T
        gradient = probe / (
            (np.maximum(steps, _GRADIENT_FLOOR) + np.minimum(steps, _GRADIENT_FLOOR))
            + _GRADIENT_OFFSET
        )
        alpha, beta = self._params.alpha, self._params.beta
        s = (
            beta * self._squared_gradient
            + (1 - beta) * (np.mean(np.abs(gradient), 1)) ** 2
        )
        s = s * np.sign(np.mean(gradient, 1))
        self._squared_gradient = s
        amplitude = alpha / np.sqrt(np.fmax(s + alpha**2, 0.1 * alpha**2))
        self._adaptive[:, :] = np.array([amplitude]).T

        self._best = candidates[:, winner]
        score[winner] = _SECOND_BEST_OFFSET + score[winner]
        self._second_best = candidates[:, int(np.argmin(score))]
