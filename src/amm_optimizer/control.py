"""Success-rate control of the random-search width (variant ``amplitude_control``).

In the original the width of a sampler's random search is an open-loop schedule
(a sawtooth in an iteration counter, scaled by hyperparameters), and nothing
measures whether the current width produces improvements. Here every sampler
keeps the logarithm of its width and moves it up when more of its samples beat
the incumbent than a target rate, and down otherwise, as in the 1/5 success rule
of a (1+1)-ES. The four samplers have different target rates, which keeps a
portfolio of widths.
"""

from __future__ import annotations

import numpy as np

from .config import Variant
from .state import SamplerState

# A width this small cannot move a coordinate of order one any more.
LOG_WIDTH_FLOOR = float(np.log(1e-15))
# The controller: ``log_width += SR_ETA * (success_rate - target)`` per sampler, the
# four samplers' target success rates, their start widths and the widths' cap.
SR_ETA = 1.5
SR_TARGETS = (0.05, 0.10, 0.20, 0.30)
SR_WIDTHS = (0.5, 1.0, 2.0, 4.0)
SR_MAX_WIDTH = 5.0
# The learned covariance's eigenvalues stay within this ratio of the largest.
COV_CAP = 1e6


def start_log_widths(variant: Variant, scale: float | None = None) -> list[float]:
    """The log-widths every sampler starts a search with (times ``scale``)."""
    shift = 0.0 if scale is None else float(np.log(scale))
    cap = float(np.log(SR_MAX_WIDTH))
    return [float(np.clip(np.log(w) + shift, LOG_WIDTH_FLOOR, cap)) for w in SR_WIDTHS]


def update_log_widths(
    samplers: list[SamplerState],
    probe: np.ndarray,
    boundaries: np.ndarray,
    variant: Variant,
) -> None:
    """Move each sampler's log-width (kept in ``random_growth``) toward its target.

    ``probe`` is every candidate's cost minus the incumbent's cost; a candidate
    succeeded if it is negative. The incumbent's own column counts as a failure.
    """
    cap = float(np.log(SR_MAX_WIDTH))
    for k, sampler in enumerate(samplers):
        segment = probe[boundaries[k] : boundaries[k + 1]]
        rate = np.count_nonzero(segment < 0) / len(segment)
        step = SR_ETA * (rate - SR_TARGETS[k])
        sampler.random_growth = float(
            np.clip(sampler.random_growth + step, LOG_WIDTH_FLOOR, cap)
        )


def update_covariance(
    cov: np.ndarray,
    steps: np.ndarray,
    cost: np.ndarray,
    variant: Variant,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-mu update of the covariance shape; returns it and its square root.

    ``steps`` holds the evaluated samples' displacements from the incumbent in
    units of their sampler's width (one row per sample), ``cost`` their costs.
    The best ``cov_select`` share, at least one, enters with log-rank weights.
    The result is rescaled to eigenvalues of geometric mean 1 whose ratio to the
    largest is at least ``1 / cov_cap``.
    """
    dim = cov.shape[0]
    count = max(1, int(round(variant.cov_select * len(cost))))
    best = np.argsort(cost, kind="stable")[:count]
    weights = np.log(count + 0.5) - np.log(np.arange(1, count + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / float(np.sum(weights**2))
    rate = min(
        1.0,
        variant.cov_rate
        * 2.0
        * max(mu_eff - 2.0 + 1.0 / mu_eff, 0.0)
        / ((dim + 2) ** 2 + mu_eff),
    )
    chosen = steps[best]
    new = (1 - rate) * cov + rate * (chosen.T * weights) @ chosen
    values, vectors = np.linalg.eigh((new + new.T) / 2)
    values = np.maximum(values, values.max() * 1e-300 + 1e-300)
    values = np.maximum(values, values.max() / COV_CAP)
    values /= np.exp(np.mean(np.log(values)))
    shape = (vectors * values) @ vectors.T
    factor = vectors * np.sqrt(values)
    return shape, factor
