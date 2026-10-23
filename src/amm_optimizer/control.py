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


# ``width_mode="ladder"``: sampler k searches at sigma times LADDER[k]; sigma starts
# at LADDER_START, so that the start widths are the original 0.5, 1, 2, 4.
LADDER = (0.25, 0.5, 1.0, 2.0)
LADDER_START = 2.0


def _rungs(variant: Variant) -> tuple[float, ...]:
    return LADDER if variant.width_mode == "ladder" else (1.0,) * len(LADDER)


def start_log_widths(variant: Variant, scale: float | None = None) -> list[float]:
    """The log-widths every sampler starts a search with (times ``scale``)."""
    shift = 0.0 if scale is None else float(np.log(scale))
    cap = float(np.log(variant.sr_max_width))
    if variant.width_mode != "per_sampler":
        start = (
            LADDER_START
            if variant.width_mode == "ladder"
            else float(np.sqrt(min(variant.sr_widths) * max(variant.sr_widths)))
        )
        log_sigma = _clip_shared(np.log(start) + shift, variant)
        return [log_sigma + float(np.log(r)) for r in _rungs(variant)]
    return [
        float(np.clip(np.log(w) + shift, LOG_WIDTH_FLOOR, cap))
        for w in variant.sr_widths
    ]


def _clip_shared(log_sigma: float, variant: Variant) -> float:
    """Keep every rung of a shared width within the floor and the cap."""
    rungs = np.log(_rungs(variant))
    cap = float(np.log(variant.sr_max_width))
    return float(np.clip(log_sigma, LOG_WIDTH_FLOOR - rungs.min(), cap - rungs.max()))


def update_log_widths(
    samplers: list[SamplerState],
    probe: np.ndarray,
    boundaries: np.ndarray,
    variant: Variant,
) -> None:
    """Move each sampler's log-width (kept in ``random_growth``) toward its target.

    ``probe`` is every candidate's cost minus the incumbent's cost; a candidate
    succeeded if it is negative. The incumbent's own column counts as a failure
    (``skip_starved``: a sampler with nothing but that column is left alone).
    A shared width (``width_mode``) moves by the success rate of all the real
    samples of the pool, the incumbent columns excluded.
    """
    if variant.width_mode != "per_sampler":
        real = np.ones(len(probe), dtype=bool)
        real[boundaries[:-1]] = False
        if not real.any():
            return
        pooled = int(np.count_nonzero(probe[real] < 0)) / int(np.count_nonzero(real))
        rungs = np.log(_rungs(variant))
        log_sigma = samplers[0].random_growth - float(rungs[0])
        log_sigma = _clip_shared(
            log_sigma + variant.sr_eta * (pooled - variant.sr_shared_target), variant
        )
        for sampler, rung in zip(samplers, rungs, strict=True):
            sampler.random_growth = log_sigma + float(rung)
        return
    cap = float(np.log(variant.sr_max_width))
    for k, sampler in enumerate(samplers):
        segment = probe[boundaries[k] : boundaries[k + 1]]
        if variant.skip_starved and len(segment) <= 1:
            continue
        rate = np.count_nonzero(segment < 0) / len(segment)
        step = variant.sr_eta * (rate - variant.sr_targets[k])
        sampler.random_growth = float(
            np.clip(sampler.random_growth + step, LOG_WIDTH_FLOOR, cap)
        )


def learned_scales(moment: np.ndarray, cap: float) -> np.ndarray:
    """Per-coordinate width multipliers: ``sqrt(moment)`` with geometric mean 1.

    The ratio between the widest and the narrowest coordinate is limited to
    ``cap ** 2`` by clipping each multiplier into ``[1 / cap, cap]``.
    """
    root = np.sqrt(moment)
    scales = root / np.exp(np.mean(np.log(root)))
    return np.asarray(np.clip(scales, 1.0 / cap, cap))


def update_scale_moment(moment: np.ndarray, change: np.ndarray, rate: float) -> None:
    """Fold an accepted move into the moving average (in place).

    The move is normalised by its own mean square, so that the average tracks the
    shape of the moves and not their size, which shrinks as the search converges.
    """
    square = change**2
    mean = float(square.mean())
    if mean > 0 and np.isfinite(mean):
        moment *= 1 - rate
        moment += rate * square / mean


def update_covariance(
    cov: np.ndarray,
    steps: np.ndarray,
    cost: np.ndarray,
    variant: Variant,
    rank_one: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank-mu update of the covariance shape; returns it and its square root.

    ``rank_one``, if given, is an evolution path added with CMA's default rate
    ``c1 = 2 / ((dim + 1.3)^2 + mu_eff)`` (variant ``cov_path``).

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
    if rank_one is None:
        new = (1 - rate) * cov + rate * (chosen.T * weights) @ chosen
    else:
        c1 = min(2.0 / ((dim + 1.3) ** 2 + mu_eff), 1.0 - rate)
        new = (
            (1 - rate - c1) * cov
            + c1 * np.outer(rank_one, rank_one)
            + rate * (chosen.T * weights) @ chosen
        )
    values, vectors = np.linalg.eigh((new + new.T) / 2)
    values = np.maximum(values, values.max() * 1e-300 + 1e-300)
    values = np.maximum(values, values.max() / variant.cov_cap)
    values /= np.exp(np.mean(np.log(values)))
    shape = (vectors * values) @ vectors.T
    factor = vectors * np.sqrt(values)
    return shape, factor
