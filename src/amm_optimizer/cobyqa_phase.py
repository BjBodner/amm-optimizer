"""Restarted SciPy COBYQA for the smallest budgets (cycle 61, small-budget toy).

COBYQA (Ragonneau and Zhang; SciPy >= 1.14) is Powell's derivative-free trust-region
SQP on quadratic models interpolating 2 D + 1 points, the family HCMA uses (NEWUOA) for
its first evaluations. In a small-budget study of the engines
an initial trust radius of 0.25 of the box width led every other engine at 10 x D; at
D 2 it leads ``c42`` there by +.020 (RESULTS, "Small-budget engines toy against c42").
"""

from __future__ import annotations

import warnings

import numpy as np

from . import qn_phase
from .bounds import Box
from .es_phase import EsResult
from .objective import CountedObjective

FINAL_RADIUS = 1e-10


class _SpentError(Exception):
    pass


def run_cobyqa_restarts(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    box: Box,
    evaluations: int,
    radius_share: float = 0.25,
) -> EsResult:
    """COBYQA from ``x`` with an initial trust radius of ``radius_share`` x the box's
    mean width (final radius ``FINAL_RADIUS``), restarted from uniform points whenever
    it stops early, until ``evaluations`` are spent or the objective runs out.

    Returns the best point, its value, sigma 0 (no step-size state to hand over) and
    the evaluations used.
    """
    from scipy.optimize import OptimizeWarning  # type: ignore[import-untyped]
    from scipy.optimize import minimize as scipy_minimize

    dim = len(x)
    radius = radius_share * float(np.mean(box.upper - box.lower))
    bounds = list(zip(box.lower, box.upper, strict=True))
    used = 0
    best_x, best_cost = np.array(x, dtype=float), np.inf

    def fun(point: np.ndarray) -> float:
        nonlocal used, best_x, best_cost
        if used >= evaluations:
            raise _SpentError
        clipped = np.clip(np.asarray(point, dtype=float), box.lower, box.upper)
        value = objective.evaluate_point(clipped)
        if value is None:
            raise _SpentError
        used += 1
        if value < best_cost:
            best_x, best_cost = clipped, float(value)
        return float(value) if np.isfinite(value) else 1e300

    start = np.clip(np.array(x, dtype=float), box.lower, box.upper)
    while used < evaluations:
        before, left = used, evaluations - used
        try:
            with warnings.catch_warnings():
                # SciPy's notices only: the objective's warnings get through (B5)
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                scipy_minimize(
                    fun,
                    start,
                    method="COBYQA",
                    bounds=bounds,
                    options={
                        "maxfev": left,
                        "maxiter": 10 * left,
                        "initial_tr_radius": radius,
                        "final_tr_radius": FINAL_RADIUS * qn_phase.X_UNIT,
                    },
                )
        except _SpentError:
            break
        if used == before:  # nothing evaluated: never loop without progress
            break
        start = box.lower + (box.upper - box.lower) * rng.rand(dim)
    return EsResult(best_x, best_cost, 0.0, used)


def run_cobyqa_from(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    box: Box,
    evaluations: int,
    radius_share: float = 0.25,
    sigma: float = 0.0,
) -> EsResult:
    """:func:`run_cobyqa_restarts` from a point whose value ``cost`` is known
    (``small_switch``, the small10 toy): COBYQA's first point, when it is ``x``
    itself, is not evaluated again (COBYQA moves a start that lies within one radius
    of a bound). The best point starts as (``x``, ``cost``).
    """
    from scipy.optimize import OptimizeWarning
    from scipy.optimize import minimize as scipy_minimize

    dim = len(x)
    radius = radius_share * float(np.mean(box.upper - box.lower))
    bounds = list(zip(box.lower, box.upper, strict=True))
    used = 0
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    known_x = np.array(x, dtype=float)
    first = [True]

    def fun(point: np.ndarray) -> float:
        nonlocal used, best_x, best_cost
        clipped = np.clip(np.asarray(point, dtype=float), box.lower, box.upper)
        if first[0]:
            first[0] = False
            if np.array_equal(clipped, known_x):
                return float(cost)
        if used >= evaluations:
            raise _SpentError
        value = objective.evaluate_point(clipped)
        if value is None:
            raise _SpentError
        used += 1
        if value < best_cost:
            best_x, best_cost = clipped, float(value)
        return float(value) if np.isfinite(value) else 1e300

    start = np.array(x, dtype=float)
    while used < evaluations:
        # one more than is left: the known first point is free
        before, left = used, evaluations - used + 1
        try:
            with warnings.catch_warnings():
                # SciPy's notices only: the objective's warnings get through (B5)
                warnings.filterwarnings("ignore", category=OptimizeWarning)
                scipy_minimize(
                    fun,
                    start,
                    method="COBYQA",
                    bounds=bounds,
                    options={
                        "maxfev": left,
                        "maxiter": 10 * left,
                        "initial_tr_radius": radius,
                        "final_tr_radius": FINAL_RADIUS * qn_phase.X_UNIT,
                    },
                )
        except _SpentError:
            break
        first[0] = False
        if used == before:  # nothing evaluated: never loop without progress
            break
        start = box.lower + (box.upper - box.lower) * rng.rand(dim)
    return EsResult(best_x, best_cost, sigma, used)
