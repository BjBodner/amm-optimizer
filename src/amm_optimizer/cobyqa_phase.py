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

from ._budget import BudgetedObjective, SpentError
from .bounds import Box
from .es_phase import EsResult
from .objective import CountedObjective

FINAL_RADIUS = 1e-10


def run_cobyqa_restarts(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    box: Box,
    evaluations: int,
    radius_share: float = 0.25,
    cost: float | None = None,
    sigma: float = 0.0,
) -> EsResult:
    """COBYQA from ``x`` with an initial trust radius of ``radius_share`` x the box's
    mean width (final radius ``FINAL_RADIUS``), restarted from uniform points whenever
    it stops early, until ``evaluations`` are spent or the objective runs out.

    Returns the best point, its value, ``sigma`` (COBYQA has no step-size state to
    hand over) and the evaluations used.

    With ``cost``, the value of ``x`` is known (``small_switch``): the best point
    starts as (``x``, ``cost``), and COBYQA's first point, when it is ``x`` itself, is
    not evaluated again (COBYQA moves a start that lies within one radius of a bound),
    so the first run may make one call more. Without it, the first run starts from
    ``x`` clipped onto the box.
    """
    from scipy.optimize import OptimizeWarning  # type: ignore[import-untyped]
    from scipy.optimize import minimize as scipy_minimize

    dim = len(x)
    radius = radius_share * float(np.mean(box.upper - box.lower))
    bounds = list(zip(box.lower, box.upper, strict=True))
    known = cost is not None
    fun = BudgetedObjective(
        objective,
        evaluations,
        box=box,
        best_x=np.array(x, dtype=float),
        best_cost=np.inf if cost is None else float(cost),
        known=None if cost is None else (np.array(x, dtype=float), float(cost)),
    )
    start = (
        np.array(x, dtype=float)
        if known
        else np.clip(np.array(x, dtype=float), box.lower, box.upper)
    )
    while fun.used < evaluations:
        # with a known first point, one more than is left: that point is free
        before, left = fun.used, evaluations - fun.used + (1 if known else 0)
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
                        "final_tr_radius": FINAL_RADIUS,
                    },
                )
        except SpentError:
            break
        fun.known = None
        if fun.used == before:  # nothing evaluated: never loop without progress
            break
        start = box.lower + (box.upper - box.lower) * rng.rand(dim)
    assert fun.best_x is not None
    return EsResult(fun.best_x, fun.best_cost, sigma, fun.used)
