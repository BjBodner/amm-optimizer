"""Restarted Nelder-Mead for very low dimensions (variant ``nm_max_dim``, cycle 5).

In 2-D the best solvers of the COCO archive are restarted simplex searches (Hansen's
NELDER, NELDERDOERR), whose strength is many short local runs. Here SciPy's
Nelder-Mead runs from the best point with a simplex of ``scale`` per axis; each
following run starts at the best point with the simplex size halved while it keeps
improving, and from a uniform point (simplex a fifth of the box) once a run fails to
improve. Points are clipped onto the box before evaluation.
"""

from __future__ import annotations

import numpy as np

from ._budget import BudgetedObjective, SpentError
from .bounds import Box
from .es_phase import INITIAL_SIGMA_SHARE, EsResult
from .objective import CountedObjective

TOLERANCE = 1e-12
MIN_SCALE_SHARE = 1e-9


def run_nm_restarts(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    box: Box,
    evaluations: int,
    scale: float,
) -> EsResult:
    """Spend up to ``evaluations`` on restarted Nelder-Mead from ``x`` at ``cost``."""
    from scipy.optimize import minimize  # type: ignore[import-untyped]

    dim = len(x)
    width = float(np.mean(box.upper - box.lower))
    fun = BudgetedObjective(
        objective,
        evaluations,
        box=box,
        best_x=np.array(x, dtype=float),
        best_cost=float(cost),
    )
    start, size = np.array(x, dtype=float), max(scale, MIN_SCALE_SHARE * width)
    while fun.used < evaluations - dim - 1:
        before = fun.best_cost
        simplex = np.vstack([start, start + size * np.eye(dim)])
        try:
            minimize(
                fun,
                start,
                method="Nelder-Mead",
                options={
                    "initial_simplex": simplex,
                    "maxfev": evaluations - fun.used,
                    "xatol": TOLERANCE * width,
                    "fatol": TOLERANCE,
                    "adaptive": True,
                },
            )
        except SpentError:
            break
        if fun.best_cost < before:
            assert fun.best_x is not None
            start, size = fun.best_x.copy(), max(size / 2, MIN_SCALE_SHARE * width)
        else:
            start = box.uniform(rng)
            size = INITIAL_SIGMA_SHARE * width
    assert fun.best_x is not None
    return EsResult(fun.best_x, fun.best_cost, size, fun.used)
