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

from .bounds import Box
from .es_phase import INITIAL_SIGMA_SHARE, EsResult
from .objective import CountedObjective

TOLERANCE = 1e-12
MIN_SCALE_SHARE = 1e-9


class _SpentError(Exception):
    pass


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
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    used = 0

    def fun(z: np.ndarray) -> float:
        nonlocal used, best_x, best_cost
        if used >= evaluations:
            raise _SpentError
        point = np.clip(z, box.lower, box.upper)
        value = objective.evaluate_point(point)
        if value is None:
            raise _SpentError
        used += 1
        if value < best_cost:
            best_x, best_cost = point.copy(), float(value)
        return float(value) if np.isfinite(value) else 1e300

    start, size = best_x.copy(), max(scale, MIN_SCALE_SHARE * width)
    while used < evaluations - dim - 1:
        before = best_cost
        simplex = np.vstack([start, start + size * np.eye(dim)])
        try:
            minimize(
                fun,
                start,
                method="Nelder-Mead",
                options={
                    "initial_simplex": simplex,
                    "maxfev": evaluations - used,
                    "xatol": TOLERANCE * width,
                    "fatol": TOLERANCE,
                    "adaptive": True,
                },
            )
        except _SpentError:
            break
        if best_cost < before:
            start, size = best_x.copy(), max(size / 2, MIN_SCALE_SHARE * width)
        else:
            start = box.lower + (box.upper - box.lower) * rng.rand(dim)
            size = INITIAL_SIGMA_SHARE * width
    return EsResult(best_x, best_cost, size, used)
