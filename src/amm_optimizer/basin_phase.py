"""A low-D multi-basin route: a design reading and MLSL (cycle 53, the D 2 toy study).

At D 2-3, ``c37`` trails the field on Gallagher's functions (bbob f21, f22) at
100-1000 x D, where many smooth basins of different depths defeat one descent and the
populations alike. MLSL (Rinnooy Kan and Timmer 1987) starts SLSQP only from sample
points that no better point within the critical radius r_N beats, and keeps sampling.
A D 2 calibration study (on bbob instances 1-3) routed to it with two readings:
* the design's rank-quadratic fit (``r2q`` at most ``GATE_R2Q``): many basins;
* ``local_reading`` at the design's best point (``lcurv`` at most ``GATE_LCURV``, not
  flat): smooth at small scales, unlike the rugged f16, f19 and f23 that read alike on
  the design.

The rule sent 92.5% of f21 / f22 runs and 0.6% of the others to MLSL. On fresh seeds it
gained +.36 on f21-f22 at D 2, 100 x D and at D 3, 300 x D, and +.024 pooled over 14
functions; the decision against a probe-and-ignore control added +.014.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import qn_phase
from ._budget import BudgetedObjective, SpentError
from .bounds import Box
from .objective import CountedObjective

DESIGN_PER_DIM = 16
MLSL_STARTS = 3  # descents per round
MLSL_SIGMA = 2.0  # the critical radius's sigma
LOCAL_SHARES = (1e-3, 1e-5)  # local_reading's steps, as shares of the box width
GATE_LCURV, GATE_R2Q = 0.1, 0.5  # the toy's rule G
DESCENT_FAR = 0.25  # cycle 60: the second descent's start, as a share of the box


def design_size(budget: int, dim: int) -> int:
    """The design's size: min(16 dim, max(dim + 1, budget // 4))."""
    return int(min(DESIGN_PER_DIM * dim, max(dim + 1, budget // 4)))


def _sobol(rng: np.random.RandomState, dim: int) -> Any:
    from scipy.stats import qmc  # type: ignore[import-untyped]

    return qmc.Sobol(
        dim, scramble=True, rng=np.random.default_rng(int(rng.randint(1, 2**31 - 1)))
    )


def _draw(sobol: Any, n: int) -> np.ndarray:
    """``n`` Sobol points in [0, 1)^dim (the design need not be a power of 2)."""
    import warnings

    with warnings.catch_warnings():
        # only SciPy's notice that n is not a power of 2 (review B5)
        warnings.filterwarnings(
            "ignore", message="The balance properties of Sobol", category=UserWarning
        )
        return np.asarray(sobol.random(n))


def rank_quadratic_r2(x: np.ndarray, f: np.ndarray, box: Box) -> float | None:
    """R^2 of a full quadratic least-squares fit to the ranks of ``f`` (None if too few
    points). Ranks make it invariant to monotone transforms of f."""
    n, dim = x.shape
    iu = np.triu_indices(dim)
    z = (x - x.mean(0)) / np.mean(box.upper - box.lower)
    design = np.hstack(
        [np.ones((n, 1)), z, (z[:, :, None] * z[:, None, :])[:, iu[0], iu[1]]]
    )
    if design.shape[1] >= n:
        return None
    rank = np.empty(n)
    rank[np.argsort(f, kind="stable")] = np.arange(n)
    coef = np.linalg.lstsq(design, rank, rcond=None)[0]
    res = rank - design @ coef
    return float(1 - res @ res / max(float(((rank - rank.mean()) ** 2).sum()), 1e-300))


def local_reading(
    fun: BudgetedObjective, x: np.ndarray, fx: float, box: Box
) -> tuple[float, float]:
    """(lcurv, flat share) at ``x``: 4 D evaluations (plus one if ``x`` moved inward).

    c_i(h) = (f(x + h e_i) + f(x - h e_i) - 2 f(x)) / h^2 at the two ``LOCAL_SHARES``
    of the box width. ``lcurv`` is the median over the axes of |log10 |c_i(small h)| /
    |c_i(large h)||: about 0 where f is smooth at these scales, large where it is rugged
    at all scales (Weierstrass, Katsuura). ``flat`` is the share of axes whose four
    differences are all 0 (plateaus).
    """
    width = box.upper - box.lower
    steps = [share * float(np.mean(width)) for share in LOCAL_SHARES]
    inner = np.clip(
        np.asarray(x, dtype=float), box.lower + steps[0], box.upper - steps[0]
    )
    if not np.array_equal(inner, x):
        fx = fun(inner)
    dim = len(inner)
    ratios, flat = [], 0
    for i in range(dim):
        cs, diffs = [], []
        for h in steps:
            e = np.zeros(dim)
            e[i] = h
            fp, fm = fun(inner + e), fun(inner - e)
            diffs += [fp - fx, fm - fx]
            cs.append((fp + fm - 2 * fx) / h**2)
        if all(v == 0 for v in diffs):
            flat += 1
            continue
        ratios.append(abs(np.log10((abs(cs[1]) + 1e-300) / (abs(cs[0]) + 1e-300))))
    return (float(np.median(ratios)) if ratios else 0.0), flat / dim


def critical_radius(n: int, box: Box, sigma: float = MLSL_SIGMA) -> float:
    """MLSL's r_N for n uniform points in the box (Rinnooy Kan and Timmer 1987)."""
    width = box.upper - box.lower
    if n < 2:
        return float(np.max(width))
    dim = len(width)
    volume = float(np.prod(width))
    return float(
        (math.gamma(1 + dim / 2) * volume * sigma * math.log(n) / n) ** (1 / dim)
        / math.sqrt(math.pi)
    )


def _slsqp(fun: BudgetedObjective, x: np.ndarray, box: Box) -> tuple[np.ndarray, float]:
    import warnings

    from scipy.optimize import OptimizeWarning  # type: ignore[import-untyped]
    from scipy.optimize import minimize as scipy_minimize

    with warnings.catch_warnings(), qn_phase.quiet_differences():
        # SciPy's own notices only: the objective's warnings get through (review B5)
        warnings.filterwarnings("ignore", category=OptimizeWarning)
        warnings.filterwarnings(
            "ignore", message="Values in x were outside bounds", category=RuntimeWarning
        )
        res = scipy_minimize(
            fun,
            np.clip(np.asarray(x, dtype=float), box.lower, box.upper),
            method="SLSQP",
            bounds=list(zip(box.lower, box.upper, strict=True)),
            options={"eps": 1e-9, "ftol": 1e-14, "maxiter": 10**5},
        )
    return np.clip(res.x, box.lower, box.upper), float(res.fun)


def _mlsl_rounds(
    fun: BudgetedObjective, sobol: Any, box: Box, x: np.ndarray, f: np.ndarray, n0: int
) -> None:
    """Rounds of at most ``MLSL_STARTS`` SLSQP descents, then ``n0`` more Sobol points,
    until ``fun`` runs out. A sample point starts a descent when it was not a start
    before and no sample point or descent end within r_N has a lower value; candidates
    in increasing f."""
    from scipy.spatial import cKDTree  # type: ignore[import-untyped]

    dim = x.shape[1]
    points, values = np.array(x, dtype=float), np.array(f, dtype=float)
    used = np.zeros(len(values), dtype=bool)
    ends, end_values = np.empty((0, dim)), np.empty(0)
    width = box.upper - box.lower
    while True:
        r = critical_radius(len(values), box)
        pairs = cKDTree(points).query_pairs(r, output_type="ndarray")
        beaten = np.zeros(len(values), dtype=bool)
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            beaten[b[values[a] < values[b]]] = True
            beaten[a[values[b] < values[a]]] = True
        free = np.flatnonzero(~beaten & ~used)
        launched = 0
        for i in free[np.argsort(values[free], kind="stable")]:
            if launched >= MLSL_STARTS:
                break
            near = np.sqrt(((ends - points[i]) ** 2).sum(1)) <= r
            if len(end_values) and np.any(near & (end_values < values[i])):
                continue
            used[i] = True
            launched += 1
            end, value = _slsqp(fun, points[i], box)
            ends = np.vstack([ends, end])
            end_values = np.append(end_values, value)
        new = box.lower + width * _draw(sobol, n0)
        new_values = np.array([fun(p) for p in new])
        points, values = np.vstack([points, new]), np.append(values, new_values)
        used = np.append(used, np.zeros(n0, dtype=bool))


def multi_basin_route(
    objective: CountedObjective,
    rng: np.random.RandomState,
    box: Box,
    budget: int,
    descents: int = 0,
    start_out: list[tuple[np.ndarray, float]] | None = None,
) -> tuple[bool, np.ndarray, float, dict[str, float]]:
    """The design, its readings and, on a multi-basin reading, MLSL for the rest of
    ``budget``: (routed, best point, value, readings).

    Not routed, the caller's path continues from the returned best point; the
    evaluations spent are counted.
    ``descents`` (cycle 60): on a smooth reading that is not multi-basin, up to that
    many SLSQP descents, from the design's best point and from the best point at
    least ``DESCENT_FAR`` of the box away; the returned best point is then the best
    seen, and ``start_out`` receives the design's best point and value, where the
    caller's path starts (a descent's local minimum is a poor start on Rastrigin or
    the Schaffers, which read smooth at the probe's scale).
    """
    dim = len(box.lower)
    fun = BudgetedObjective(objective, budget, as_float=True)
    readings: dict[str, float] = {}
    try:
        sobol = _sobol(rng, dim)
        n0 = design_size(budget, dim)
        x = box.lower + (box.upper - box.lower) * _draw(sobol, n0)
        f = np.array([fun(p) for p in x])
        r2q = rank_quadratic_r2(x, f, box)
        best = int(np.argmin(f))
        lcurv, flat = local_reading(fun, x[best], float(f[best]), box)
        readings = {"r2q": np.nan if r2q is None else r2q, "lcurv": lcurv, "flat": flat}
        multi = (
            r2q is not None and r2q <= GATE_R2Q and lcurv <= GATE_LCURV and flat < 0.5
        )
        readings["multi"] = float(multi)
        if not multi:
            if descents and lcurv <= GATE_LCURV and flat < 0.5:
                if start_out is not None:
                    start_out.append((x[best].copy(), float(f[best])))
                width = box.upper - box.lower
                order = np.argsort(f, kind="stable")
                # RMS distance per coordinate, in box widths
                gap = np.sqrt((((x[order] - x[best]) / width) ** 2).mean(axis=1))
                far = [
                    int(i) for i, g in zip(order, gap, strict=True) if g > DESCENT_FAR
                ]
                before = fun.used
                for i in [best, *far][:descents]:
                    _slsqp(fun, x[i], box)
                readings["descents"] = float(len([best, *far][:descents]))
                readings["descent_calls"] = float(fun.used - before)
            assert fun.best_x is not None
            return False, fun.best_x, fun.best_cost, readings
        _mlsl_rounds(fun, sobol, box, x, f, n0)
    except SpentError:
        pass
    best_x = (
        fun.best_x
        if fun.best_x is not None
        else box.lower + 0.5 * (box.upper - box.lower)
    )
    return bool(readings.get("multi", 0.0)), best_x, fun.best_cost, readings
