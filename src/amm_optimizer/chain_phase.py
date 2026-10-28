"""Additivity readings and the decoupled coordinate search (cycle 50, the f20 toy).

A function separable in w = M x, with M unit triangular, has the Hessian
H = M^T D M at every point, and that factorisation is unique. bbob f20 (Schwefel)
is such a chain: each coordinate's term also takes a quarter of the previous one.
That coupling defeats every per-coordinate search, and CMA-ES as well.

``pair_test`` reads, in 3 D evaluations, whether a function is additive across far
and adjacent coordinate pairs. Its readings are "separable", "chain" (only
neighbours interact) or "nonsep". ``band_consistency`` (6 D) then separates a
linear chain, whose banded factor is the same at any two points (f20), from a
nonlinear one (Rosenbrock f8). ``run_coord_hess`` runs global line searches along
the columns of M^-1, from a finite-difference Hessian. Each such direction moves one
decoupled coordinate.

A calibration study on bbob instances 1-3 (cold start):
* ``run_coord_hess`` solved f20 in 120 of 120 runs at 3000 x D, D 5-40, a median of
  168-289 x D, +.78 over ``c36`` at 300-3000 x D.
* The pair test read f1-f5 separable 300 of 300 times, f20 a chain 60 of 60 times,
  and the other 17 functions non-separable; f8 also reads as a chain.
* The band check (threshold ``BAND_TOL``) sent all 480 f20 runs to the search and
  all 240 f8 runs back.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .bounds import Box, clip
from .es_phase import EsResult
from .objective import CountedObjective

SEP_TOL = 1e-6  # a pair's additivity residual below this reads additive
BAND_TOL = 0.02  # f20 0-0.0023, f8 0.089-4.7 (toy dev seeds 90-94, D 5-40)
GOLD = 0.3819660112501051


class _SpentError(Exception):
    """The objective (or the phase's own budget) ran out."""


class _Counted:
    """``objective`` as a plain function: counts, keeps the best, stops at a cap."""

    def __init__(self, objective: CountedObjective, evaluations: int) -> None:
        self.objective, self.evaluations = objective, evaluations
        self.used = 0
        self.best_x: np.ndarray | None = None
        self.best_cost = np.inf

    def __call__(self, x: np.ndarray) -> float:
        if self.used >= self.evaluations:
            raise _SpentError
        value = self.objective.evaluate_point(x)
        if value is None:
            raise _SpentError
        self.used += 1
        if value < self.best_cost:
            self.best_x, self.best_cost = np.array(x, dtype=float), float(value)
        return float(value) if np.isfinite(value) else 1e300


# ------------------------------------------------------------- readings ----
def pair_test(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    box: Box,
    strict: bool = False,
) -> tuple[str, np.ndarray, float] | None:
    """Two-axis additivity at ``x``: (reading, best point, its value), 3 D evaluations.

    f0 = f(x); f_i = f(x with x_i moved to a uniform t_i); then f_ij for the D - 1
    adjacent pairs (i, i + 1) and for D random pairs with |i - j| >= 2. The residual
    |f_ij - f_i - f_j + f0| / (|f_i - f0| + |f_j - f0|) is 0, up to rounding, for an
    additive pair. A far residual of at least ``SEP_TOL`` reads "nonsep"; otherwise
    an adjacent one reads "chain"; otherwise "separable". Pairs whose single moves
    leave f unchanged are skipped. None when the objective runs out.
    ``strict`` (research, gaps3): when every pair was skipped (a plateau, f7 at D 2,
    where the only pair is often flat), the reading is "flat", not "separable".
    """
    dim = len(x)
    fun = _Counted(objective, 3 * dim + 1)
    x = np.array(x, dtype=float)
    t = box.lower + (box.upper - box.lower) * rng.rand(dim)
    try:
        f0 = fun(x)
        fi = []
        for i in range(dim):
            y = x.copy()
            y[i] = t[i]
            fi.append(fun(y))

        def residual(i: int, j: int) -> float | None:
            y = x.copy()
            y[i], y[j] = t[i], t[j]
            fij = fun(y)
            den = abs(fi[i] - f0) + abs(fi[j] - f0)
            if den <= 1e-12 * max(1.0, abs(f0)):
                return None
            return abs(fij - fi[i] - fi[j] + f0) / den

        adjacent = [residual(i, i + 1) for i in range(dim - 1)]
        far = []
        if dim >= 3:
            for _ in range(dim):
                while True:
                    i, j = sorted(rng.choice(dim, 2, replace=False))
                    if j - i >= 2:
                        break
                far.append(residual(int(i), int(j)))
    except _SpentError:
        return None
    r_adj = max((r for r in adjacent if r is not None), default=0.0)
    r_far = max((r for r in far if r is not None), default=0.0)
    reading = (
        "nonsep" if r_far >= SEP_TOL else "chain" if r_adj >= SEP_TOL else "separable"
    )
    if strict and all(r is None for r in adjacent + far):
        reading = "flat"
    assert fun.best_x is not None
    return reading, fun.best_x, fun.best_cost


def _band_factors(
    fun: _Counted, x: np.ndarray, box: Box, h: float
) -> dict[str, np.ndarray | None]:
    """M's off-diagonal for both orientations of H = M^T D M from one banded FD
    Hessian at ``x`` (central diagonal, forward adjacent mixed terms, 3 D
    evaluations).

    The orientations are "lower" (M unit lower bidiagonal, the reversed LDL^T) and
    "upper" (unit upper, the LDL^T). Both exist for any tridiagonal H; only the true
    one is the same at every point. An orientation reads None on a zero pivot.
    """
    dim = len(x)
    x = np.clip(np.array(x, dtype=float), box.lower + 2 * h, box.upper - 2 * h)
    f0 = fun(x)
    fp, fm = np.empty(dim), np.empty(dim)
    for i in range(dim):
        y = x.copy()
        y[i] += h
        fp[i] = fun(y)
        y[i] -= 2 * h
        fm[i] = fun(y)
    diag = (fp - 2 * f0 + fm) / h**2
    off = np.empty(dim - 1)
    for i in range(dim - 1):
        y = x.copy()
        y[i] += h
        y[i + 1] += h
        off[i] = (fun(y) - fp[i] - fp[i + 1] + f0) / h**2
    return {"lower": _lower_band(diag, off), "upper": _upper_band(diag, off)}


def _lower_band(diag: np.ndarray, off: np.ndarray) -> np.ndarray | None:
    """M[i + 1, i] of the reversed LDL^T of a tridiagonal H (from the last row)."""
    sub = np.empty(len(off))
    p = diag[-1]
    for i in range(len(off) - 1, -1, -1):
        if p == 0 or not np.isfinite(p):
            return None
        sub[i] = off[i] / p
        p = diag[i] - sub[i] ** 2 * p
    return sub


def _upper_band(diag: np.ndarray, off: np.ndarray) -> np.ndarray | None:
    """M[i, i + 1] of the LDL^T of a tridiagonal H (from the first row)."""
    sup = np.empty(len(off))
    p = diag[0]
    for i in range(len(off)):
        if p == 0 or not np.isfinite(p):
            return None
        sup[i] = off[i] / p
        p = diag[i + 1] - sup[i] ** 2 * p
    return sup


def band_consistency(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    box: Box,
    h: float = 1e-4,
) -> tuple[float, str, np.ndarray, float] | None:
    """(spread, orientation, best point, value) of a banded factorisation at ``x``
    and at a uniform x2: 6 D evaluations.

    The spread is the median |M_off(x) - M_off(x2)| over the band, in the
    orientation ("lower" or "upper") where it is smaller. A chain whose coupling is
    linear (f20, separable after a triangular map) has the same factor everywhere in
    its true orientation; a nonlinear chain (Rosenbrock, f8) has it in neither. A
    zero pivot reads inf. None when the objective runs out.
    """
    dim = len(x)
    fun = _Counted(objective, 6 * dim)
    x2 = box.lower + (box.upper - box.lower) * rng.rand(dim)
    try:
        a = _band_factors(fun, x, box, h)
        b = _band_factors(fun, x2, box, h)
    except _SpentError:
        return None
    spreads = {}
    for order in ("lower", "upper"):
        fa, fb = a[order], b[order]
        spreads[order] = (
            np.inf if fa is None or fb is None else float(np.median(np.abs(fa - fb)))
        )
    order = min(spreads, key=lambda k: spreads[k])
    spread = spreads[order]
    assert fun.best_x is not None
    return spread, order, fun.best_x, fun.best_cost


# ------------------------------------------------------------ 1-D search ----
def _refine_bracket(
    h: Callable[[float], float],
    a: float,
    fa: float,
    m: float,
    fm: float,
    b: float,
    fb: float,
    n: int,
) -> tuple[float, float]:
    """At most ``n`` evaluations of parabolic steps (golden fallback) in [a, b].

    Needs a < m < b with fm <= fa, fb. Returns the best point and value.
    """
    widths = [b - a]
    for _ in range(n):
        width = b - a
        if width <= 1e-13 * max(1.0, abs(m)):
            break
        u = None
        den = (m - a) * (fm - fb) - (m - b) * (fm - fa)
        slow = len(widths) >= 3 and width > 0.5 * widths[-3]
        if den < 0 and not slow:
            num = (m - a) ** 2 * (fm - fb) - (m - b) ** 2 * (fm - fa)
            u = m - 0.5 * num / den
            gap = 0.01 * width
            if not (a + gap < u < b - gap) or abs(u - m) < gap:
                u = None
        if u is None:  # a golden step into the larger part
            u = m + GOLD * (b - m) if b - m >= m - a else m - GOLD * (m - a)
        fu = h(u)
        if fu <= fm:
            if u > m:
                a, fa = m, fm
            else:
                b, fb = m, fm
            m, fm = u, fu
        elif u > m:
            b, fb = u, fu
        else:
            a, fa = u, fu
        widths.append(b - a)
    return m, fm


def _refine_at(
    h: Callable[[float], float], ts: np.ndarray, vals: np.ndarray, k: int, n: int
) -> tuple[float, float]:
    """Refine around sampled point ``k`` of a profile with ``n`` evaluations."""
    t, v = float(ts[k]), float(vals[k])
    if n <= 0:
        return t, v
    if 0 < k < len(ts) - 1:
        lo, hi = k - 1, k + 1
        while lo > 0 and ts[lo] == ts[k]:
            lo -= 1
        while hi < len(ts) - 1 and ts[hi] == ts[k]:
            hi += 1
        if ts[lo] < t < ts[hi] and vals[k] <= vals[lo] and vals[k] <= vals[hi]:
            return _refine_bracket(h, ts[lo], vals[lo], t, v, ts[hi], vals[hi], n)
    nb = 1 if k == 0 else len(ts) - 2  # at a window end: towards the neighbour
    if nb < 0 or nb >= len(ts) or ts[nb] == t:
        return t, v
    mid = 0.5 * (t + ts[nb])
    fm = h(mid)
    if fm < v:
        if t < ts[nb]:
            a, fa, b, fb = t, v, ts[nb], vals[nb]
        else:
            a, fa, b, fb = ts[nb], vals[nb], t, v
        if fm <= fa and fm <= fb:
            return _refine_bracket(h, a, fa, mid, fm, b, fb, n - 1)
        return mid, fm
    return t, v


def _local_minima(vals: np.ndarray) -> list[int]:
    """Indices of a sampled profile's local minima (on a plateau, its first index)."""
    n = len(vals)
    out = []
    for k in range(n):
        left = vals[k - 1] if k > 0 else np.inf
        right = vals[k + 1] if k < n - 1 else np.inf
        if vals[k] < left and vals[k] <= right:
            out.append(k)
    return out


def _segment(x: np.ndarray, v: np.ndarray, box: Box) -> tuple[float, float]:
    """The t range keeping x + t v inside the box."""
    lo, hi = -np.inf, np.inf
    for xj, vj, lj, uj in zip(x, v, box.lower, box.upper, strict=True):
        if vj > 1e-15:
            lo, hi = max(lo, (lj - xj) / vj), min(hi, (uj - xj) / vj)
        elif vj < -1e-15:
            lo, hi = max(lo, (uj - xj) / vj), min(hi, (lj - xj) / vj)
    return min(lo, 0.0), max(hi, 0.0)


# ------------------------------------------------------- the basis ----
def _fd_hessian(
    fun: _Counted, x: np.ndarray, fx: float, box: Box, h: float
) -> np.ndarray:
    """Central diagonal, forward off-diagonal; 2 D + D (D - 1) / 2 evaluations.

    Steps point into the box (a forward second difference next to a bound).
    """
    dim = len(x)
    sign = np.where(x + 2 * h <= box.upper, 1.0, -1.0)
    fp = np.empty(dim)
    hess = np.zeros((dim, dim))
    for i in range(dim):
        y = x.copy()
        y[i] += sign[i] * h
        fp[i] = fun(y)
        y = x.copy()
        if box.lower[i] <= x[i] - h and x[i] + h <= box.upper[i]:
            y[i] -= h
            hess[i, i] = (fp[i] - 2 * fx + fun(y)) / h**2
        else:
            y[i] += 2 * sign[i] * h
            hess[i, i] = (fun(y) - 2 * fp[i] + fx) / h**2
    for i in range(dim):
        for j in range(i + 1, dim):
            y = x.copy()
            y[i] += sign[i] * h
            y[j] += sign[j] * h
            mixed = (fun(y) - fp[i] - fp[j] + fx) / (h**2 * sign[i] * sign[j])
            hess[i, j] = hess[j, i] = mixed
    return hess


def _ldl_unit(a: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """a = L diag(p) L^T with L unit lower triangular; None on a tiny pivot."""
    dim = len(a)
    low = np.eye(dim)
    p = np.zeros(dim)
    scale = max(float(np.max(np.abs(np.diag(a)))), 1e-300)
    for j in range(dim):
        p[j] = a[j, j] - np.sum(low[j, :j] ** 2 * p[:j])
        if not np.isfinite(p[j]) or abs(p[j]) <= 1e-8 * scale:
            return None
        for i in range(j + 1, dim):
            low[i, j] = (a[i, j] - np.sum(low[i, :j] * low[j, :j] * p[:j])) / p[j]
    return low, p


def decoupling_basis(
    hess: np.ndarray, order: str | None = None
) -> tuple[np.ndarray, bool]:
    """The columns of M^-1 for H = M^T D M, M unit triangular, and D > 0 or not.

    The LDL^T of the reversed H gives a lower M, that of H an upper one. ``order``
    ("lower" or "upper", from ``band_consistency``) takes that one when it exists.
    Otherwise the sparser factor is taken, as in the toy: the true M of a banded
    coupling is banded and the other order's is dense, except for a tridiagonal H,
    where both are bidiagonal and only ``order`` tells them apart. Without a
    factorisation it returns the axes and False.
    """
    dim = len(hess)
    hess = 0.5 * (hess + hess.T)
    rev = np.arange(dim)[::-1]
    options = []
    got = _ldl_unit(hess[np.ix_(rev, rev)])
    if got is not None:  # J H J = L D L^T, so M = J L^T J (lower triangular)
        lower = got[0].T[np.ix_(rev, rev)]
        options.append(("lower", lower, bool(np.all(got[1] > 0))))
    got = _ldl_unit(hess)
    if got is not None:  # H = L D L^T, so M = L^T (upper triangular)
        options.append(("upper", got[0].T, bool(np.all(got[1] > 0))))
    if not options:
        return np.eye(dim), False
    named = [option for option in options if option[0] == order]
    if named:
        _, m, positive = named[0]
    else:
        mass = [float(np.sum(np.abs(m)) - dim) for _, m, _ in options]
        _, m, positive = options[int(np.argmin(mass))]
    return np.linalg.inv(m), positive


# ------------------------------------------------------------- the search ----
def _progressed(f_new: float, f_old: float) -> bool:
    return f_new < f_old - 1e-12 * max(1.0, abs(f_new))


class _CoordHess:
    """The calibration study's ``coord_hess`` search, one run per ``solve``.

    1. Warm-up: at least ``warm`` cheap sweeps along the axes (``warm_grid`` points,
       ``warm_refine`` evaluations), towards a coordinate-wise minimum, where the
       factorisation's curvatures are positive and not small.
    2. Basis: an FD Hessian and ``decoupling_basis``. If its pivots are not all
       positive, one more warm-up sweep and a new Hessian, up to ``warm_max``.
    3. Cycles along the basis:
       * a global sweep: ``grid`` points over each direction's feasible segment,
         offset by a random fraction of the spacing; the ``basins`` best local
         minima are refined with ``refine`` evaluations each;
       * then local sweeps: ``local_grid`` points in a window starting at two global
         spacings and halving per sweep, until a sweep makes no progress.
       A cycle whose global sweep makes no progress ends the run.

    A restart skips the warm-up and takes its basis from a Hessian at the previous
    run's end point, a local minimum, where the factorisation is most accurate.
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        box: Box,
        grid: int = 81,
        refine: int = 8,
        basins: int = 3,
        local_grid: int = 5,
        warm_grid: int = 21,
        warm_refine: int = 6,
        warm: int = 2,
        warm_max: int = 4,
        h: float = 1e-4,
        order: str | None = None,
    ) -> None:
        self.rng, self.box, self.order = rng, box, order
        self.grid, self.refine, self.basins, self.local_grid = (
            grid,
            refine,
            basins,
            local_grid,
        )
        self.warm_grid, self.warm_refine, self.warm, self.warm_max = (
            warm_grid,
            warm_refine,
            warm,
            warm_max,
        )
        self.h = h
        self.previous: tuple[np.ndarray, float] | None = None
        self.cycles = 0

    def basis(self, fun: _Counted, x: np.ndarray, fx: float) -> tuple[np.ndarray, bool]:
        return decoupling_basis(_fd_hessian(fun, x, fx, self.box, self.h), self.order)

    def line(
        self,
        fun: _Counted,
        x: np.ndarray,
        fx: float,
        v: np.ndarray,
        a: float,
        b: float,
        grid: int,
        refine: int,
        basins: int,
    ) -> tuple[np.ndarray, float]:
        """Sample [a, b] along v, refine the best local minima: the new (x, f)."""
        base = x.copy()
        lo, hi = self.box.lower, self.box.upper

        def h(t: float) -> float:
            return fun(clip(base + t * v, lo, hi))

        ts = np.linspace(a, b, grid)
        vals = np.array([h(t) for t in ts])
        ts, vals = np.append(ts, 0.0), np.append(vals, fx)
        order = np.argsort(ts, kind="stable")
        ts, vals = ts[order], vals[order]
        minima = sorted(_local_minima(vals), key=lambda k: vals[k])[:basins]
        t, val = 0.0, fx
        for k in minima or [int(np.argmin(vals))]:
            tk, vk = _refine_at(h, ts, vals, k, refine)
            if vk < val:
                t, val = tk, vk
        if val < fx:
            return np.clip(base + t * v, lo, hi), val
        return x, fx

    def global_sweep(
        self, fun: _Counted, x: np.ndarray, fx: float, basis: np.ndarray
    ) -> tuple[np.ndarray, float]:
        for i in range(len(x)):
            lo, hi = _segment(x, basis[:, i], self.box)
            step = (hi - lo) / (self.grid - 1)
            shift = (self.rng.rand() - 0.5) * step
            a, b = max(lo, lo + shift), min(hi, hi + shift)
            x, fx = self.line(
                fun, x, fx, basis[:, i], a, b, self.grid, self.refine, self.basins
            )
        return x, fx

    def local_sweeps(
        self, fun: _Counted, x: np.ndarray, fx: float, basis: np.ndarray
    ) -> tuple[np.ndarray, float]:
        dim = len(x)
        width = np.array(
            [
                2
                * np.subtract(*_segment(x, basis[:, i], self.box)[::-1])
                / (self.grid - 1)
                for i in range(dim)
            ]
        )
        while True:
            f_sweep = fx
            for i in range(dim):
                lo, hi = _segment(x, basis[:, i], self.box)
                w = min(width[i], hi - lo)
                a = min(max(-0.5 * w, lo), hi - w)
                x, fx = self.line(
                    fun, x, fx, basis[:, i], a, a + w, self.local_grid, self.refine, 1
                )
            width = np.maximum(0.5 * width, 1e-12)
            if not _progressed(fx, f_sweep):
                return x, fx

    def solve(self, fun: _Counted, x0: np.ndarray) -> None:
        dim = len(x0)
        x = np.array(x0, dtype=float)
        fx = fun(x)
        if self.previous is not None:  # a restart: the basis from the last end point
            basis, _ = self.basis(fun, *self.previous)
        else:
            for sweep in range(1, self.warm_max + 1):
                for i in range(dim):  # a warm-up sweep along the axes
                    e = np.zeros(dim)
                    e[i] = 1.0
                    lo, hi = self.box.lower[i] - x[i], self.box.upper[i] - x[i]
                    x, fx = self.line(
                        fun, x, fx, e, lo, hi, self.warm_grid, self.warm_refine, 1
                    )
                if sweep < self.warm:
                    continue
                basis, positive = self.basis(fun, x, fx)
                if positive:
                    break
        cycles = 0
        while True:
            cycles += 1
            self.cycles += 1
            f_cycle = fx
            x, fx = self.global_sweep(fun, x, fx, basis)
            if cycles > 1 and not _progressed(fx, f_cycle):
                break
            x, fx = self.local_sweeps(fun, x, fx, basis)
            self.previous = (x, fx)
        self.previous = (x, fx)


def run_coord_hess(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    box: Box,
    evaluations: int,
    order: str | None = None,
) -> EsResult:
    """Restarted ``_CoordHess`` runs, the first from ``x``, then from uniform points,
    until ``evaluations`` are spent; ``order`` goes to ``decoupling_basis``. Returns
    the best point, its value, sigma 0 and the evaluations used.
    """
    fun = _Counted(objective, evaluations)
    engine = _CoordHess(rng, box, order=order)
    start = np.array(x, dtype=float)
    try:
        while True:
            engine.solve(fun, start)
            start = box.lower + (box.upper - box.lower) * rng.rand(len(x))
    except _SpentError:
        pass
    best = fun.best_x if fun.best_x is not None else np.array(x, dtype=float)
    return EsResult(best, fun.best_cost, 0.0, fun.used)
