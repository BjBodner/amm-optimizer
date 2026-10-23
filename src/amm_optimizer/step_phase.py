"""A separable coordinate search (variant ``sep_probe``, audit cycle 4).

Round-robin STEP as in BSrr (Pošík and Baudiš 2015): every coordinate keeps the
points sampled along its axis through the current best point ``x*`` and, at each
turn, bisects the interval whose "difficulty" is smallest. That is the curvature a
parabola through the interval's end values needs to dip ``EPS`` below the
coordinate's best (STEP, Swarzberg et al. 1994). A sample that improves moves
``x*`` along that axis at once.

Under separability ``f(x* with x_i = t) - f(x*) = f_i(t) - f_i(x*_i)`` whatever the
other coordinates are, so each coordinate stores its values relative to its own best
(shifted when that best moves, the BSrr trick). After every sweep one stored point is
re-evaluated: if ``f`` differs from that prediction by more than ``CHECK_TOL`` (in
relative terms), the function is not separable and the phase stops, having spent
about ``3 dim`` evaluations (HCMA's non-separability test, Loshchilov et al. 2013).

That check only sees an interaction if ``x*`` has moved along another axis since the
re-evaluated point was sampled. On a rugged rotated function STEP soon sticks in a
local minimum, ``x*`` stops moving and the check passes for ever (cycle 23: on
rotated Rastrigin / Schaffer at D 5 the phase took 40% of a 3000 x D budget).
``pair_check`` moves two axes at once to stored points and tests additivity,
``f(x*; x_i = a, x_j = b) = f* + h_i(a) + h_j(b)``, which holds exactly under
separability whether or not ``x*`` has moved.
"""

from __future__ import annotations

import numpy as np

from . import qn_phase
from .bounds import Box
from .es_phase import EsResult
from .objective import CountedObjective

EPS = 1e-8
MIN_WIDTH = 1e-10
CHECK_TOL = 1e-7


def run_step_phase(
    objective: CountedObjective,
    x: np.ndarray,
    cost: float,
    box: Box,
    evaluations: int,
    parabolic: bool = False,
    parabolic_late: bool = False,
    pair_check: bool = False,
    pair_when_stuck: bool = False,
    check_tol: float = CHECK_TOL,
    box_check: bool = False,
    ties_out: list[int] | None = None,
) -> EsResult:
    """Spend up to ``evaluations`` on round-robin STEP from ``x`` (cost ``cost``).

    Returns the best point and value, the evaluations used and, as ``stalled``,
    whether the phase stopped because the separability check failed.

    ``parabolic`` (cycle 14, after Brent-STEP): every other turn of an axis probes the
    vertex of the parabola through its best stored point and the two neighbours
    when that parabola is convex and the vertex lies strictly inside their bracket,
    away from both, instead of STEP's least-difficult midpoint. On a separable
    quadratic (bbob f2) one such step lands on an axis's minimum. ``parabolic_late``:
    only once the separability check has passed, so a non-separable function sees
    exactly the plain search (Batch 37's noise).
    ``pair_when_stuck`` (cycle 29): the check uses ``pair_check``'s two-axis
    additivity test only when the best value has not improved over the last two
    sweeps. Re-evaluating a stored point of one axis at an unchanged best returns the
    stored value, so the one-axis check cannot fail once STEP is stuck in a local
    minimum (f15-f18 at D 5 spent 600-1800 x dim there).
    ``check_tol`` replaces ``CHECK_TOL`` (``sep_route``, cycle 50: inf turns the
    check off once a pair test has read the function separable).
    ``box_check`` (``Variant.step_box_check``, the novel-ideas toy): the separability
    check also runs once right after the box ends, on axis 0, when the best point has
    moved along another axis since axis 0's ends were probed (otherwise it could not
    fail); a non-separable function then stops a sweep (``dim`` evaluations) earlier.
    ``ties_out`` (research, parked2), if given, receives the index of every probe
    whose value repeats the incumbent's exactly (a plateau reading at no cost: only
    bbob f7's steps tie inside the box).

    Each axis keeps its samples in arrays (cycle 50; they were lists converted at
    every turn, O(samples) Python work per evaluation: 4 ms per evaluation at D 40,
    3000 x D). The points evaluated are the same.
    """
    dim = len(x)
    best = np.array(x, dtype=float)
    f_best = float(cost)
    used = 0
    # Per coordinate: sampled positions and their values relative to its best.
    ts = [np.array([float(best[i])]) for i in range(dim)]
    hs = [np.array([0.0]) for _ in range(dim)]
    moved: set[int] = set()  # ``box_check``: axes the best point moved along

    def probe(i: int, t: float) -> float | None:
        nonlocal used, f_best
        point = best.copy()
        point[i] = t
        value = objective.evaluate_point(point)
        if value is None:
            return None
        used += 1
        if ties_out is not None and float(value) == f_best:
            ties_out.append(used)
        h = float(value) - f_best if np.isfinite(value) else np.inf
        if h < 0:  # a new best along axis i: re-base coordinate i's values
            moved.add(i)
            best[i] = t
            f_best = float(value)
            hs[i] = hs[i] - h
            h = 0.0
        # Slice and concatenate: what ``np.insert`` does, without its Python overhead.
        t_i, h_i = ts[i], hs[i]
        position = int(t_i.searchsorted(t))
        ts[i] = np.concatenate((t_i[:position], [float(t)], t_i[position:]))
        hs[i] = np.concatenate((h_i[:position], [h], h_i[position:]))
        return float(value)

    for i in range(dim):  # the box ends of every axis
        for end in (box.lower[i], box.upper[i]):
            if used >= evaluations or not np.isfinite(f_best):
                return EsResult(best, f_best, 0.0, used, False)
            if end != best[i] and probe(i, float(end)) is None:
                return EsResult(best, f_best, 0.0, used, False)
    if box_check and dim > 1 and moved - {0} and used < evaluations:
        early = np.flatnonzero((hs[0] > 0) & np.isfinite(hs[0]))
        if len(early):
            k0 = int(early[len(early) // 2])
            predicted = f_best + hs[0][k0]
            point = best.copy()
            point[0] = ts[0][k0]
            value = objective.evaluate_point(point)
            if value is None:
                return EsResult(best, f_best, 0.0, used, False)
            used += 1
            scale = max(abs(predicted), abs(float(value)), 1.0)
            if not abs(float(value) - predicted) <= check_tol * scale:
                if value < f_best:
                    best, f_best = point, float(value)
                return EsResult(best, f_best, 0.0, used, True)
    sweep = 0
    turns = [0] * dim
    checks_passed = 0
    after_sweep: list[float] = []  # the best value after each sweep
    while used < evaluations:
        progressed = False
        for i in range(dim):
            if used >= evaluations:
                break
            t_i, h_i = ts[i], hs[i]  # probe replaces, never mutates, these
            turns[i] += 1
            if (
                parabolic
                and turns[i] % 2 == 1
                and (not parabolic_late or checks_passed)
            ):
                vertex = _parabola_vertex(t_i, h_i)
                if vertex is not None:
                    if probe(i, vertex) is None:
                        return EsResult(best, f_best, 0.0, used, False)
                    progressed = True
                    continue
            widths = np.diff(t_i)
            ok = widths > MIN_WIDTH * qn_phase.X_UNIT
            if not ok.any():
                continue
            ha, hb = np.maximum(h_i[:-1], 0) + EPS, np.maximum(h_i[1:], 0) + EPS
            difficulty = ((np.sqrt(ha) + np.sqrt(hb)) / np.where(ok, widths, 1.0)) ** 2
            difficulty = np.where(ok & np.isfinite(difficulty), difficulty, np.inf)
            k = int(np.argmin(difficulty))
            if not np.isfinite(difficulty[k]):
                continue
            if probe(i, 0.5 * (t_i[k] + t_i[k + 1])) is None:
                return EsResult(best, f_best, 0.0, used, False)
            progressed = True
        if not progressed:
            break
        after_sweep.append(f_best)
        stuck = len(after_sweep) >= 3 and after_sweep[-1] >= after_sweep[-3]
        # The separability check: re-evaluate a stored point of one axis
        # (``pair_check``, or ``pair_when_stuck`` once stuck: of two axes at once,
        # an additivity test).
        j = sweep % dim
        sweep += 1
        candidates = np.flatnonzero((hs[j] > 0) & np.isfinite(hs[j]))
        partner = (j + 1 + (sweep // dim) % max(dim - 1, 1)) % dim
        partners = np.flatnonzero((hs[partner] > 0) & np.isfinite(hs[partner]))
        if len(candidates) and used < evaluations:
            k = int(candidates[len(candidates) // 2])
            predicted = f_best + hs[j][k]
            point = best.copy()
            point[j] = ts[j][k]
            if (pair_check or (pair_when_stuck and stuck)) and (
                dim > 1 and partner != j and len(partners)
            ):
                k2 = int(partners[len(partners) // 2])
                predicted += hs[partner][k2]
                point[partner] = ts[partner][k2]
            value = objective.evaluate_point(point)
            if value is None:
                break
            used += 1
            scale = max(abs(predicted), abs(float(value)), 1.0)
            if not abs(float(value) - predicted) <= check_tol * scale:
                if value < f_best:
                    best, f_best = point, float(value)
                return EsResult(best, f_best, 0.0, used, True)
            checks_passed += 1
    return EsResult(best, f_best, 0.0, used, False)


def _parabola_vertex(t: np.ndarray, h: np.ndarray) -> float | None:
    """The vertex of the parabola through the best stored point and its neighbours.

    None unless the three points exist, the parabola is convex, and the vertex lies
    inside the bracket at least 1% of its width away from every stored point.
    """
    m = int(np.argmin(h))
    if m == 0 or m == len(t) - 1:
        return None
    (a, b, c), (fa, fb, fc) = t[m - 1 : m + 2], h[m - 1 : m + 2]
    if not np.all(np.isfinite((fa, fb, fc))):
        return None
    num = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)
    den = (b - a) * (fb - fc) - (b - c) * (fb - fa)
    if den >= 0:  # not convex (the vertex formula's denominator is negative then)
        return None
    vertex = b - 0.5 * num / den
    gap = 0.01 * (c - a)
    if not a + gap < vertex < c - gap or np.min(np.abs(t - vertex)) < max(
        gap, MIN_WIDTH * qn_phase.X_UNIT
    ):
        return None
    return float(vertex)
