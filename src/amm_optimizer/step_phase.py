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

import bisect
import math
from typing import Any

import numpy as np

from . import qn_phase
from .bounds import Box
from .es_phase import EsResult
from .objective import CountedObjective

EPS = 1e-8
MIN_WIDTH = 1e-10
CHECK_TOL = 1e-7
CHUNK = 4096  # an axis keeps its samples in chunks of CHUNK to 2 CHUNK (``_Axis``)


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
    3000 x D). The points evaluated are the same. Since 2026-10-23 (post-freeze)
    the arrays are buffers that also hold every interval's difficulty, updated
    where a probe lands (``_Axis``): recomputing them at every turn cost O(samples)
    per evaluation, so a run's cost grew with the square of its length (bbob f21
    at D 20: 7000 evaluations per second at the start, 1200 after 600 000). The
    points evaluated are again the same.
    """
    dim = len(x)
    best = np.array(x, dtype=float)
    f_best = float(cost)
    used = 0
    # Per coordinate: sampled positions and their values relative to its best.
    axes = [_Axis(float(best[i])) for i in range(dim)]
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
            axes[i].insert(float(t), 0.0, rebase=h)
        else:
            axes[i].insert(float(t), h)
        return float(value)

    for i in range(dim):  # the box ends of every axis
        for end in (box.lower[i], box.upper[i]):
            if used >= evaluations or not np.isfinite(f_best):
                return EsResult(best, f_best, 0.0, used, False)
            if end != best[i] and probe(i, float(end)) is None:
                return EsResult(best, f_best, 0.0, used, False)
    if box_check and dim > 1 and moved - {0} and used < evaluations:
        early = axes[0].median_stored()
        if early is not None:
            predicted = f_best + early[1]
            point = best.copy()
            point[0] = early[0]
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
            turns[i] += 1
            if (
                parabolic
                and turns[i] % 2 == 1
                and (not parabolic_late or checks_passed)
            ):
                vertex = axes[i].vertex()
                if vertex is not None:
                    if probe(i, vertex) is None:
                        return EsResult(best, f_best, 0.0, used, False)
                    progressed = True
                    continue
            middle = axes[i].easiest_middle()
            if middle is None:
                continue
            if probe(i, middle) is None:
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
        stored = axes[j].median_stored()
        partner = (j + 1 + (sweep // dim) % max(dim - 1, 1)) % dim
        if stored is not None and used < evaluations:
            predicted = f_best + stored[1]
            point = best.copy()
            point[j] = stored[0]
            other = (
                axes[partner].median_stored()
                if (pair_check or (pair_when_stuck and stuck))
                and (dim > 1 and partner != j)
                else None
            )
            if other is not None:
                predicted += other[1]
                point[partner] = other[0]
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


def _difficulty(t: np.ndarray, h: np.ndarray) -> np.ndarray:
    """STEP's difficulty of each interval between neighbouring samples ``t``.

    Inf where the interval is narrower than ``MIN_WIDTH`` or the value is not finite.
    Every operation is elementwise, so a slice gives the values the whole axis would.
    """
    widths = np.diff(t)
    ok = widths > MIN_WIDTH * qn_phase.x_unit()
    ha, hb = np.maximum(h[:-1], 0) + EPS, np.maximum(h[1:], 0) + EPS
    difficulty = ((np.sqrt(ha) + np.sqrt(hb)) / np.where(ok, widths, 1.0)) ** 2
    return np.where(ok & np.isfinite(difficulty), difficulty, np.inf)


def _pair_difficulty(t1: Any, h1: Any, t2: Any, h2: Any) -> float:
    """``_difficulty`` of the one interval from ``(t1, h1)`` to ``(t2, h2)``.

    The same operations on scalars (correctly rounded square roots, ``q * q`` for
    ``** 2``), so the same value as the vectorised form gives for that interval.
    """
    width = t2 - t1
    if not width > MIN_WIDTH * qn_phase.x_unit():
        return math.inf
    q = (math.sqrt(max(h1, 0.0) + EPS) + math.sqrt(max(h2, 0.0) + EPS)) / width
    difficulty = float(q * q)
    return difficulty if math.isfinite(difficulty) else math.inf


class _Chunk:
    """A run of an axis's samples, in order: positions ``t``, values ``h`` and ``d``,
    the difficulty of the interval from each sample to the next one of the axis
    (inf after its last sample). The first ``m`` entries are live."""

    __slots__ = ("d", "h", "m", "t")

    def __init__(self, capacity: int) -> None:
        self.t = np.empty(capacity)
        self.h = np.empty(capacity)
        self.d = np.empty(capacity)
        self.m = 0

    def grow(self) -> None:
        """Double the room (up to ``2 CHUNK``: a full chunk is split instead)."""
        room = min(2 * len(self.t), 2 * CHUNK)
        for name in ("t", "h", "d"):
            bigger = np.empty(room)
            bigger[: self.m] = getattr(self, name)[: self.m]
            setattr(self, name, bigger)


class _Axis:
    """One coordinate's samples, sorted by position, with their values relative to
    its best and the difficulty of every interval between neighbours.

    The samples are held in chunks of ``CHUNK`` to ``2 CHUNK`` (``_Chunk``), with each
    chunk's last position (``last``, to find where a position goes) and its least
    difficulty (``least``, to find the easiest interval). An insert shifts one chunk
    and recomputes the one or two intervals the new sample borders, so its cost does
    not grow with the number of samples. A re-base (a new best: every value moves)
    recomputes every interval. Values are never below 0 and are 0 at the axis's best,
    so ``zero``, the smallest position whose value is 0, locates ``np.argmin`` of the
    values without a scan.
    """

    def __init__(self, t0: float) -> None:
        first = _Chunk(64)
        first.t[0], first.h[0], first.d[0], first.m = t0, 0.0, np.inf, 1
        self.chunks = [first]
        self.last = [t0]
        self.least = np.array([np.inf])
        self.zero = t0

    def _locate(self, t: float) -> tuple[int, int]:
        """(chunk, index) where ``t`` goes: before every sample at or above it."""
        b = bisect.bisect_left(self.last, t)
        if b == len(self.chunks):
            return b - 1, self.chunks[b - 1].m
        chunk = self.chunks[b]
        return b, int(chunk.t[: chunk.m].searchsorted(t))

    def _before(self, b: int, j: int) -> tuple[Any, Any] | None:
        """(position, value) of the sample before sample ``j`` of chunk ``b``."""
        if j > 0:
            chunk = self.chunks[b]
            return chunk.t[j - 1], chunk.h[j - 1]
        if b > 0:
            chunk = self.chunks[b - 1]
            return chunk.t[chunk.m - 1], chunk.h[chunk.m - 1]
        return None

    def _after(self, b: int, j: int) -> tuple[Any, Any] | None:
        """(position, value) of the sample after sample ``j`` of chunk ``b``."""
        chunk = self.chunks[b]
        if j + 1 < chunk.m:
            return chunk.t[j + 1], chunk.h[j + 1]
        if b + 1 < len(self.chunks):
            chunk = self.chunks[b + 1]
            return chunk.t[0], chunk.h[0]
        return None

    def _interval(self, b: int, j: int) -> float:
        after = self._after(b, j)
        if after is None:
            return math.inf
        chunk = self.chunks[b]
        return _pair_difficulty(chunk.t[j], chunk.h[j], after[0], after[1])

    def insert(self, t: float, h: float, rebase: float = 0.0) -> None:
        """Insert ``(t, h)``, first subtracting ``rebase`` (< 0: a new best) from every
        stored value, as ``hs[i] = hs[i] - h`` did."""
        if rebase:
            for chunk in self.chunks:
                np.subtract(chunk.h[: chunk.m], rebase, out=chunk.h[: chunk.m])
            self.zero = t
        elif h == 0.0 and t < self.zero:
            self.zero = t
        b, q = self._locate(t)
        chunk = self.chunks[b]
        m = chunk.m
        if m == len(chunk.t):
            chunk.grow()
        for column in (chunk.t, chunk.h, chunk.d):
            column[q + 1 : m + 1] = column[q:m]
        chunk.t[q], chunk.h[q], chunk.m = t, h, m + 1
        if q == m:
            self.last[b] = t
        if rebase:
            for k, each in enumerate(self.chunks):
                live = each.m
                each.d[: live - 1] = _difficulty(each.t[:live], each.h[:live])
                each.d[live - 1] = self._interval(k, live - 1)
                self.least[k] = np.min(each.d[:live])
        else:
            chunk.d[q] = self._interval(b, q)
            if q > 0:
                chunk.d[q - 1] = self._interval(b, q - 1)
            elif b > 0:
                previous = self.chunks[b - 1]
                previous.d[previous.m - 1] = self._interval(b - 1, previous.m - 1)
                self.least[b - 1] = np.min(previous.d[: previous.m])
            self.least[b] = np.min(chunk.d[: chunk.m])
        if chunk.m == 2 * CHUNK:
            self._split(b)

    def _split(self, b: int) -> None:
        chunk = self.chunks[b]
        rest = _Chunk(2 * CHUNK)
        rest.m = chunk.m - CHUNK
        for name in ("t", "h", "d"):
            getattr(rest, name)[: rest.m] = getattr(chunk, name)[CHUNK : chunk.m]
        chunk.m = CHUNK
        self.chunks.insert(b + 1, rest)
        self.last[b] = float(chunk.t[CHUNK - 1])
        self.last.insert(b + 1, float(rest.t[rest.m - 1]))
        self.least = np.insert(self.least, b + 1, np.min(rest.d[: rest.m]))
        self.least[b] = np.min(chunk.d[: chunk.m])

    def easiest_middle(self) -> Any:
        """The middle of the least difficult interval (the first of equals), or None
        if every interval is too narrow or has a value that is not finite."""
        b = int(np.argmin(self.least))
        chunk = self.chunks[b]
        j = int(np.argmin(chunk.d[: chunk.m]))
        if not np.isfinite(chunk.d[j]):
            return None
        after = self._after(b, j)
        assert after is not None  # a finite difficulty has an interval
        return 0.5 * (chunk.t[j] + after[0])

    def vertex(self) -> float | None:
        """The vertex of the parabola through the best stored point and its
        neighbours on the axis, or None unless the three points exist, the parabola
        is convex, and the vertex lies inside the bracket at least 1% of its width
        away from every stored point."""
        b = bisect.bisect_left(self.last, self.zero)
        chunk = self.chunks[b]
        j = int(chunk.t[: chunk.m].searchsorted(self.zero))
        while chunk.h[j] != 0.0:  # an equal position with another value (never seen)
            j += 1
            if j == chunk.m:
                b, j = b + 1, 0
                chunk = self.chunks[b]
        before, after = self._before(b, j), self._after(b, j)
        if before is None or after is None:
            return None
        found = _bracket_vertex(
            before[0], chunk.t[j], after[0], before[1], chunk.h[j], after[1]
        )
        if found is None:
            return None
        vertex, keep = found
        b2, q2 = self._locate(vertex)  # a stored position is above it, one below
        right = self.chunks[b2].t[q2]
        left = self._before(b2, q2)
        assert left is not None
        if min(abs(left[0] - vertex), abs(right - vertex)) < keep:
            return None
        return float(vertex)

    def median_stored(self) -> tuple[Any, Any] | None:
        """(position, value) of the middle sample (by position) among those with a
        value above 0 and finite, as the separability check picks it, or None."""
        masks = [
            (chunk.h[: chunk.m] > 0) & np.isfinite(chunk.h[: chunk.m])
            for chunk in self.chunks
        ]
        counts = [int(np.count_nonzero(mask)) for mask in masks]
        rank = sum(counts) // 2
        if not sum(counts):
            return None
        for chunk, mask, count in zip(self.chunks, masks, counts, strict=True):
            if rank < count:
                k = int(np.flatnonzero(mask)[rank])
                return chunk.t[k], chunk.h[k]
            rank -= count
        return None


def _bracket_vertex(
    a: Any, b: Any, c: Any, fa: Any, fb: Any, fc: Any
) -> tuple[Any, float] | None:
    """The vertex of the parabola through ``(a, fa), (b, fb), (c, fc)`` and how far
    it must stay from every stored point, or None unless the values are finite, the
    parabola is convex and the vertex lies inside the bracket, 1% of it from both
    ends."""
    if not np.all(np.isfinite((fa, fb, fc))):
        return None
    num = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)
    den = (b - a) * (fb - fc) - (b - c) * (fb - fa)
    if den >= 0:  # not convex (the vertex formula's denominator is negative then)
        return None
    vertex = b - 0.5 * num / den
    gap = 0.01 * (c - a)
    if not a + gap < vertex < c - gap:
        return None
    return vertex, max(gap, MIN_WIDTH * qn_phase.x_unit())
