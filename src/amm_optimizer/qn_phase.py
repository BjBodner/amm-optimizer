"""Restarted SLSQP, a quasi-Newton local search (cycle 11).

On fresh bbob instances, SciPy's SLSQP restarted from uniform points beat the hybrid
`c23` at 10-300 x D on the smooth functions (f6, f8-f14) and on Gallagher's f21-f22
(RESULTS, pip-installable comparison). Its BFGS update learns a full Hessian from
finite-difference gradients (dim + 1 evaluations each), so it converges
superlinearly where the function is smooth. Here it runs from the best point inside
the box; after each run it restarts from a uniform point, until the evaluations are
spent. Points are clipped onto the box before evaluation.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ._budget import SpentError
from .bounds import Box
from .es_phase import EsResult
from .objective import CountedObjective

FTOL = 1e-14
# The curvature probe (cycle 12): scales as shares of the box width, directions, the
# beta threshold (cycle 12 calibration ``prefix_beta.py``: beta over absolute scales
# 0.3-3 is 1.5-2.4 on f1, f2, f6, f8-f14 and f20-f22, and -0.1 to 1.4 on f7, f15-f19,
# f23 and f24, at D 5-40).
BETA_SCALES = (0.03, 0.1, 0.3)
BETA_DIRECTIONS = 4
BETA_THRESHOLD = 1.45
# ``es_kind="slsqplq"``: below this many times dim evaluations the prefix stays the
# (1+1) (Batch 32: the 24-evaluation probe cost up to .05 at 10 x D).
QN_MIN_BUDGET_PER_DIM = 30.0
# ``cma_local_nonsep="qn2"`` (cycle 13): plateau check at this share of the box width
# (Batch 34 diagnosis: mirrored probes at absolute h = .003 repeat f(x) exactly in
# every probe on f7 and in none on the other 23 bbob functions, D 5-40), and the
# share of the local budget SLSQP may use before the CMA finisher takes over.
TIE_SCALE = 3e-4
TIE_DIRECTIONS = 2
QN_LOCAL_SHARE = 1.0 / 3.0
# ``"qn3"`` (Batch 36): the SLSQP stage is also capped at this many times dim, and
# the finisher keeps the warm-start shape (Batch 35 lost f10, f11 and f14 at
# 1000 x D when SLSQP took a third of the budget and the shape was dropped).
QN_LOCAL_CAP_PER_DIM = 100.0
# ``multi_basin``: ends within ``SEED_GAP`` x width x sqrt(dim)
# of a basin already descended.
SEED_GAP = 0.02
# ``"qn6"`` (cycle 16): a capped stage whose first descent is still running when the
# cap is reached (f8, f9, f13 at D 20-40) continues for this share of what is left
# before the CMA finisher; the hill-valley test (Maree et al. 2018) checks these
# fractions of the segment between two descent ends, for up to this many pairs.
QN_EXTEND_SHARE = 2.0 / 3.0
# ``es_kind="slsqplq10"`` (cycle 17): the curvature-gated prefix also below
# ``QN_MIN_BUDGET_PER_DIM`` x dim once the budget reaches this many evaluations, where
# the 24-evaluation probe is at most an eighth of it (Batch 32 at 10 x D: D 20 +.026,
# D 40 +.031; D 5 -.052 and D 10 -.006, where the probe costs a quarter or more).
QN_MIN_BUDGET = 200
# ``"slsqplq10c"`` (cycle 17, after Batch 43): below 30 x dim the probe uses this many
# directions (12 evaluations). On calibration instances 1-3 and 8-10 at D 20 / 40, two
# directions route 125 / 126 of 144 (function, instance) pairs as the calibration table,
# four directions 125 / 124.
CHEAP_BETA_DIRECTIONS = 2
# ``cma_local_nonsep="qn9"`` and ``es_kind="slsqplq10h"`` (cycle 20, detector report):
# SLSQP's forward-difference step. SciPy's default 1.49e-8 leaves a truncation-bias
# floor of (h^2 / 8) d^T H^-1 d, about 1e-5 on the discus f11; on calibration ids 1-3
# (D 5-20) a step of 1e-9 lifted restarted SLSQP by +.19 / +.22 / +.13-.22 / +.07-.10
# on f10 / f11 / f12 / f13-f14 at 100-1000 x D.
QN_FD_STEP = 1e-9
# The status-4 fallback (``Variant.qn_fallback="lbfgsb"``, cycle 28): once at least
# ``FALLBACK_MIN_DESCENTS`` SLSQP descents have completed and more of them failed
# (SciPy status != 0) than converged, the call's remaining evaluations go to
# restarted L-BFGS-B with ``LBFGSB_MAXCOR`` correction pairs and no tolerance stop
# (ftol = gtol = 0). In c31's own runs only f12 (bent cigar) shows the majority
# (cycle 27 status trace), where bounded SLSQP never converges at D 20-40.
FALLBACK_MIN_DESCENTS = 2
LBFGSB_MAXCOR = 10
# ``QnRoute.rugged_ipop`` (cycle 29): the budget left (x dim) from which a rugged
# reading runs IPOP from a uniform point (Hansen & Kern 2004's global structure
# needs wide, large populations; c31's plateau route for f7 is the template), and
# where it starts at 8x instead of 4x (one 8x run gave .97 / .88 on f17 / f18 at
# D 40, 3000 x D, against .78 / .58 for plain IPOP; COCO leaders report).
RUGGED_IPOP_MIN_LEFT_PER_DIM = 640
# ``rugged_ipop_low_dim`` (cycle 46): the low rugged route opens with this many x dim
# left (Batch 63's 2000, 3000 x D budgets)
RUGGED_IPOP_LOW_LEFT_PER_DIM = 2000.0
RUGGED_IPOP8_MIN_DIM = 20
RUGGED_IPOP8_MIN_LEFT_PER_DIM = 2500
# ``run_sampled_restarts`` (cycle 35, ``gal3c``; the multi-basin route's restarts):
# before each descent ``SAMPLE_PER_DIM`` x dim uniform points join a sample, and the
# descent starts from the best sample point farther than ``SAMPLE_GAP_SHARE`` x box
# width x sqrt(dim) from every earlier start and end; a descent above the incumbent
# stops once its last 3 (dim + 1) evaluations gained less than ``GAP_STOP_SHARE`` of
# its gap to the incumbent. Scratch (Gallagher, ids 1-3): 2-13x more global-basin
# hits per descent than uniform starts, descents 1.5-2x cheaper with the stop.
SAMPLE_PER_DIM = 10
SAMPLE_GAP_SHARE = 0.1
GAP_STOP_SHARE = 0.01
# ``micro_slopes`` (cycle 35, ``fract``): along ``MICRO_DIRECTIONS`` directions fixed
# per dimension, the absolute mirrored second difference at the scales ``MICRO_SCALES``
# x box width; its log10 slope between neighbouring scales is 2 where the function is
# smooth at those scales and about 1 on Katsuura's fractal (f23). A rugged reading is
# fractal when the median over directions of both of the two smallest slopes is below
# ``MICRO_FRACTAL_SLOPE``, unless at least ``MICRO_KINK_DIRECTIONS`` directions read a
# smallest slope within ``MICRO_KINK_SPREAD`` of exactly 1 (a kink in the probe line:
# f16-f18) or the smallest scale reads no difference at all (a plateau: f7). In c34's
# own rugged readings at D 20-40 (ids 1-3, 12 seeds) that flags 56 of 67 f23, 0 of 71
# f16 and 0 of 313 others; no f23 reading had two directions within .02 of 1.
MICRO_SCALES = (1e-7, 1e-6, 1e-5, 1e-4)
MICRO_DIRECTIONS = 4
MICRO_FRACTAL_SLOPE = 1.5
MICRO_KINK_SPREAD = 0.02
MICRO_KINK_DIRECTIONS = 2
MICRO_SEED = 4_243
# ``kink_check`` (cycle 30, Batch 56; reused in cycle 36): mirrored second
# differences at ``KINK_H_SHARE`` of the box width along ``KINK_DIRECTIONS`` directions
# fixed per dimension (no draws from the run's stream). On a kink the difference grows
# like 1 / h; on a smooth function it is the curvature. At c31's hand-over points
# (calibration ids 1-3, D 5-40, 300-1000 x D) f13 (sharp ridge) read 9.3e7-3.4e8 and
# every other bbob function at most 2e6 (one f16 run 2.1e7). ``KINK_D2`` is an
# absolute threshold tied to bbob's scaling.
KINK_H_SHARE = 1e-7
KINK_DIRECTIONS = 3
KINK_D2 = 5e7
KINK_SEED = 7_919
KINK_MIN_LEFT_PER_DIM = 100
HILL_VALLEY_POINTS = (0.25, 0.5, 0.75)
HILL_VALLEY_PAIRS = 3


@dataclass(frozen=True)
class Descent:
    """One completed SLSQP descent: its best point, that point's value and SciPy's
    exit status (0: converged; 4, 8, ...: stopped on a failure)."""

    x: np.ndarray
    value: float
    status: int


# ``cma_local_nonsep="qn10"`` (cycle 21, detector report's ``qnstall``): past the stage
# cap, the first SLSQP descent continues (no restart, so its quasi-Newton matrix is
# kept) while the last ``STALL_WINDOW`` iterations look healthy: median ratio of
# actual to predicted decrease at the first line-search trial at least ``STALL_RHO``
# and at most ``STALL_BACKTRACK`` of them backtracking. The report's thresholds (.25,
# .3) came from simplified functions; on bbob (calibration ids 1-2, D 20 / 40, 100 x D
# cap) progressing f8 / f9 descents read median rho .15-1.2 with 40-50% backtracks,
# and descents stuck at a floor (f11, f13) read -1 to -300 with 80-100%, so the
# thresholds are 0 and .6. A reserve of the budget stays with the CMA finisher.
STALL_WINDOW = 10
STALL_RHO = 0.0
STALL_BACKTRACK = 0.6
QN_RESERVE_SHARE = 0.1
QN_RESERVE_PER_DIM = 20.0
SCIPY_FD_STEP = 1.4901161193847656e-08  # SLSQP's default ``eps``


def _eps_option(fd_step: float | None) -> dict[str, float]:
    """SLSQP's ``eps`` option: ``fd_step``, or nothing (SciPy's own default)."""
    return {"eps": fd_step} if fd_step is not None else {}


def _step_range() -> tuple[float, float]:
    return SCALED_STEP_RANGE


class StallMonitor:
    """SLSQP's iterations rebuilt from its evaluation stream, at no extra cost.

    A forward-difference probe moves one coordinate of the current iterate by the
    finite-difference step; every other evaluation is a line-search trial. The
    first trial after a complete gradient opens an iteration, whose ratio
    rho = (f(trial) - f(iterate)) / (g . d / 2) compares the actual decrease with
    the quadratic model's prediction for the full step d; later trials before the
    next probes are backtracks.
    """

    def __init__(self, step: float) -> None:
        self.step = step
        self.base: tuple[np.ndarray, float] | None = None
        self.grad: np.ndarray | None = None
        self.probes = 0
        self.open: list[float] | None = None  # [rho, trials] of the current iteration
        self.records: list[tuple[float, int]] = []

    def observe(self, z: np.ndarray, value: float) -> None:
        if self.base is not None:
            diff = z - self.base[0]
            moved = np.flatnonzero(diff)
            if (
                len(moved) == 1
                and abs(abs(diff[moved[0]]) - self.step) <= 1e-3 * self.step
            ):
                if self.grad is None:
                    self.grad, self.probes = np.zeros(len(z)), 0
                self.grad[moved[0]] = (value - self.base[1]) / diff[moved[0]]
                self.probes += 1
                return
        if self.grad is not None and self.probes == len(z) and self.base is not None:
            if self.open is not None:
                self.records.append((self.open[0], int(self.open[1])))
            predicted = 0.5 * float(self.grad @ (z - self.base[0]))
            rho = (value - self.base[1]) / predicted if predicted < 0 else float("nan")
            self.open = [rho, 1]
            self.grad = None
        elif self.open is not None:
            self.open[1] += 1
        self.base = (np.array(z, dtype=float), float(value))

    def progressing(self) -> bool:
        recent = self.records[-STALL_WINDOW:]
        if len(recent) < STALL_WINDOW:
            return True
        rhos = np.array([r for r, _ in recent])
        rho = float(np.median(np.where(np.isfinite(rhos), rhos, -1.0)))
        backtracked = float(np.mean([t > 1 for _, t in recent]))
        return rho > STALL_RHO and backtracked <= STALL_BACKTRACK


@dataclass(frozen=True)
class SmallSwitch:
    """``small_switch`` (research option, from a study of budgets up to 10 x D).

    The prefix's first SLSQP descent is read at two checkpoints, the first probe of
    the gradient after its first and after its second line search (no extra
    evaluations). It stops, and restarted COBYQA (initial trust radius ``radius`` x
    the box width) spends the rest from the best point, at checkpoint 1 when the first
    line search took at least ``trials`` trials (0: off), or at checkpoint 2 when the
    two gradients are not parallel, ``|cos| < parallel_cos`` (0: off), unless
    ``keep_unit`` and the second line search accepted its first trial. ``rest``:
    COBYQA also instead of the prefix's (1+1) after a descent that ended before its
    cap or after the curvature probe's rugged reading.
    """

    trials: float = 0.0
    parallel_cos: float = 0.99
    keep_unit: bool = False
    radius: float = 0.25
    rest: bool = True


class SwitchReader:
    """``small_switch``'s reading of the first SLSQP descent (``StallMonitor``'s rule).

    :meth:`before` is asked before each evaluation and returns True to stop the
    descent (the evaluation is not made); :meth:`after` sees each evaluation.
    """

    def __init__(self, step: float, dim: int, rule: SmallSwitch) -> None:
        self.step = step
        self.dim = dim
        self.rule = rule
        self.reading = True
        self.base: tuple[np.ndarray, float] | None = None
        self.grad: np.ndarray | None = None
        self.probes = 0
        self.grads: list[np.ndarray] = []
        self.trials: list[int] = []
        self.current = 0
        self.stopped_at = 0

    def start_descent(self, index: int) -> None:
        """Only the first descent of the call is read."""
        self.reading = index == 0

    def _probe_axis(self, point: np.ndarray) -> int | None:
        if self.base is None:
            return None
        diff = point - self.base[0]
        moved = np.flatnonzero(diff)
        if len(moved) == 1 and abs(abs(diff[moved[0]]) - self.step) <= 1e-3 * self.step:
            return int(moved[0])
        return None

    def _switch(self, n_searches: int) -> bool:
        rule = self.rule
        if n_searches == 1:
            return rule.trials > 0 and self.trials[0] >= rule.trials
        if n_searches == 2 and rule.parallel_cos > 0:
            a, b = self.grads[1], self.grads[0]
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na <= 0 or nb <= 0:
                return False
            cos = float(a @ b) / (na * nb)
            if not -rule.parallel_cos < cos < rule.parallel_cos:
                return False
            return not (rule.keep_unit and self.trials[1] < 2)
        return False

    def before(self, point: np.ndarray) -> bool:
        if self.stopped_at:
            return True
        if not self.reading or self.grad is not None or not self.grads:
            return False
        if self._probe_axis(point) is None:
            return False
        # the first probe of a new gradient: the line search before it has ended
        self.trials.append(self.current)
        self.current = 0
        if self._switch(len(self.trials)):
            self.stopped_at = len(self.trials)
            return True
        return False

    def after(self, point: np.ndarray, value: float) -> None:
        if not self.reading:
            return
        axis = self._probe_axis(point)
        if axis is not None:
            assert self.base is not None
            if self.grad is None:
                self.grad, self.probes = np.zeros(self.dim), 0
            self.grad[axis] = (value - self.base[1]) / (
                point[axis] - self.base[0][axis]
            )
            self.probes += 1
            if self.probes == self.dim:
                self.grads.append(self.grad.copy())
            return
        self.grad, self.probes = None, 0
        self.base = (np.array(point, dtype=float), float(value))
        if self.grads:
            self.current += 1


class FlatGradientReader:
    """Reads whether a descent's first finite-difference gradient was exactly zero.

    Research (``prefix_descent_retries``, the parked2 toy): SLSQP's first ``dim + 1``
    evaluations are its start and the forward-difference probes of its first gradient;
    when all of them repeat one value the start sits on a plateau (bbob f7), a reading
    the descent already paid for. Same interface as :class:`SwitchReader`; it never
    stops a descent.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.reading = True
        self.values: list[float] = []

    def start_descent(self, index: int) -> None:
        self.reading = index == 0

    def before(self, point: np.ndarray) -> bool:
        return False

    def after(self, point: np.ndarray, value: float) -> None:
        if self.reading and len(self.values) <= self.dim:
            self.values.append(float(value))

    @property
    def flat(self) -> bool:
        """The first gradient was read and every one of its values tied."""
        v = self.values
        return len(v) == self.dim + 1 and all(x == v[0] for x in v)


# ``ridge_reading`` (the f13 toy, ``Variant.prefix_ridge_after``): on f13's sharp
# ridge every full SLSQP step overshoots across the ridge by a similar factor (the
# median ratio of actual to predicted decrease at the first trial is -1.7 to -3.0 at
# D 20 / 40, 30 x D, calibration ids 1-3) and about one backtrack recovers it (2.3-2.8
# trials per iteration). Rosenbrock, f6 and f20 overshoot far more (median ratio
# below -3.6), the rugged functions need 3.7-11 trials, and the smooth well-scaled
# ones rarely backtrack.
RIDGE_RHO = (-3.3, -1.5)
RIDGE_MAX_TRIALS = 3.2
RIDGE_MIN_BACKTRACK = 0.8


def ridge_reading(records: list[tuple[float, int]]) -> bool:
    """Whether a descent's iteration records (``StallMonitor.records``, the first
    left out) read a sharp ridge (``RIDGE_RHO``, ``RIDGE_MAX_TRIALS``,
    ``RIDGE_MIN_BACKTRACK``)."""
    recent = records[1:]
    if not recent:
        return False
    rhos = np.array([r if np.isfinite(r) else -1e9 for r, _ in recent])
    rho = float(np.median(rhos))
    trials = float(np.mean([t for _, t in recent]))
    backtracked = float(np.mean([t > 1 for _, t in recent]))
    return (
        RIDGE_RHO[0] <= rho <= RIDGE_RHO[1]
        and trials <= RIDGE_MAX_TRIALS
        and backtracked >= RIDGE_MIN_BACKTRACK
    )


def run_slsqp_restarts(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    box: Box,
    evaluations: int,
    ends_out: list[Descent] | None = None,
    max_descents: int | None = None,
    fd_step: float | None = None,
    extend_to: int | None = None,
    fallback: bool = False,
    gate_once: bool = False,
    gate_out: list[bool] | None = None,
    reader: SwitchReader | FlatGradientReader | None = None,
    ridge_after: int = 0,
    ridge_out: list[bool] | None = None,
) -> EsResult:
    """Spend up to ``evaluations`` on restarted SLSQP from ``x`` (cost ``cost``).

    Each restart starts from a uniform point. ``ends_out``, if given, receives a
    ``Descent`` for every completed descent (not for one cut short by the evaluation
    budget). ``max_descents``: stop after that many completed descents (cycle 17's
    single descent before the simplex). ``fd_step``: SLSQP's forward-difference
    step (SciPy's default 1.49e-8 when None; cycle 20).
    ``extend_to`` (cycle 21): the first descent may go on past ``evaluations``, up to
    ``extend_to``, while a ``StallMonitor`` reads it as progressing; no restart
    follows it. ``fallback`` (cycle 28): once failed descents outnumber converged
    ones (``failure_majority``), the rest goes to ``run_lbfgsb_restarts`` from the
    best point. ``gate_once`` (cycle 33, ``capgate``): with ``extend_to``, the
    ``StallMonitor`` is read once, when the first descent reaches ``evaluations``;
    its verdict (appended to ``gate_out``) holds for the rest of that descent.
    ``reader`` (``small_switch``, the small10 toy): reads the first descent and may
    stop the call at one of its checkpoints (``reader.stopped_at``).
    ``ridge_after`` (the f13 toy): when the first descent's ``StallMonitor`` has
    recorded that many iterations, ``ridge_reading`` reads them once (the verdict is
    appended to ``ridge_out``); a ridge stops the call there.

    The returned sigma (the scale ATM's tail starts from) is a tenth of the per-axis
    distance the best run travelled from its start, within [1e-8, 0.2] of the box
    width. Until 2026-10-18 it was 0, which pinned ATM's widths to their floor
    (Batches 31-32).
    """
    from scipy.optimize import minimize  # type: ignore[import-untyped]

    dim = len(x)
    bounds = list(zip(box.lower, box.upper, strict=True))
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    width = float(np.mean(box.upper - box.lower))
    used = 0
    start = best_x.copy()
    best_start = start
    ends: list[np.ndarray] = []
    statuses: list[int] = []
    fallback_sigma: float | None = None
    run_best: list[float | np.ndarray] = [np.inf, start]
    monitor = (
        StallMonitor(fd_step if fd_step is not None else SCIPY_FD_STEP)
        if extend_to is not None or ridge_after > 0
        else None
    )
    ridge: list[bool] = []

    gate: list[bool] = []

    def extend_ok() -> bool:
        assert monitor is not None
        if not gate_once:
            return monitor.progressing()
        if not gate:  # read once, at the cap
            gate.append(monitor.progressing())
            if gate_out is not None and not ends:
                gate_out.append(gate[0])
        return gate[0]

    def extending() -> bool:
        """Whether the descent now running may go past ``evaluations``."""
        return not ends

    def fun(z: np.ndarray) -> float:
        nonlocal used, best_x, best_cost, best_start
        if used >= evaluations and not (
            monitor is not None
            and extending()
            and extend_to is not None
            and used < extend_to
            and extend_ok()
        ):
            raise SpentError
        point = np.clip(z, box.lower, box.upper)
        if reader is not None and reader.before(point):
            raise SpentError
        value = objective.evaluate_point(point)
        if value is None:
            raise SpentError
        used += 1
        if reader is not None:
            reader.after(point, float(value))
        if monitor is not None and extending():
            monitor.observe(point, float(value))
        if value < best_cost:
            best_x, best_cost, best_start = point.copy(), float(value), start
        if value < run_best[0]:
            run_best[0], run_best[1] = float(value), point.copy()
        # The c45 candidate's composition order of the first descent's readers (each
        # only when on, so any one alone runs exactly as on its own branch):
        # ``reader`` (small_switch) before the evaluation, the ``StallMonitor`` after
        # it, then the ridge reading (``ridge_after``, once; a ridge stops the call)
        if (
            ridge_after > 0
            and monitor is not None
            and not ends
            and not ridge
            and len(monitor.records) >= ridge_after
        ):
            ridge.append(ridge_reading(monitor.records[:ridge_after]))
            if ridge_out is not None:
                ridge_out.append(ridge[0])
            if ridge[0]:
                raise SpentError
        return float(value) if np.isfinite(value) else 1e300

    while used < evaluations - dim - 1:
        if reader is not None:
            reader.start_descent(len(ends))
        try:
            with quiet_differences():
                res = minimize(
                    fun,
                    start,
                    method="SLSQP",
                    bounds=bounds,
                    options={"maxiter": evaluations, "ftol": FTOL}
                    | _eps_option(fd_step),
                )
        except SpentError:
            break
        ends.append(np.asarray(run_best[1]))
        status = int(res.status)
        statuses.append(status)
        if ends_out is not None:
            ends_out.append(
                Descent(np.asarray(run_best[1]), float(run_best[0]), status)
            )
        run_best[0] = np.inf
        if max_descents is not None and len(ends) >= max_descents:
            break
        if fallback and failure_majority(statuses):
            rest = run_lbfgsb_restarts(
                objective, rng, best_x, best_cost, box, evaluations - used
            )
            used += rest.used
            if rest.cost < best_cost:
                best_x, best_cost, fallback_sigma = rest.x, rest.cost, rest.sigma
            break
        start = box.lower + (box.upper - box.lower) * rng.rand(dim)
        run_best[1] = start
    if fallback_sigma is not None:
        return EsResult(best_x, best_cost, fallback_sigma, used)
    travelled = float(np.linalg.norm(best_x - best_start)) / np.sqrt(dim)
    sigma = float(np.clip(0.1 * travelled, 1e-8 * width, 0.2 * width))
    return EsResult(best_x, best_cost, sigma, used)


def kink_check(
    objective: CountedObjective, x: np.ndarray, cost: float, box: Box
) -> tuple[bool, float, np.ndarray, float] | None:
    """Whether ``x`` sits on a kink: (verdict, second difference, best point, value).

    For each of ``KINK_DIRECTIONS`` unit directions u (fixed per dimension), the
    mirrored second difference (f(x + h u) + f(x - h u) - 2 f(x)) / h^2 at h =
    ``KINK_H_SHARE`` x the mean box width (points clipped onto the box); a median
    absolute value of at least ``KINK_D2`` reads a kink. Spends 2 evaluations per
    direction. None if the objective ran out.
    """
    dim = len(x)
    h = KINK_H_SHARE * float(np.mean(box.upper - box.lower))
    gen = np.random.RandomState(KINK_SEED + dim)
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    seconds: list[float] = []
    for _ in range(KINK_DIRECTIONS):
        u = gen.randn(dim)
        u /= np.linalg.norm(u)
        values = []
        for sign in (1.0, -1.0):
            point = np.clip(x + sign * h * u, box.lower, box.upper)
            value = objective.evaluate_point(point)
            if value is None:
                return None
            values.append(float(value))
            if value < best_cost:
                best_x, best_cost = point, float(value)
        seconds.append((values[0] + values[1] - 2 * cost) / h**2)
    d2 = float(np.median(np.abs(seconds)))
    return d2 >= KINK_D2, d2, best_x, best_cost


def micro_slopes(
    objective: CountedObjective,
    x: np.ndarray,
    cost: float,
    box: Box,
    centered: bool = False,
) -> tuple[bool, tuple[float, ...], np.ndarray, float] | None:
    """Whether ``x`` reads fractal: (verdict, slopes, best point, its value).

    For each of ``MICRO_DIRECTIONS`` fixed unit directions u and each scale s in
    ``MICRO_SCALES`` (x the mean box width), |f(x + s u) + f(x - s u) - 2 f(x)|; a
    direction's slopes are log10 of the ratios between neighbouring scales (2 where f
    is smooth at those scales), and the returned slopes are their medians over the
    directions. Fractal when the first two medians are below ``MICRO_FRACTAL_SLOPE``,
    the smallest scale reads a difference (not a plateau) and fewer than
    ``MICRO_KINK_DIRECTIONS`` directions have a first slope within
    ``MICRO_KINK_SPREAD`` of 1 (a kink, whose second difference is exactly linear in
    the step). Spends 2 x scales x directions evaluations. None if the objective ran
    out.

    ``centered`` (cycle 37): a point within the largest scale of
    a bound is first moved inward by at most that scale (one more evaluation), so no
    probe point is clipped. A clipped probe is one-sided, and its second difference
    grows linearly in the step: Batch 59's f16 false exemptions sat on the bound.
    """
    dim = len(x)
    width = float(np.mean(box.upper - box.lower))
    gen = np.random.RandomState(MICRO_SEED + dim)
    dirs = gen.randn(MICRO_DIRECTIONS, dim)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    if centered:
        margin = MICRO_SCALES[-1] * width
        inner = np.clip(x, box.lower + margin, box.upper - margin)
        if not np.array_equal(inner, x):
            moved = objective.evaluate_point(inner)
            if moved is None:
                return None
            if moved < best_cost:
                best_x, best_cost = inner, float(moved)
            x, cost = inner, float(moved)
    table = np.zeros((MICRO_DIRECTIONS, len(MICRO_SCALES)))
    for j, share in enumerate(MICRO_SCALES):
        h = share * width
        for i, u in enumerate(dirs):
            values = []
            for sign in (1.0, -1.0):
                point = np.clip(x + sign * h * u, box.lower, box.upper)
                value = objective.evaluate_point(point)
                if value is None:
                    return None
                values.append(float(value))
                if value < best_cost:
                    best_x, best_cost = point, float(value)
            table[i, j] = abs(values[0] + values[1] - 2 * cost)
    floored = np.maximum(table, 1e-300)
    per_direction = np.log10(floored[:, 1:] / floored[:, :-1])
    slopes = tuple(float(s) for s in np.median(per_direction, axis=0))
    near_one = np.abs(per_direction[:, 0] - 1.0) < MICRO_KINK_SPREAD
    fractal = (
        slopes[0] < MICRO_FRACTAL_SLOPE
        and slopes[1] < MICRO_FRACTAL_SLOPE
        and float(np.max(table[:, 0])) > 0
        and int(near_one.sum()) < MICRO_KINK_DIRECTIONS
    )
    return fractal, slopes, best_x, best_cost


class _AbortDescentError(Exception):
    pass


# ``run_scaled_bfgs`` (cycle 44, idea #58 from the local-engine toy): a Lewis-Overton
# nonsmooth BFGS on forward differences taken along the eigenvectors of its inverse
# Hessian, each with step 2 sqrt(eps |f| mu_i) clipped to ``SCALED_STEP_RANGE``: long
# along a smooth valley, tiny across a ridge (f13), where no fixed step suits both.
# Calibration-id toy (cold start, D 5-20, 300-1000 x D): f13 1.00 in every cell
# against .55-.66 for SLSQP at a 1e-9 step; f12 +.53, f11 1.00; behind SLSQP at 30-100
# x D (f8, f10).
SCALED_STEP_RANGE = (1e-12, 1e-6)
_EPS = float(np.finfo(float).eps)


def _scaled_grad(
    fun: Any, x: np.ndarray, fx: float, inv_h: np.ndarray, box: Box
) -> np.ndarray:
    """Forward differences along the eigenvectors of the inverse Hessian ``inv_h``."""
    mu, vecs = np.linalg.eigh(0.5 * (inv_h + inv_h.T))
    steps = np.clip(
        2 * np.sqrt(_EPS * max(abs(fx), 1.0) * np.maximum(mu, 0.0)),
        *_step_range(),
    )
    lo, hi = box.lower[:, None], box.upper[:, None]
    plus = x[:, None] + steps * vecs
    minus = x[:, None] - steps * vecs
    ok_plus = np.all((plus >= lo) & (plus <= hi), axis=0)
    ok_minus = np.all((minus >= lo) & (minus <= hi), axis=0)
    if not np.all(
        ok_plus | ok_minus
    ):  # on a face: per-coordinate steps from diag inv_h
        diag = np.clip(
            2 * np.sqrt(_EPS * max(abs(fx), 1.0) * np.maximum(np.diag(inv_h), 0.0)),
            *_step_range(),
        )
        g = np.empty(len(x))
        for i in range(len(x)):
            z = x.copy()
            step = diag[i] if x[i] + diag[i] <= box.upper[i] else -diag[i]
            z[i] = x[i] + step
            g[i] = (fun(z) - fx) / step
        return g
    dd = np.empty(len(x))
    for i in range(len(x)):
        if ok_plus[i]:
            dd[i] = (fun(plus[:, i]) - fx) / steps[i]
        else:
            dd[i] = (fun(minus[:, i]) - fx) / -steps[i]
    return np.asarray(vecs @ dd, dtype=float)


def _box_qp(g: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """argmin_d g.d + d.B.d / 2 over lo <= d <= hi (lo <= 0 <= hi), B SPD.

    A primal active-set method from d = 0 (the BFGS engine toy, 2026-10-21): solve on
    the free set with the bound coordinates fixed, step to the first blocking bound and
    add it; once the free solve is feasible, release the bound with the most negative
    multiplier or stop. On 300 random box QPs (D 2-40, condition up to 1e6) the toy's
    copy was always feasible and never worse than L-BFGS-B's solution by more than
    3.2e-12 relative.
    """
    n = len(g)
    d: np.ndarray = np.zeros(n)
    at_lo = np.zeros(n, dtype=bool)
    at_hi = np.zeros(n, dtype=bool)
    for _ in range(6 * n + 10):
        fixed = at_lo | at_hi
        free = ~fixed
        target = d.copy()
        if free.any():
            rhs = -(g[free] + b[np.ix_(free, fixed)] @ d[fixed])
            try:
                target[free] = np.linalg.solve(b[np.ix_(free, free)], rhs)
            except np.linalg.LinAlgError:
                return d
        step = target - d
        # ratio test: the largest alpha in [0, 1] keeping d + alpha step in the box
        alpha, block = 1.0, -1
        for i in np.flatnonzero(free & (step != 0)):
            lim = (hi[i] - d[i]) / step[i] if step[i] > 0 else (lo[i] - d[i]) / step[i]
            if lim < alpha:
                alpha, block = max(lim, 0.0), int(i)
        d = d + alpha * step
        if block >= 0:
            if step[block] > 0:
                d[block], at_hi[block] = hi[block], True
            else:
                d[block], at_lo[block] = lo[block], True
            continue
        # free solve feasible: a bound coordinate whose multiplier has the wrong sign
        # (the model decreases inward) is released
        grad = g + b @ d
        viol = np.minimum(
            np.where(at_lo & (grad < 0), grad, 0.0),
            np.where(at_hi & (grad > 0), -grad, 0.0),
        )
        worst = int(np.argmin(viol))
        if viol[worst] >= 0:
            return d
        at_lo[worst] = at_hi[worst] = False
    return d


# ``run_scaled_bfgs``'s line search: Armijo and weak-Wolfe constants, and the caps on
# its doubling and bisection steps
BFGS_C1, BFGS_C2 = 1e-4, 0.5
BFGS_MAX_EXPAND, BFGS_MAX_BISECT = 30, 40


def run_scaled_bfgs(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    box: Box,
    evaluations: int,
    max_descents: int | None = None,
    box_qp: bool = False,
    extend_to: int | None = None,
    extend_gate: Any = None,
) -> EsResult:
    """Restarted scaled-FD nonsmooth BFGS (``run_scaled_bfgs``), the first from ``x``.

    Each descent: an identity inverse Hessian, a projected quasi-Newton direction, an
    Armijo plus weak-Wolfe bracketing line search (doubling, then bisection; the
    slope is one forward difference along the direction), the BFGS update when
    s.y > 0, and no small-step
    stop: a descent ends only when the line search finds no Armijo point or the
    projected direction is not a descent direction. Later descents start from uniform
    points. ``box_qp`` (cycle 59): where -H g leaves the box, the direction is the
    quadratic model's minimiser over the box (``_box_qp``), as in SLSQP. Returns the
    best point, its value, a sigma from the first descent's
    travel (as ``run_slsqp_restarts``) and the evaluations used.
    ``extend_to`` (the f13 toy, ``Variant.rugged_try_rest``): a descent still running
    when ``evaluations`` are used goes on, the same descent, up to ``extend_to``
    evaluations, if ``extend_gate(best_x, best_cost)`` (read once; None: always) says
    so.
    """
    dim = len(x)
    width = float(np.mean(box.upper - box.lower))
    used = 0
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    start, start_cost = best_x.copy(), best_cost
    first_end = best_x.copy()
    limit = [evaluations]
    extended: list[bool] = []

    def fun(point: np.ndarray) -> float:
        nonlocal used, best_x, best_cost
        if used >= limit[0] and extend_to is not None and not extended:
            extended.append(extend_gate is None or bool(extend_gate(best_x, best_cost)))
            if extended[0]:
                limit[0] = extend_to
        if used >= limit[0]:
            raise SpentError
        value = objective.evaluate_point(point)
        if value is None:
            raise SpentError
        used += 1
        if value < best_cost:
            best_x, best_cost = point.copy(), float(value)
        return float(value) if np.isfinite(value) else 1e300

    def free_dir(z: np.ndarray, d: np.ndarray) -> np.ndarray:
        out = np.array(d, dtype=float)
        out[(z <= box.lower) & (d < 0)] = 0.0
        out[(z >= box.upper) & (d > 0)] = 0.0
        return out

    descents = 0
    try:
        while max_descents is None or descents < max_descents:
            z = np.clip(start, box.lower, box.upper)
            f = start_cost if np.isfinite(start_cost) else fun(z)
            g = np.empty(dim)
            for i in range(dim):  # the first gradient: forward steps of 1e-9
                w = z.copy()
                step = 1e-9 if z[i] + 1e-9 <= box.upper[i] else -1e-9
                w[i] = z[i] + step
                g[i] = (fun(w) - f) / step
            inv_h = np.eye(dim)
            while True:
                d = -inv_h @ g
                if not box_qp:
                    d = free_dir(z, d)
                elif np.any((z + d < box.lower) | (z + d > box.upper)):
                    # ``box_qp`` (cycle 59, the engine toy): SLSQP's bound handling.
                    # The model's minimiser over the box replaces zeroing the components
                    # that point out of a face, so t <= 1 stays inside the box and no H
                    # reset is needed at a face (f10: 4-20 resets per run, all at a
                    # face, and 12-13 bisections per line search)
                    try:
                        b = np.linalg.inv(0.5 * (inv_h + inv_h.T))
                        d = _box_qp(g, 0.5 * (b + b.T), box.lower - z, box.upper - z)
                    except np.linalg.LinAlgError:
                        d = free_dir(z, d)
                gd = float(g @ d)
                if not gd < 0:
                    inv_h = np.eye(dim)
                    d = free_dir(z, -g)
                    gd = float(g @ d)
                    if not gd < 0:
                        break
                lo, hi_t, t = 0.0, np.inf, 1.0
                lo_state = accepted = None
                expands = bisects = 0
                dnorm = max(float(np.linalg.norm(d)), 1e-300)
                u = d / dnorm
                hu = 2 * np.sqrt(
                    _EPS * max(abs(f), 1.0) * max(float(u @ inv_h @ u), 0.0)
                )
                tau = float(np.clip(hu, *_step_range())) / dnorm
                while True:
                    zt = np.clip(z + t * d, box.lower, box.upper)
                    moved = not np.array_equal(zt, z)
                    ft = fun(zt) if moved else f
                    if not moved or ft >= f + BFGS_C1 * float(g @ (zt - z)):
                        hi_t = t
                    else:
                        zs = np.clip(z + (t + tau) * d, box.lower, box.upper)
                        slope = 0.0 if np.array_equal(zs, zt) else (fun(zs) - ft) / tau
                        if slope >= BFGS_C2 * gd:
                            accepted = (zt, ft)
                            break
                        lo, lo_state = t, (zt, ft)
                    if hi_t < np.inf:
                        bisects += 1
                        if bisects > BFGS_MAX_BISECT:
                            break
                        t = 0.5 * (lo + hi_t)
                    else:
                        expands += 1
                        if expands > BFGS_MAX_EXPAND:
                            break
                        t *= 2.0
                if accepted is None:
                    accepted = lo_state
                if accepted is None:
                    break
                zn, fn = accepted
                gn = _scaled_grad(fun, zn, fn, inv_h, box)
                s, y = zn - z, gn - g
                sy = float(s @ y)
                if sy > 0 and np.isfinite(sy):
                    rho = 1.0 / sy
                    hy = inv_h @ y
                    inv_h = (
                        inv_h
                        - rho * (np.outer(s, hy) + np.outer(hy, s))
                        + (rho * rho * float(y @ hy) + rho) * np.outer(s, s)
                    )
                z, f, g = zn, fn, gn
            descents += 1
            if descents == 1:
                first_end = z.copy()
            start = box.lower + (box.upper - box.lower) * rng.rand(dim)
            start_cost = np.inf
    except SpentError:
        pass
    travelled = float(np.linalg.norm(first_end - x)) / np.sqrt(dim)
    sigma = float(np.clip(0.1 * travelled, 1e-8 * width, 0.2 * width))
    return EsResult(best_x, best_cost, sigma, used)


def _clear_of(
    point: np.ndarray,
    claimed: list[np.ndarray],
    claimed_rows: np.ndarray,
    first: int,
    radius: float,
) -> bool:
    """``all(np.linalg.norm(point - c) > radius for c in claimed[first:])``, faster.

    The distances come from one vectorised pass (``claimed_rows`` holds the claimed
    points as rows). Its sum runs in another order than ``np.linalg.norm``'s, so a
    distance within a relative 1e-9 of ``radius`` (or NaN) is recomputed with
    ``np.linalg.norm`` and decided as there; every other one is far from where
    rounding could matter.
    """
    rows = claimed_rows[first : len(claimed)]
    if not len(rows):
        return True
    distance = np.sqrt(((rows - point) ** 2).sum(axis=1))
    if np.any(distance < radius * (1 - 1e-9)):
        return False
    unsure = np.flatnonzero(~(distance > radius * (1 + 1e-9)))
    return all(
        np.linalg.norm(point - claimed[first + j]) > radius for j in unsure.tolist()
    )


def run_sampled_restarts(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    box: Box,
    evaluations: int,
    fd_step: float | None = None,
    samples: Any = None,  # (points, values) | None (Any for the callers' **options)
) -> EsResult:
    """Restarted SLSQP with sampled starts and a gap-scaled stop (``gal3c``).

    Before each descent ``SAMPLE_PER_DIM`` x dim uniform points are evaluated and
    join a sample; the descent starts from the best sample point farther than
    ``SAMPLE_GAP_SHARE`` x box width x sqrt(dim) from ``x`` and from every earlier
    start and end (a uniform point if none is). A descent whose best is above the
    incumbent at its start stops once its last 3 (dim + 1) evaluations gained less
    than ``GAP_STOP_SHARE`` of that gap (checked every dim + 1 evaluations), so
    descents into known or worse basins end early and the incumbent's basin keeps
    its full precision. The returned sigma follows ``run_slsqp_restarts``.

    ``samples`` (the anytime hand-over): points and values evaluated before this
    call, in the sample from the start (they cost nothing here).

    The start is found incrementally (post-freeze, 2026-10-23; the same starts as
    scanning every sample in sorted order against every claimed point, which cost
    O(descents^2) per descent and held FT4's 10^5 x D runs at D 20-40 to a few
    evaluations per second): claimed points are never removed, so a sample within
    the radius of one is out for good, and a sample clear of the first ``seen[k]``
    claimed points is only checked against the later ones (``_clear_of``).
    """
    from scipy.optimize import minimize

    dim = len(x)
    bounds = list(zip(box.lower, box.upper, strict=True))
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    width = float(np.mean(box.upper - box.lower))
    used = 0
    best_start = best_x.copy()
    sample_x: list[np.ndarray] = []
    sample_f: list[float] = []
    claimed: list[np.ndarray] = []
    claimed_rows = np.empty((64, dim))
    # Samples not yet within the radius of a claimed point, by (NaN last, value,
    # index): the order of ``np.argsort(sample_f, kind="stable")``.
    queue: list[tuple[bool, float, int]] = []
    seen: list[int] = []  # per sample: how many claimed points it is clear of
    radius = SAMPLE_GAP_SHARE * width * np.sqrt(dim)
    window = 3 * (dim + 1)

    def claim(point: np.ndarray) -> None:
        nonlocal claimed_rows
        if len(claimed) == len(claimed_rows):
            claimed_rows = np.concatenate((claimed_rows, np.empty_like(claimed_rows)))
        claimed_rows[len(claimed)] = point
        claimed.append(point)

    claim(best_x.copy())
    if samples is not None:  # the anytime hand-over: earlier stages' points join
        for point, value in zip(*samples, strict=True):
            missing = value != value
            key = (missing, 0.0 if missing else float(value), len(sample_f))
            heapq.heappush(queue, key)
            sample_f.append(float(value))
            sample_x.append(np.array(point, dtype=float))
            seen.append(0)

    def evaluate(point: np.ndarray) -> float:
        nonlocal used, best_x, best_cost
        if used >= evaluations:
            raise SpentError
        value = objective.evaluate_point(point)
        if value is None:
            raise SpentError
        used += 1
        if value < best_cost:
            best_x, best_cost = point.copy(), float(value)
        return float(value)

    while used < evaluations - dim - 1:
        try:
            for _ in range(SAMPLE_PER_DIM * dim):
                point = box.lower + (box.upper - box.lower) * rng.rand(dim)
                value = evaluate(point)
                missing = value != value
                key = (missing, 0.0 if missing else value, len(sample_f))
                heapq.heappush(queue, key)
                sample_f.append(value)
                sample_x.append(point)
                seen.append(0)
        except SpentError:
            break
        start = None
        while queue:
            k = queue[0][2]
            if _clear_of(sample_x[k], claimed, claimed_rows, seen[k], radius):
                seen[k] = len(claimed)
                start = sample_x[k]
                break
            heapq.heappop(queue)
        if start is None:
            start = box.lower + (box.upper - box.lower) * rng.rand(dim)
        claim(np.array(start))
        incumbent = best_cost
        local: dict[str, Any] = {"best": np.inf, "x": np.array(start), "hist": []}

        def fun(
            z: np.ndarray,
            local: dict[str, Any] = local,
            incumbent: float = incumbent,
            start: np.ndarray = start,
        ) -> float:
            nonlocal best_start
            point = np.clip(z, box.lower, box.upper)
            before = best_cost
            value = evaluate(point)
            if best_cost < before:
                best_start = start
            if value < local["best"]:
                local["best"], local["x"] = value, point.copy()
            local["hist"].append(local["best"])
            k = len(local["hist"])
            if k > window and k % (dim + 1) == 0 and local["best"] > incumbent:
                gained = local["hist"][-window] - local["best"]
                if gained < GAP_STOP_SHARE * (local["best"] - incumbent):
                    raise _AbortDescentError
            return value if np.isfinite(value) else 1e300

        try:
            with quiet_differences():
                minimize(
                    fun,
                    start,
                    method="SLSQP",
                    bounds=bounds,
                    options={"maxiter": evaluations, "ftol": FTOL}
                    | _eps_option(fd_step),
                )
        except _AbortDescentError:
            pass
        except SpentError:
            break
        claim(np.array(local["x"]))
    travelled = float(np.linalg.norm(best_x - best_start)) / np.sqrt(dim)
    sigma = float(np.clip(0.1 * travelled, 1e-8 * width, 0.2 * width))
    return EsResult(best_x, best_cost, sigma, used)


def quiet_differences() -> np.errstate:
    """``np.errstate`` for SciPy's quasi-Newton methods: their finite differences
    over the 1e300 that stands for a failed evaluation overflow, and the warning
    tells the caller nothing (review B5). Only warnings change, never a value; the
    objective's own overflow and divide warnings are silenced inside it too."""
    return np.errstate(over="ignore", divide="ignore")


def failure_majority(statuses: list[int]) -> bool:
    """At least ``FALLBACK_MIN_DESCENTS`` descents, and more failed than converged."""
    failed = sum(s != 0 for s in statuses)
    return len(statuses) >= FALLBACK_MIN_DESCENTS and failed > len(statuses) - failed


def run_lbfgsb_restarts(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    box: Box,
    evaluations: int,
    fd_step: float = QN_FD_STEP,
) -> EsResult:
    """Spend up to ``evaluations`` on restarted L-BFGS-B from ``x`` (cost ``cost``).

    SciPy's L-BFGS-B with the box as bounds, ``LBFGSB_MAXCOR`` correction pairs, a
    forward-difference gradient with step ``fd_step``, and ftol = gtol = 0: a
    descent ends only on a line-search failure or a zero projected gradient
    (SciPy's ftol is relative to max(|f|, 1), which stops descents early on
    instances with a large |f_opt|). Later starts are uniform in the box. The
    returned sigma follows ``run_slsqp_restarts``.
    """
    from scipy.optimize import minimize

    dim = len(x)
    bounds = list(zip(box.lower, box.upper, strict=True))
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    width = float(np.mean(box.upper - box.lower))
    used = 0
    start = best_x.copy()
    best_start = start

    def fun(z: np.ndarray) -> float:
        nonlocal used, best_x, best_cost, best_start
        if used >= evaluations:
            raise SpentError
        point = np.clip(z, box.lower, box.upper)
        value = objective.evaluate_point(point)
        if value is None:
            raise SpentError
        used += 1
        if value < best_cost:
            best_x, best_cost, best_start = point.copy(), float(value), start
        return float(value) if np.isfinite(value) else 1e300

    while used < evaluations - dim - 1:
        try:
            with quiet_differences():
                minimize(
                    fun,
                    start,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxcor": LBFGSB_MAXCOR,
                        "ftol": 0.0,
                        "gtol": 0.0,
                        "eps": fd_step,
                        "maxfun": evaluations,
                        "maxiter": evaluations,
                    },
                )
        except SpentError:
            break
        start = box.lower + (box.upper - box.lower) * rng.rand(dim)
    travelled = float(np.linalg.norm(best_x - best_start)) / np.sqrt(dim)
    sigma = float(np.clip(0.1 * travelled, 1e-8 * width, 0.2 * width))
    return EsResult(best_x, best_cost, sigma, used)


def curvature_beta(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    box: Box,
    directions: int = BETA_DIRECTIONS,
) -> tuple[float, np.ndarray, float, int] | None:
    """beta at ``x``: the log-log slope of the mirrored loss against the probe scale.

    Mirrored pairs ``x +- h u`` (``directions`` random unit directions, ``h`` in
    ``BETA_SCALES`` times the box width, clipped onto the box). Returns (beta, best
    point, its value, evaluations used), or None if the objective ran out. beta is
    NaN with fewer than two positive losses.
    """
    dim = len(x)
    width = float(np.mean(box.upper - box.lower))
    dirs = rng.randn(directions, dim)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    used = 0
    losses = []
    for share in BETA_SCALES:
        loss = []
        for u in dirs:
            pair = []
            for sign in (1.0, -1.0):
                point = np.clip(x + sign * share * width * u, box.lower, box.upper)
                value = objective.evaluate_point(point)
                if value is None:
                    return None
                used += 1
                if value < best_cost:
                    best_x, best_cost = point, float(value)
                pair.append(float(value))
            loss.append(0.5 * (pair[0] + pair[1]) - float(cost))
        losses.append(float(np.mean(loss)))
    h = np.array(BETA_SCALES)
    l_arr = np.array(losses)
    ok = np.isfinite(l_arr) & (l_arr > 0)
    beta = (
        float(np.polyfit(np.log10(h[ok]), np.log10(l_arr[ok]), 1)[0])
        if ok.sum() >= 2
        else float("nan")
    )
    return beta, best_x, best_cost, used


def plateau_check(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    box: Box,
) -> tuple[bool, int] | None:
    """True if at least half of ``2 * TIE_DIRECTIONS`` tiny mirrored probes repeat
    ``cost`` exactly (a plateau function such as the step ellipsoid); also returns the
    evaluations used. None if the objective ran out. The probe points are never
    better than ``x`` by more than rounding, so the incumbent is kept.
    """
    dim = len(x)
    width = float(np.mean(box.upper - box.lower))
    dirs = rng.randn(TIE_DIRECTIONS, dim)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ties = used = 0
    for u in dirs:
        for sign in (1.0, -1.0):
            point = np.clip(x + sign * TIE_SCALE * width * u, box.lower, box.upper)
            value = objective.evaluate_point(point)
            if value is None:
                return None
            used += 1
            ties += int(value == cost)
    return 2 * ties >= used, used


def multi_basin(
    objective: CountedObjective, descents: list[Descent], box: Box
) -> bool | None:
    """True if the converged descents found at least two separate basins.

    Hill-valley test (Maree, Alderliesten & Bosman, HillVallEA, GECCO 2018): the best
    converged end and up to ``HILL_VALLEY_PAIRS`` others farther than the seed gap
    are in different basins if a point at ``HILL_VALLEY_POINTS`` of the segment
    between them is worse than both ends. A convex function never passes, however
    far apart its ends are (f10's early stops along flat axes). Needs at least two
    converged descents and no more failed descents than converged ones (f12 fails
    most descents with status 4). None if the objective ran out.
    """
    converged = sorted((d for d in descents if d.status == 0), key=lambda d: d.value)
    if len(converged) < 2 or 2 * len(converged) < len(descents):
        return False
    best = converged[0]
    width = float(np.mean(box.upper - box.lower))
    gap = SEED_GAP * width * np.sqrt(len(best.x))
    others = [d for d in converged[1:] if np.linalg.norm(d.x - best.x) > gap]
    for other in others[:HILL_VALLEY_PAIRS]:
        top = max(best.value, other.value)
        for t in HILL_VALLEY_POINTS:
            value = objective.evaluate_point(best.x + t * (other.x - best.x))
            if value is None:
                return None
            if value > top + 1e-12 * max(abs(top), 1.0):
                return True
    return False


@dataclass(frozen=True)
class QnRoute:
    """The switches of one SLSQP route of ``cma_local_nonsep``.

    ``api._qn_local_phase`` runs the steps in order; each switch turns one on:

    Every route first runs tiny mirrored probes (``plateau_check``; a plateau skips
    the curvature probe), and on a smooth reading SLSQP runs for ``QN_LOCAL_SHARE`` of
    the local budget, at most ``QN_LOCAL_CAP_PER_DIM`` x dim, before the CMA finisher
    continues from its best point at its hand-over sigma (``"qn3"``).

    * ``extend``: a stage whose first descent is still running at the cap continues
      for ``QN_EXTEND_SHARE`` of what is left, for at most another capped stage
      (``"capped"``, ``"qn7"``), or not (``"none"``).
    * ``basins``: converged descents in separate basins (``multi_basin``) keep
      restarting SLSQP to the end (from ``"qn6"``).
    * ``rugged_steps``: a rugged or plateau reading gives the finisher ``steps``
      populations (from ``"qn6"``).
    * ``plateau_ipop4``: a plateau runs IPOP populations with 4x the default lambda
      from a uniform point to the end (from ``"qn8"``).
    * ``fd_step``: the stage and its extension use SLSQP with the finite-difference
      step ``QN_FD_STEP`` instead of SciPy's (from ``"qn9"``, cycle 20).
    * ``extend_in_descent``: the stage's first descent goes on past the cap while a
      ``StallMonitor`` reads it as progressing, keeping a reserve for the finisher,
      instead of the separate extension (cycle 21).
    * ``basins_scipy_step``: the multi-basin SLSQP restarts use SciPy's step, for
      more and cheaper descents (from ``"qn12"``, cycle 24).
    * ``rugged_ipop``: a rugged reading with at least ``RUGGED_IPOP_MIN_LEFT_PER_DIM``
      x dim left runs the plateau route's schedule instead of the finisher: IPOP
      populations from a uniform point at sigma0 = box / 5, 4x the default lambda
      doubling per restart (8x from ``RUGGED_IPOP8_MIN_DIM`` with at least
      ``RUGGED_IPOP8_MIN_LEFT_PER_DIM`` x dim left), no protection, to the end
      (cycle 29; COCO leaders report rank 2). ``rugged_ipop_min_dim``:
      only from that dimension (``"qn17"``: D >= 20, cycle 31; Batch 55 lost f15-f16
      at D 5 and f18 at D 10, where c31's (1+1)-first path is strong, and no in-state
      probe separates the fractal f23 from Schaffer's f17-f18).
    * ``cap_gate`` (with ``extend_in_descent``; cycle 33): the stage's
      first descent, still running at the cap, goes on to SciPy's own exit (or the
      budget minus the reserve) when the ``StallMonitor`` reads its last iterations
      as progressing at the cap, read once (``run_slsqp_restarts``'s ``gate_once``);
      otherwise, or when the extended descent ends on a failure (f12's fallback),
      the capped continuation runs as in ``"qn17"``.
    * ``second_descent_min_left_per_dim`` (``"qn19"``): after a gated descent that
      converged with at least this many x dim evaluations left, one more SLSQP
      descent from a uniform point before the finisher (Rosenbrock's second minimum).
    * ``basins_sampled`` (``"qn20"``, cycle 35, ``gal3c``): the multi-basin restarts
      are ``run_sampled_restarts`` (sampled starts and the gap-scaled stop).
    * ``fractal_exempt`` (``"qn29"``, cycles 35 and 37, ``fract2``): before the rugged
      route's populations, ``micro_slopes`` tests the reading (centred: the probe
      moved off the bounds first); a fractal one (f23-like) runs restarted SLSQP
      (SciPy's step) for the rest instead.
    * ``kink_small_step`` (``"qn38"``, cycles 36, 44 and 47, f13): where the cap gate
      read the stage's first descent as not progressing, from ``kink_min_dim``,
      ``kink_check`` tests the hand-over point; a kink with at least
      ``KINK_MIN_LEFT_PER_DIM`` x dim left runs the scaled-FD BFGS for the rest
      instead of the finisher.
    """

    extend: str = "none"
    basins: bool = False
    rugged_steps: bool = False
    plateau_ipop4: bool = False
    fd_step: bool = False
    extend_in_descent: bool = False
    basins_scipy_step: bool = False
    rugged_ipop: bool = False
    rugged_ipop_min_dim: int = 0
    cap_gate: bool = False
    second_descent_min_left_per_dim: float = 0.0
    basins_sampled: bool = False
    fractal_exempt: bool = False
    kink_small_step: bool = False
    kink_min_dim: int = 20
    rugged_try_per_dim: float = 0.0
    rugged_try_max_dim: int = 0
    rugged_try_max_budget_per_dim: float = 0.0
    rugged_ipop_low_dim: int = 0
    smooth_bfgs_max_per_dim: float = 0.0
    smooth_bfgs_descents: int = 0
    smooth_bfgs_box_qp: bool = False

    def __post_init__(self) -> None:
        if self.extend not in ("none", "capped"):
            raise ValueError(f"extend must be none or capped, got {self.extend!r}")


# ``Variant.cma_local_nonsep``'s SLSQP routes (cycles 13-25). A new route that
# recombines these steps is one row here (and its name in ``config.py``'s choices).
_QN8 = QnRoute(extend="capped", basins=True, rugged_steps=True, plateau_ipop4=True)
_QN12 = replace(_QN8, fd_step=True, basins_scipy_step=True)
_QN17 = replace(_QN12, rugged_ipop=True, rugged_ipop_min_dim=20)
_QN18 = replace(_QN17, extend_in_descent=True, cap_gate=True)
_QN19 = replace(_QN18, second_descent_min_left_per_dim=400.0)
_QN29 = replace(_QN19, basins_sampled=True, fractal_exempt=True)
QN_ROUTES: dict[str, QnRoute] = {
    "qn3": QnRoute(),
    "qn7": QnRoute(extend="capped", basins=True, rugged_steps=True),
    "qn8": _QN8,
    "qn9": replace(_QN8, fd_step=True),
    "qn12": _QN12,
    "qn17": _QN17,
    "qn19": _QN19,
    "qn20": replace(_QN19, basins_sampled=True),
    # cycle 39's promoted route: ``gal3c``'s sampled restarts plus ``fract2`` (the
    # centred fractal probe) on ``qn19``; cycle 36's f13 kink route (``qn24``, SLSQP
    # at a smaller step) did not confirm twice and left with dead code 7
    "qn29": _QN29,
    # cycle 48 (Batch 65): ``kinklow`` + ``tryd5`` + ``lowipop`` on one route. The
    # try (D <= 9, up to 500 x D) and the low rugged IPOP (2000 x D left) never meet
    "qn38": replace(
        _QN29,
        kink_small_step=True,
        kink_min_dim=5,
        rugged_try_per_dim=100.0,
        rugged_try_max_dim=9,
        rugged_try_max_budget_per_dim=500.0,
        rugged_ipop_low_dim=5,
    ),
    # cycle 59: on a smooth reading up to 300 x D, one descent of the scaled-FD BFGS
    # with SLSQP's bound handling (``box_qp``; the engine toy's f10 fix), then the usual
    # staged route from its best point (the parked study's ``pk40d1``: the descent's
    # own end decides, so Gallagher keeps its basin restarts)
    "qn44": replace(
        _QN29,
        kink_small_step=True,
        kink_min_dim=5,
        rugged_try_per_dim=100.0,
        rugged_try_max_dim=9,
        rugged_try_max_budget_per_dim=500.0,
        rugged_ipop_low_dim=5,
        smooth_bfgs_max_per_dim=300.0,
        smooth_bfgs_box_qp=True,
        smooth_bfgs_descents=1,
    ),
}
