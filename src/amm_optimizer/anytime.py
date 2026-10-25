"""The anytime mode (``Variant.anytime="replay"``): plan small first, then hand over.

A configuration plans every phase from the budget it is given, so a run cut short
of its budget is not the run that budget would have made. The anytime mode does
not plan the whole budget at the start.

**Replay, up to ``anytime_switch`` evaluations (10 000).**
* Stage ``k`` plans ``anytime_start_per_dim`` x ``anytime_factor``^k x dim
  evaluations in total: 10, 20, 40, ... x dim.
* Each stage reruns the configuration (with ``anytime="off"``) from the same start
  point and seed for its larger plan. A point the previous stage evaluated is
  answered from memory, without a call, and still counts against the stage's plan,
  since the planned budget is cumulative.
* Whatever the larger plan does as the smaller one did (probes, designs, descents,
  ATM's and CMA's states) is therefore rebuilt exactly and at no cost. From the
  first decision the larger budget changes, the larger plan takes over.
* At the end of stage ``k`` the run has been exactly the configuration's run at
  ``B_k``. Until the hand-over, the run never reads ``budget``, which only caps the
  calls.

**The hand-over.** Once the next plan would exceed ``anytime_switch``, one
ordinary run of the configuration gets every evaluation left of ``budget``: the
normal path, which knows its budget. It starts from ``x0``. Every point evaluated
before is fed into it (``Warm``):
* the phases that would draw fresh points start from that archive (the prefix's
  design, ATM's first pool, the first CMA run's shape, the sampled restarts'
  sample);
* a point it asks for again is answered from memory.

Only the points of the replay part are kept (a few thousand per D), so the memory
stays small whatever the budget.

The objective must be deterministic: a remembered value stands for a new call.
``STAGE_LOG`` (one dict per stage, cleared by nobody) records what each stage
planned, called and replayed, for the benchmark's diagnostics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np

from .config import ATMConfig
from .profiling import Profiler, resolve_profiler
from .result import OptimizeResult

STAGE_LOG: list[dict[str, Any]] = []
# Where the hand-over run starts: ``"start"`` (the run's own x0) or ``"best"``
# (the best point of the replay part). Fixed by the calibration of 2026-10-24
# (experiment 106; see ``docs/ANYTIME.md``).
HANDOVER_FROM = "start"


@dataclasses.dataclass(frozen=True)
class Warm:
    """What the replay part evaluated: points as rows of ``x``, their values ``f``,
    in the order they were called (the hand-over run's archive)."""

    x: np.ndarray
    f: np.ndarray

    def _order(self) -> np.ndarray:
        return np.argsort(np.where(np.isnan(self.f), np.inf, self.f), kind="stable")

    @property
    def best_x(self) -> np.ndarray:
        return np.asarray(self.x[self._order()[0]])

    @property
    def best_f(self) -> float:
        return float(self.f[self._order()[0]])

    def best(self, count: int) -> np.ndarray:
        """The ``count`` best points (the first of equal values first)."""
        return np.asarray(self.x[self._order()[:count]])

    def mapped(self, lo: np.ndarray, span: np.ndarray) -> Warm:
        """The same points in ``box_normalize``'s coordinates ([-5, 5]^D)."""
        return Warm((self.x - lo) / span - 5.0, self.f)


class _CapReachedError(Exception):
    """The calls allowed by ``budget`` are spent (raised to end the run)."""


class _Store:
    """The objective as the stages see it: counts real calls, keeps the best point
    and the replay part's points, and answers remembered points from memory."""

    def __init__(self, fun: Callable[[np.ndarray], float], cap: int) -> None:
        self.fun = fun
        self.cap = cap
        self.calls = 0
        self.best_x: np.ndarray | None = None
        self.best_f = float("inf")
        self.earlier: dict[bytes, float] = {}  # what the previous stage asked for
        self.current: dict[bytes, float] = {}
        self.replayed = self.requested = self.remembered = 0
        self.xs: list[np.ndarray] = []  # every point called in the replay part
        self.fs: list[float] = []
        self.memory: dict[bytes, float] = {}
        self.keeping = True  # off for the hand-over: nothing new is kept

    def begin_stage(self) -> None:
        """What the stage just ended asked for is what the next one may replay."""
        self.earlier, self.current = self.current, {}
        self.replayed = self.requested = 0

    def call(self, x: np.ndarray) -> float:
        if self.calls >= self.cap:
            raise _CapReachedError
        self.calls += 1
        value = float(self.fun(x))
        if self.best_x is None or value < self.best_f or self.best_f != self.best_f:
            self.best_x, self.best_f = np.array(x, dtype=float), value
        if self.keeping:
            self.xs.append(np.array(x, dtype=float))
            self.fs.append(value)
            self.memory[np.asarray(x, dtype=float).tobytes()] = value
        return value

    def replaying(self, x: np.ndarray) -> float:
        key = np.asarray(x, dtype=float).tobytes()
        self.requested += 1
        value = self.earlier.get(key)
        if value is None:
            value = self.call(x)
        else:
            self.replayed += 1
        self.current[key] = value
        return value

    def remembering(self, x: np.ndarray) -> float:
        value = self.memory.get(np.asarray(x, dtype=float).tobytes())
        if value is not None:
            self.remembered += 1
            return value
        return self.call(x)

    def warm(self) -> Warm | None:
        if not self.xs:
            return None
        return Warm(np.array(self.xs), np.array(self.fs))


def _plan(variant: Any, dim: int, stage: int) -> int:
    """The total budget stage ``stage`` plans."""
    per_dim = variant.anytime_start_per_dim * variant.anytime_factor**stage
    return max(1, int(per_dim * dim))


def minimize_anytime(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    budget: int,
    *,
    seed: int | None,
    rng: np.random.RandomState | None,
    config: ATMConfig,
    profile: bool | Profiler,
    bounds: tuple[object, object] | None,
) -> OptimizeResult:
    """:func:`~.api.minimize` for ``config.variant.anytime="replay"``. ``budget``
    caps the calls and is planned only by the hand-over; ``nrestarts`` counts the
    stages after the first (the hand-over included)."""
    from .api import _minimize, minimize

    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")
    if rng is None:
        rng = np.random.RandomState(seed)
    variant = config.variant
    inner = dataclasses.replace(
        config, variant=dataclasses.replace(variant, anytime="off")
    )
    x0 = np.asarray(x0, dtype=float)
    dim = len(x0)
    stage_seed = int(rng.randint(2**31 - 1))  # one draw; each stage seeds from it
    profiler = resolve_profiler(profile)
    store = _Store(fun, budget)
    stage = 0
    try:
        while store.calls < budget:
            plan = _plan(variant, dim, stage)
            if plan > variant.anytime_switch:
                break
            before = store.calls
            entry: dict[str, Any] = {"stage": stage, "plan": plan, "before": before}
            store.begin_stage()
            STAGE_LOG.append(entry)
            try:
                minimize(
                    store.replaying,
                    x0,
                    plan,
                    rng=np.random.RandomState(stage_seed),
                    config=inner,
                    profile=profiler,
                    bounds=bounds,
                )
            finally:
                entry.update(
                    calls=store.calls - before,
                    replayed=store.replayed,
                    requested=store.requested,
                )
            stage += 1
            if store.calls == before and store.requested < plan:
                break  # the configuration ends early whatever its plan
        if store.calls < budget:
            # the hand-over: the normal path on everything left, fed the archive
            entry = {"stage": stage, "handover": budget - store.calls}
            entry["before"] = store.calls
            STAGE_LOG.append(entry)
            warm = store.warm()
            store.keeping = False
            start = (
                x0 if HANDOVER_FROM == "start" or store.best_x is None else store.best_x
            )
            stage += 1
            try:
                _minimize(
                    store.remembering,
                    start,
                    budget - store.calls,
                    seed=None,
                    rng=np.random.RandomState(stage_seed),
                    config=inner,
                    profile=profiler,
                    bounds=bounds,
                    warm=warm,
                )
            finally:
                entry.update(
                    calls=store.calls - entry["before"], remembered=store.remembered
                )
    except _CapReachedError:
        pass
    best_x = x0 if store.best_x is None else store.best_x
    return OptimizeResult(
        x=np.array(best_x),
        fun=float(store.best_f),
        nfev=store.calls,
        nrestarts=max(0, stage - 1),
        timings=profiler.report(),
    )
