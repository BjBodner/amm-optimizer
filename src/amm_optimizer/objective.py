"""Objective-function wrapper that does the budget bookkeeping."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from time import perf_counter_ns
from typing import Any

import numpy as np

from .profiling import NULL_PROFILER, Profiler
from .result import OptimizeResult


def real_scalar(value: Any) -> float:
    """``value`` (what ``fun`` returned) as a real number, or a plain ``TypeError``.

    A float (NumPy's ``float64`` included) is returned as it is; any other real
    scalar (``float32``, ``int``, ``Fraction``) as ``float(value)``.
    """
    if isinstance(value, float):
        return value
    if np.ndim(value) != 0:
        raise TypeError(
            f"fun must return a real scalar, got {type(value).__name__} of shape "
            f"{np.shape(value)}"
        )
    if np.iscomplexobj(value):
        raise TypeError(f"fun must return a real scalar, got {type(value).__name__}")
    try:
        return float(value)
    except TypeError:
        raise TypeError(
            f"fun must return a real scalar, got {type(value).__name__}"
        ) from None


def copying(fun: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
    """``fun`` called on a fresh float copy of its argument, as SciPy does, and its
    value checked by :func:`real_scalar`.

    The optimizer passes views of its own buffers (a column of the sample matrix,
    an incumbent); a ``fun`` that writes into its argument, or keeps it, must not
    see or change them.
    """

    def call(x: np.ndarray) -> float:
        return real_scalar(fun(np.array(x, dtype=float)))

    return call


class _Stop(Exception):  # noqa: N818  (a signal, not an error)
    """Ends a run early (``ftarget`` reached, or the callback raised ``StopIteration``).

    Raised from inside ``fun`` by :class:`Watch` and caught by
    :func:`amm_optimizer.api.minimize`. Like ``_budget.SpentError``, no
    phase catches it, so it unwinds every phase and SciPy call at once.
    ``nrestarts`` is filled in on the way out by the run that knows it.
    """

    def __init__(self, status: int, watch: Watch) -> None:
        super().__init__(status)
        self.status = status
        self.watch = watch  # the run it stops
        self.nrestarts = 0


class Watch:
    """The user's ``fun`` as a run calls it: counts the calls and the improvements.

    With ``keep`` (or a ``callback`` or an ``ftarget``) it also keeps a copy of the
    best point. Each evaluation that improves the best finite value calls
    ``callback`` with an :class:`~amm_optimizer.OptimizeResult` holding ``x``,
    ``fun``, ``nfev`` and ``nit``; ``StopIteration`` from it, or a best value at or
    below ``ftarget``, raises :class:`_Stop`. It never draws random numbers, so the
    run's evaluations are the same with or without it.
    """

    def __init__(
        self,
        fun: Callable[[np.ndarray], Any],
        callback: Callable[[Any], object] | None = None,
        ftarget: float | None = None,
        keep: bool = False,
    ) -> None:
        self._fun = fun
        self._callback = callback
        self._ftarget = ftarget
        self._keep = keep or callback is not None or ftarget is not None
        self.calls = 0
        self.improvements = 0
        self.best_f = float("inf")
        self.best_x: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> float:
        if not self._keep:
            value = real_scalar(self._fun(x))
            self.calls += 1
            if value < self.best_f and value > -math.inf:  # NaN compares False
                self.best_f = value
                self.improvements += 1
            return value
        kept = np.array(x, dtype=float)  # before ``fun``, which may write into x
        value = real_scalar(self._fun(x))
        self.calls += 1
        if value < self.best_f and value > -math.inf:
            self.best_f, self.best_x = value, kept
            self.improvements += 1
            self._improved()
        return value

    def _improved(self) -> None:
        status = None
        if self._callback is not None:
            assert self.best_x is not None
            try:
                self._callback(
                    OptimizeResult(
                        x=self.best_x.copy(),
                        fun=self.best_f,
                        nfev=self.calls,
                        nit=self.improvements,
                    )
                )
            except StopIteration:
                status = 2
        if self._ftarget is not None and self.best_f <= self._ftarget:
            status = 1
        if status is not None:
            raise _Stop(status, self)


class CountedObjective:
    """Counts calls to ``fun`` against a budget; ``fun`` gets a copy of each point.

    ``calls`` is the true number of calls. With ``strict`` accounting every call
    is charged and at most ``allowed`` calls are made. Without it the original's
    bookkeeping applies: the call made at the start of every search is free
    (audit B4b) and the budget only counts as spent once ``allowed + 1`` charged
    calls have been made (audit B4).
    """

    def __init__(
        self,
        fun: Callable[[np.ndarray], float],
        allowed: int,
        strict: bool = True,
        profiler: Profiler = NULL_PROFILER,
    ) -> None:
        self._fun = copying(fun)
        self._profiler = profiler
        self._strict = strict
        self._limit = allowed if strict else allowed + 1
        self.calls = 0
        self._charged = 0

    def set_limit(self, calls: int) -> None:
        """Make ``calls`` charged calls the end of the budget (a phase's share).

        Only for strict accounting; a later call can raise the limit again.
        """
        assert self._strict
        self._limit = calls

    def evaluate_start(self, x: np.ndarray) -> float:
        """Evaluate the point a search starts from."""
        self.calls += 1
        if self._strict:
            self._charged += 1
        if self._profiler.enabled:
            start = perf_counter_ns()
            value = float(self._fun(x))
            self._profiler.record("objective", perf_counter_ns() - start)
            return value
        return float(self._fun(x))

    @property
    def remaining(self) -> int:
        """Evaluations that can still be charged."""
        return max(self._limit - self._charged, 0)

    def evaluate_point(self, x: np.ndarray) -> float | None:
        """Charge and evaluate one point; ``None`` when the budget is spent."""
        if self._charged >= self._limit:
            return None
        self._charged += 1
        self.calls += 1
        if self._profiler.enabled:
            start = perf_counter_ns()
            try:
                return float(self._fun(x))
            finally:
                self._profiler.record("objective", perf_counter_ns() - start)
        return float(self._fun(x))

    def evaluate_columns(
        self,
        samples: np.ndarray,
        known: Mapping[int, float] | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Charge and evaluate the columns of ``samples`` in order.

        Returns the costs evaluated so far and whether every column was reached;
        evaluation stops as soon as the budget is spent. ``known`` maps column
        indices to costs that are already known: those columns are neither
        evaluated nor charged.
        """
        if known:
            return self._evaluate_columns_known(samples, known)
        if self._profiler.enabled:
            return self._evaluate_columns_timed(samples)
        fun = self._fun
        cost = np.zeros(samples.shape[1])
        for j, column in enumerate(samples.T):
            if self._charged >= self._limit:
                return cost[:j], False
            self._charged += 1
            self.calls += 1
            cost[j] = fun(column)
        return cost, True

    def _evaluate_columns_known(
        self,
        samples: np.ndarray,
        known: Mapping[int, float],
    ) -> tuple[np.ndarray, bool]:
        """:meth:`evaluate_columns` with some costs given (slower, per-call timing)."""
        fun = self._fun
        profiler = self._profiler
        cost = np.zeros(samples.shape[1])
        for j, column in enumerate(samples.T):
            if j in known:
                cost[j] = known[j]
                continue
            if self._charged >= self._limit:
                return cost[:j], False
            self._charged += 1
            self.calls += 1
            if profiler.enabled:
                start = perf_counter_ns()
                try:
                    cost[j] = fun(column)
                finally:
                    profiler.record("objective", perf_counter_ns() - start)
            else:
                cost[j] = fun(column)
        return cost, True

    def _evaluate_columns_timed(self, samples: np.ndarray) -> tuple[np.ndarray, bool]:
        """Same loop as :meth:`evaluate_columns`, timing the user's function."""
        fun = self._fun
        cost = np.zeros(samples.shape[1])
        spent = 0
        count = 0
        try:
            for j, column in enumerate(samples.T):
                if self._charged >= self._limit:
                    return cost[:j], False
                self._charged += 1
                self.calls += 1
                count += 1
                start = perf_counter_ns()
                cost[j] = fun(column)
                spent += perf_counter_ns() - start
        finally:
            # Also reached when the user's function raises (the call counts).
            if count:
                self._profiler.record("objective", spent, count)
        return cost, True


class restarts_on_stop:  # noqa: N801  (used as a context manager, like a function)
    """On a :class:`_Stop` leaving the block, record ``count()`` as its ``nrestarts``.

    ``count`` reads the run's restart counter when the stop passes, so the result
    of a stopped run reports the restarts made up to then.
    """

    def __init__(self, count: Callable[[], int]) -> None:
        self._count = count

    def __enter__(self) -> None:
        return None

    def __exit__(
        self, kind: object, error: BaseException | None, trace: object
    ) -> None:
        if isinstance(error, _Stop):
            error.nrestarts = self._count()
