"""Objective-function wrapper that does the budget bookkeeping."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter_ns

import numpy as np

from .profiling import NULL_PROFILER, Profiler


class CountedObjective:
    """Counts calls to ``fun`` against a budget.

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
        memo: bool = False,
    ) -> None:
        self._fun = fun
        self._profiler = profiler
        self._strict = strict
        self._limit = allowed if strict else allowed + 1
        self.calls = 0
        self._charged = 0
        self._held = 0
        # ``Variant.eval_memo`` (the novel-ideas toy): exactly repeated points return
        # their stored value, uncharged and uncalled (``memo_hits`` counts them)
        self._memo: dict[bytes, float] | None = {} if memo else None
        self.memo_hits = 0

    def _recall(self, x: np.ndarray) -> tuple[bytes, float | None]:
        """(key, stored value or None) of ``x`` in the memo (which must be on)."""
        assert self._memo is not None
        key = np.asarray(x, dtype=float).tobytes()
        value = self._memo.get(key)
        if value is not None:
            self.memo_hits += 1
        return key, value

    def _evaluate_memo(self, x: np.ndarray) -> float | None:
        """:meth:`evaluate_point` with the memo on."""
        assert self._memo is not None
        key, value = self._recall(x)
        if value is not None:
            return value
        if self._charged >= self._limit:
            return None
        self._charged += 1
        self.calls += 1
        value = float(self._fun(x))
        self._memo[key] = value
        return value

    def set_limit(self, calls: int) -> None:
        """Make ``calls`` charged calls the end of the budget (a phase's share).

        Only for strict accounting; a later call can raise the limit again.
        """
        assert self._strict
        self._limit = calls

    def evaluate_start(self, x: np.ndarray) -> float:
        """Evaluate the point a search starts from."""
        if self._memo is not None:
            key, known = self._recall(x)
            if known is not None:
                return known
            self.calls += 1
            if self._strict:
                self._charged += 1
            value = float(self._fun(x))
            self._memo[key] = value
            return value
        self.calls += 1
        if self._strict:
            self._charged += 1
        if self._profiler.enabled:
            start = perf_counter_ns()
            value = self._fun(x)
            self._profiler.record("objective", perf_counter_ns() - start)
            return value
        return self._fun(x)

    @property
    def remaining(self) -> int:
        """Evaluations that can still be charged."""
        return max(self._limit - self._charged, 0)

    def hold_back(self, n: int) -> None:
        """Make the last ``n`` evaluations unavailable until :meth:`release`."""
        self._held = n
        self._limit -= n

    def release(self) -> None:
        """Make the evaluations kept back by :meth:`hold_back` available again."""
        self._limit += self._held
        self._held = 0

    def evaluate_point(self, x: np.ndarray) -> float | None:
        """Charge and evaluate one point; ``None`` when the budget is spent."""
        if self._memo is not None:
            return self._evaluate_memo(x)
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
        if self._memo is not None:
            cost = np.zeros(samples.shape[1])
            for j, column in enumerate(samples.T):
                if known and j in known:
                    cost[j] = known[j]
                    continue
                value = self._evaluate_memo(column)
                if value is None:
                    return cost[:j], False
                cost[j] = value
            return cost, True
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
