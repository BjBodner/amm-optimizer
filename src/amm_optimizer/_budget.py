"""A phase's share of the objective, as a plain function for SciPy's methods."""

from __future__ import annotations

import numpy as np

from .bounds import Box
from .objective import CountedObjective


class SpentError(Exception):
    """The objective, or a phase's own share of it, ran out.

    Raised from inside the function a SciPy method calls, and caught by the phase
    that started the method, so the method stops at once.
    """


class BudgetedObjective:
    """``objective`` as a plain function of a point, with a cap of its own.

    Each call charges one evaluation (``used``) until ``evaluations`` calls are made
    or the objective's own budget is spent; then it raises :class:`SpentError`. It
    keeps the best point and value it saw (``best_x``, ``best_cost``; only a strictly
    better value replaces them) and returns a failed value as 1e300, so SciPy's
    methods step away from it.

    ``box``: each point is clipped onto the box before it is evaluated (and the
    clipped point is the one kept). ``as_float``: without a box, the objective gets
    ``np.asarray(x, dtype=float)`` rather than ``x`` itself. ``known``: a point whose
    value is known; the first call, if it asks for that point (after the clip), gets
    the value without a charge.
    """

    def __init__(
        self,
        objective: CountedObjective,
        evaluations: int,
        *,
        box: Box | None = None,
        as_float: bool = False,
        best_x: np.ndarray | None = None,
        best_cost: float = np.inf,
        known: tuple[np.ndarray, float] | None = None,
    ) -> None:
        self.objective, self.evaluations = objective, evaluations
        self.box, self.as_float = box, as_float
        self.used = 0
        self.best_x, self.best_cost = best_x, best_cost
        self.known = known

    def __call__(self, x: np.ndarray) -> float:
        if self.known is not None:
            known_x, known_cost = self.known
            self.known = None
            if np.array_equal(self._point(x), known_x):
                return float(known_cost)
        if self.used >= self.evaluations:
            raise SpentError
        point = self._point(x)
        value = self.objective.evaluate_point(point)
        if value is None:
            raise SpentError
        self.used += 1
        if value < self.best_cost:
            self.best_x, self.best_cost = np.array(point, dtype=float), float(value)
        return float(value) if np.isfinite(value) else 1e300

    def _point(self, x: np.ndarray) -> np.ndarray:
        if self.box is not None:
            return np.asarray(
                np.clip(np.asarray(x, dtype=float), self.box.lower, self.box.upper)
            )
        return np.asarray(x, dtype=float) if self.as_float else x
