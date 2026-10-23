"""A bounded Nelder-Mead simplex search with restarts (variant ``local_method``).

On smooth landscapes in a few dimensions a restarted simplex search converges to
high precision in fewer evaluations than ATM's sampling does (measured on bbob:
in 2-D from 100 x D, in 5-D from 300 x D). This module is a small dependency-free
implementation with the dimension-adaptive coefficients of Gao and Han (2012),
candidates clipped onto the box, and restarts when the simplex has collapsed.
"""

from __future__ import annotations

import numpy as np

from .bounds import Box
from .es_phase import EsResult
from .objective import CountedObjective

# The simplex has converged when it is this small (relative to the box) and its
# values agree to this relative precision; a restart follows.
SIZE_TOLERANCE = 1e-12
VALUE_TOLERANCE = 1e-14
# Initial simplex edge as a share of the box width, and the factor by which the
# edge shrinks on every restart around the best point.
INITIAL_EDGE = 0.05
RESTART_SHRINK = 0.3


def run_nelder_mead(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    edge: float,
    box: Box,
    evaluations: int,
) -> EsResult:
    """Nelder-Mead from ``x`` (cost known) for up to ``evaluations``.

    ``edge`` is the initial edge of the simplex (absolute). When the simplex
    collapses, the search restarts around the best point with a smaller edge, and
    after every third restart from a uniformly drawn point. Failed or NaN values
    never replace the best point. Returns the best point, its cost and the last
    edge used.
    """
    dim = len(x)
    width = float(np.mean(box.upper - box.lower))
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    used = 0
    restarts = 0
    while used < evaluations and edge > SIZE_TOLERANCE * width:
        if restarts and restarts % 3 == 0:
            start = box.lower + (box.upper - box.lower) * rng.rand(dim)
            start_cost = objective.evaluate_point(start)
            if start_cost is None:
                break
            used += 1
        else:
            start, start_cost = best_x, best_cost
        used_here, vertex, value = _simplex(
            objective,
            np.array(start, dtype=float),
            start_cost,
            edge,
            box,
            evaluations - used,
        )
        used += used_here
        if value < best_cost:
            best_x, best_cost = vertex, value
        restarts += 1
        edge *= RESTART_SHRINK
    return EsResult(best_x, best_cost, edge)


def _simplex(
    objective: CountedObjective,
    start: np.ndarray,
    start_cost: float,
    edge: float,
    box: Box,
    evaluations: int,
) -> tuple[int, np.ndarray, float]:
    """One simplex run until convergence or the budget; returns (used, x, f)."""
    simplex = _Simplex(objective, box, len(start), evaluations)
    if simplex.build(start, start_cost, edge):
        while simplex.used < evaluations and simplex.step():
            pass
    return simplex.used, *simplex.best()


def _finite(value: float) -> float:
    """Failed or NaN values count as infinitely bad."""
    return value if np.isfinite(value) else np.inf


class _Simplex:
    """The vertices of one Nelder-Mead run and its moves (Gao-Han coefficients)."""

    def __init__(
        self, objective: CountedObjective, box: Box, dim: int, evaluations: int
    ) -> None:
        self._objective = objective
        self._box = box
        self._dim = dim
        self._evaluations = evaluations
        self._width = float(np.mean(box.upper - box.lower))
        self.alpha, self.beta = 1.0, 1.0 + 2.0 / dim
        self.gamma = 0.75 - 1.0 / (2 * dim)
        self.delta = max(1.0 - 1.0 / dim, 0.5)
        self.used = 0
        self.vertices: list[np.ndarray] = []
        self.values: list[float] = []
        self._shrink_ran_out = False

    def evaluate(self, point: np.ndarray) -> float | None:
        """Evaluate within this run's share of the budget (``None``: spent)."""
        if self.used >= self._evaluations:
            return None
        value = self._objective.evaluate_point(point)
        if value is not None:
            self.used += 1
        return value

    def clip(self, point: np.ndarray) -> np.ndarray:
        return np.asarray(np.clip(point, self._box.lower, self._box.upper))

    def best(self) -> tuple[np.ndarray, float]:
        if self._shrink_ran_out:  # the shrink was cut short: first vertex, as sorted
            return self.vertices[0], self.values[0]
        i = int(np.argmin(self.values))
        return self.vertices[i], self.values[i]

    def build(self, start: np.ndarray, start_cost: float, edge: float) -> bool:
        """The start point plus one vertex per coordinate; ``False``: budget spent."""
        box = self._box
        self.vertices = [start]
        self.values = [start_cost if np.isfinite(start_cost) else np.inf]
        for i in range(self._dim):
            point = start.copy()
            step = edge if start[i] + edge <= box.upper[i] else -edge
            point[i] = np.clip(start[i] + step, box.lower[i], box.upper[i])
            value = self.evaluate(point)
            if value is None:
                return False
            self.vertices.append(point)
            self.values.append(_finite(value))
        return True

    def _sort(self) -> None:
        order = np.argsort(self.values, kind="stable")
        self.vertices = [self.vertices[i] for i in order]
        self.values = [self.values[i] for i in order]

    def _converged(self) -> bool:
        best = self.vertices[0]
        size = max(float(np.max(np.abs(v - best))) for v in self.vertices[1:])
        spread = abs(self.values[-1] - self.values[0])
        return size < SIZE_TOLERANCE * self._width and spread <= VALUE_TOLERANCE * (
            abs(self.values[0]) + 1e-300
        )

    def _replace_worst(self, point: np.ndarray, value: float) -> None:
        self.vertices[-1], self.values[-1] = point, value

    def step(self) -> bool:
        """One reflection, expansion, contraction or shrink; ``False``: stop."""
        self._sort()
        if self._converged():
            return False
        centroid = np.mean(self.vertices[:-1], axis=0)
        worst = self.vertices[-1]
        reflected = self.clip(centroid + self.alpha * (centroid - worst))
        f_reflected = self.evaluate(reflected)
        if f_reflected is None:
            return False
        f_reflected = _finite(f_reflected)
        if f_reflected < self.values[0]:
            return self._expand(centroid, reflected, f_reflected)
        if f_reflected < self.values[-2]:
            self._replace_worst(reflected, f_reflected)
            return True
        return self._contract(centroid, reflected, f_reflected)

    def _expand(
        self, centroid: np.ndarray, reflected: np.ndarray, f_reflected: float
    ) -> bool:
        expanded = self.clip(centroid + self.beta * (reflected - centroid))
        f_expanded = self.evaluate(expanded)
        if f_expanded is None:
            self._replace_worst(reflected, f_reflected)
            return False
        if np.isfinite(f_expanded) and f_expanded < f_reflected:
            self._replace_worst(expanded, f_expanded)
        else:
            self._replace_worst(reflected, f_reflected)
        return True

    def _contract(
        self, centroid: np.ndarray, reflected: np.ndarray, f_reflected: float
    ) -> bool:
        outside = f_reflected < self.values[-1]
        target = reflected if outside else self.vertices[-1]
        contracted = self.clip(centroid + self.gamma * (target - centroid))
        f_contracted = self.evaluate(contracted)
        if f_contracted is None:
            return False
        f_contracted = _finite(f_contracted)
        if f_contracted < (f_reflected if outside else self.values[-1]):
            self._replace_worst(contracted, f_contracted)
            return True
        return self._shrink()

    def _shrink(self) -> bool:
        """Move every vertex toward the best one."""
        best = self.vertices[0]
        for i in range(1, self._dim + 1):
            self.vertices[i] = self.clip(best + self.delta * (self.vertices[i] - best))
            value = self.evaluate(self.vertices[i])
            if value is None:
                self._shrink_ran_out = True
                return False
            self.values[i] = _finite(value)
        return True
