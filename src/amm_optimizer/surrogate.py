"""A linear-quadratic surrogate for population CMA-ES runs (``surrogate="lq2"``).

After Hansen's lq-CMA-ES (GECCO 2019, "A global surrogate assisted CMA-ES"): a
regression model of f, linear, then diagonal-quadratic, then full quadratic as data
accumulate, ranks each generation's offspring. Only the model's best few are
evaluated; more are evaluated, in model order, until the model's ranking of the
latest evaluated points agrees with their true values (Kendall tau at least
``TAU_TRUTH``) or the whole generation is evaluated. The CMA-ES update then ranks the
generation by true values where known and by the bias-corrected model elsewhere.

Our own choices: the fit is in the run's current frame ((x - mean) / sigma) by
weighted normal equations with a tiny ridge (after Jacobi scaling); the full model is
used only up to ``full_max_dim`` (its cost grows as dim^6). After a generation the
model needed in full, the next ``misses`` generations (at most ``MAX_PAUSE``) are
evaluated without it. Otherwise it follows pycma's implementation
(``cma.fitness_models``):
* the fit uses only the best ``TRUNCATION`` share of the recent points, weighted
  linearly from ``MAX_WEIGHT`` (best) to 1 (worst used);
* after each batch of evaluations the model is refitted and its Kendall tau is taken
  in sample on the latest points; the batch grows by half while tau stays below
  ``TAU_TRUTH``;
* evaluated candidates are ranked by their true values and the others by the refitted
  model plus its median bias (pycma ranks all by the model, which on the calibration
  instances lost f9 and f10 runs; injecting the model's minimizer, as pycma does, broke
  f7 and f12 and was removed);
* every step used in the update has its Mahalanobis length capped (``pop_cma``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache

import numpy as np
import scipy.linalg  # type: ignore[import-untyped]

TAU_TRUTH = 0.85
MIN_SIZE_FACTOR = 1.1
MAX_SIZE_FACTOR = 2.0
RIDGE = 1e-10
MAX_PAUSE = 8
TRUNCATION = 0.75
MAX_WEIGHT = 20.0


@cache
def _triangle(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``np.triu_indices(n, 1)``, made once per length (read-only)."""
    i, j = np.triu_indices(n, 1)
    i.setflags(write=False)
    j.setflags(write=False)
    return i, j


def _solve_pos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``scipy.linalg.solve(a, b, assume_a="pos")``: the same bits, less overhead.

    The solution is LAPACK ``dposv`` on the upper triangle, as in scipy's solve
    (bit-equal with SciPy 1.18 on every fit of the recorded runs; ``lower=True`` and
    ``cho_factor`` are not), without scipy's fixed cost of about 0.14 ms a call
    (and without its condition estimate, so no ``LinAlgWarning``). A 1 x 1 system
    (scipy divides), non-finite input (scipy's ``check_finite``), a failed
    factorisation or a non-finite solution go to scipy's solve, which raises or
    returns as before.
    """
    if len(b) > 1 and np.isfinite(a).all() and np.isfinite(b).all():
        _, x, info = scipy.linalg.lapack.dposv(a, b, lower=False)
        if info == 0 and np.isfinite(x).all():
            return np.asarray(x)
    return np.asarray(scipy.linalg.solve(a, b, assume_a="pos"))


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall's tau-a of two equally long arrays."""
    i, j = _triangle(len(a))
    if not len(i):
        return 1.0
    return float(np.mean(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])))


class QuadraticSurrogate:
    """Linear / diagonal / full quadratic least squares on the most recent points."""

    def __init__(self, dim: int, full_max_dim: int = 20) -> None:
        self.dim = dim
        self.sizes = [dim + 1, 2 * dim + 1]
        if dim <= full_max_dim:
            self.sizes.append((dim + 1) * (dim + 2) // 2)
        self.keep = int(np.ceil(MAX_SIZE_FACTOR * self.sizes[-1]))
        self.xs: list[np.ndarray] = []
        self.fs: list[float] = []
        self.coef: np.ndarray | None = None
        self.terms = 0
        self.center = np.zeros(dim)
        self.scale = 1.0
        # Generations to evaluate in full without fitting, after the model failed
        # (it needed the whole generation) ``misses`` times in a row.
        self.misses = 0
        self.pause = 0
        self._iu = np.triu_indices(dim, 1)

    def add(self, x: np.ndarray, f: float) -> None:
        if np.isfinite(f):
            self.xs.append(np.array(x, dtype=float))
            self.fs.append(float(f))
            if len(self.xs) > self.keep:
                del self.xs[0], self.fs[0]

    def features(self, x: np.ndarray, terms: int) -> np.ndarray:
        u = (np.atleast_2d(x) - self.center) / self.scale
        cols = [np.ones((len(u), 1)), u]
        if terms > self.dim + 1:
            cols.append(u**2)
        if terms > 2 * self.dim + 1:
            cols.append(u[:, self._iu[0]] * u[:, self._iu[1]])
        return np.hstack(cols)

    def fit(self, center: np.ndarray, scale: float) -> bool:
        """Refit in the frame (x - center) / scale; False if too few points."""
        n = len(self.xs)
        usable = int(TRUNCATION * (n + 1))
        fitting = [p for p in self.sizes if MIN_SIZE_FACTOR * p <= usable]
        if not fitting:
            self.coef = None
            return False
        self.terms = fitting[-1]
        self.center, self.scale = np.array(center, dtype=float), float(scale)
        size = min(n, int(np.ceil(MAX_SIZE_FACTOR * self.terms)))
        x = np.array(self.xs[-size:])
        f = np.array(self.fs[-size:])
        # The best share only, weighted linearly from MAX_WEIGHT down to 1.
        used = min(size, max(self.terms + 2, int(TRUNCATION * (size + 1))))
        best = np.argsort(f, kind="stable")[:used]
        x, f = x[best], f[best]
        # Rows scaled by the weight, as in pycma: squared residuals by its square.
        w = np.linspace(MAX_WEIGHT, 1.0, used) ** 2
        phi = self.features(x, self.terms)
        offset, spread = float(np.median(f)), float(np.std(f)) or 1.0
        weighted = phi.T * w
        gram = weighted @ phi
        rhs = weighted @ ((f - offset) / spread)
        try:
            # Jacobi scaling: old points far out in the current frame make the
            # columns' scales differ by many orders of magnitude.
            col = 1.0 / np.sqrt(np.maximum(np.diag(gram), 1e-300))
            scaled = gram * col[:, None] * col[None, :]
            scaled[np.diag_indices_from(scaled)] += RIDGE
            self.coef = col * _solve_pos(scaled, col * rhs)
        except (np.linalg.LinAlgError, ValueError):
            self.coef = None
            return False
        self._offset, self._spread = offset, spread
        return True

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.coef is not None
        values = self.features(x, self.terms) @ self.coef
        return np.asarray(values * self._spread + self._offset)

    def recent(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.xs[-n:]), np.array(self.fs[-n:])


def evaluate_generation(
    evaluate: Callable[[np.ndarray], float | None],
    shown: np.ndarray,
    model: QuadraticSurrogate,
    center: np.ndarray,
    scale: float,
    left: int,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Rank ``shown`` with as few true evaluations as the model allows.

    Returns (ranking values, true values with NaN where not evaluated and inf where
    not finite, evaluations used), or None if ``evaluate`` reports the budget spent.
    At most ``left`` points are evaluated; the generation is cut there.
    """
    lam = len(shown)
    true = np.full(lam, np.nan)
    known = np.zeros(lam, dtype=bool)
    used = 0

    def run(k: int) -> bool:
        nonlocal used
        value = evaluate(shown[k])
        if value is None:
            return False
        true[k] = value if np.isfinite(value) else np.inf
        known[k] = True
        model.add(shown[k], true[k])
        used += 1
        return True

    if model.pause > 0 or (model.coef is None and not model.fit(center, scale)):
        model.pause = max(model.pause - 1, 0)
        for k in range(min(lam, left)):
            if not run(k):
                return None
        model.fit(center, scale)
        return np.where(known, true, np.inf), true, used
    pending = list(np.argsort(model.predict(shown), kind="stable"))
    target = 1
    while pending and used < left:
        take, pending = pending[: target - used], pending[target - used :]
        for k in take[: left - used]:
            if not run(int(k)):
                return None
        fitted = model.fit(center, scale)
        if not pending or not fitted:
            break
        n_tau = max(15, min(int(1.2 * used), int(0.75 * lam)))
        x_recent, f_recent = model.recent(n_tau)
        if kendall_tau(model.predict(x_recent), f_recent) >= TAU_TRUTH:
            break
        target += int(np.ceil(target / 2))
        rest = np.array(pending)
        pending = list(rest[np.argsort(model.predict(shown[rest]), kind="stable")])
    if known.all():
        model.misses += 1
        model.pause = min(model.misses, MAX_PAUSE)
        return np.where(np.isfinite(true), true, np.inf), true, used
    model.misses = 0
    if model.coef is None:
        return np.where(known, true, np.inf), true, used
    pred = model.predict(shown)
    # evaluated candidates by their true values, the rest by the model plus its
    # median bias on them
    finite = known & np.isfinite(true)
    bias = float(np.median(true[finite] - pred[finite])) if finite.any() else 0.0
    return np.where(known, true, pred + bias), true, used
