"""A restarted (1+1)-Cholesky-CMA-ES inside the box (``es_kind``, ``cma_handoff``).

The (1+1)-CMA-ES of Igel, Suttorp and Hansen (2006) keeps the Cholesky factor
``A`` of its covariance (and the inverse) and updates it in O(dim^2) after every
success, with a smoothed success rate driving the step size. Here it works inside
the box like the ES phase: candidates are clipped onto the box, a pinned
coordinate's inward noise is mirrored with probability ``PIN_STICK``, and the
covariance learns from the step actually taken. On stagnation it restarts from a
uniform point of the box and keeps the best point found.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from .bounds import Box, clip
from .es_phase import INITIAL_SIGMA_SHARE, PIN_STICK, EsResult
from .objective import CountedObjective

# Restart when sigma falls below this share of the mean box width ...
SIGMA_FLOOR_SHARE = 1e-12
# ... or when the best value improved by at most TOLFUN over 100 + 50 dim steps.
TOLFUN = 1e-11
# Quadratic model steps (``model=True``): a trust radius of MODEL_RADIUS sqrt(dim)
# in CMA coordinates, fitted to the last 2 (dim + 1)(dim + 2) / 2 points.
MODEL_RADIUS = 3.0
# ``restart_rule="short"``: the stall window.
SHORT_WINDOW = (10, 30)  # 10 + 30 dim evaluations
# The weighted model fit (``_lq_fit``, after lq-CMA-ES): the points per parameter
# a model needs, the best share it fits and the rank weights.
LQ_POINTS_PER_PARAMETER = 1.47
LQ_BEST_SHARE = 0.75
LQ_WEIGHTS = (20.0, 1.0)
# ``active=True``: the negative update of Arnold and Hansen (2010) after a
# candidate worse than the fifth-last accepted point.
ACTIVE_LAG = 5
# The gated model (``model_kind="diagonal"``): the period's start and cap, the
# ratios of actual to predicted decrease that lengthen (two in a row below
# RATIO_LOW) or shorten (one above RATIO_HIGH) it, and the step cap
# sqrt(dim) + GATED_RADIUS_ADD.
GATE_PERIOD = (2, 16)
RATIO_LOW, RATIO_HIGH = 0.1, 0.5
GATED_RADIUS_ADD = 2.0
# ``model_kind="diagonal"``: fit once LQ_POINTS_PER_PARAMETER (2 dim + 1) points
# exist, to the last DIAGONAL_WINDOW (2 dim + 1).
DIAGONAL_WINDOW = 3
# ``probe_at``: probe scales (times sigma, along A z directions) and directions.
PROBE_SCALES = (10.0, 30.0, 100.0)
PROBE_DIRECTIONS = 4
# Diagnostics: beta at each probe, recorded only while ``RECORD_PROBES`` is true
# (the log is process-global; off, it stays empty and no run grows it).
RECORD_PROBES = False
PROBE_LOG: list[float] = []


class _Recent:
    """A run's latest points and values, kept in a buffer of twice the window.

    ``latest(k)`` returns the same arrays as ``np.array(points[-k:])`` and
    ``np.array(values[-k:])`` of the lists the run kept before (the same rows in the
    same order, fresh copies), without converting a list of up to ``keep`` points
    at every model step. ``count`` is the lists' length.
    """

    def __init__(self, dim: int, keep: int) -> None:
        self._keep = keep
        self._x = np.empty((2 * keep, dim))
        self._f = np.empty(2 * keep)
        self._end = 0
        self.count = 0

    def reset(self, point: np.ndarray, value: float) -> None:
        self._end = self.count = 0
        self.append(point, value)

    def append(self, point: np.ndarray, value: float) -> None:
        if self._end == len(self._f):  # full: the latest ``keep`` rows to the front
            self._x[: self._keep] = self._x[self._end - self._keep :]
            self._f[: self._keep] = self._f[self._end - self._keep :]
            self._end = self._keep
        self._x[self._end] = point
        self._f[self._end] = value
        self._end += 1
        self.count += 1

    def latest(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        assert k <= self._keep
        start = max(self._end - k, 0)
        return self._x[start : self._end].copy(), self._f[start : self._end].copy()


@cache
def _upper(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """``np.triu_indices(dim)``, made once per dimension (read-only).

    The models below index their quadratic terms by it on every fit, and building it
    cost about a tenth of a full-quadratic model step at D 5-10.
    """
    rows, cols = np.triu_indices(dim)
    rows.setflags(write=False)
    cols.setflags(write=False)
    return rows, cols


def _pairs(z: np.ndarray) -> np.ndarray:
    """The products ``z_i z_j`` for ``i <= j``, one row per point.

    The same products as ``(z[:, :, None] * z[:, None, :])[:, i, j]`` (the same
    operands, so the same bits) without the ``n x dim x dim`` intermediate.
    """
    rows, cols = _upper(z.shape[1])
    return np.asarray(z[:, rows] * z[:, cols])


def cholesky_update(
    a: np.ndarray, a_inv: np.ndarray, v: np.ndarray, alpha: float, beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Factors of ``alpha A A^T + beta v v^T`` and its inverse (Igel et al. 2006)."""
    w = a_inv @ v
    norm2 = float(w @ w)
    if norm2 <= 0 or not np.isfinite(norm2):
        return a, a_inv
    root_a = np.sqrt(alpha)
    root = np.sqrt(1 + beta / alpha * norm2)
    a_new = root_a * a + root_a / norm2 * (root - 1) * np.outer(v, w)
    a_inv_new = a_inv / root_a - (1 - 1 / root) / (root_a * norm2) * np.outer(
        w, w @ a_inv
    )
    return a_new, a_inv_new


def quadratic_step(
    points: np.ndarray,
    values: np.ndarray,
    x: np.ndarray,
    sigma: float,
    a: np.ndarray,
    a_inv: np.ndarray,
) -> np.ndarray | None:
    """The minimiser of a least-squares quadratic model, or ``None``.

    The model is fitted to ``points`` (one per row) and ``values`` in the CMA
    coordinates ``z = A^-1 (p - x) / sigma``, where the model is best conditioned.
    ``None`` when the fitted Hessian is not positive definite. The step is cut to
    the trust radius ``MODEL_RADIUS * sqrt(dim)`` in those coordinates.
    """
    dim = len(x)
    z = ((points - x) / sigma) @ a_inv.T
    upper = _upper(dim)
    features = np.hstack([np.ones((len(z), 1)), z, _pairs(z)])
    span = float(values.max() - values.min())
    if not span > 0 or not np.isfinite(span):
        return None
    scaled = (values - values.min()) / span
    coef = np.linalg.lstsq(features, scaled, rcond=None)[0]
    g = coef[1 : dim + 1]
    h = np.zeros((dim, dim))
    h[upper] = coef[dim + 1 :]
    h = h + h.T  # the diagonal doubles: d^2/dz_i^2 of h_ii z_i^2 is 2 h_ii
    eig = np.linalg.eigvalsh(h)
    if not np.isfinite(eig).all() or eig.min() <= 1e-12 * np.abs(eig).max():
        return None
    step = -np.linalg.solve(h, g)
    radius = MODEL_RADIUS * np.sqrt(dim)
    norm = float(np.linalg.norm(step))
    if norm > radius:
        step *= radius / norm
    return np.asarray(x + sigma * (a @ step))


def _lq_features(z: np.ndarray, rung: str) -> np.ndarray:
    ones = np.ones((len(z), 1))
    if rung == "linear":
        return np.hstack([ones, z])
    if rung == "diagonal":
        return np.hstack([ones, z, z * z])
    return np.hstack([ones, z, _pairs(z)])


def _lq_fit(z: np.ndarray, values: np.ndarray, rung: str) -> np.ndarray:
    """Rank-weighted least squares on the best ``LQ_BEST_SHARE`` of the points."""
    order = np.argsort(values, kind="stable")
    count = max(int(np.ceil(LQ_BEST_SHARE * len(values))), 2)
    chosen = order[:count]
    hi, lo = LQ_WEIGHTS
    weights = np.sqrt(np.linspace(hi, lo, count))
    features = _lq_features(z[chosen], rung)
    span = float(values.max() - values.min()) or 1.0
    target = (values[chosen] - values.min()) / span
    solution = np.linalg.lstsq(
        features * weights[:, None], target * weights, rcond=None
    )[0]
    return np.asarray(solution)


def model_step(
    points: np.ndarray,
    values: np.ndarray,
    x: np.ndarray,
    sigma: float,
    a: np.ndarray,
    a_inv: np.ndarray,
    rung: str,
    weighted: bool,
    radius: float,
    box: Box,
) -> tuple[np.ndarray, float] | None:
    """A model candidate (clipped onto the box) and the decrease the model predicts.

    The model (``rung`` "full" or "diagonal") is fitted in the CMA coordinates
    ``z = A^-1 (p - x) / sigma``, by least squares on all points or, if
    ``weighted``, as ``_lq_fit`` does. A full model needs a positive definite
    Hessian; a diagonal one takes the Newton step if every curvature is positive
    and otherwise the step of length ``radius`` along the negative gradient. The
    step is cut to ``radius``. The predicted decrease, in units of the values, is
    the model at ``x`` minus the model at the clipped candidate. ``None`` when
    there is no usable step.
    """
    dim = len(x)
    if not np.isfinite(values).all() or not values.max() > values.min():
        return None
    span = float(values.max() - values.min())
    z = ((points - x) / sigma) @ a_inv.T
    if weighted:
        coef = _lq_fit(z, values, rung)
    else:
        scaled = (values - values.min()) / span
        coef = np.linalg.lstsq(_lq_features(z, rung), scaled, rcond=None)[0]
    g = coef[1 : dim + 1]
    if rung == "diagonal":
        curvature = 2 * coef[dim + 1 :]
        if (curvature > 1e-12 * np.abs(curvature).max()).all():
            step = -g / curvature
        else:
            norm_g = float(np.linalg.norm(g))
            if not norm_g > 0:
                return None
            step = -g / norm_g * radius
    else:
        upper = _upper(dim)
        h = np.zeros((dim, dim))
        h[upper] = coef[dim + 1 :]
        h = h + h.T
        eig = np.linalg.eigvalsh(h)
        if not np.isfinite(eig).all() or eig.min() <= 1e-12 * np.abs(eig).max():
            return None
        step = -np.linalg.solve(h, g)
    norm = float(np.linalg.norm(step))
    if not np.isfinite(norm) or norm == 0:
        return None
    if norm > radius:
        step *= radius / norm
    candidate = clip(x + sigma * (a @ step), box.lower, box.upper)
    z_c = (a_inv @ ((candidate - x) / sigma))[None, :]
    predicted = float(coef[0] - (_lq_features(z_c, rung) @ coef)[0]) * span
    return candidate, predicted


def warm_factor(points: np.ndarray, values: np.ndarray) -> np.ndarray | None:
    """A Cholesky factor of unit determinant from the best evaluated points.

    The covariance of the best max(2 dim, n / 8) points (regularised by a
    millionth of its mean variance) is scaled to unit determinant; ``None`` when
    there are too few finite points or it is degenerate (``cma_warm_start``).
    """
    dim = points.shape[1]
    finite = np.isfinite(values)
    points, values = points[finite], values[finite]
    count = max(2 * dim, len(values) // 8)
    if len(values) < count or count < dim + 1:
        return None
    best = points[np.argsort(values, kind="stable")[:count]]
    cov = np.atleast_2d(np.cov(best.T))  # np.cov is 0-d for one dimension
    cov = cov + 1e-6 * np.trace(cov) / dim * np.eye(dim)
    eig = np.linalg.eigvalsh(cov)
    if not (np.isfinite(eig).all() and eig.min() > 0):
        return None
    shape = cov / np.exp(np.mean(np.log(eig)))
    return np.asarray(np.linalg.cholesky(shape))


def run_cma(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    sigma: float,
    box: Box,
    evaluations: int,
    restarts: bool = True,
    model: bool = False,
    *,
    model_kind: str | None = None,
    restart_rule: str = "legacy",
    active: bool = False,
    a0: np.ndarray | None = None,
    stop_when_left: int = 0,
    run_cap: int = 0,
    probe_at: int = 0,
    probe_beta: float = 0.0,
) -> EsResult:
    """Spend up to ``evaluations`` on the restarted (1+1)-CMA-ES from ``x``.

    ``cost`` is the known cost of ``x``. Returns the best point, its cost, the
    width scale of the last run (``sigma * det(A) ** (1 / dim)``) and the
    evaluations used. ``restarts=False`` returns at the first stagnation instead
    of restarting (``stalled`` is then true). ``model=True`` makes every other
    candidate the minimiser of a quadratic model of the run's recent points
    (``quadratic_step``) once there are enough of them; a model candidate that
    improves is accepted but does not adapt sigma or the covariance.

    ``model_kind="diagonal"``: the gated, rank-weighted diagonal model
    (``model_step``) instead. ``restart_rule``: a stall window of 100 + 50 dim
    (``"legacy"``) or 10 + 30 dim (``"short"``). ``active``: the negative covariance
    update of Arnold and Hansen (2010). ``a0``: the first run's Cholesky factor.
    ``stop_when_left > 0`` returns at a stagnation (``stalled``) instead of
    restarting if at least that many evaluations are left; ``run_cap`` and
    ``probe_at`` / ``probe_beta`` hand over earlier (``Variant`` explains them).
    """
    kind = model_kind if model_kind is not None else ("quadratic" if model else None)
    dim = len(x)
    width = float(np.mean(box.upper - box.lower))
    d = 1 + dim / 2
    p_target, c_p, c_c = 2 / 11, 1 / 12, 2 / (dim + 2)
    c_cov, p_thresh = 2 / (dim**2 + 6), 0.44
    window = 100 + 50 * dim
    short_window = SHORT_WINDOW[0] + SHORT_WINDOW[1] * dim
    c_minus = 0.4 / (dim**1.6 + 1)
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    x, cost = best_x.copy(), best_cost
    if a0 is not None:
        a, a_inv = np.array(a0, dtype=float), np.linalg.inv(a0)
    else:
        a, a_inv = np.eye(dim), np.eye(dim)
    parents = [cost]
    p_c, p_succ = np.zeros(dim), p_target
    history: list[float] = []
    used = 0
    models = _Models(kind, dim)
    models.recent.reset(x, cost)
    probed = False
    ties: list[bool] = []

    while used < evaluations:
        candidate, predicted = models.candidate(x, sigma, a, a_inv, box)
        if candidate is not None:
            value = objective.evaluate_point(candidate)
            if value is None:
                break
            used += 1
            models.recent.append(candidate, value)
            if models.diagonal:
                models.gate.read(
                    (cost - value) / predicted if predicted > 0 else -np.inf
                )
            if value <= cost:
                x, cost = candidate, value
                parents.append(cost)
                if cost < best_cost:
                    best_x, best_cost = x.copy(), cost
            history.append(cost)
            continue
        candidate = _random_candidate(rng, x, sigma, a, box)
        value = objective.evaluate_point(candidate)
        if value is None:
            break
        used += 1
        if kind is not None:
            models.recent.append(candidate, value)
        success = value <= cost
        ties.append(value == cost)
        p_succ = (1 - c_p) * p_succ + c_p * success
        if success:
            taken = (candidate - x) / sigma
            x, cost = candidate, value
            p_c, a, a_inv = _success_update(
                p_c, a, a_inv, taken, p_succ < p_thresh, c_c, c_cov
            )
            parents.append(cost)
            if cost < best_cost:
                best_x, best_cost = x.copy(), cost
        elif active and len(parents) >= ACTIVE_LAG and value > parents[-ACTIVE_LAG]:
            a, a_inv = _active_update(a, a_inv, (candidate - x) / sigma, c_minus)
        sigma *= float(np.exp((p_succ - p_target) / (d * (1 - p_target))))
        history.append(cost)
        span = short_window if restart_rule == "short" else window
        stalled = len(history) > span and history[-span - 1] - cost <= TOLFUN
        left = evaluations - used
        if probe_at and not probed and len(history) >= probe_at:
            probed = True
            beta, spent, best_x, best_cost = _probe_beta(
                objective, rng, x, cost, sigma, a, box, best_x, best_cost
            )
            used, left = used + spent, left - spent
            if beta is None:
                break
            if probe_beta > 0 and beta < probe_beta and left >= max(stop_when_left, 1):
                return EsResult(best_x, float(best_cost), sigma, used, True)
        # ``run_cap``: hand over after this many evaluations of the run, stall or not
        if run_cap and len(history) >= run_cap and left >= max(stop_when_left, 1):
            return _capped(best_x, best_cost, sigma, a, used, ties)
        if (sigma < SIGMA_FLOOR_SHARE * width or stalled) and used < evaluations:
            if not restarts or (
                stop_when_left and evaluations - used >= stop_when_left
            ):
                return EsResult(best_x, float(best_cost), sigma, used, True)
            x = box.uniform(rng)
            value = objective.evaluate_point(x)
            if value is None:
                break
            used += 1
            cost = value
            sigma = INITIAL_SIGMA_SHARE * width
            if cost < best_cost:
                best_x, best_cost = x.copy(), cost
            a, a_inv = np.eye(dim), np.eye(dim)
            p_c, p_succ = np.zeros(dim), p_target
            history = []
            parents = [cost]
            models.restart(x, cost)
            ties = []
    sign, log_det = np.linalg.slogdet(a)
    scale = sigma * float(np.exp(log_det / dim)) if sign != 0 else sigma
    return EsResult(best_x, float(best_cost), scale, used)


def _random_candidate(
    rng: np.random.RandomState, x: np.ndarray, sigma: float, a: np.ndarray, box: Box
) -> np.ndarray:
    """The (1+1)'s random candidate ``x + sigma A z``, clipped onto the box; a step
    that would pull a coordinate on a wall inside is mirrored outward with
    probability ``PIN_STICK``."""
    dim = len(x)
    z_draw = rng.randn(dim)
    step = sigma * (a @ z_draw)
    inward = ((x <= box.lower) & (step > 0)) | ((x >= box.upper) & (step < 0))
    if inward.any():
        step = np.where(inward & (rng.rand(dim) < PIN_STICK), -step, step)
    return np.asarray(clip(x + step, box.lower, box.upper))


def _success_update(
    p_c: np.ndarray,
    a: np.ndarray,
    a_inv: np.ndarray,
    taken: np.ndarray,
    below_threshold: bool,
    c_c: float,
    c_cov: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """After a success: the evolution path and the Cholesky factor's rank-one update
    (the path learns from the step only while the success rate is below the
    threshold)."""
    if below_threshold:
        p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c)) * taken
        alpha = 1 - c_cov
    else:
        p_c = (1 - c_c) * p_c
        alpha = 1 - c_cov + c_cov * c_c * (2 - c_c)
    a, a_inv = cholesky_update(a, a_inv, p_c, alpha, c_cov)
    return p_c, a, a_inv


class _Gate:
    """The diagonal model's period: lengthened by two low ratios of actual to
    predicted decrease in a row (below ``RATIO_LOW``), shortened by one high one
    (above ``RATIO_HIGH``)."""

    def __init__(self) -> None:
        self.period, self.countdown, self.low_streak = (
            GATE_PERIOD[0],
            GATE_PERIOD[0],
            0,
        )

    def due(self) -> bool:
        """Count one candidate; whether a model step is due."""
        self.countdown -= 1
        if self.countdown <= 0:
            self.countdown = self.period
            return True
        return False

    def read(self, ratio: float) -> None:
        if ratio < RATIO_LOW:
            self.low_streak += 1
            if self.low_streak >= 2:
                self.period, self.low_streak = min(GATE_PERIOD[1], 2 * self.period), 0
                self.countdown = self.period
        elif ratio > RATIO_HIGH:
            self.period, self.low_streak = max(GATE_PERIOD[0], self.period // 2), 0
            self.countdown = self.period
        else:
            self.low_streak = 0


class _Models:
    """The model candidates of a CMA run (``kind``: ``"quadratic"``, ``"diagonal"``
    or None) and the recent points they are fitted to."""

    def __init__(self, kind: str | None, dim: int) -> None:
        self.kind, self.dim = kind, dim
        self.diagonal = kind == "diagonal"
        self.n_model = (dim + 1) * (dim + 2) // 2
        if self.diagonal:
            n_diag = 2 * dim + 1
            self.need = int(np.ceil(LQ_POINTS_PER_PARAMETER * n_diag))
            self.keep_fit = DIAGONAL_WINDOW * n_diag
        else:
            self.need, self.keep_fit = self.n_model + dim, 2 * self.n_model
        # The run's start and candidates: up to the largest window a model reads
        # (``keep_fit``, or ``2 n_model`` for ``quadratic_step``).
        self.recent = _Recent(dim, max(self.keep_fit, 2 * self.n_model))
        self.radius_fit = np.sqrt(dim) + GATED_RADIUS_ADD
        self.gate = _Gate()
        self.turn = False

    def restart(self, x: np.ndarray, cost: float) -> None:
        self.recent.reset(x, cost)
        self.gate = _Gate()

    def candidate(
        self,
        x: np.ndarray,
        sigma: float,
        a: np.ndarray,
        a_inv: np.ndarray,
        box: Box,
    ) -> tuple[np.ndarray | None, float]:
        """This turn's model candidate and its predicted decrease (None: a random
        step); the quadratic model takes every other turn."""
        candidate, predicted = None, 0.0
        recent = self.recent
        if self.diagonal:
            if self.gate.due() and recent.count >= self.need:
                found_step = model_step(
                    *recent.latest(self.keep_fit),
                    x,
                    sigma,
                    a,
                    a_inv,
                    "diagonal",
                    True,
                    self.radius_fit,
                    box,
                )
                if found_step is not None:
                    candidate, predicted = found_step
        elif (
            self.kind == "quadratic"
            and self.turn
            and recent.count >= self.n_model + self.dim
        ):
            proposal = quadratic_step(
                *recent.latest(2 * self.n_model), x, sigma, a, a_inv
            )
            if proposal is not None:
                candidate = clip(proposal, box.lower, box.upper)
        self.turn = not self.turn
        return candidate, predicted


def _active_update(
    a: np.ndarray, a_inv: np.ndarray, taken: np.ndarray, c_minus: float
) -> tuple[np.ndarray, np.ndarray]:
    """The active (negative) update of the Cholesky factor along ``taken``."""
    z = a_inv @ taken
    norm2 = float(z @ z)
    rate = c_minus
    if 2 * norm2 - 1 > 0:
        rate = min(rate, 1 / (2 * norm2 - 1))
    return cholesky_update(a, a_inv, taken, 1 + rate, -rate)


def _probe_beta(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    sigma: float,
    a: np.ndarray,
    box: Box,
    best_x: np.ndarray,
    best_cost: float,
) -> tuple[float | None, int, np.ndarray, float]:
    """``probe_at``: the curvature probe; (beta, evaluations used, best
    point, its value), beta None when the objective ran out.

    Mirrored pairs along ``PROBE_DIRECTIONS`` directions of the learned metric at
    ``PROBE_SCALES`` times the step; beta is the slope of log(mean loss) against
    log(scale): 2 on any quadratic, lower on rugged landscapes.
    """
    dim = len(x)
    used = 0
    losses: list[list[float]] = [[] for _ in PROBE_SCALES]
    dirs = [a @ rng.randn(dim) for _ in range(PROBE_DIRECTIONS)]
    for j, scale in enumerate(PROBE_SCALES):
        for d_k in dirs:
            pair = []
            for sign in (1.0, -1.0):
                point = clip(x + sign * scale * sigma * d_k, box.lower, box.upper)
                value = objective.evaluate_point(point)
                if value is None:
                    return None, used, best_x, best_cost
                used += 1
                pair.append(float(value))
                if value < best_cost:
                    best_x, best_cost = point.copy(), float(value)
            losses[j].append(0.5 * (pair[0] + pair[1]) - cost)
    mean_loss = [float(np.mean(v)) for v in losses]
    ok = [j for j, v in enumerate(mean_loss) if v > 0]
    beta = (
        float(
            np.polyfit(
                np.log10([PROBE_SCALES[j] for j in ok]),
                np.log10([mean_loss[j] for j in ok]),
                1,
            )[0]
        )
        if len(ok) >= 2
        else float("nan")
    )
    if RECORD_PROBES:
        PROBE_LOG.append(beta)
    return beta, used, best_x, best_cost


def _capped(
    best_x: np.ndarray,
    best_cost: float,
    sigma: float,
    a: np.ndarray,
    used: int,
    ties: list[bool],
) -> EsResult:
    """``run_cap``'s hand-over: the width scale (sigma det(A)^(1/dim)), the shape
    A A^T at unit determinant and the share of the last 10 dim samples that tied."""
    dim = len(best_x)
    sign, log_det = np.linalg.slogdet(a)
    scale = sigma * float(np.exp(log_det / dim)) if sign != 0 else sigma
    shape = a @ a.T
    shape = shape / np.exp(np.linalg.slogdet(shape)[1] / dim)
    tied = float(np.mean(ties[-10 * dim :])) if ties else 0.0
    return EsResult(best_x, float(best_cost), scale, used, True, True, shape, tied)
