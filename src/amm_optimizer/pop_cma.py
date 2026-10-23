"""A (mu/mu_w, lambda)-CMA-ES run inside the box (variant ``cma_regimes``).

The standard non-elitist CMA-ES of Hansen's tutorial (arXiv 1604.00772):
weighted recombination of the best half, cumulative step-size adaptation, rank-one
and rank-mu covariance updates with the default learning rates. It is the "large
regime" of a BIPOP-style restart schedule: a population that averages over the
ruggedness of multimodal landscapes, which an elitist (1+1) cannot. Candidates
are clipped onto the box for evaluation, and the update uses the clipped points.
One call is one run; it ends at stagnation or when its evaluations are spent.
"""

from __future__ import annotations

import numpy as np

from .bounds import Box, clip
from .es_phase import EsResult
from .objective import CountedObjective

# Stop a run when the best value of the last 10 + 30 dim / lambda generations
# spans at most TOLHISTFUN, when sigma times the largest axis falls below TOLX of
# the box width, or when the covariance's condition number exceeds MAX_CONDITION.
TOLHISTFUN = 1e-11
TOLX_SHARE = 1e-12
MAX_CONDITION = 1e14
# ``lambda_rule="budget"``: a run is planned to last about this many times dim
# generations' worth of evaluations per lambda (research notes, cycle 3).
BUDGET_RUN_LENGTH = 35


GROW_FROM_PER_DIM = 2000.0


def run_population_restarts(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    sigma: float,
    box: Box,
    evaluations: int,
    doubling: bool,
    *,
    lambda_rule: str = "default",
    active: bool = False,
    min_left: int = 0,
    bounds_mode: str = "clip",
    must_beat: float | None = None,
    cov0: np.ndarray | None = None,
    surrogate: str = "none",
) -> EsResult:
    """Restarted CMA-ES runs for ``evaluations``; the first from ``x`` at ``sigma``.

    Later runs start from uniform points of the box at ``INITIAL_SIGMA_SHARE`` of
    its width, with the default population, doubled at every restart if
    ``doubling`` (IPOP). Returns the best point and value (``cost`` is the known
    value of ``x``) and the evaluations used.

    Batch 12 options: ``lambda_rule`` ``"cap4"`` cycles the population through 1, 2
    and 4 times the default, ``"fixed2"`` keeps it at twice the default;
    ``active`` adds the negative covariance update; ``min_left > 0`` launches a
    run only while at least that many evaluations are left (the rest is
    returned unused). ``lambda_rule="budget"`` sizes each run's population as
    max(default, left / (BUDGET_RUN_LENGTH dim)); ``"steps"`` uses 1, 2 or 4 times
    the default for fewer than 150, 150-640 and at least 640 dim evaluations left;
    ``"ipop4"``: 4 times the default, doubled at every restart (``"ipop8"``: from 8
    times); ``"stepsgrow"``:
    ``"steps"``, doubling from 4x per restart while at least ``GROW_FROM_PER_DIM``
    dim evaluations are left.
    ``bounds_mode`` is passed to ``run_population_cma``. ``must_beat``
    (protection): stop after the first run that ends without a value below both it
    and every earlier run's best.
    """
    from .es_phase import INITIAL_SIGMA_SHARE

    dim = len(x)
    width = float(np.mean(box.upper - box.lower))
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    start, start_sigma = best_x.copy(), sigma
    used, restarts = 0, 0
    while used < evaluations:
        if min_left and evaluations - used < min_left:
            break
        if lambda_rule == "cap4":
            lam = default_population(dim) * 2 ** (restarts % 3)
        elif lambda_rule == "fixed2":
            lam = 2 * default_population(dim)
        elif lambda_rule == "budget":
            lam = max(
                default_population(dim),
                int((evaluations - used) / (BUDGET_RUN_LENGTH * dim)),
            )
        elif lambda_rule == "steps":
            # Cycle 14 (literature report, rank 2; sweep at D 5, 20, 40): the
            # multiple of the default lambda that wins with R x dim evaluations left.
            left_per_dim = (evaluations - used) / dim
            multiple = 4 if left_per_dim >= 640 else 2 if left_per_dim >= 150 else 1
            lam = multiple * default_population(dim)
        elif lambda_rule == "ipop4":  # cycle 18's plateau route: 4x, doubling
            lam = 4 * default_population(dim) * 2**restarts
        elif lambda_rule == "ipop8":  # cycle 29's rugged route at D >= 20: 8x, doubling
            lam = 8 * default_population(dim) * 2**restarts
        elif lambda_rule == "stepsgrow":
            # Cycle 22: ``"steps"``, but with at least GROW_FROM_PER_DIM x dim
            # evaluations left the population doubles from 4x at every restart
            # (a 2500 x D sweep: f15 at D 5 needs 8-16x, f16-f18 stay solved).
            left_per_dim = (evaluations - used) / dim
            if left_per_dim >= GROW_FROM_PER_DIM:
                multiple = 4 * 2**restarts
            else:
                multiple = 4 if left_per_dim >= 640 else 2 if left_per_dim >= 150 else 1
            lam = multiple * default_population(dim)
        else:
            lam = default_population(dim) * (2**restarts if doubling else 1)
        lam = min(lam, evaluations - used)
        if lam < 4:
            break
        run = run_population_cma(
            objective,
            rng,
            start,
            start_sigma,
            box,
            evaluations - used,
            lam,
            active,
            bounds_mode,
            cov0 if restarts == 0 else None,
            surrogate,
        )
        used += run.used
        improved = run.cost < best_cost and (must_beat is None or run.cost < must_beat)
        if run.cost < best_cost:
            best_x, best_cost = run.x, run.cost
        if must_beat is not None and not improved:
            break
        if run.used == 0 or not run.stalled:
            break
        restarts += 1
        start = box.lower + (box.upper - box.lower) * rng.rand(dim)
        start_sigma = INITIAL_SIGMA_SHARE * width
    return EsResult(best_x, best_cost, start_sigma, used)


def reflect_into(x: np.ndarray, box: Box) -> np.ndarray:
    """``x`` folded into the box by repeated mirroring at its faces."""
    width = box.upper - box.lower
    t = np.mod((x - box.lower) / width, 2.0)
    return np.asarray(box.lower + width * np.where(t > 1.0, 2.0 - t, t))


def default_population(dim: int) -> int:
    """``4 + floor(3 ln dim)``, the CMA-ES default lambda."""
    return 4 + int(3 * np.log(dim))


def run_population_cma(
    objective: CountedObjective,
    rng: np.random.RandomState,
    mean: np.ndarray,
    sigma: float,
    box: Box,
    evaluations: int,
    population: int,
    active: bool = False,
    bounds_mode: str = "clip",
    cov0: np.ndarray | None = None,
    surrogate: str = "none",
) -> EsResult:
    """One CMA-ES run of at most ``evaluations`` from ``mean`` with ``population``.

    Returns the best point and value of the run, its final sigma, the evaluations
    used, and whether it ended by stagnation (``stalled``) rather than budget.
    ``active`` gives the worst ``lambda - mu`` steps negative weights, scaled as in
    Hansen's tutorial (arXiv 1604.00772, eq. 50-53) and pycma, with the
    ``dim / |C^-1/2 y|^2`` rescaling of their outer products.
    ``bounds_mode="reflect"``: the mean and the steps stay unbounded and the
    evaluated point is ``reflect_into(x, box)``; the update uses the sampled steps.
    ``surrogate="lq"``: generations are ranked by ``surrogate.evaluate_generation``
    (true values where evaluated, the model elsewhere); the run may end inside a
    generation when its evaluations are spent.
    """
    dim = len(mean)
    width = float(np.mean(box.upper - box.lower))
    lam = max(int(population), 4)
    mu = lam // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mu_eff = 1.0 / float(np.sum(weights**2))
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
    chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
    negative = np.zeros(lam - mu)
    if active and lam > mu:
        raw = np.log(mu + 0.5) - np.log(np.arange(mu + 1, lam + 1))
        mu_eff_neg = float(raw.sum() ** 2 / np.sum(raw**2))
        alpha = min(
            1 + c_1 / c_mu,
            1 + 2 * mu_eff_neg / (mu_eff + 2),
            (1 - c_1 - c_mu) / (dim * c_mu),
        )
        negative = alpha * raw / np.abs(raw).sum()
    mean = np.clip(np.array(mean, dtype=float), box.lower, box.upper)
    p_sigma, p_c = np.zeros(dim), np.zeros(dim)
    cov = np.eye(dim) if cov0 is None else np.array(cov0, dtype=float)
    if cov0 is None:
        basis, axes = np.eye(dim), np.ones(dim)
    else:
        eig0, basis = np.linalg.eigh((cov + cov.T) / 2)
        axes = np.sqrt(np.maximum(eig0, 1e-300))
    best_x, best_cost = mean.copy(), np.inf
    history: list[float] = []
    window = 10 + int(np.ceil(30 * dim / lam))
    used, generation = 0, 0
    stalled = False
    model = None
    if surrogate in ("lq", "lq2"):
        from .surrogate import QuadraticSurrogate, evaluate_generation

        model = QuadraticSurrogate(dim, style=surrogate)
    # lq2: every step used in the update is capped at this Mahalanobis length
    # (Hansen 2011, injecting external solutions into CMA-ES). The cap covers
    # box-clipped points: with a thin covariance, clipping
    # turns a step into one along a near-null axis, and once the model ranks such a
    # point first, the step-size path explodes.
    step_cap = np.sqrt(dim) + 2 * dim / (dim + 2)
    while used + (1 if model is not None else lam) <= evaluations:
        z = rng.randn(lam, dim)
        y = (z * axes) @ basis.T
        reflect = bounds_mode == "reflect"
        if reflect:
            x = mean + sigma * y
            shown = reflect_into(x, box)
        else:
            x = clip(mean + sigma * y, box.lower, box.upper)
            shown = x
        if model is not None:
            got = evaluate_generation(
                objective.evaluate_point, shown, model, mean, sigma, evaluations - used
            )
            if got is None:
                return EsResult(best_x, float(best_cost), sigma, used, False)
            values, true, spent = got
            used += spent
            truth = np.where(np.isnan(true), np.inf, true)
        else:
            values = np.empty(lam)
            for k in range(lam):
                value = objective.evaluate_point(shown[k])
                if value is None:
                    return EsResult(best_x, float(best_cost), sigma, used, False)
                values[k] = value if np.isfinite(value) else np.inf
                used += 1
            truth = values
        order = np.argsort(values, kind="stable")
        top = int(np.argmin(truth))
        if truth[top] < best_cost:
            best_x, best_cost = shown[top].copy(), float(truth[top])
        inv_sqrt = (basis / axes) @ basis.T
        all_steps = (x - mean) / sigma  # the clipped points' steps
        if surrogate == "lq2":
            lengths = np.linalg.norm(all_steps @ inv_sqrt.T, axis=1)
            all_steps *= np.minimum(1.0, step_cap / np.maximum(lengths, 1e-300))[
                :, None
            ]
        steps = all_steps[order[:mu]]
        step_w = weights @ steps
        mean = mean + sigma * step_w
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
            c_sigma * (2 - c_sigma) * mu_eff
        ) * (inv_sqrt @ step_w)
        generation += 1
        h_sigma = float(
            np.linalg.norm(p_sigma)
            / np.sqrt(1 - (1 - c_sigma) ** (2 * generation))
            / chi_n
            < 1.4 + 2 / (dim + 1)
        )
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * step_w
        rank_mu = (steps.T * weights) @ steps
        decay = 1 - c_1 - c_mu
        if active and len(negative):
            worst = all_steps[order[mu:]]
            whitened = np.sum((worst @ inv_sqrt.T) ** 2, axis=1)
            scaled = negative * dim / np.maximum(whitened, 1e-300)
            rank_mu = rank_mu + (worst.T * scaled) @ worst
            decay = 1 - c_1 - c_mu * (1 + float(negative.sum()))
        cov = (
            decay * cov
            + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * cov)
            + c_mu * rank_mu
        )
        sigma *= float(
            np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
        )
        cov = (cov + cov.T) / 2
        eig, basis = np.linalg.eigh(cov)
        eig = np.maximum(eig, 1e-300)
        axes = np.sqrt(eig)
        history.append(float(truth[top]))
        recent = history[-window:]
        if (
            (len(history) >= window and max(recent) - min(recent) <= TOLHISTFUN)
            or sigma * float(np.max(axes)) < TOLX_SHARE * width
            or float(np.max(eig)) / float(np.min(eig)) > MAX_CONDITION
            or not np.isfinite(sigma)
        ):
            stalled = True
            break
    return EsResult(best_x, float(best_cost), sigma, used, stalled)
