"""BIPOP-style restart regimes for the hand-over phase (variant ``cma_regimes``).

After the first (1+1)-CMA-ES run from ATM's best point stalls, every further run
starts from a uniform point of the box and is one of two kinds:

* a small-regime run ("S"): the (1+1)-CMA-ES of ``cma_phase`` with a random
  start step ``0.2 * width * 10 ** (-2 U)``, at most half as long as the latest
  large-regime run (Hansen 2009);
* a large-regime run ("L"): the non-elitist (mu/mu_w, lambda)-CMA-ES of
  ``pop_cma`` at ``0.2 * width``, with lambda doubling from the default at
  every large run (the population that smooths rugged landscapes).

The next run goes to the regime that has used fewer evaluations; with
``reward=True`` (NBIPOP, Loshchilov et al. 2012) the regime that found the best
point counts its evaluations at half.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .bounds import Box
from .cma_phase import run_cma
from .es_phase import INITIAL_SIGMA_SHARE
from .objective import CountedObjective
from .pop_cma import default_population, run_population_cma


def run_regimes(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    sigma: float,
    box: Box,
    budget: int,
    reward: bool,
    cma_kwargs: dict[str, Any],
) -> tuple[np.ndarray, float, int]:
    """Spend the rest of ``budget``; returns the best point, its cost, and runs."""
    dim = len(x)
    width = float(np.mean(box.upper - box.lower))
    best_x, best_cost = np.array(x, dtype=float), float(cost)
    first = run_cma(
        objective,
        rng,
        best_x,
        best_cost,
        sigma,
        box,
        budget - objective.calls,
        restarts=False,
        **cma_kwargs,
    )
    if first.cost <= best_cost:
        best_x, best_cost = first.x, first.cost
    used = {"S": first.used, "L": 0}
    holder = "S"
    large_runs, last_large, runs = 0, 0, 1
    while objective.calls < budget:
        remaining = budget - objective.calls
        weight = {r: (0.5 if reward and holder == r else 1.0) for r in used}
        lam = default_population(dim) * 2**large_runs
        go_large = used["L"] * weight["L"] <= used["S"] * weight["S"] and (
            remaining >= lam
        )
        start = box.lower + (box.upper - box.lower) * rng.rand(dim)
        if go_large:
            run = run_population_cma(
                objective, rng, start, INITIAL_SIGMA_SHARE * width, box, remaining, lam
            )
            regime = "L"
            large_runs += 1
            last_large = run.used
        else:
            value = objective.evaluate_point(start)
            if value is None:
                break
            if value < best_cost:
                best_x, best_cost = start, float(value)
            cap = remaining - 1
            if last_large:
                cap = max(1, min(cap, last_large // 2))
            start_sigma = INITIAL_SIGMA_SHARE * width * 10 ** (-2 * rng.rand())
            run = run_cma(
                objective,
                rng,
                start,
                float(value),
                start_sigma,
                box,
                cap,
                restarts=False,
                **cma_kwargs,
            )
            regime = "S"
            run = type(run)(run.x, run.cost, run.sigma, run.used + 1, run.stalled)
        used[regime] += run.used
        runs += 1
        if run.cost < best_cost:
            best_x, best_cost = run.x, run.cost
            holder = regime
        if run.used == 0:
            break
    return best_x, best_cost, runs
