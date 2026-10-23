"""The (mu/mu_w, lambda)-CMA-ES and the BIPOP-style regimes (``cma_regimes``)."""

import dataclasses

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.bounds import Box
from amm_optimizer.config import ATMConfig, Variant
from amm_optimizer.objective import CountedObjective
from amm_optimizer.pop_cma import default_population, run_population_cma
from amm_optimizer.regimes import run_regimes
from preset_cases import wave5_variant

BOX = (-5.0, 5.0)


def rotated_ellipsoid(dim, seed=0):
    rng = np.random.RandomState(seed)
    q = np.linalg.qr(rng.randn(dim, dim))[0]
    scales = 1e4 ** (np.arange(dim) / (dim - 1))
    return lambda x: float(np.sum(scales * (q @ (np.asarray(x) - 1.0)) ** 2))


def rastrigin(x):
    z = np.asarray(x) - 1.0
    return float(10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z)))


def test_default_population():
    assert [default_population(d) for d in (2, 5, 10, 20)] == [6, 8, 10, 12]


def test_population_cma_solves_a_rotated_ellipsoid_inside_the_box():
    fun = rotated_ellipsoid(6)
    objective = CountedObjective(fun, 8000)
    result = run_population_cma(
        objective, np.random.RandomState(1), np.zeros(6), 2.0, Box.make(BOX, 6), 8000, 8
    )
    assert result.cost < 1e-8
    assert result.used == objective.calls <= 8000
    assert (np.abs(result.x) <= 5).all()


def test_population_cma_stalls_on_a_plateau():
    objective = CountedObjective(lambda x: 1.0, 5000)
    result = run_population_cma(
        objective, np.random.RandomState(2), np.zeros(3), 2.0, Box.make(BOX, 3), 5000, 7
    )
    assert result.stalled
    assert result.used < 5000


@pytest.mark.parametrize("reward", [False, True])
def test_regimes_spend_the_budget_and_keep_the_best(reward):
    seen = []

    def counted(x):
        seen.append(rastrigin(x))
        return seen[-1]

    objective = CountedObjective(counted, 6000)
    x0 = np.full(4, 3.0)
    cost = counted(x0)
    objective.calls = 1
    x, best, runs = run_regimes(
        objective,
        np.random.RandomState(3),
        x0,
        cost,
        1.0,
        Box.make(BOX, 4),
        6000,
        reward,
        {},
    )
    assert objective.calls == 6000
    assert best == min(seen)
    assert runs >= 3


def hyb(**changes):
    variant = dataclasses.replace(
        wave5_variant(), es_kind="cma", cma_handoff=0.2, **changes
    )
    return ATMConfig(iterations_per_hyperparameter_set=1000000, variant=variant)


@pytest.mark.parametrize("regimes", ["bipop", "nbipop"])
def test_minimize_with_regimes(regimes):
    seen = []

    def counted(x):
        seen.append(rastrigin(x))
        return seen[-1]

    result = atm.api.minimize(
        counted,
        np.full(5, 3.0),
        5000,
        seed=4,
        config=hyb(cma_regimes=regimes),
        bounds=BOX,
    )
    assert result.nfev == len(seen) == 5000
    assert result.fun == min(seen)


def test_regimes_need_the_handoff():
    with pytest.raises(ValueError, match="cma_handoff"):
        Variant(amplitude_control="success_rate_targeted", cma_regimes="bipop")


@pytest.mark.parametrize("doubling", [False, True])
def test_population_restarts_spend_the_budget_and_keep_the_best(doubling):
    from amm_optimizer.pop_cma import run_population_restarts

    seen = []

    def counted(x):
        seen.append(rastrigin(x))
        return seen[-1]

    objective = CountedObjective(counted, 4000)
    x0 = np.full(4, 3.0)
    result = run_population_restarts(
        objective,
        np.random.RandomState(5),
        x0,
        rastrigin(x0),
        1.0,
        Box.make(BOX, 4),
        4000,
        doubling,
    )
    assert objective.calls <= 4000
    assert result.used == objective.calls
    assert result.cost == min(seen)


@pytest.mark.parametrize("local", ["population", "ipop"])
def test_minimize_with_a_population_local_phase(local):
    seen = []

    def counted(x):
        seen.append(rastrigin(x))
        return seen[-1]

    config = hyb(cma_local=local, cma_local_min_dim=2)
    result = atm.api.minimize(
        counted, np.full(5, 3.0), 5000, seed=6, config=config, bounds=BOX
    )
    assert result.nfev == len(seen) <= 5000
    assert result.fun == min(seen)


def test_minimize_with_the_one_plus_one_first_then_ipop():
    seen = []

    def counted(x):
        seen.append(rastrigin(x))
        return seen[-1]

    config = hyb(cma_local="then_ipop", cma_local_min_dim=2, cma_restart_rule="short")
    result = atm.api.minimize(
        counted, np.full(5, 3.0), 5000, seed=7, config=config, bounds=BOX
    )
    assert result.nfev == len(seen) <= 5000
    assert result.fun == min(seen)
