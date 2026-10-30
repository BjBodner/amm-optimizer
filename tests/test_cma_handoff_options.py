"""Options of the CMA phase: the restart rules, the active update and the warm
start."""

import dataclasses

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.bounds import Box
from amm_optimizer.cma_phase import run_cma, warm_factor
from amm_optimizer.config import ATMConfig
from amm_optimizer.objective import CountedObjective
from preset_cases import wave5_variant

BOX = (-5.0, 5.0)


def ellipsoid(dim, seed=0, rotate=True):
    rng = np.random.RandomState(seed)
    q = np.linalg.qr(rng.randn(dim, dim))[0] if rotate else np.eye(dim)
    centre = rng.uniform(-1, 1, dim)
    scales = 1e3 ** (np.arange(dim) / (dim - 1))
    return lambda x: float(np.sum(scales * (q @ (np.asarray(x) - centre)) ** 2)), centre


def test_warm_factor_has_unit_determinant():
    rng = np.random.RandomState(4)
    points = rng.randn(200, 4) * np.array([1.0, 2.0, 3.0, 4.0])
    values = rng.rand(200)
    factor = warm_factor(points, values)
    assert factor is not None
    assert np.linalg.det(factor) == pytest.approx(1.0)
    assert warm_factor(points[:3], values[:3]) is None


@pytest.mark.parametrize(
    ("rule", "window"), [("legacy", 100 + 50 * 2), ("short", 10 + 30 * 2)]
)
def test_restart_rule_sets_the_stall_window(rule, window):
    objective = CountedObjective(lambda x: 1.0, 5000)
    result = run_cma(
        objective,
        np.random.RandomState(0),
        np.zeros(2),
        1.0,
        1.0,
        Box.make(BOX, 2),
        5000,
        restarts=False,
        restart_rule=rule,
    )
    assert result.stalled
    assert result.used == window + 1


def test_economy_keeps_the_long_window_for_the_run_holding_the_best():
    objective = CountedObjective(lambda x: 1.0, 5000)
    result = run_cma(
        objective,
        np.random.RandomState(0),
        np.zeros(2),
        1.0,
        1.0,
        Box.make(BOX, 2),
        5000,
        restarts=False,
        restart_rule="economy",
    )
    assert result.used == 100 + 50 * 2 + 1


def test_active_update_keeps_learning_a_rotated_ellipsoid():
    fun, _ = ellipsoid(6, seed=5)
    box = Box.make(BOX, 6)
    finals = {}
    for active in (False, True):
        objective = CountedObjective(fun, 3000)
        x0 = np.full(6, 2.0)
        result = run_cma(
            objective,
            np.random.RandomState(6),
            x0,
            fun(x0),
            2.0,
            box,
            3000,
            active=active,
        )
        finals[active] = result.cost
    assert np.isfinite(finals[True])
    assert finals[True] < 1e-6


def hyb(**changes):
    variant = dataclasses.replace(
        wave5_variant(), es_kind="cma", cma_handoff=0.2, **changes
    )
    return ATMConfig(iterations_per_hyperparameter_set=1000000, variant=variant)


@pytest.mark.parametrize("changes", [dict(cma_active=True), dict(cma_warm_start=True)])
def test_options_spend_the_budget_and_keep_the_best(changes):
    fun, _ = ellipsoid(6, seed=3)
    seen = []

    def counted(x):
        seen.append(fun(x))
        return seen[-1]

    result = atm.api.minimize(
        counted, np.full(6, 3.0), 3000, seed=4, config=hyb(**changes), bounds=BOX
    )
    assert result.nfev == len(seen) == 3000
    assert result.fun == min(seen)
