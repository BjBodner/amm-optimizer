"""The CMA phase's diagonal model, the population runs and their options, the
reflection into the box and the separable search."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer import cma_phase
from amm_optimizer.bounds import Box
from amm_optimizer.cma_phase import model_step
from amm_optimizer.config import ATMConfig
from amm_optimizer.objective import CountedObjective
from amm_optimizer.pop_cma import run_population_cma

BOX = (-5.0, 5.0)
# a recorded hybrid configuration with the CMA hand-over
FINAL = (
    "variant.amplitude_control=success_rate_targeted,variant.initial_design"
    "=uniform,variant.bounds=fold,variant.corner_clip=0.25,variant.pool_rul"
    "e=budget,variant.pool_min=12,variant.reevaluate_incumbent=false,varian"
    "t.es_phase=prefix,variant.es_fraction=0.9,variant.es_max_budget_per_di"
    "m=100,variant.covariance=rank_mu,variant.cov_rate=3,variant.cov_select"
    "=0.5,variant.es_kind=cma,variant.cma_handoff=0.2,iterations_per_hyperp"
    "arameter_set=1000000,variant.cma_model=quadratic,variant.cma_active=tr"
    "ue,variant.cma_warm_start=true,variant.cma_restart_rule=short,variant."
    "cma_handoff_cap_per_dim=60"
)


def final(**changes):
    """``FINAL`` plus ``changes``."""
    fields: dict[str, object] = {}
    top: dict[str, object] = {}
    for item in FINAL.split(","):
        key, value = item.split("=")
        parsed: object = value
        if value in ("true", "false"):
            parsed = value == "true"
        else:
            for kind in (int, float):
                try:
                    parsed = kind(value)
                    break
                except ValueError:
                    continue
        if key.startswith("variant."):
            fields[key[len("variant.") :]] = parsed
        else:
            top[key] = parsed
    fields.update(changes)
    return ATMConfig(variant=atm.config.Variant(**fields), **top)


def sphere_shift(dim):
    shift = np.linspace(-2, 2, dim)
    return lambda x: float(np.sum((np.asarray(x) - shift) ** 2))


def test_model_step_full_and_diagonal_reach_a_quadratic_minimum():
    rng = np.random.RandomState(0)
    dim = 4
    box = Box.make(BOX, dim)
    target = np.array([1.0, -1.0, 0.5, 2.0])
    scales = np.array([1.0, 4.0, 9.0, 16.0])
    points = rng.uniform(-3, 3, (60, dim))
    values = np.sum(scales * (points - target) ** 2, axis=1)
    x = np.zeros(dim)
    for rung, weighted in (("full", False), ("full", True), ("diagonal", True)):
        found = model_step(
            points, values, x, 1.0, np.eye(dim), np.eye(dim), rung, weighted, 10.0, box
        )
        assert found is not None
        candidate, predicted = found
        assert np.allclose(candidate, target, atol=1e-6)
        actual = float(np.sum(scales * (x - target) ** 2))
        assert predicted == pytest.approx(actual, rel=1e-6)


def test_diagonal_model_falls_back_to_a_gradient_step():
    dim = 3
    box = Box.make(BOX, dim)
    points = np.random.RandomState(1).uniform(-1, 1, (30, dim))
    values = points @ np.array([1.0, 2.0, 3.0])  # linear: no curvature
    found = model_step(
        points,
        values,
        np.zeros(dim),
        1.0,
        np.eye(dim),
        np.eye(dim),
        "diagonal",
        False,
        2.0,
        box,
    )
    assert found is not None
    step = found[0]
    assert np.linalg.norm(step) == pytest.approx(2.0)
    assert step @ np.array([1.0, 2.0, 3.0]) < 0


def test_active_population_cma_solves_an_ellipsoid():
    dim = 8
    box = Box.make(BOX, dim)
    scales = 10.0 ** np.linspace(0, 3, dim)

    def ellipsoid(x):
        return float(np.sum(scales * (np.asarray(x) - 1.0) ** 2))

    finals = {}
    for active in (False, True):
        objective = CountedObjective(ellipsoid, 12000)
        run = run_population_cma(
            objective,
            np.random.RandomState(3),
            np.full(dim, -2.0),
            2.0,
            box,
            12000,
            10,
            active,
        )
        finals[active] = run.cost
    assert finals[True] < 1e-8
    assert finals[True] <= finals[False] * 10


def test_pop_gate_needs_then_ipop():
    with pytest.raises(ValueError):
        dataclasses.replace(final().variant, pop_gate_per_dim=400.0)


def test_reflect_into_stays_in_the_box_and_is_identity_inside():
    from amm_optimizer.pop_cma import reflect_into

    box = Box.make(BOX, 3)
    x = np.array([[0.5, -4.0, 4.9], [6.0, -7.0, 25.0]])
    y = reflect_into(x, box)
    assert np.allclose(y[0], x[0])
    assert np.allclose(y[1], [4.0, -3.0, 5.0])


def test_reflected_active_population_solves_a_corner_slope():
    """The defect of 11c: clipped steps made the active update collapse on f5."""
    dim = 6
    box = Box.make(BOX, dim)

    def slope(x):
        return float(np.sum(5.0 - np.asarray(x)))  # minimum at the upper corner

    objective = CountedObjective(slope, 6000)
    run = run_population_cma(
        objective,
        np.random.RandomState(2),
        np.zeros(dim),
        2.0,
        box,
        6000,
        12,
        True,
        "reflect",
    )
    assert run.cost < 1e-6


def test_step_phase_solves_a_separable_rastrigin_and_rejects_a_rotated_one():
    from amm_optimizer.step_phase import run_step_phase

    dim = 8
    box = Box.make(BOX, dim)
    shift = np.linspace(-3, 3, dim)

    def rastrigin(x):
        z = np.asarray(x) - shift
        return float(10 * dim + np.sum(z**2 - 10 * np.cos(2 * np.pi * z)))

    q, _ = np.linalg.qr(np.random.RandomState(0).randn(dim, dim))

    def rotated(x):
        return rastrigin(q @ (np.asarray(x) - shift) + shift)

    x0 = np.zeros(dim)
    sep = run_step_phase(
        CountedObjective(rastrigin, 3000), x0, rastrigin(x0), box, 3000
    )
    assert sep.cost < 1e-6 and not sep.stalled
    rot = run_step_phase(CountedObjective(rotated, 3000), x0, rotated(x0), box, 3000)
    assert rot.stalled and rot.used <= 4 * dim


@pytest.mark.parametrize(
    "changes",
    [
        dict(
            cma_local="then_ipop",
            cma_local_min_dim=5,
            pop_lambda="fixed2",
            pop_gate_per_dim=50.0,
        ),
    ],
)
def test_options_spend_the_budget_and_keep_the_best(changes):
    fun = sphere_shift(6)
    seen = []

    def counted(x):
        seen.append(fun(x))
        return seen[-1]

    result = atm.api.minimize(
        counted, np.full(6, 3.0), 3000, seed=4, config=final(**changes), bounds=BOX
    )
    assert result.nfev == len(seen) == 3000
    assert result.fun == min(seen)
    assert result.fun < 1e-6


def test_the_diagonal_model_runs_above_the_full_models_dimension(monkeypatch):
    """``cma_model_high="diagonal"`` above ``CMA_MODEL_MAX_DIM`` (10): the hand-over's
    CMA phase takes the diagonal model's steps, spends the budget and keeps the best."""
    dim = 12
    fun = sphere_shift(dim)
    seen = []

    def counted(x):
        seen.append(fun(x))
        return seen[-1]

    rungs = []
    real = cma_phase.model_step

    def spy(*args, **kwargs):
        rungs.append(args[6])
        return real(*args, **kwargs)

    monkeypatch.setattr(cma_phase, "model_step", spy)
    config = final(cma_model_high="diagonal", es_max_budget_per_dim=10.0)
    result = atm.api.minimize(
        counted, np.full(dim, 3.0), 300 * dim, seed=4, config=config, bounds=BOX
    )
    assert result.nfev == len(seen) == 300 * dim
    assert result.fun == min(seen)
    assert rungs and set(rungs) == {"diagonal"}


@pytest.mark.parametrize(
    "changes",
    [
        dict(sep_probe=True),
        dict(
            cma_local="then_ipop",
            cma_local_min_dim=5,
            pop_gate_per_dim=50.0,
            cma_probe_at_per_dim=20.0,
            cma_probe_beta=1.5,
        ),
        dict(nm_max_dim=6),
        dict(
            cma_local="then_ipop",
            cma_local_min_dim=5,
            pop_lambda="fixed2",
            pop_protect=True,
            pop_gate_per_dim=50.0,
            pop_switch_per_dim=20.0,
        ),
        dict(
            cma_local="then_ipop",
            cma_local_min_dim=5,
            pop_protect=True,
            pop_active=True,
            pop_gate_per_dim=50.0,
        ),
    ],
)
def test_population_options_spend_the_budget(changes):
    fun = sphere_shift(6)
    seen = []

    def counted(x):
        seen.append(fun(x))
        return seen[-1]

    result = atm.api.minimize(
        counted, np.full(6, 3.0), 3000, seed=5, config=final(**changes), bounds=BOX
    )
    assert result.nfev == len(seen) == 3000
    assert result.fun == min(seen) < 1e-6
