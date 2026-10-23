"""``atm.cobyqa_phase``: restarted COBYQA and the ``small_engine`` route (cycle 61)."""

from dataclasses import replace

import numpy as np
import pytest

from amm_optimizer import api, cobyqa_phase, minimize, presets
from amm_optimizer.bounds import Box
from amm_optimizer.objective import CountedObjective

BOX2 = Box.make((-5.0, 5.0), 2)


def counted(fun, allowed):
    calls = []

    def logged(x):
        calls.append(np.array(x))
        return fun(x)

    return CountedObjective(logged, allowed), calls


def ellipse(x):
    x = np.asarray(x, dtype=float)
    return float((x[0] - 1.0) ** 2 + 100.0 * (x[1] + 2.0) ** 2)


def wells(x):
    """Many shallow basins: COBYQA stops early and restarts from uniform points."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2) + 10.0 * np.sum(np.cos(3.0 * x)))


@pytest.mark.parametrize("budget", [5, 20, 200])
def test_cobyqa_spends_at_most_its_budget_inside_the_box(budget):
    objective, calls = counted(wells, 10_000)
    out = cobyqa_phase.run_cobyqa_restarts(
        objective, np.random.RandomState(0), np.zeros(2), BOX2, budget
    )
    assert out.used == len(calls) <= budget
    assert all(np.all(np.abs(c) <= 5.0) for c in calls)
    assert out.cost == min(wells(c) for c in calls) and wells(out.x) == out.cost


def test_cobyqa_solves_a_smooth_basin_from_the_start_point():
    objective, calls = counted(ellipse, 10_000)
    out = cobyqa_phase.run_cobyqa_restarts(
        objective, np.random.RandomState(0), np.array([4.0, 4.0]), BOX2, 200
    )
    # SciPy may shift the start so that the first trust region fits the box
    assert len(calls) <= 200
    assert out.cost < 1e-8 and np.allclose(out.x, [1.0, -2.0], atol=1e-4)


def test_cobyqa_stops_when_the_objective_runs_out():
    objective, calls = counted(wells, 7)
    out = cobyqa_phase.run_cobyqa_restarts(
        objective, np.random.RandomState(0), np.zeros(2), BOX2, 200
    )
    assert out.used == len(calls) == 7


@pytest.mark.parametrize(
    ("dim", "budget_per_dim", "routed"),
    [(2, 10, True), (2, 11, False), (3, 10, False), (2, 30, False)],
)
def test_the_small_engine_route_takes_only_its_envelope(
    monkeypatch, dim, budget_per_dim, routed
):
    seen = []
    real = cobyqa_phase.run_cobyqa_restarts

    def spy(*args, **kwargs):
        seen.append(args[4])
        return real(*args, **kwargs)

    monkeypatch.setattr(api.cobyqa_phase, "run_cobyqa_restarts", spy)
    c42 = presets.get("c42")
    config = replace(c42, variant=replace(c42.variant, small_engine="cobyqa"))
    out = minimize(
        lambda x: float(np.sum(np.square(x))),
        np.full(dim, 3.0),
        budget_per_dim * dim,
        seed=0,
        config=config,
        bounds=(-5, 5),
    )
    assert bool(seen) is routed
    if routed:
        assert seen == [budget_per_dim * dim] and out.nfev <= budget_per_dim * dim


def test_the_small_engine_is_off_by_default():
    assert presets.get("c42").variant.small_engine == "none"
