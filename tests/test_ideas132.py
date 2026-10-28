"""The research options of a screen of 132 ideas: the
central-difference polish of the sampled restarts (#33), the false-success guard
(#23) and the floor hand-over (#24 / #32); off by default, so the presets are
unchanged."""

import numpy as np
import pytest

from amm_optimizer import presets, qn_phase
from amm_optimizer.api import minimize
from amm_optimizer.bounds import Box
from amm_optimizer.config import Variant
from amm_optimizer.objective import CountedObjective

# Pins research-only options: deselected in the default tier (tools/ci.py).
pytestmark = pytest.mark.research

BOX5 = Box.make((-5.0, 5.0), 5)


def different_powers(x):
    """bbob f14's shape: a kink at the optimum (the square root of a sum of powers)."""
    z = np.asarray(x, dtype=float) - 0.5
    powers = 2 + 4 * np.arange(len(z)) / (len(z) - 1)
    return float(np.sqrt(np.sum(np.abs(z) ** powers)))


def two_basins(x):
    """Two quadratic basins, the better one at +2 (a Gallagher-like reading)."""
    x = np.asarray(x, dtype=float)
    return float(min(np.sum((x - 2.0) ** 2), 1.0 + np.sum((x + 2.0) ** 2)))


def test_the_options_default_off():
    variant = Variant()
    assert variant.sampled_polish_per_dim == 0.0
    assert variant.qn_false_success is False
    assert variant.qn_floor == "none"
    with pytest.raises(ValueError):
        Variant(qn_floor="maybe")
    with pytest.raises(ValueError):
        Variant(sampled_polish_per_dim=-1.0)


def test_the_polish_spends_at_most_its_cap_and_never_loses():
    rng = np.random.RandomState(3)
    x0 = np.full(5, -3.0)
    plain = qn_phase.run_sampled_restarts(
        CountedObjective(two_basins, 4000),
        np.random.RandomState(3),
        x0,
        two_basins(x0),
        BOX5,
        400,
    )
    polished = qn_phase.run_sampled_restarts(
        CountedObjective(two_basins, 4000),
        rng,
        x0,
        two_basins(x0),
        BOX5,
        400,
        polish_per_dim=10.0,
    )
    assert polished.used <= 400 and plain.used <= 400
    assert np.isfinite(polished.cost)


def test_the_central_polish_stops_at_its_cap():
    calls = []

    def evaluate(z):
        calls.append(np.array(z))
        return different_powers(z)

    qn_phase._central_polish(evaluate, np.zeros(5), BOX5, 1e-3, 37)
    assert 0 < len(calls) <= 37


def test_the_false_success_guard_relabels_an_unmoved_exit():
    flat = lambda x: 1.0  # noqa: E731 - SLSQP exits at once with status 0
    ends: list = []
    qn_phase.run_slsqp_restarts(
        CountedObjective(flat, 1000),
        np.random.RandomState(0),
        np.zeros(5),
        1.0,
        BOX5,
        60,
        ends_out=ends,
        max_descents=1,
        false_success=True,
    )
    assert [d.status for d in ends] == [-1]
    ends = []
    qn_phase.run_slsqp_restarts(
        CountedObjective(flat, 1000),
        np.random.RandomState(0),
        np.zeros(5),
        1.0,
        BOX5,
        60,
        ends_out=ends,
        max_descents=1,
    )
    assert [d.status for d in ends] == [0]


@pytest.mark.parametrize("action", ["bfgs", "cd"])
def test_the_floor_hand_over_keeps_the_budget_and_the_best(action):
    x0 = np.full(5, 3.0)
    for floor in ("", action):
        objective = CountedObjective(different_powers, 3000)
        out = qn_phase.run_slsqp_restarts(
            objective,
            np.random.RandomState(0),
            x0,
            different_powers(x0),
            BOX5,
            3000,
            fd_step=qn_phase.QN_FD_STEP,
            floor_bfgs=floor,
        )
        assert out.used <= 3000 and objective.calls == out.used
        assert out.cost <= different_powers(x0)


@pytest.mark.parametrize(
    "change",
    [
        {"sampled_polish_per_dim": 200.0},
        {"qn_false_success": True},
        {"qn_floor": "bfgs"},
    ],
)
def test_the_options_run_inside_the_c43_composite(change):
    config = presets.with_changes("c43", **change)
    result = minimize(
        different_powers,
        np.full(5, 3.0),
        1500,
        seed=1,
        config=config,
        bounds=(-5.0, 5.0),
    )
    assert result.nfev <= 1500 and np.isfinite(result.fun)


def test_the_low_budget_atm_cap_matches_the_plain_cap_up_to_its_budget():
    def sphere(x):
        return float(np.sum((np.asarray(x) - 0.3) ** 2))

    x0 = np.full(10, 2.0)
    low = presets.with_changes("c43", cma_handoff_cap_low_per_dim=20.0)
    plain20 = presets.with_changes("c43", cma_handoff_cap_per_dim=20.0)
    base = presets.get("c43")

    def run(config, budget):
        r = minimize(sphere, x0, budget, seed=4, config=config, bounds=(-5.0, 5.0))
        return r.fun, r.nfev

    assert run(low, 3000) == run(plain20, 3000)  # 300 x D: the low cap applies
    assert run(low, 3100) == run(base, 3100)  # above 300 x D: c43's cap
    with pytest.raises(ValueError):
        Variant(cma_handoff_cap_low_per_dim=-1.0)
