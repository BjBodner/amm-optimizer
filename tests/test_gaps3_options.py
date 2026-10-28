"""The gaps3 study's opt-in options: ``sep_route_low``
and ``rugged_direct_beta`` / ``rugged_direct_min_beta``. Each default is c44's."""

import numpy as np
import pytest

from amm_optimizer import chain_phase, presets
from amm_optimizer.api import minimize
from amm_optimizer.bounds import Box
from amm_optimizer.config import Variant
from amm_optimizer.objective import CountedObjective

# Pins research-only options: deselected in the default tier (tools/ci.py).
pytestmark = [
    pytest.mark.research,
    # the options it pins are research-only, deprecated in 0.6.0 (FutureWarning)
    pytest.mark.filterwarnings("ignore:Variant:FutureWarning"),
]


def separable(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - 1.0) ** 2 + 2.0 * np.sin(3.0 * x)))


def rotated(x):
    c, s = np.cos(0.7), np.sin(0.7)
    y = np.array([[c, -s], [s, c]]) @ np.asarray(x, dtype=float)
    return float(y[0] ** 2 + 100.0 * y[1] ** 2)


def stepped(x):
    """Separable plateaus: every single move from the origin is flat."""
    return float(np.sum(np.floor(np.abs(np.asarray(x, dtype=float)) / 20.0)))


def run(config, fun, dim=2, budget=200, seed=3):
    # x0 from another stream than the optimizer's (as in the harness): with the same
    # seed the pair test's random point t would be x0 itself
    x0 = np.random.RandomState(seed + 100).uniform(-5, 5, dim)
    calls = []

    def logged(x):
        calls.append(np.array(x))
        return fun(x)

    res = minimize(logged, x0, budget, seed=seed, config=config, bounds=(-5.0, 5.0))
    return res, np.array(calls)


def test_defaults_are_off():
    v = Variant()
    assert (v.sep_route_low, v.sep_route_low_dim) == ("off", 2)
    assert (v.rugged_direct_beta, v.rugged_direct_min_beta) == (0.0, 0.0)


@pytest.mark.parametrize(
    "bad",
    [
        {"sep_route_low": "yes"},
        {"sep_route_low_dim": 1},
        {"rugged_direct_beta": -1.0},
        {"rugged_direct_min_beta": -0.5},
    ],
)
def test_bad_values_are_refused(bad):
    with pytest.raises(ValueError):
        presets.with_changes("c44", **bad)


def test_strict_pair_test_reads_an_all_flat_test_as_flat():
    box = Box.make((-5.0, 5.0), 2)
    x = np.zeros(2)
    plain = chain_phase.pair_test(
        CountedObjective(stepped, 100), np.random.RandomState(0), x, box
    )
    strict = chain_phase.pair_test(
        CountedObjective(stepped, 100), np.random.RandomState(0), x, box, strict=True
    )
    assert plain is not None and strict is not None
    assert plain[0] == "separable" and strict[0] == "flat"


def test_low_sep_route_takes_separable_d2_runs_and_leaves_the_others():
    c44 = presets.get("c44")
    route = presets.with_changes("c44", sep_route_low="route")
    probe = presets.with_changes("c44", sep_route_low="probe")
    strict = presets.with_changes("c44", sep_route_low="strict")
    for config in (route, probe, strict):
        res, calls = run(config, separable)
        assert len(calls) <= 200 and np.isfinite(res.fun)
    # a rotated quadratic reads non-separable: route and probe are the same run
    _, a = run(route, rotated)
    _, b = run(probe, rotated)
    _, s = run(strict, rotated)
    assert np.array_equal(a, b) and np.array_equal(a, s)
    # the probe's first four evaluations are the pair test, then c44's path
    _, base = run(c44, rotated)
    assert not np.array_equal(a, base)
    # below sep_route_min_per_dim (100 x D) nothing changes
    _, small_a = run(route, separable, budget=150)
    _, small_b = run(c44, separable, budget=150)
    assert np.array_equal(small_a, small_b)


def chain3(x):
    """An f20-like linear chain at D 3 (reads "chain" in the pair test)."""
    m = np.eye(3) + np.diag(np.full(2, 0.25), -1)
    w = m @ (np.asarray(x, dtype=float) - 0.5)
    return float(np.sum(w**2 + 3.0 * (1.0 - np.cos(2.0 * np.pi * w))))


def test_low_sep_route_never_takes_the_chain_branch():
    route = presets.with_changes("c44", sep_route_low="route")
    probe = presets.with_changes("c44", sep_route_low="probe")
    # 300 x D: the chain branch's budget at D >= 5
    _, a = run(route, chain3, dim=3, budget=900)
    _, b = run(probe, chain3, dim=3, budget=900)
    assert np.array_equal(a, b)


def rugged(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2 + 10.0 * (1.0 - np.cos(2.0 * np.pi * x))))


def test_rugged_direct_runs_and_is_off_outside_its_band():
    base = presets.get("c44")
    on = presets.with_changes("c44", rugged_direct_beta=1.0, rugged_direct_min_beta=0.5)
    never = presets.with_changes(
        "c44", rugged_direct_beta=1.0, rugged_direct_min_beta=0.99
    )
    res, calls = run(on, rugged, dim=5, budget=1500)
    assert len(calls) <= 1500 and np.isfinite(res.fun)
    # a band no reading falls into leaves c44's run unchanged on a smooth function
    _, a = run(never, separable, dim=5, budget=1500)
    _, b = run(base, separable, dim=5, budget=1500)
    assert np.array_equal(a, b)
