"""Objectives that fail (NaN, inf) in part of the domain must not crash ATM."""

import warnings
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.objective import CountedObjective
from amm_optimizer.profiling import TimingProfiler


def sphere(x):
    return float(np.sum((np.asarray(x) - 2) ** 2))


def nan_region(x):
    return np.nan if x[0] > 3 else sphere(x)


def inf_region(x):
    return np.inf if x[0] > 3 else sphere(x)


def neg_inf_region(x):
    return -np.inf if x[0] > 3 else sphere(x)


@pytest.mark.parametrize("bad", [nan_region, inf_region, neg_inf_region])
def test_failed_evaluations_do_not_crash_and_still_improve(bad):
    x0 = np.array([-2.0, 1.0, 0.5])
    result = atm.api.minimize(bad, x0, 1000, seed=6)
    assert np.isfinite(result.fun)
    assert result.fun < sphere(x0)
    assert result.x[0] <= 3  # never settles inside the failing region


def test_only_the_start_point_is_valid():
    def only_start(x):
        return 5.0 if np.allclose(x, 1.0) else np.nan

    result = atm.api.minimize(only_start, np.ones(3), 400, seed=0)
    np.testing.assert_array_equal(result.x, np.ones(3))
    assert result.fun == 5.0


@pytest.mark.parametrize("start_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("config", [None, "atm", "composite", "c23"])
def test_a_non_finite_start_value_is_a_failed_evaluation(start_value, config):
    """Decision D2 (b): the run carries on from a start where ``fun`` fails."""

    def undefined_at_the_centre(x):
        return start_value if np.all(x == 0.0) else float(np.sum((x - 1.0) ** 2))

    result = atm.minimize(
        undefined_at_the_centre, bounds=[(-5, 5)] * 3, budget=600, seed=0, config=config
    )
    assert np.isfinite(result.fun) and result.fun < 1.0
    assert result.nfev == 600


def test_a_run_where_every_value_fails_returns_inf_at_the_start():
    result = atm.api.minimize(lambda x: np.nan, np.zeros(2), 200, seed=0)
    assert result.fun == np.inf
    np.testing.assert_array_equal(result.x, np.zeros(2))
    assert (result.nfev, result.nit, result.status) == (200, 0, 0)


def test_failure_handling_does_not_change_results_for_finite_objectives():
    """Same input, same stream: unchanged versus the recorded trajectory."""
    rng = np.random.RandomState(0)
    x0 = 10 * (rng.rand(2) - 0.5)
    legacy = atm.ATMConfig(strict_budget=False)  # the recorded value uses it
    result = atm.api.minimize(sphere, x0, 1000, rng=rng, config=legacy)
    assert result.fun == pytest.approx(2.0437191838110754e-06, rel=1e-12)


def _vandal(x):
    """Writes into its argument after evaluating it (review B1)."""
    value = float(np.sum((x - 1.0) ** 2))
    x[:] = 99.0
    return value


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"anytime": True},
        {"config": atm.ATMConfig()},
        {"config": atm.ATMConfig.composite()},
    ],
    ids=["default", "anytime", "atm", "composite"],
)
def test_fun_writing_into_x_does_not_change_the_search(kwargs):
    """``fun`` gets its own copy of every point, as in SciPy: the run is the run of
    the same objective that leaves ``x`` alone."""
    box = [(-5, 5)] * 3
    result = atm.minimize(_vandal, bounds=box, budget=300, seed=0, **kwargs)
    clean = atm.minimize(
        lambda x: float(np.sum((x - 1.0) ** 2)),
        bounds=box,
        budget=300,
        seed=0,
        **kwargs,
    )
    assert not np.any(result.x == 99.0)
    assert result.fun == clean.fun
    np.testing.assert_array_equal(result.x, clean.x)
    if "config" not in kwargs or kwargs["config"] != atm.ATMConfig():
        assert result.fun < 1e-6  # AMM solves it; the original sampler alone is slow


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"anytime": True}, {"config": atm.ATMConfig()}],
    ids=["default", "anytime", "atm"],
)
def test_fun_that_keeps_x_sees_distinct_arrays(kwargs):
    kept = []

    def keeper(x):
        kept.append(x)
        return float(np.sum((x - 1.0) ** 2))

    atm.minimize(keeper, bounds=[(-5, 5)] * 3, budget=120, seed=0, **kwargs)
    assert len({id(x) for x in kept}) == len(kept)
    assert not any(np.shares_memory(a, b) for a, b in pairwise(kept))
    assert all(x.base is None and x.flags.writeable for x in kept)


@pytest.mark.parametrize("kind", [np.float32, Fraction], ids=["float32", "Fraction"])
def test_the_start_value_is_cast_to_float_like_every_other_value(kind):
    """Review B3: ``evaluate_start`` returned ``fun``'s raw value."""
    objective = CountedObjective(lambda x: kind(3), 10)
    assert type(objective.evaluate_start(np.zeros(2))) is float
    with_profile = CountedObjective(lambda x: kind(3), 10, profiler=TimingProfiler())
    assert type(with_profile.evaluate_start(np.zeros(2))) is float


@pytest.mark.parametrize("kind", [np.float32, Fraction], ids=["float32", "Fraction"])
@pytest.mark.parametrize("config", [None, atm.ATMConfig()], ids=["default", "atm"])
def test_real_scalars_of_any_type_are_accepted(kind, config):
    def f(x):
        return kind(float(np.sum((x - 1.0) ** 2)))

    result = atm.minimize(f, bounds=[(-5, 5)] * 2, budget=60, seed=0, config=config)
    assert type(result.fun) is float and np.isfinite(result.fun)


def _warns_on_every_call(x):
    warnings.warn("from the objective", UserWarning, stacklevel=2)
    return float(np.sum((x - 1.0) ** 2) + np.sum(np.cos(3.0 * x)))


@pytest.mark.parametrize(
    ("dim", "per_dim"),
    [(2, 10), (3, 100), (5, 10)],
    ids=["small-engine COBYQA", "basin route (SLSQP, Sobol)", "COBYQA from a point"],
)
def test_warnings_from_fun_reach_the_caller_on_every_route(dim, per_dim):
    """Review B5: the SciPy blocks ignored every warning, the user's too."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = atm.minimize(
            _warns_on_every_call, bounds=[(-5, 5)] * dim, budget=per_dim * dim, seed=0
        )
    ours = [w for w in caught if str(w.message) == "from the objective"]
    assert len(ours) == result.nfev


def test_a_warning_from_fun_is_seen_by_pytest_warns():
    with pytest.warns(UserWarning, match="from the objective"):
        atm.minimize(_warns_on_every_call, bounds=[(-5, 5)] * 2, budget=20, seed=0)


def _inf_beyond_one(x):
    return float(np.sum((x - 1.0) ** 2)) if x[0] < 1.0 else np.inf


def _inf_outside_a_ball(x):
    return float(np.sum((x - 2.0) ** 2)) if np.sum(x**2) < 9.0 else np.inf


@pytest.mark.parametrize(
    ("fun", "dim", "budget"),
    [
        (_inf_beyond_one, 3, 300),
        (_inf_beyond_one, 10, 300),
        (_inf_outside_a_ball, 10, 1000),
    ],
)
def test_an_objective_returning_inf_raises_no_runtime_warning(fun, dim, budget):
    """SLSQP's finite differences over a failed evaluation overflowed, and warned."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = atm.minimize(fun, bounds=[(-5, 5)] * dim, budget=budget, seed=1)
    assert np.isfinite(result.fun)
    assert [str(w.message) for w in caught if w.category is RuntimeWarning] == []
