"""Objectives that fail (NaN, inf) in part of the domain must not crash ATM."""

import numpy as np
import pytest

import amm_optimizer as atm


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
def test_start_point_must_have_a_finite_objective_value(start_value):
    with pytest.raises(ValueError, match="finite"):
        atm.api.minimize(lambda x: start_value, np.zeros(2), 200, seed=0)


def test_failure_handling_does_not_change_results_for_finite_objectives():
    """Same input, same stream: unchanged versus the recorded trajectory."""
    rng = np.random.RandomState(0)
    x0 = 10 * (rng.rand(2) - 0.5)
    legacy = atm.ATMConfig(strict_budget=False)  # the recorded value uses it
    result = atm.api.minimize(sphere, x0, 1000, rng=rng, config=legacy)
    assert result.fun == pytest.approx(2.0437191838110754e-06, rel=1e-12)
