import numpy as np
import pytest

import amm_optimizer as atm


def sphere(x):
    return float(np.sum((np.asarray(x) - 2) ** 2))


def test_public_api():
    assert atm.__all__ == [
        "ATMConfig",
        "OptimizeResult",
        "Profiler",
        "Timings",
        "TimingProfiler",
        "minimize",
        "presets",
        "__version__",
    ]
    assert isinstance(atm.__version__, str)


def test_minimizes_the_sphere():
    x0 = np.zeros(5)
    result = atm.minimize(sphere, x0, budget=2000, seed=0)
    assert isinstance(result, atm.OptimizeResult)
    assert result.fun < 1e-3 * sphere(x0)
    assert result.fun == pytest.approx(sphere(result.x))


def test_seed_makes_runs_reproducible():
    first = atm.minimize(sphere, np.zeros(3), 600, seed=7)
    second = atm.minimize(sphere, np.zeros(3), 600, seed=7)
    other = atm.minimize(sphere, np.zeros(3), 600, seed=8)
    np.testing.assert_array_equal(first.x, second.x)
    assert not np.array_equal(first.x, other.x)


def test_explicit_rng_continues_its_stream():
    rng = np.random.RandomState(5)
    a = atm.minimize(sphere, np.zeros(3), 300, rng=rng)
    b = atm.minimize(sphere, np.zeros(3), 300, rng=rng)
    assert not np.array_equal(a.x, b.x)


def test_input_is_not_modified_and_lists_are_accepted():
    x0 = np.array([5.0, -3.0])
    before = x0.copy()
    atm.minimize(sphere, x0, 300, seed=0)
    np.testing.assert_array_equal(x0, before)
    assert atm.minimize(sphere, [5.0, -3.0], 300, seed=0).x.shape == (2,)


def test_one_dimensional_problems_work():
    result = atm.minimize(sphere, np.array([8.0]), 600, seed=0)
    assert result.x.shape == (1,)
    assert result.fun < sphere(np.array([8.0]))


def test_nfev_counts_every_call_to_the_objective():
    calls = []

    def counting(x):
        calls.append(1)
        return sphere(x)

    result = atm.minimize(counting, np.zeros(4), budget=500, seed=1)
    assert result.nfev == len(calls)
    assert result.nfev == 500


def test_flat_objective_triggers_restarts_and_still_finishes():
    result = atm.minimize(lambda x: 1.0, np.zeros(3), budget=9000, seed=0)
    assert result.nrestarts >= 1
    assert result.fun == 1.0


def test_a_worse_later_search_never_replaces_the_best_result():
    """With a drifting objective a restart can end worse than an earlier one."""
    calls = []

    def drifting(x):
        calls.append(1)
        return 1.0 if len(calls) < 7000 else 5.0

    result = atm.minimize(drifting, np.zeros(3), budget=14000, seed=0)
    assert result.nrestarts >= 1
    assert result.fun == 1.0


def test_custom_config_changes_the_trajectory():
    default = atm.minimize(sphere, np.zeros(3), 600, seed=0)
    fewer = atm.minimize(
        sphere,
        np.zeros(3),
        600,
        seed=0,
        config=atm.ATMConfig(iterations_per_hyperparameter_set=3),
    )
    assert not np.array_equal(default.x, fewer.x)


def test_does_not_leak_floating_point_error_state():
    before = np.geterr()
    atm.minimize(sphere, np.zeros(2), 200, seed=0)
    assert np.geterr() == before


def test_import_has_no_side_effects_on_the_global_random_state():
    np.random.seed(123)
    expected = np.random.rand()
    np.random.seed(123)
    import importlib

    importlib.reload(atm)
    assert np.random.rand() == expected
