"""Budget semantics: a budget of B means at most B calls, all of them useful."""

import numpy as np
import pytest

import amm_optimizer as atm


def sphere(x):
    return float(np.sum((np.asarray(x) - 2) ** 2))


class Recorder:
    def __init__(self, fun):
        self.fun = fun
        self.costs = []

    def __call__(self, x):
        self.costs.append(float(self.fun(x)))
        return self.costs[-1]


@pytest.mark.parametrize("budget", [1, 2, 5, 10, 59, 60, 61, 100, 500, 1234])
@pytest.mark.parametrize("dim", [1, 3])
def test_exactly_the_budget_is_used(budget, dim):
    recorder = Recorder(sphere)
    result = atm.minimize(recorder, np.full(dim, 5.0), budget, seed=1)
    assert len(recorder.costs) == budget
    assert result.nfev == budget


def test_restarts_do_not_waste_evaluations():
    recorder = Recorder(lambda x: 1.0)  # flat: forces several restarts
    result = atm.minimize(recorder, np.zeros(3), 14000, seed=0)
    assert result.nrestarts >= 1
    assert len(recorder.costs) == 14000


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("budget", [7, 30, 64, 130, 700])
def test_result_is_the_best_point_ever_evaluated(seed, budget):
    """Includes the partial batch at the end, which the original discarded."""
    recorder = Recorder(sphere)
    x0 = 10 * (np.random.RandomState(seed).rand(3) - 0.5)
    result = atm.minimize(recorder, x0, budget, seed=seed)
    assert result.fun == min(recorder.costs)
    assert sphere(result.x) == result.fun


def test_tiny_budgets_still_improve_on_the_start_point():
    x0 = np.array([5.0, -3.0])
    result = atm.minimize(sphere, x0, 10, seed=0)
    assert result.fun < sphere(x0)


def test_strict_trace_is_a_prefix_of_the_legacy_trace():
    """Same steps as the original; the new accounting just stops earlier."""
    strict, legacy = Recorder(sphere), Recorder(sphere)
    x0 = np.linspace(-4, 4, 6)
    atm.minimize(strict, x0, 900, seed=3)
    atm.minimize(legacy, x0, 900, seed=3, config=atm.ATMConfig(strict_budget=False))
    assert len(strict.costs) == 900
    assert len(legacy.costs) == 902
    assert strict.costs == legacy.costs[:900]


def test_legacy_accounting_is_still_available():
    recorder = Recorder(sphere)
    config = atm.ATMConfig(strict_budget=False)
    result = atm.minimize(recorder, np.zeros(4), 500, seed=1, config=config)
    assert result.nfev == len(recorder.costs) == 502


@pytest.mark.parametrize("budget", [0, -5])
def test_budget_must_be_positive(budget):
    with pytest.raises(ValueError, match="budget"):
        atm.minimize(sphere, np.zeros(2), budget, seed=0)
