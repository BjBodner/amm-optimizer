"""Parent selection: the fast best-eighth must equal the three-median definition."""

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import amm_optimizer as atm
from amm_optimizer.state import SearchState, _best_eighth_by_medians, best_eighth

costs = st.lists(
    st.one_of(
        st.floats(-1e12, 1e12, allow_nan=False),
        st.sampled_from([0.0, -0.0, 1.0, 2.0, np.inf, -np.inf]),  # ties, extremes
    ),
    min_size=1,
    max_size=80,
).map(np.array)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@settings(max_examples=600, deadline=None)
@given(costs)
def test_best_eighth_equals_the_median_definition(cost):
    np.testing.assert_array_equal(best_eighth(cost), _best_eighth_by_medians(cost))


@pytest.mark.parametrize("n", range(1, 70))
def test_best_eighth_on_every_size_with_heavy_ties(n):
    rng = np.random.RandomState(n)
    cost = rng.randint(0, 4, n).astype(float)
    np.testing.assert_array_equal(best_eighth(cost), _best_eighth_by_medians(cost))


def test_best_eighth_is_never_empty_and_contains_the_winner():
    rng = np.random.RandomState(0)
    for _ in range(200):
        cost = rng.randn(rng.randint(1, 70))
        chosen = best_eighth(cost)
        assert chosen.size >= 1
        assert int(np.argmin(cost)) in chosen


def test_best_eighth_with_nan_keeps_numpys_semantics():
    cost = np.array([1.0, np.nan, 0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.testing.assert_array_equal(best_eighth(cost), _best_eighth_by_medians(cost))
        assert best_eighth(np.array([])).size == 0


def test_multi_parent_starts_with_the_second_cost():
    state = SearchState(
        best=np.zeros(2),
        second_best=np.ones(2),
        previous_cost=0.0,
        change_vector=np.zeros(2),
        current_cost=0.0,
    )
    assert not state.multi_parent
    np.testing.assert_array_equal(state.parents, np.ones(2))
    state.cost_history.append(1.0)
    assert not state.multi_parent
    state.cost_history.append(0.5)
    state.best_eighth = np.zeros((2, 3))
    assert state.multi_parent
    assert state.parents.shape == (2, 3)


def test_only_the_parent_kind_that_is_read_is_selected(monkeypatch):
    """After the first iteration the runner-up is never computed again."""
    from amm_optimizer import section

    calls = {"runner_up": 0, "best_eighth": 0}
    runner_up, eighth = section.runner_up, section.best_eighth

    def counting_runner_up(*args):
        calls["runner_up"] += 1
        return runner_up(*args)

    def counting_eighth(*args):
        calls["best_eighth"] += 1
        return eighth(*args)

    monkeypatch.setattr(section, "runner_up", counting_runner_up)
    monkeypatch.setattr(section, "best_eighth", counting_eighth)
    atm.minimize(lambda x: float(np.sum(x**2)), np.ones(3), 700, seed=0)
    assert calls["runner_up"] == 1  # one search, one first iteration
    assert calls["best_eighth"] > 5
