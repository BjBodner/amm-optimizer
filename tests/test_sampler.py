import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from amm_optimizer import sampler
from amm_optimizer.sampler import ModeParams

PARAMS = ModeParams(
    random_growth_rate=5.0,
    random_period=5.0,
    max_random_search=2.0,
    targeted_growth_rate=0.5,
    targeted_decay_rate=2.0,
    significant_change=1e-6,
    alpha=100.0,
    beta=0.6,
)


def draw(n_params=4, n_samples=6, seed=0, change=None, **kwargs):
    rng = np.random.RandomState(seed)
    base = np.linspace(-1, 1, n_params)
    return sampler.two_mode_samples(
        base,
        kwargs.get("parents", base + 0.5),
        n_samples,
        np.zeros(n_params) if change is None else change,
        kwargs.get("random_growth", 1.0),
        kwargs.get("targeted_growth", 1.0),
        kwargs.get("params", PARAMS),
        np.ones((n_params, n_samples)),
        rng,
    )


def test_mode_params_ignore_legacy_padding_rows():
    vector = np.arange(11, dtype=float)
    params = ModeParams.from_vector(vector)
    assert tuple(params) == tuple(range(8))


def test_first_column_is_the_seed_and_shape_is_right():
    samples, _, _ = draw(n_params=5, n_samples=7)
    assert samples.shape == (5, 7)
    np.testing.assert_array_equal(samples[:, 0], np.linspace(-1, 1, 5))


def test_same_rng_state_gives_same_samples():
    np.testing.assert_array_equal(draw(seed=3)[0], draw(seed=3)[0])
    assert not np.array_equal(draw(seed=3)[0], draw(seed=4)[0])


def test_random_growth_increases_by_one_and_resets_on_significant_move():
    _, random_growth, _ = draw(random_growth=7.0)
    assert random_growth == 8.0
    _, random_growth, targeted_growth = draw(
        random_growth=7.0, change=np.array([1.0, 0, 0, 0])
    )
    assert random_growth == 1.0  # reset to 0, then incremented
    assert targeted_growth == 2.0


def test_targeted_growth_resets_without_movement():
    _, _, targeted_growth = draw(targeted_growth=9.0)
    assert targeted_growth == 0


def test_update_growth_factors_does_nothing_exactly_at_threshold():
    assert sampler.update_growth_factors(np.array([0.5, 0.5]), 1.0, 3, 4) == (3, 4)


def test_targeted_search_moves_along_the_change_vector():
    change = np.array([1.0, 0.0, -1.0, 0.0])
    samples_with, _, _ = draw(change=change, seed=1)
    samples_without, _, _ = draw(seed=1)
    assert not np.array_equal(samples_with[:, 1:], samples_without[:, 1:])


def test_targeted_amplitude_falls_back_when_exponent_is_huge():
    rng = np.random.RandomState(0)
    out = sampler.targeted_search(
        np.ones(3), 4, PARAMS, 0.0, 0.0, targeted_growth=1000.0, rng=rng
    )
    expected = 100 * np.random.RandomState(0).rand(4, 1)
    np.testing.assert_allclose(out, np.ones((3, 1)) * expected.T)


def test_targeted_amplitude_is_reset_when_it_grows_past_the_cap():
    big = PARAMS._replace(max_random_search=1e9)
    rng = np.random.RandomState(0)
    out = sampler.targeted_search(
        np.ones(2), 3, big, 0.0, 0.0, targeted_growth=100.0, rng=rng
    )
    expected = 100 * np.random.RandomState(0).rand(3, 1)
    np.testing.assert_allclose(out, np.ones((2, 1)) * expected.T)


def test_nan_amplitude_is_reset():
    nan_params = PARAMS._replace(max_random_search=np.nan)
    out = sampler.targeted_search(
        np.ones(2), 3, nan_params, 0.0, 0.0, 1.0, np.random.RandomState(0)
    )
    assert np.isfinite(out).all()


@given(
    growth=st.floats(0, 1e6),
    period=st.floats(0.1, 10),
    max_search=st.floats(1e-7, 10),
    rate=st.floats(0.01, 10),
)
@settings(max_examples=200, deadline=None)
def test_random_amplitude_is_capped_and_nonnegative(growth, period, max_search, rate):
    amplitude = sampler.random_search_amplitude(
        max_search, rate, growth, period, np.full((2, 3), 50.0)
    )
    assert (amplitude >= 0).all()
    assert (amplitude <= sampler.MAX_RANDOM_AMPLITUDE).all()


@given(growth=st.floats(0, 1e6), period=st.floats(0.1, 10))
def test_periodic_decay_is_negative(growth, period):
    assert sampler.periodic_decay(growth, period) < 0


@given(
    seed=st.integers(0, 2**31 - 1),
    n_params=st.integers(1, 8),
    n_samples=st.integers(1, 12),
    n_parents=st.integers(1, 6),
)
@settings(max_examples=100, deadline=None)
def test_multi_parent_children_only_use_parent_genes(
    seed, n_params, n_samples, n_parents
):
    rng = np.random.RandomState(seed)
    parents = rng.randn(n_params, n_parents)
    _, child = sampler.multi_parent_crossover(
        np.zeros(n_params), parents, n_samples, rng
    )
    for row in range(n_params):
        assert set(child[row]) <= set(parents[row])


@given(seed=st.integers(0, 2**31 - 1), n_params=st.integers(1, 8))
@settings(max_examples=100, deadline=None)
def test_two_parent_children_only_use_parent_genes(seed, n_params):
    rng = np.random.RandomState(seed)
    first, second = rng.randn(n_params), rng.randn(n_params) + 10
    _, child = sampler.two_parent_crossover(first, second, 9, rng)
    for row in range(n_params):
        assert set(child[row]) <= {first[row], second[row]}


@pytest.mark.parametrize("n_params", [1, 2, 11])
def test_works_for_any_dimension(n_params):
    samples, _, _ = draw(n_params=n_params, n_samples=3)
    assert samples.shape == (n_params, 3)
    assert np.isfinite(samples).all()
