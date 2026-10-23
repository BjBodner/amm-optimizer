import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from amm_optimizer import hyperparams


def test_bounds_have_legacy_padding_rows_of_zero():
    bounds = hyperparams.bounds()
    assert bounds.shape == (11, 2)
    np.testing.assert_array_equal(bounds[8:], 0)
    assert (bounds[:8, 0] < bounds[:8, 1]).all()


def test_meta_hyperparameters_are_positive_and_use_a_wider_search_than_the_bounds():
    meta = hyperparams.meta_hyperparameters()
    bounds = hyperparams.bounds()[:8]
    assert meta.shape == (8,)
    assert (meta > 0).all()
    # Only candidates are clipped; the meta search itself runs with a maximal
    # random search (20.02) above the candidates' upper bound (10).
    outside = (meta < bounds[:, 0]) | (meta > bounds[:, 1])
    assert list(np.flatnonzero(outside)) == [2]


def test_meta_significant_change_matches_the_documented_fit():
    meta = hyperparams.meta_hyperparameters()
    assert meta[5] == 0.00001 * hyperparams.mean_change_value_fit(20.02, 8)
    assert meta[6] == 10**4 / 20.02


def test_mean_change_fit_is_the_published_polynomial():
    value = hyperparams.mean_change_value_fit(2.0, 10)
    expected = (
        -0.007921662 * 2
        + 0.003755446 * 4
        + 5.92e-5 * 10
        - 2.45e-7 * 100
        + 0.250014494 * 20
    )
    assert abs(value - expected) < 1e-15


@given(
    seed=st.integers(0, 2**31 - 1),
    scale=st.floats(0.1, 1e6),
    n_samples=st.integers(1, 8),
)
@settings(max_examples=100, deadline=None)
def test_clipped_samples_always_respect_the_bounds(seed, scale, n_samples):
    rng = np.random.RandomState(seed)
    bounds = hyperparams.bounds()
    samples = scale * rng.randn(11, n_samples)
    clipped = hyperparams.clip_to_bounds(samples, bounds)
    assert clipped.shape == samples.shape
    assert (clipped >= bounds[:, 0:1]).all()
    assert (clipped <= bounds[:, 1:2]).all()
