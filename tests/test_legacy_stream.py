"""The stream-alignment helpers draw exactly what the original drew."""

import numpy as np

from amm_optimizer import legacy_stream


def test_sampler_init_draws_use_five_vectors_and_keep_the_fourth():
    rng, ref = np.random.RandomState(4), np.random.RandomState(4)
    adaptive = legacy_stream.sampler_init_draws(6, rng)
    vectors = [ref.rand(6) for _ in range(5)]
    np.testing.assert_array_equal(adaptive, vectors[3] - 0.5)
    assert rng.rand() == ref.rand()  # the same number of draws was consumed


def test_initial_hyperparameters_are_scaled_and_padded():
    drawn = legacy_stream.draw_initial_hyperparameters(np.random.RandomState(1), 4)
    expected = 10 * np.random.RandomState(1).rand(11, 4)
    np.testing.assert_array_equal(drawn, expected)
    assert legacy_stream.N_ROWS == 8 + legacy_stream.LEGACY_PADDING_ROWS


def test_burning_the_pool_advances_the_stream_by_one_matrix():
    rng, ref = np.random.RandomState(2), np.random.RandomState(2)
    legacy_stream.burn_directional_pool(rng, 5, 64)
    ref.rand(5, 64)
    assert rng.rand() == ref.rand()
