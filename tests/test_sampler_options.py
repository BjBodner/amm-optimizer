"""``SampleOptions``: the sampler variants (E06 ...) and their no-op default."""

import numpy as np
import pytest

from amm_optimizer.config import Variant
from amm_optimizer.sampler import (
    TARGETED_MULTIPLIER_RANGE,
    ModeParams,
    SampleOptions,
    sample_options,
    targeted_search,
    two_mode_samples,
)
from test_rng_ledger import LoggingRandomState


def params(**kw):
    base = dict(
        random_growth_rate=2.0,
        random_period=3.0,
        max_random_search=1.0,
        targeted_growth_rate=1.0,
        targeted_decay_rate=1.0,
        significant_change=0.01,
        alpha=1.0,
        beta=0.5,
    )
    return ModeParams(**{**base, **kw})


def draw(change, options=None, seed=3, n=6, dim=3, **kw):
    rng = LoggingRandomState(seed)
    out = two_mode_samples(
        np.zeros(dim),
        np.zeros(dim),
        n,
        np.asarray(change, dtype=float),
        kw.get("random_growth", 1.0),
        kw.get("targeted_growth", 1.0),
        params(),
        np.broadcast_to(
            np.asarray(kw.get("adaptive", np.full(dim, 0.25)))[:, np.newaxis], (dim, n)
        ),
        rng,
        options=options,
        noise_factor=kw.get("noise_factor"),
    )
    return out, rng.ledger


def test_default_variant_asks_for_no_options():
    assert sample_options(Variant()) is None


def test_an_all_default_options_object_is_the_original_path():
    for change in ([0, 0, 0], [1.0, -2.0, 0.5], [-1.0, -2.0, -3.0]):
        (a, *rest_a), ledger_a = draw(change, None)
        (b, *rest_b), ledger_b = draw(change, SampleOptions())
        np.testing.assert_array_equal(a, b)
        assert rest_a == rest_b
        assert ledger_a == ledger_b


def test_the_original_gate_skips_the_targeted_step_without_a_positive_coordinate():
    change = [-1.0, -2.0, 0.0]
    (samples, *_), ledger = draw(change)
    assert ("rand", (6, 1)) not in ledger  # no targeted draw
    (gated, *_), ledger_any = draw(change, SampleOptions(targeted_gate_any=True))
    assert ("rand", (6, 1)) in ledger_any
    # the targeted term moves samples along the (negative) move
    assert (gated[:, 1:] - samples[:, 1:])[:2].max() < 0


@pytest.mark.parametrize("flag", [False, True])
def test_a_zero_move_has_no_targeted_step_either_way(flag):
    options = SampleOptions(targeted_gate_any=flag)
    _, ledger = draw([0, 0, 0], options)
    assert ("rand", (6, 1)) not in ledger


def test_a_positive_coordinate_opens_both_gates():
    change = [0.0, 0.5, -1.0]
    (a, *_), ledger_a = draw(change)
    (b, *_), ledger_b = draw(change, SampleOptions(targeted_gate_any=True))
    np.testing.assert_array_equal(a, b)
    assert ledger_a == ledger_b


# research-only options (deprecated in 0.6.0)
# research-only options (deprecated in 0.6.0)
def test_loguniform_targeted_step_is_a_bounded_multiple_of_the_move():
    change = np.array([1.0, -2.0, 0.5])
    low, high = TARGETED_MULTIPLIER_RANGE
    rng = LoggingRandomState(0)
    # growth factors that make the original's amplitude astronomically large
    steps = targeted_search(change, 500, params(), 5.0, 5.0, 50.0, rng, True)
    ratio = steps / change[:, None]  # (3, 500): one multiplier per sample
    np.testing.assert_allclose(ratio, np.broadcast_to(ratio[:1], ratio.shape))
    assert ratio.min() >= low * (1 - 1e-12)
    assert ratio.max() <= high * (1 + 1e-12)
    # log-uniform: about half the multipliers are below 1
    assert 0.4 < (ratio[0] < 1).mean() < 0.6
    assert rng.ledger == [("rand", (500, 1))]  # one draw, like the original
    original = targeted_search(change, 500, params(), 5.0, 5.0, 50.0, rng)
    assert np.abs(original).max() > 10 * np.abs(change).max()  # unbounded by comparison


# research-only options (deprecated in 0.6.0)
