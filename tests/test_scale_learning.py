"""``scale_learning="diagonal"`` (E14): widths learned from accepted moves."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.config import Variant
from amm_optimizer.control import learned_scales, update_scale_moment
from amm_optimizer.sampler import SampleOptions, sample_options
from test_sampler_options import draw

DIAG = Variant(amplitude_control="success_rate_targeted", scale_learning="diagonal")


def test_the_scales_have_geometric_mean_one_and_equal_moments_give_ones():
    np.testing.assert_allclose(learned_scales(np.ones(5), 100.0), np.ones(5))
    scales = learned_scales(np.array([1.0, 4.0, 16.0, 0.25]), 1e6)
    np.testing.assert_allclose(np.exp(np.mean(np.log(scales))), 1.0)
    assert scales[2] / scales[1] == pytest.approx(2.0)  # sqrt of the moment ratio


def test_the_scales_are_capped():
    scales = learned_scales(np.array([1e-12, 1.0, 1e12, 1.0]), 10.0)
    assert scales.min() >= 0.1 and scales.max() <= 10.0


def test_the_moment_follows_the_shape_of_the_moves_not_their_size():
    one, two = np.ones(3), np.ones(3)
    update_scale_moment(one, np.array([1.0, 0.0, 3.0]), 0.5)
    update_scale_moment(two, np.array([1e-9, 0.0, 3e-9]), 0.5)
    np.testing.assert_allclose(one, two)
    assert one[2] > one[0] > one[1]
    assert one.mean() == pytest.approx(1.0)  # relative squares have mean 1


def test_a_move_of_zero_or_nan_changes_nothing():
    moment = np.array([1.0, 2.0, 3.0])
    before = moment.copy()
    update_scale_moment(moment, np.zeros(3), 0.5)
    update_scale_moment(moment, np.array([np.nan, 1.0, 1.0]), 0.5)
    update_scale_moment(moment, np.array([np.inf, 1.0, 1.0]), 0.5)
    np.testing.assert_array_equal(moment, before)


def test_repeated_moves_along_one_coordinate_concentrate_the_scale():
    moment = np.ones(4)
    for _ in range(60):
        update_scale_moment(moment, np.array([1.0, 0.1, 0.1, 0.1]), 0.15)
    scales = learned_scales(moment, 100.0)
    assert scales[0] > 5 * scales[1:].max()


def test_the_variant_asks_for_scales_and_they_multiply_the_width():
    assert sample_options(DIAG).use_scales
    options = SampleOptions(scalar_width=True, no_targeted=True, use_scales=True)
    scales = np.array([0.1, 1.0, 10.0])
    (wide, _, _), _ = draw(
        [0, 0, 0], options, n=400, random_growth=np.log(0.5), adaptive=scales
    )
    spread = np.abs(wide[:, 1:] - wide[:, :1]).max(axis=1)
    np.testing.assert_allclose(spread, 0.5 * scales, rtol=0.05)


def test_scale_parameters_are_validated():
    for kwargs, match in [
        ({"scale_learning": "full"}, "scale_learning"),
        ({"scale_rate": 0.0}, "scale_rate"),
        ({"scale_cap": 0.5}, "scale_cap"),
    ]:
        with pytest.raises(ValueError, match=match):
            Variant(amplitude_control="success_rate", **kwargs)
    with pytest.raises(ValueError, match="amplitude_control"):
        Variant(scale_learning="diagonal")


def test_the_separable_ellipsoid_is_solved_far_better_with_learned_scales():
    d = 5
    weights = 10.0 ** (6 * np.arange(d) / (d - 1))

    def ellipsoid(x):
        return float(np.sum(weights * (np.asarray(x) - 1.0) ** 2))

    base = Variant(amplitude_control="success_rate_targeted", pool_rule="budget")
    plain = atm.minimize(
        ellipsoid, np.zeros(d), 4000, seed=0, config=atm.ATMConfig(variant=base)
    )
    learned = atm.minimize(
        ellipsoid, np.zeros(d), 4000, seed=0, config=atm.ATMConfig(variant=DIAG)
    )
    assert learned.nfev == 4000
    assert learned.fun < plain.fun * 1e-2
