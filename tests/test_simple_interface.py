"""``amm_optimizer.minimize``: the arguments a caller can leave out, and ``bounds``."""

import numpy as np
import pytest
from scipy.optimize import Bounds

import amm_optimizer as amm
from amm_optimizer import ATMConfig, presets


def sphere(x):
    return float(np.sum((np.asarray(x) - 1.0) ** 2))


def same(a, b):
    assert a.nfev == b.nfev
    assert a.fun == b.fun
    np.testing.assert_array_equal(a.x, b.x)


def explicit(dim, budget, **kwargs):
    """The call spelled out in full: x0 the centre of (-5, 5)^dim."""
    return amm.minimize(
        sphere, np.zeros(dim), budget, bounds=(-5.0, 5.0), seed=3, **kwargs
    )


def test_the_shortest_call_is_the_spelled_out_call():
    short = amm.minimize(sphere, bounds=[(-5, 5)] * 3, seed=3)
    same(short, explicit(3, 300))


def test_default_budget_is_100_per_dimension_with_a_floor_of_100():
    assert amm.minimize(sphere, bounds=[(-5, 5)] * 4, seed=0).nfev == 400
    assert amm.minimize(sphere, bounds=[(-5, 5)], seed=0).nfev == 100


def test_the_start_defaults_to_the_centre_of_the_box():
    seen = []

    def spy(x):
        seen.append(np.array(x, dtype=float))
        return sphere(x)

    amm.minimize(spy, bounds=[(0, 4), (-2, 6)], budget=40, seed=0)
    np.testing.assert_array_equal(seen[0], [2.0, 2.0])


def test_x0_alone_with_scalar_bounds_still_works():
    same(
        amm.minimize(sphere, np.zeros(3), bounds=(-5, 5), seed=3),
        explicit(3, 300),
    )


@pytest.mark.parametrize(
    "bounds",
    [
        (-5, 5),
        (-5.0, 5.0),
        (np.full(3, -5.0), np.full(3, 5.0)),
        [(-5, 5), (-5, 5), (-5, 5)],
        [[-5, 5]] * 3,
        np.array([[-5.0, 5.0]] * 3),
        Bounds([-5, -5, -5], [5, 5, 5]),
        Bounds(-5, 5),
        ((-5, 5), (-5, 5), (-5, 5)),
    ],
)
def test_every_form_of_bounds_gives_the_same_run(bounds):
    x0 = np.zeros(3)
    run = amm.minimize(sphere, x0, 300, seed=3, bounds=bounds)
    same(run, explicit(3, 300))


def test_a_float_budget_that_is_whole_is_accepted():
    assert amm.minimize(sphere, np.zeros(2), 3e2, bounds=(-5, 5), seed=0).nfev == 300


def test_a_budget_that_is_not_whole_is_refused():
    with pytest.raises(ValueError, match="whole number"):
        amm.minimize(sphere, np.zeros(2), 2.5, bounds=(-5, 5))
    with pytest.raises(ValueError, match="budget"):
        amm.minimize(sphere, np.zeros(2), "100", bounds=(-5, 5))


def test_two_coordinates_given_as_a_tuple_of_pairs_are_read_as_pairs():
    # ((-5, 5), (-5, 5)) is not a box as (lower, upper), so it is two pairs
    a = amm.minimize(sphere, bounds=((-5, 5), (-5, 5)), seed=1)
    b = amm.minimize(sphere, bounds=[(-5, 5), (-5, 5)], seed=1)
    same(a, b)


def test_two_coordinates_given_as_two_arrays_are_read_as_arrays():
    # ((-5, -5), (5, 5)) is not a box as pairs, so it is (lower, upper)
    a = amm.minimize(sphere, bounds=((-5, -5), (5, 5)), seed=1)
    b = amm.minimize(sphere, bounds=[(-5, 5), (-5, 5)], seed=1)
    same(a, b)


def test_a_tuple_that_reads_both_ways_is_refused():
    with pytest.raises(ValueError, match="can mean"):
        amm.minimize(sphere, bounds=((-5, 5), (-4, 6)), seed=0)


def test_dimension_must_be_known():
    with pytest.raises(ValueError, match="number of variables"):
        amm.minimize(sphere, bounds=(-5, 5))
    with pytest.raises(ValueError, match="x0 .* or bounds"):
        amm.minimize(sphere, config=ATMConfig())


def test_x0_and_bounds_must_agree_on_the_dimension():
    with pytest.raises(ValueError, match="2 coordinates but x0 has 3"):
        amm.minimize(sphere, np.zeros(3), bounds=[(-5, 5)] * 2)


def test_bad_inputs_get_plain_errors():
    with pytest.raises(TypeError, match="callable"):
        amm.minimize(5, bounds=[(-5, 5)])
    with pytest.raises(ValueError, match="1-d"):
        amm.minimize(sphere, np.zeros((2, 2)), bounds=(-5, 5))
    with pytest.raises(ValueError, match="finite"):
        amm.minimize(sphere, bounds=[(None, 5)] * 2)
    with pytest.raises(ValueError, match="shape"):
        amm.minimize(sphere, np.zeros(2), bounds=[(1, 2, 3)] * 2)


def test_anytime_true_is_the_anytime_variant_of_the_default():
    flag = amm.minimize(sphere, bounds=[(-5, 5)] * 2, budget=200, seed=2, anytime=True)
    config = presets.with_changes("c46", anytime="replay")
    spelled = amm.minimize(
        sphere, np.zeros(2), 200, seed=2, bounds=(-5, 5), config=config
    )
    same(flag, spelled)


def test_anytime_true_applies_to_a_given_config_too():
    base = presets.get("c26")
    flag = amm.minimize(
        sphere, np.zeros(2), 120, seed=2, bounds=(-5, 5), config=base, anytime=True
    )
    spelled = amm.minimize(
        sphere,
        np.zeros(2),
        120,
        seed=2,
        bounds=(-5, 5),
        config=presets.with_changes("c26", anytime="replay"),
    )
    same(flag, spelled)


def test_anytime_false_changes_nothing():
    same(
        amm.minimize(sphere, bounds=[(-5, 5)] * 2, budget=150, seed=2, anytime=False),
        amm.minimize(sphere, bounds=[(-5, 5)] * 2, budget=150, seed=2),
    )
