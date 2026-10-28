"""``amm_optimizer.minimize``: the arguments a caller can leave out, and ``bounds``."""

import warnings

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


def spied(seen):
    def spy(x):
        seen.append(np.array(x, dtype=float))
        return sphere(x)

    return spy


@pytest.mark.parametrize("config", [None, ATMConfig()], ids=["default", "atm"])
def test_an_x0_outside_the_bbob_box_is_clipped_with_a_warning(config):
    """Review B2 / decision D1: ``fun`` only sees points inside the box."""
    seen = []
    with pytest.warns(UserWarning, match="x0 is outside bounds; clipped to the box"):
        result = amm.minimize(
            spied(seen), np.full(4, 15.0), 200, bounds=(-5, 5), seed=0, config=config
        )
    np.testing.assert_array_equal(seen[0], np.full(4, 5.0))
    if config is None:  # the original sampler alone does not keep to the box
        assert np.all(np.abs(np.array(seen)) <= 5.0)
    same(
        result,
        amm.minimize(
            sphere, np.full(4, 5.0), 200, bounds=(-5, 5), seed=0, config=config
        ),
    )


def test_an_x0_outside_any_other_box_runs_as_before_and_now_warns():
    box = [(0, 10), (0, 10), (-1, 1)]
    x0 = np.array([15.0, -3.0, 0.5])
    inside = np.array([10.0, 0.0, 0.5])
    with pytest.warns(UserWarning, match="outside bounds"):
        result = amm.minimize(sphere, x0, 300, bounds=box, seed=1)
    same(result, amm.minimize(sphere, inside, 300, bounds=box, seed=1))


def test_an_x0_inside_or_on_the_box_gives_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        amm.minimize(sphere, np.array([5.0, -5.0, 0.0]), 60, bounds=(-5, 5), seed=0)


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


class Counting:
    def __init__(self, returns=None):
        self.calls, self.returns = 0, returns

    def __call__(self, x):
        self.calls += 1
        if self.returns is not None and self.calls >= self.returns[0]:
            return self.returns[1]
        return sphere(x)


BAD_INPUTS = [
    # (keyword arguments, exception, message fragment)
    ({"x0": [np.nan, 0.0]}, ValueError, "x0 must be finite"),
    ({"x0": [np.inf, 0.0]}, ValueError, "x0 must be finite"),
    ({"x0": [0.0, -np.inf]}, ValueError, "x0 must be finite"),
    ({"config": "c99"}, ValueError, "unknown preset 'c99'"),
    ({"config": {"variant": None}}, TypeError, "ATMConfig or a preset name"),
    ({"config": 46}, TypeError, "ATMConfig or a preset name"),
    ({"rng": 0}, TypeError, "Generator or RandomState.*seed="),
    ({"rng": np.random.PCG64(0)}, TypeError, "Generator or RandomState"),
    ({"seed": -1}, ValueError, r"seed must be an integer in \[0, 2\*\*32\)"),
    ({"seed": 2**32}, ValueError, r"seed must be an integer in \[0, 2\*\*32\)"),
    ({"seed": 1.5}, TypeError, "seed must be an integer"),
    ({"seed": "0"}, TypeError, "seed must be an integer"),
    ({"seed": np.random.default_rng(0)}, TypeError, "SeedSequence"),
    ({"seed": True}, TypeError, "seed must be an integer"),
    ({"budget": True}, TypeError, "budget"),
    ({"budget": False}, TypeError, "budget"),
]


@pytest.mark.parametrize(("kwargs", "error", "match"), BAD_INPUTS)
def test_bad_inputs_fail_before_any_evaluation(kwargs, error, match):
    """Review B6 / D2 (a): plain errors, and no evaluation spent first."""
    fun = Counting()
    arguments = {"bounds": [(-5, 5)] * 2, "budget": 50} | kwargs
    with pytest.raises(error, match=match):
        amm.minimize(fun, **arguments)
    assert fun.calls == 0


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (
            np.array([1.0]),
            r"fun must return a real scalar, got ndarray of shape \(1,\)",
        ),
        (np.zeros(2), r"got ndarray of shape \(2,\)"),
        ([1.0], "fun must return a real scalar, got list"),
        (1j, "fun must return a real scalar, got complex"),
        (None, "fun must return a real scalar, got NoneType"),
    ],
    ids=["size-1 array", "array", "list", "complex", "None"],
)
@pytest.mark.parametrize("first_bad", [1, 7], ids=["at the start", "later"])
@pytest.mark.parametrize("config", [None, ATMConfig()], ids=["default", "atm"])
def test_fun_must_return_a_real_scalar(value, match, first_bad, config):
    fun = Counting(returns=(first_bad, value))
    with pytest.raises(TypeError, match=match):
        amm.minimize(fun, bounds=[(-5, 5)] * 2, budget=50, seed=0, config=config)
    assert fun.calls == first_bad


@pytest.mark.parametrize("seed", [0, 2**32 - 1, np.int64(7), np.uint32(9)])
def test_every_seed_numpy_accepts_as_an_integer_is_accepted(seed):
    result = amm.minimize(sphere, bounds=[(-5, 5)] * 2, budget=20, seed=seed)
    assert result.nfev == 20


def test_a_numpy_integer_budget_is_accepted():
    assert amm.minimize(sphere, bounds=[(-5, 5)] * 2, budget=np.int64(20)).nfev == 20
