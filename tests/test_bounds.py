"""The box and its repair rules (``bounds="fold"``)."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.bounds import Box, reflect, repair
from amm_optimizer.config import Variant

BOX = Box.make((-5.0, 5.0), 3)


def test_box_from_scalars_and_vectors():
    box = Box.make((-5, [1, 2, 3]), 3)
    np.testing.assert_array_equal(box.lower, [-5, -5, -5])
    np.testing.assert_array_equal(box.upper, [1, 2, 3])
    assert box.lower.flags.writeable  # a copy, not a read-only broadcast view


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ((5.0, -5.0), "below"),
        ((0.0, 0.0), "below"),
        ((-np.inf, 5.0), "finite"),
        ((-5.0, np.nan), "finite"),
    ],
)
def test_box_rejects_bad_bounds(bounds, message):
    with pytest.raises(ValueError, match=message):
        Box.make(bounds, 3)


def test_reflect_known_values():
    x = np.array([[5.5, -6.0, 15.0, -15.0, 25.5, 0.0]] * 3)
    want = np.array([[4.5, -4.0, -5.0, 5.0, 4.5, 0.0]] * 3)
    np.testing.assert_allclose(reflect(x, BOX), want)


def test_reflect_is_identity_inside_and_idempotent_and_in_range():
    rng = np.random.RandomState(0)
    inside = rng.uniform(-5, 5, (3, 50))
    np.testing.assert_allclose(reflect(inside, BOX), inside, atol=1e-12)
    far = rng.uniform(-1e3, 1e3, (3, 500))
    once = reflect(far, BOX)
    assert once.min() >= -5 and once.max() <= 5
    np.testing.assert_allclose(reflect(once, BOX), once, atol=1e-12)


def test_repair_leaves_inside_points_and_the_incumbent_bit_identical():
    rng = np.random.RandomState(0)
    samples = np.concatenate(
        [np.array([[9.0], [9.0], [9.0]]), rng.uniform(-5, 5, (3, 20))], axis=1
    )
    before = samples.copy()
    state = rng.get_state()[1].copy()
    repair(samples, BOX, 0.5, rng)
    np.testing.assert_array_equal(samples, before)  # the incumbent (outside) stays
    np.testing.assert_array_equal(rng.get_state()[1], state)  # no draw needed


def test_repair_reflects_outside_coordinates_only():
    samples = np.array([[0.0, 5.5, 1.0], [0.0, 2.0, -7.0], [0.0, 0.0, 0.0]])
    repair(samples, BOX, 0.0, np.random.RandomState(0))
    np.testing.assert_allclose(samples, [[0, 4.5, 1], [0, 2, -3], [0, 0, 0]])


def test_corner_clip_clips_a_share_of_the_leaving_samples_as_a_whole():
    samples = np.zeros((3, 4001))
    samples[:, 1:] = [[5.5], [7.0], [0.5]]  # inside in the third coordinate
    repair(samples, BOX, 0.25, np.random.RandomState(0))
    body = samples[:, 1:]
    clipped = (body[0] == 5.0).mean()
    assert 0.22 < clipped < 0.28
    # a clipped sample is in the corner in every coordinate that left
    corner = body[:, body[0] == 5.0]
    assert (corner[1] == 5.0).all() and (corner[2] == 0.5).all()
    # the others are reflected: 5.5 -> 4.5, 7 -> 3
    reflected = body[:, body[0] != 5.0]
    assert (reflected[0] == 4.5).all() and (reflected[1] == 3.0).all()


def test_a_leaving_sample_costs_one_corner_draw():
    samples = np.zeros((3, 5))
    samples[:, 1:] = 1.0
    samples[0, 2] = 6.0
    rng = np.random.RandomState(0)
    state = rng.get_state()[1].copy()
    repair(samples, BOX, 0.5, rng)
    assert not np.array_equal(rng.get_state()[1], state)  # one sample left: a draw


def test_repair_always_clips_infinite_and_ignores_nan():
    samples = np.zeros((3, 4))
    samples[:, 1] = np.inf
    samples[:, 2] = -np.inf
    samples[:, 3] = np.nan
    repair(samples, BOX, 0.0, np.random.RandomState(0))
    np.testing.assert_array_equal(samples[:, 1], [5, 5, 5])
    np.testing.assert_array_equal(samples[:, 2], [-5, -5, -5])
    assert np.isnan(samples[:, 3]).all()


def sphere(x):
    return float(np.sum((x - 2) ** 2))


def test_a_variant_that_needs_the_box_refuses_to_run_without_it():
    with pytest.raises(ValueError, match="bounds"):
        atm.minimize(
            sphere,
            np.zeros(2),
            100,
            seed=0,
            config=atm.ATMConfig(variant=Variant(bounds="fold")),
        )


def test_bounds_are_not_enforced_for_the_default_variant():
    seen = []

    def spy(x):
        seen.append(np.array(x))
        return sphere(x)

    with_box = atm.minimize(spy, np.full(3, 4.0), 600, seed=1, bounds=(-5, 5))
    out = np.abs(np.array(seen)).max()
    without = atm.minimize(sphere, np.full(3, 4.0), 600, seed=1)
    assert out > 5  # the default leaves the box
    np.testing.assert_array_equal(with_box.x, without.x)


def test_folding_keeps_the_incumbent_in_the_box_even_with_a_far_optimum():
    def far(x):
        return float(np.sum((x - 40) ** 2))  # optimum far outside the box

    config = atm.ATMConfig(variant=Variant(bounds="fold", corner_clip=0.25))
    result = atm.minimize(far, np.zeros(4), 1500, seed=0, config=config, bounds=(-5, 5))
    assert np.abs(result.x).max() <= 5
    assert result.fun < far(np.zeros(4))
