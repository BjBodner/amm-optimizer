"""``initial_design="uniform"`` (E01): a global first pool."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.bounds import Box, uniform_columns
from amm_optimizer.config import Variant

UNIFORM = atm.ATMConfig(variant=Variant(initial_design="uniform"))


def sphere(x):
    return float(np.sum((x - 2) ** 2))


def run(config, budget, dim=4, seed=0, **kw):
    points = []

    def spy(x):
        points.append(np.array(x))
        return sphere(x)

    atm.minimize(spy, np.full(dim, -3.0), budget, seed=seed, config=config, **kw)
    return np.array(points)


def test_uniform_columns_cover_the_box_evenly():
    box = Box.make(([-5.0, 0.0], [5.0, 2.0]), 2)
    draws = uniform_columns(20_000, box, np.random.RandomState(0))
    assert draws.shape == (2, 20_000)
    np.testing.assert_allclose(draws.min(axis=1), box.lower, atol=0.01)
    np.testing.assert_allclose(draws.max(axis=1), box.upper, atol=0.01)
    np.testing.assert_allclose(draws.mean(axis=1), [0.0, 1.0], atol=0.05)


def test_the_first_pool_is_spread_over_the_box_not_around_the_start():
    legacy = run(atm.ATMConfig(), 66, bounds=(-5, 5))
    uniform = run(UNIFORM, 66, bounds=(-5, 5))
    start = np.full(4, -3.0)
    # point 0 is the start; points 1-64 are the first pool (incumbent copies
    # included: four of them, equal to the start)
    legacy_spread = np.linalg.norm(legacy[1:65] - start, axis=1)
    uniform_spread = np.linalg.norm(uniform[1:65] - start, axis=1)
    assert np.median(legacy_spread) < 0.5
    assert np.median(uniform_spread) > 3.0
    assert uniform[1:65].min() >= -5 and uniform[1:65].max() <= 5
    assert (uniform[1:65] == start).all(axis=1).sum() == 4  # incumbent columns kept


def test_only_the_first_iteration_is_uniform():
    """The second pool follows the winner of the first (it is not drawn again)."""
    uniform = run(UNIFORM, 400, bounds=(-5, 5))
    second = uniform[65:125]
    # a fresh uniform design would put every coordinate inside the box
    assert (np.abs(second) > 5).mean() > 0.05


def test_without_bounds_the_variant_refuses_to_run():
    with pytest.raises(ValueError, match="bounds"):
        atm.minimize(sphere, np.zeros(2), 50, seed=0, config=UNIFORM)


def test_a_restart_gets_a_uniform_pool_again(monkeypatch):
    from amm_optimizer.search import Search

    starts = []
    original = Search.run

    def counting(self, *args):
        starts.append(1)
        return original(self, *args)

    monkeypatch.setattr(Search, "run", counting)
    config = atm.ATMConfig(variant=Variant(initial_design="uniform"))
    # flat objective: the restart policy fires, each search opens with a pool
    atm.minimize(
        lambda x: 1.0, np.zeros(3), 6000, seed=0, config=config, bounds=(-5, 5)
    )
    assert len(starts) >= 2
