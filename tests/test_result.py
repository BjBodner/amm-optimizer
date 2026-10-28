"""The result is SciPy's ``OptimizeResult`` (WP4.1, D3), with the seed on it (WP4.3,
D4), and ``config=`` takes preset names (WP4.4)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
import scipy.optimize

import amm_optimizer as amm
from amm_optimizer import presets
from amm_optimizer.config import ATMConfig

FIELDS = {
    "x",
    "fun",
    "nfev",
    "nit",
    "nrestarts",
    "success",
    "status",
    "message",
    "seed",
    "timings",
}
RUNS = {
    "default": {},
    "anytime": {"anytime": True},
    "atm": {"config": ATMConfig()},
    "engine": None,
}


def sphere(x):
    return float(np.sum((x - 1.0) ** 2))


def run(name, fun=sphere, **kwargs):
    if RUNS[name] is None:
        return amm.api.minimize(fun, np.zeros(3), 300, **kwargs)
    return amm.minimize(fun, bounds=[(-5, 5)] * 3, **RUNS[name], **kwargs)


@pytest.mark.parametrize("name", list(RUNS))
def test_the_result_is_scipys_with_every_field(name):
    result = run(name, seed=3)
    assert isinstance(result, scipy.optimize.OptimizeResult)
    assert isinstance(result, amm.OptimizeResult)
    assert set(result) == FIELDS
    for key in FIELDS:
        assert result[key] is getattr(result, key)
    assert result["x"] is result.x
    assert result.x.shape == (3,) and result.fun == sphere(result.x)
    assert (result.status, result.success) == (0, True)
    assert result.message.startswith("The run ended normally")
    assert result.seed == 3
    assert result.timings is None
    assert 1 <= result.nit <= result.nfev == 300


@pytest.mark.parametrize("name", list(RUNS))
def test_nit_counts_the_improvements_of_the_best_value(name):
    values = []

    def f(x):
        values.append(sphere(x))
        return values[-1]

    result = run(name, f, seed=0)
    best = np.minimum.accumulate(values)
    assert result.nit == 1 + int(np.sum(best[1:] < best[:-1]))


def test_a_result_behaves_like_a_dict_and_pickles():
    result = amm.minimize(sphere, bounds=[(-5, 5)] * 2, seed=0, profile=True)
    assert result.timings is not None
    plain = dict(result)
    assert plain["nfev"] == result.nfev == 200
    copy = pickle.loads(pickle.dumps(result))
    assert isinstance(copy, amm.OptimizeResult)
    np.testing.assert_array_equal(copy.x, result.x)
    assert copy.seed == result.seed
    result.x = np.zeros(2)  # attributes and keys stay one
    assert result["x"] is result.x
    assert "message" in repr(result)


def test_scipy_style_code_reads_both_results_alike():
    def summary(r):
        return r.x.shape, bool(r.success), int(r.nfev) > 0, r["fun"] == r.fun

    ours = amm.minimize(sphere, bounds=[(-5, 5)] * 2, seed=0)
    theirs = scipy.optimize.minimize(sphere, np.zeros(2), method="Nelder-Mead")
    assert summary(ours) == summary(theirs)


# --- seeds (WP4.3) --------------------------------------------------------------


@pytest.mark.parametrize("name", list(RUNS))
def test_rerunning_with_the_recorded_seed_repeats_the_run(name):
    first = run(name)  # a fresh seed
    assert isinstance(first.seed, int) and 0 <= first.seed < 2**32
    again = run(name, seed=first.seed)
    np.testing.assert_array_equal(again.x, first.x)
    assert (again.fun, again.nfev, again.seed) == (first.fun, first.nfev, first.seed)


def test_fresh_seeds_differ():
    seeds = {run("default", budget=20).seed for _ in range(5)}
    assert len(seeds) == 5


@pytest.mark.parametrize("name", list(RUNS))
def test_a_generator_gives_one_draw_as_the_seed(name):
    a = run(name, rng=np.random.default_rng(0))
    b = run(name, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(a.x, b.x)
    assert a.seed == b.seed == int(np.random.default_rng(0).integers(2**32))
    same = run(name, seed=a.seed)
    np.testing.assert_array_equal(same.x, a.x)
    generator = np.random.default_rng(0)
    run(name, rng=generator)
    assert (
        generator.integers(2**32) == np.random.default_rng(0).integers(2**32, size=2)[1]
    )


def test_a_seed_sequence_is_a_seed():
    sequence = np.random.SeedSequence(12345)
    a = run("default", seed=sequence)
    expected = int(np.random.SeedSequence(12345).generate_state(1)[0])
    assert a.seed == expected
    np.testing.assert_array_equal(run("default", seed=expected).x, a.x)


def test_a_random_state_is_continued_and_no_seed_is_recorded():
    a = run("default", rng=np.random.RandomState(4))
    b = run("default", rng=np.random.RandomState(4))
    np.testing.assert_array_equal(a.x, b.x)
    assert a.seed is None


def test_an_int_seed_runs_exactly_as_before():
    """The seed's ``RandomState`` is the one 0.5.0 made from it."""
    a = run("engine", seed=7)
    b = run("engine", rng=np.random.RandomState(7))
    np.testing.assert_array_equal(a.x, b.x)
    assert (a.seed, b.seed) == (7, None)


# --- preset names in ``config=`` (WP4.4) ------------------------------------------


@pytest.mark.parametrize(
    ("name", "config"),
    [
        ("default", None),
        ("c46", presets.get("c46")),
        ("atm", ATMConfig()),
        ("composite", ATMConfig.composite()),
        ("c30", presets.get("c30")),
    ],
)
def test_a_preset_name_runs_its_configuration(name, config):
    by_name = amm.minimize(sphere, bounds=[(-5, 5)] * 3, seed=2, config=name)
    by_value = amm.minimize(sphere, bounds=[(-5, 5)] * 3, seed=2, config=config)
    np.testing.assert_array_equal(by_name.x, by_value.x)
    assert (by_name.fun, by_name.nfev) == (by_value.fun, by_value.nfev)
    engine = amm.api.minimize(
        sphere, np.zeros(3), 300, seed=2, config=name, bounds=(-5, 5)
    )
    np.testing.assert_array_equal(engine.x, by_name.x)


def test_atm_by_name_needs_no_bounds():
    result = amm.minimize(sphere, np.zeros(2), 200, seed=0, config="atm")
    assert result.nfev == 200
