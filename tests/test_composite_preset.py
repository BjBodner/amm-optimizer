"""``ATMConfig.composite()`` is the composite that was confirmed on three holdouts."""

import numpy as np
import pytest

import amm_optimizer as atm
import characterisation_cases

# The exact string the first held-out evaluation was run with.
HOLDOUT_CONFIG = (
    "variant.amplitude_control=success_rate_targeted,"
    "variant.initial_design=uniform,variant.bounds=fold,variant.corner_clip=0.25,"
    "variant.pool_rule=budget,variant.pool_min=12,"
    "variant.reevaluate_incumbent=false"
)


def rastrigin(x):
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def test_preset_is_exactly_the_configuration_that_was_held_out():
    # the holdout's configuration text, item by item, as the snapshots spell it
    text = characterisation_cases.text_of(atm.ATMConfig.composite())
    assert set(text.split(",")) == set(HOLDOUT_CONFIG.split(","))


def test_defaults_are_untouched_by_the_preset():
    assert atm.ATMConfig() != atm.ATMConfig.composite()
    assert atm.ATMConfig().variant.bounds == "none"
    assert not atm.ATMConfig().variant.needs_box


def test_preset_survives_a_round_trip():
    config = atm.ATMConfig.composite()
    assert atm.ATMConfig.from_dict(config.to_dict()) == config


def test_preset_needs_the_search_box_and_says_so():
    with pytest.raises(ValueError, match="bounds"):
        atm.api.minimize(
            rastrigin, np.zeros(3), 300, seed=0, config=atm.ATMConfig.composite()
        )


def test_preset_runs_inside_the_box_and_spends_exactly_the_budget():
    seen = []

    def recording(x):
        seen.append(np.array(x))
        return rastrigin(x)

    x0 = np.full(4, 3.0)
    result = atm.api.minimize(
        recording,
        x0,
        800,
        seed=1,
        bounds=(-5.0, 5.0),
        config=atm.ATMConfig.composite(),
    )
    points = np.array(seen)
    assert len(points) == result.nfev == 800
    assert points.min() >= -5.0 and points.max() <= 5.0
    assert result.fun == min(rastrigin(p) for p in points)
    assert result.fun < rastrigin(x0)


def test_preset_beats_the_defaults_on_a_smooth_box_problem():
    """A sanity check with a fixed seed, not a benchmark."""

    def sphere(x):
        return float(np.sum((np.asarray(x) - 1.0) ** 2))

    x0 = np.full(5, 4.0)
    kwargs = {"bounds": (-5.0, 5.0)}
    default = atm.api.minimize(sphere, x0, 1000, seed=3, **kwargs).fun
    tuned = atm.api.minimize(
        sphere, x0, 1000, seed=3, config=atm.ATMConfig.composite(), **kwargs
    ).fun
    assert tuned < default
