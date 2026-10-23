"""``ATMConfig.improved()``: the composite plus local stages (the Tier 5 candidate)."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.config import Variant


def sphere(x):
    return float(np.sum((np.asarray(x) - 2) ** 2))


def test_the_preset_is_a_non_default_strict_variant_and_the_default_is_untouched():
    improved = atm.ATMConfig.improved()
    assert improved.variant != Variant()
    assert improved.strict_budget
    assert atm.ATMConfig().variant == Variant()
    assert improved.variant.needs_box


def test_the_preset_needs_the_box_and_beats_the_original_on_a_sphere():
    improved = atm.ATMConfig.improved()
    with pytest.raises(ValueError, match="bounds"):
        atm.minimize(sphere, np.zeros(3), 300, seed=0, config=improved)
    x0 = np.full(5, -3.0)
    new = atm.minimize(sphere, x0, 800, seed=1, config=improved, bounds=(-5, 5))
    old = atm.minimize(sphere, x0, 800, seed=1)
    assert new.nfev == old.nfev == 800
    assert new.fun < old.fun * 1e-3


def test_the_preset_runs_in_ten_and_twenty_dimensions():
    for dim in (10, 20):
        result = atm.minimize(
            sphere,
            np.full(dim, -3.0),
            60 * dim,
            seed=2,
            config=atm.ATMConfig.improved(),
            bounds=(-5, 5),
        )
        assert result.nfev == 60 * dim
        assert result.fun < sphere(np.full(dim, -3.0))


def test_the_wave6_preset_is_kept_and_differs_only_in_the_small_dimension_limit():
    wave6 = atm.ATMConfig.improved_wave6()
    improved = atm.ATMConfig.improved()
    assert wave6.variant.es_small_dim == 0
    assert improved.variant.es_small_dim == 5
    assert wave6.variant.es_max_budget_per_dim == 100.0
    assert wave6.config_hash() != improved.config_hash()
    assert (
        improved.variant.refine_max_dim == 10 and wave6.variant.refine_max_dim > 10**5
    )
    other = Variant(
        **{**improved.variant.to_dict(), "es_small_dim": 0, "refine_max_dim": 1_000_000}
    )
    assert other == wave6.variant


def test_the_preset_skips_the_local_prefix_in_small_dimensions_at_100_x_d():
    sphere_cost = lambda x: float(np.sum((np.asarray(x) - 2) ** 2))  # noqa: E731
    box = (-5, 5)
    x0 = np.full(3, -3.0)

    def first_points(config, budget):
        seen = []

        def spy(x):
            seen.append(np.array(x))
            return sphere_cost(x)

        atm.minimize(spy, x0, budget, seed=3, config=config, bounds=box)
        return seen

    wave6 = first_points(atm.ATMConfig.improved_wave6(), 300)
    improved = first_points(atm.ATMConfig.improved(), 300)
    short_wave6 = first_points(atm.ATMConfig.improved_wave6(), 60)
    short_improved = first_points(atm.ATMConfig.improved(), 60)
    assert any(not np.array_equal(a, b) for a, b in zip(wave6, improved, strict=False))
    assert all(
        np.array_equal(a, b) for a, b in zip(short_wave6, short_improved, strict=True)
    )
