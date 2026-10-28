"""``amm_optimizer.minimize`` runs AMM (c46) by default; the engine keeps ATM."""

import numpy as np
import pytest

import amm_optimizer as amm
from amm_optimizer import ATMConfig, presets

BOX = (-5.0, 5.0)


def shifted_sphere(x):
    return float(np.sum((np.asarray(x) - 1.0) ** 2))


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def same(a, b):
    assert a.nfev == b.nfev
    assert a.fun == b.fun
    np.testing.assert_array_equal(a.x, b.x)


def test_the_public_function_is_the_wrapper_and_the_engine_is_unchanged():
    assert amm.minimize is not amm.api.minimize
    assert amm.minimize.__module__ == "amm_optimizer.default"
    assert amm.api.minimize.__module__ == "amm_optimizer.api"
    assert "minimize" in amm.__all__


def test_default_call_is_the_c46_preset():
    x0 = np.zeros(3)
    default = amm.minimize(shifted_sphere, x0, 300, seed=4, bounds=BOX)
    explicit = amm.minimize(
        shifted_sphere, x0, 300, seed=4, bounds=BOX, config=presets.get("c46")
    )
    same(default, explicit)
    engine = amm.api.minimize(
        shifted_sphere, x0, 300, seed=4, bounds=BOX, config=presets.get("c46")
    )
    same(default, engine)


def test_default_call_accepts_an_rng_and_profile():
    a = amm.minimize(
        shifted_sphere,
        np.zeros(3),
        300,
        rng=np.random.RandomState(2),
        bounds=BOX,
        profile=True,
    )
    b = amm.minimize(
        shifted_sphere, np.zeros(3), 300, rng=np.random.RandomState(2), bounds=BOX
    )
    same(a, b)
    assert a.timings is not None


def test_no_bounds_and_no_config_raises_a_clear_error():
    with pytest.raises(ValueError, match="bounds") as info:
        amm.minimize(shifted_sphere, np.zeros(3), 300, seed=0)
    assert "ATMConfig()" in str(info.value)


def test_original_sampler_needs_no_bounds_and_equals_the_engine():
    x0 = np.zeros(3)
    via_public = amm.minimize(shifted_sphere, x0, 300, seed=1, config=ATMConfig())
    via_engine = amm.api.minimize(shifted_sphere, x0, 300, seed=1)
    same(via_public, via_engine)
    with_bounds = amm.minimize(
        shifted_sphere, x0, 300, seed=1, config=ATMConfig(), bounds=BOX
    )
    same(
        with_bounds,
        amm.api.minimize(
            shifted_sphere, x0, 300, seed=1, config=ATMConfig(), bounds=BOX
        ),
    )


def test_the_engine_default_is_still_the_original_sampler():
    x0 = np.zeros(3)
    same(
        amm.api.minimize(shifted_sphere, x0, 300, seed=3),
        amm.api.minimize(shifted_sphere, x0, 300, seed=3, config=ATMConfig()),
    )


@pytest.mark.parametrize("name", ["composite", "c26"])
def test_an_explicit_config_passes_through_unchanged(name):
    assert name in presets.names(research=True)
    config = ATMConfig.composite() if name == "composite" else presets.get(name)
    x0 = np.zeros(3)
    kwargs = {"seed": 5, "bounds": BOX, "config": config}
    same(
        amm.minimize(shifted_sphere, x0, 300, **kwargs),
        amm.api.minimize(shifted_sphere, x0, 300, **kwargs),
    )


def test_explicit_config_without_bounds_defers_to_the_engine():
    # a preset that needs the box still raises the engine's own error
    with pytest.raises(ValueError, match="bounds"):
        amm.minimize(
            shifted_sphere, np.zeros(3), 300, seed=0, config=presets.get("c26")
        )


def test_home_page_example_solves_rosenbrock():
    result = amm.minimize(rosenbrock, np.zeros(5), 2000, seed=0, bounds=(-5, 5))
    assert result.nfev == 2000
    assert result.fun < 1e-10
    np.testing.assert_allclose(result.x, np.ones(5), atol=1e-4)
