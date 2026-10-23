"""Variant ``restart_mode``: a stagnation test that fires, and where to restart."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.config import ATMConfig, Variant
from amm_optimizer.plan import ipop_pool, with_pool
from amm_optimizer.restart import stagnated, stagnation_window

BASE = dict(
    amplitude_control="success_rate_targeted",
    bounds="fold",
    pool_rule="budget",
    pool_min=12,
    reevaluate_incumbent=False,
)
BOX = (-5.0, 5.0)


def config(**changes) -> ATMConfig:
    return ATMConfig(variant=Variant(**{**BASE, **changes}))


def test_window():
    assert stagnation_window(10, 64) == 15
    assert stagnation_window(2, 12) == 15
    assert stagnation_window(2, 64) == 11


def test_stagnated_needs_a_full_window():
    flat = [1.0] * 9
    assert not stagnated(flat, [0.0], 10, 1e-11, -30.0)
    assert stagnated(flat + [1.0], [0.0], 10, 1e-11, -30.0)


def test_stagnated_by_progress_and_by_width():
    moving = list(np.linspace(1.0, 0.0, 20))
    assert not stagnated(moving, [0.0], 10, 1e-11, -30.0)
    assert stagnated(moving, [-31.0, -40.0], 10, 1e-11, -30.0)
    assert not stagnated(moving, [-31.0, -29.0], 10, 1e-11, -30.0)


def test_needs_a_success_rate_controller_and_a_box():
    with pytest.raises(ValueError, match="success-rate"):
        Variant(restart_mode="uniform")
    with pytest.raises(ValueError, match="bounds"):
        atm.minimize(
            lambda x: 0.0,
            np.zeros(2),
            50,
            seed=0,
            config=config(restart_mode="uniform"),
        )


def plateau(x):
    return 1.0


@pytest.mark.parametrize("mode", ["incumbent", "uniform", "ipop"])
def test_a_plateau_restarts_and_spends_the_budget(mode):
    result = atm.minimize(
        plateau, np.zeros(3), 3000, seed=1, config=config(restart_mode=mode), bounds=BOX
    )
    assert result.nrestarts >= 2
    assert result.nfev == 3000


def test_legacy_mode_does_not_restart_on_a_plateau():
    result = atm.minimize(
        plateau, np.zeros(3), 3000, seed=1, config=config(), bounds=BOX
    )
    assert result.nrestarts == 0


@pytest.mark.parametrize("mode", ["uniform", "ipop"])
def test_best_point_is_kept_across_restarts(mode):
    seen = []

    def rastrigin(x):
        value = float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))
        seen.append(value)
        return value

    result = atm.minimize(
        rastrigin,
        np.full(4, 3.0),
        4000,
        seed=2,
        config=config(restart_mode=mode),
        bounds=BOX,
    )
    assert result.nrestarts >= 1
    assert result.nfev == len(seen) <= 4000
    assert result.fun == min(seen)


def test_ipop_pool_doubles_up_to_the_cap():
    resolved = with_pool(config(restart_mode="ipop"), 12)
    variant = resolved.variant
    assert [ipop_pool(resolved, variant, k) for k in range(4)] == [12, 24, 48, 96]
    capped = Variant(**{**BASE, "restart_mode": "ipop", "ipop_max_pool": 40})
    assert ipop_pool(resolved, capped, 3) == 40
