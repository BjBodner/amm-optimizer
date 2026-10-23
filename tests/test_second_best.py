"""Runner-up selection (audit B21)."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.state import runner_up


@pytest.mark.parametrize(
    "cost",
    [
        [-5.0, -1.0, 3.0],  # negative best: x10000 makes it more negative
        [0.0, 2.0, 3.0],  # zero best: x10000 does nothing
        [1e-9, 1e-3, 1.0],  # spread above 1e4: inflated best is still lowest
    ],
)
def test_legacy_runner_up_is_the_winner_itself_in_these_cases(cost):
    """Documents the defect that the original has."""
    cost = np.array(cost)
    assert runner_up(cost, 0, "legacy") == 0


@pytest.mark.parametrize(
    "cost",
    [[-5.0, -1.0, 3.0], [0.0, 2.0, 3.0], [1e-9, 1e-3, 1.0], [4.0, 2.0, 3.0]],
)
def test_exclude_winner_returns_the_real_runner_up(cost):
    cost = np.array(cost)
    winner = int(np.argmin(cost))
    second = runner_up(cost, winner, "exclude_winner")
    assert second != winner
    assert cost[second] == np.sort(cost)[1]


def test_both_modes_agree_in_the_ordinary_case():
    cost = np.array([3.0, 1.0, 2.0, 5.0])
    assert runner_up(cost, 1, "legacy") == runner_up(cost, 1, "exclude_winner") == 2


def test_single_candidate_is_its_own_runner_up():
    assert runner_up(np.array([1.0]), 0, "exclude_winner") == 0


def test_runner_up_does_not_modify_its_input():
    cost = np.array([-5.0, -1.0, 3.0])
    runner_up(cost, 0, "legacy")
    runner_up(cost, 0, "exclude_winner")
    np.testing.assert_array_equal(cost, [-5.0, -1.0, 3.0])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="second_best"):
        atm.ATMConfig(second_best="nonsense")


def test_the_option_changes_the_search_and_still_works():
    def shifted(x):
        return float(np.sum((x - 2) ** 2)) - 100.0  # negative costs

    x0 = np.zeros(5)
    legacy = atm.minimize(shifted, x0, 1500, seed=2)
    fixed = atm.minimize(
        shifted, x0, 1500, seed=2, config=atm.ATMConfig(second_best="exclude_winner")
    )
    assert not np.array_equal(legacy.x, fixed.x)
    assert fixed.fun < shifted(x0)
    assert fixed.nfev == 1500


def test_default_is_legacy_so_results_are_unchanged():
    assert atm.ATMConfig().second_best == "legacy"
