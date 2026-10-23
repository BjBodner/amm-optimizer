"""``pool_rule="budget"`` (E04) and ``resolve_config``: one place for the pool size."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.config import N_SAMPLERS, ATMConfig, Variant
from amm_optimizer.plan import iteration_limit, pool_size, resolve_config

BUDGET_RULE = Variant(pool_rule="budget")


@pytest.mark.parametrize(
    ("budget", "pool"),
    [(1, 24), (100, 24), (719, 24), (1500, 52), (1860, 64), (10_000, 64), (10**7, 64)],
)
def test_pool_size_follows_the_budget(budget, pool):
    assert pool_size(BUDGET_RULE, budget) == pool


@pytest.mark.parametrize(("lo", "hi"), [(12, 64), (24, 64), (8, 32)])
def test_pool_size_is_a_multiple_of_four_between_the_limits_and_monotone(lo, hi):
    variant = Variant(pool_rule="budget", pool_min=lo, pool_max=hi)
    sizes = [pool_size(variant, b) for b in range(1, 3000, 7)]
    assert min(sizes) == lo and max(sizes) == hi
    assert all(s % N_SAMPLERS == 0 for s in sizes)
    assert sizes == sorted(sizes)


@pytest.mark.parametrize("budget", [50, 600, 1500, 30_000])
def test_the_three_population_numbers_move_together(budget):
    resolved = resolve_config(ATMConfig(variant=BUDGET_RULE), budget, 5)
    pool = pool_size(BUDGET_RULE, budget)
    assert resolved.initial_population == pool
    assert resolved.evaluations_per_iteration == pool
    assert resolved.conductor.total_resources == pool
    assert resolved.variant == BUDGET_RULE


def test_resolving_does_not_touch_the_given_config():
    config = ATMConfig(variant=BUDGET_RULE)
    resolve_config(config, 600, 5)
    assert config == ATMConfig(variant=BUDGET_RULE)


def test_other_fields_survive_resolving():
    config = ATMConfig(variant=BUDGET_RULE, second_best="exclude_winner")
    resolved = resolve_config(config, 600, 5)
    assert resolved.second_best == "exclude_winner"


def test_a_fixed_pool_is_resolved_to_itself_for_any_other_flag():
    config = ATMConfig(variant=Variant(targeted_gate="any"))
    assert resolve_config(config, 600, 5) is config


def test_the_first_iteration_uses_the_resolved_pool():
    points = []

    def spy(x):
        points.append(np.array(x))
        return float(np.sum(x**2))

    atm.api.minimize(
        spy, np.full(3, 2.0), 200, seed=0, config=ATMConfig(variant=BUDGET_RULE)
    )
    # start, then a first iteration of 24 columns (4 of them the incumbent)
    first = np.array(points[1:25])
    assert (first == 2.0).all(axis=1).sum() == N_SAMPLERS
    # the second iteration begins with the first one's winner: a 64-sample first
    # pool would have put it 40 evaluations later
    winner = first[np.argmin(np.sum(first**2, axis=1))]
    np.testing.assert_array_equal(points[25], winner)


def test_the_rule_combines_with_skipping_the_incumbent_down_to_the_minimum():
    config = ATMConfig(
        variant=Variant(pool_rule="budget", pool_min=8, reevaluate_incumbent=False)
    )
    resolved = resolve_config(config, 10, 2)
    assert resolved.initial_population == 8
    result = atm.api.minimize(
        lambda x: float(np.sum(x**2)), np.ones(2), 300, seed=1, config=config
    )
    assert result.nfev == 300


def test_the_pool_rule_runs_to_the_budget_without_a_silent_restart():
    assert iteration_limit(ATMConfig(variant=BUDGET_RULE), 100) is None
