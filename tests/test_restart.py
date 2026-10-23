import numpy as np

from amm_optimizer.restart import RestartPolicy, check_restart

GROWTH_OK = [1.0, 2.0, 3.0, 4.0]


def history(values):
    return [np.float64(v) for v in values]


def test_default_policy_is_used_when_none_is_given():
    assert check_restart(GROWTH_OK, history([1.0] * 5)) == (False, "")


def test_all_samplers_stuck_in_random_growth_triggers_restart():
    restart, reason = check_restart([31, 40, 35, 50], history([1.0]))
    assert restart
    assert reason == "Random Search Growth Factor"


def test_one_sampler_below_the_limit_prevents_restart():
    assert check_restart([31, 40, 30, 50], history([1.0]))[0] is False


def test_short_history_never_restarts_on_cost():
    assert check_restart(GROWTH_OK, history([1.0] * 100)) == (False, "")


def test_no_progress_is_insufficient_absolute_improvement():
    restart, reason = check_restart(GROWTH_OK, history([5.0] * 101))
    assert (restart, reason) == (True, "Insufficient Absolute Improvement")


def test_worsening_cost_counts_as_insufficient_improvement():
    costs = [1.0] * 100 + [2.0]
    assert check_restart(GROWTH_OK, history(costs))[0]


def test_tiny_progress_relative_to_a_huge_start_is_insufficient_relative():
    costs = [1e30] + [1.0] * 70 + [1.0 - 1e-9] * 30
    costs = costs[:100] + [1.0 - 1e-9] * 1
    costs[-30:] = [1.0 - 1e-9] * 30
    costs[-30] = 1.0  # past: 1.0, recent: 1 - 1e-9 -> absolute 1e-9
    restart, reason = check_restart(GROWTH_OK, history(costs))
    assert (restart, reason) == (True, "Insufficient Relative Improvement")


def test_stagnation_is_reachable_with_a_zero_threshold():
    policy = RestartPolicy(improvement_threshold=0.0)
    restart, reason = check_restart(GROWTH_OK, history([3.0] * 101), policy)
    assert (restart, reason) == (True, "Cost Stagnation")


def test_steady_progress_does_not_restart():
    costs = np.linspace(100.0, 1.0, 150)
    assert check_restart(GROWTH_OK, history(costs)) == (False, "")


def test_zero_initial_cost_uses_numpy_semantics_instead_of_raising():
    costs = [0.0] + [1.0] * 99 + [0.5, 0.5]
    costs[-30] = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        restart, _ = check_restart(GROWTH_OK, history(costs))
    assert restart is False
