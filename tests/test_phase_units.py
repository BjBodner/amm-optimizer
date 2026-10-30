"""Unit tests of the composites' phases: budget exhaustion, rare branches, contracts.

Every phase must stop cleanly when the objective's budget runs out mid-step (the
``CountedObjective`` returns ``None``), keep the best point it saw, and never spend
more than it was given. These tests reach the branches the end-to-end tests rarely
hit (``tests/test_characterisation.py`` pins the composites as a whole).
"""

import numpy as np
import pytest

from amm_optimizer import cma_phase, pop_cma, qn_phase, step_phase, surrogate
from amm_optimizer.bounds import Box
from amm_optimizer.es_phase import EsResult, run_es_phase
from amm_optimizer.objective import CountedObjective

BOX5 = Box.make((-5.0, 5.0), 5)
BOX2 = Box.make((-5.0, 5.0), 2)


def sphere(x):
    return float(np.sum((np.asarray(x) - 1.0) ** 2))


def rastrigin(x):
    x = np.asarray(x)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def rotated_ellipsoid(x):
    x = np.asarray(x)
    q = np.linalg.qr(np.random.RandomState(7).randn(len(x), len(x)))[0]
    y = q @ (x - 0.5)
    return float(np.sum(10.0 ** (3 * np.arange(len(x)) / (len(x) - 1)) * y**2))


def counted(fun, allowed):
    calls = []

    def logged(x):
        calls.append(np.array(x))
        return fun(x)

    return CountedObjective(logged, allowed), calls


# ---------------------------------------------------------------- qn_phase ----
def test_slsqp_restarts_stop_when_the_objective_runs_out():
    objective, calls = counted(sphere, 15)
    x = np.full(5, 3.0)
    result = qn_phase.run_slsqp_restarts(
        objective, np.random.RandomState(0), x, sphere(x), BOX5, 500
    )
    assert result.used == len(calls) == 15
    assert result.cost == min([sphere(x)] + [sphere(c) for c in calls])


def test_slsqp_restarts_report_completed_descents_only():
    objective, _ = counted(sphere, 10_000)
    ends: list[qn_phase.Descent] = []
    x = np.full(5, 3.0)
    result = qn_phase.run_slsqp_restarts(
        objective, np.random.RandomState(0), x, sphere(x), BOX5, 400, None, ends
    )
    assert ends and all(d.status == 0 for d in ends)
    assert result.used <= 400
    assert result.cost < 1e-8
    assert 1e-8 * 10 <= result.sigma <= 0.2 * 10


@pytest.mark.parametrize("allowed", [0, 3, 23])
def test_curvature_beta_returns_none_when_the_objective_runs_out(allowed):
    objective, _ = counted(sphere, allowed)
    x = np.zeros(5)
    assert (
        qn_phase.curvature_beta(objective, np.random.RandomState(0), x, sphere(x), BOX5)
        is None
    )


def test_curvature_beta_reads_two_on_a_quadratic_and_respects_the_direction_count():
    objective, calls = counted(sphere, 100)
    x = np.zeros(5)
    beta, best_x, best_cost, used = qn_phase.curvature_beta(
        objective, np.random.RandomState(0), x, sphere(x), BOX5, 2
    )
    assert used == len(calls) == 2 * 2 * len(qn_phase.BETA_SCALES)
    assert abs(beta - 2.0) < 0.2
    assert best_cost <= sphere(x) and sphere(best_x) == best_cost


def test_plateau_check_returns_none_when_the_objective_runs_out():
    objective, _ = counted(sphere, 1)
    x = np.zeros(5)
    assert (
        qn_phase.plateau_check(objective, np.random.RandomState(0), x, sphere(x), BOX5)
        is None
    )


def test_plateau_check_finds_a_step_function_flat():
    def steps(x):
        return float(np.sum(np.floor(np.asarray(x))))

    objective, _ = counted(steps, 100)
    x = np.full(5, 0.5)
    flat, used = qn_phase.plateau_check(
        objective, np.random.RandomState(0), x, steps(x), BOX5
    )
    assert flat and used == 2 * qn_phase.TIE_DIRECTIONS


def test_multi_basin_returns_none_when_the_objective_runs_out():
    ends = [
        qn_phase.Descent(np.full(2, -3.0), 0.0, 0),
        qn_phase.Descent(np.full(2, 3.0), 0.0, 0),
    ]
    objective, _ = counted(sphere, 0)
    assert qn_phase.multi_basin(objective, ends, BOX2) is None


def test_multi_basin_vetoes_stages_where_failures_dominate():
    ends = [
        qn_phase.Descent(np.full(2, -3.0), 0.0, 0),
        qn_phase.Descent(np.full(2, 3.0), 0.0, 0),
        qn_phase.Descent(np.full(2, 1.0), 0.0, 4),
        qn_phase.Descent(np.full(2, 2.0), 0.0, 4),
        qn_phase.Descent(np.full(2, 0.0), 0.0, 4),
    ]
    objective, calls = counted(sphere, 100)
    assert qn_phase.multi_basin(objective, ends, BOX2) is False
    assert not calls


def bent_cigar(dim, seed=3):
    """bbob f12's shape: a rotated bent cigar (condition 1e6) with T_asy^0.5."""
    q = np.linalg.qr(np.random.RandomState(seed).randn(dim, dim))[0]
    ramp = np.arange(dim) / (dim - 1)

    def fun(x):
        z = q @ (np.asarray(x, dtype=float) - 0.7)
        pos = z > 0
        z[pos] = z[pos] ** (1 + 0.5 * ramp[pos] * np.sqrt(z[pos]))
        z = q @ z
        return float(z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2))

    return fun


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], False),
        ([4], False),
        ([0, 4], False),
        ([4, 4], True),
        ([0, 4, 8], True),
        ([0, 0, 4, 4], False),
    ],
)
def test_a_failure_majority_needs_two_descents_and_more_failures(statuses, expected):
    assert qn_phase.failure_majority(statuses) is expected


def test_lbfgsb_restarts_stop_when_the_objective_runs_out():
    objective, calls = counted(sphere, 15)
    x = np.full(5, 3.0)
    result = qn_phase.run_lbfgsb_restarts(
        objective, np.random.RandomState(0), x, sphere(x), BOX5, 500
    )
    assert result.used == len(calls) == 15
    assert result.cost == min([sphere(x)] + [sphere(c) for c in calls])


def test_lbfgsb_restarts_spend_at_most_their_budget_inside_the_box():
    fun = bent_cigar(10)
    objective, calls = counted(fun, 10_000)
    box = Box.make((-5.0, 5.0), 10)
    x = np.full(10, 3.0)
    result = qn_phase.run_lbfgsb_restarts(
        objective, np.random.RandomState(0), x, fun(x), box, 3000
    )
    assert result.used == len(calls) <= 3000
    assert all(np.all(np.abs(c) <= 5.0) for c in calls)
    assert result.cost < 1e-8
    assert 1e-8 * 10 <= result.sigma <= 0.2 * 10


def test_the_fallback_is_inert_while_descents_converge():
    runs = []
    for fallback in (False, True):
        objective, calls = counted(rotated_ellipsoid, 10_000)
        x = np.full(5, 3.0)
        result = qn_phase.run_slsqp_restarts(
            objective,
            np.random.RandomState(0),
            x,
            rotated_ellipsoid(x),
            BOX5,
            1500,
            fd_step=qn_phase.QN_FD_STEP,
            fallback=fallback,
        )
        runs.append((result, np.array(calls)))
    (plain, plain_calls), (guarded, guarded_calls) = runs
    assert np.array_equal(plain_calls, guarded_calls)
    assert (plain.cost, plain.used, plain.sigma) == (
        guarded.cost,
        guarded.used,
        guarded.sigma,
    )


@pytest.mark.snapshot
def test_the_fallback_hands_a_failing_call_to_lbfgsb(monkeypatch):
    fun = bent_cigar(20)
    box = Box.make((-5.0, 5.0), 20)
    x = np.full(20, 3.0)
    handed = []
    real = qn_phase.run_lbfgsb_restarts

    def spy(*args, **kwargs):
        handed.append(args[5])
        return real(*args, **kwargs)

    monkeypatch.setattr(qn_phase, "run_lbfgsb_restarts", spy)
    results = {}
    for fallback in (False, True):
        objective, calls = counted(fun, 100_000)
        ends: list[qn_phase.Descent] = []
        results[fallback] = qn_phase.run_slsqp_restarts(
            objective,
            np.random.RandomState(0),
            x,
            fun(x),
            box,
            6000,
            None,
            ends,
            fd_step=qn_phase.QN_FD_STEP,
            fallback=fallback,
        )
        assert results[fallback].used == len(calls) <= 6000
        if fallback:
            assert qn_phase.failure_majority([d.status for d in ends])
    assert len(handed) == 1 and 0 < handed[0] < 6000
    assert results[True].cost < 1e-9 < results[False].cost


@pytest.mark.parametrize(("gate_once", "longer"), [(True, True), (False, False)])
def test_the_cap_gate_is_read_once_and_then_holds(monkeypatch, gate_once, longer):
    verdicts = iter([True] + [False] * 10_000)
    monkeypatch.setattr(
        qn_phase.StallMonitor, "progressing", lambda self: next(verdicts)
    )
    objective, calls = counted(rotated_ellipsoid, 10_000)
    x = np.full(5, 3.0)
    gate: list[bool] = []
    result = qn_phase.run_slsqp_restarts(
        objective,
        np.random.RandomState(0),
        x,
        rotated_ellipsoid(x),
        BOX5,
        30,
        fd_step=qn_phase.QN_FD_STEP,
        extend_to=600,
        gate_once=gate_once,
        gate_out=gate,
    )
    # read once: the first verdict (True) carries the descent past the cap
    assert (result.used > 31) == longer
    assert result.used <= 600
    assert gate == ([True] if gate_once else [])


# ---------------------------------------------------------------- es_phase ----
def test_the_gated_prefix_returns_the_design_when_the_probe_runs_out():
    objective, calls = counted(sphere, 10)
    result = run_es_phase(
        objective, np.random.RandomState(0), np.full(5, 3.0), BOX5, 300, 4, "slsqplq"
    )
    assert len(calls) == 10
    assert result.cost == min(sphere(c) for c in calls[:5])
    assert result.used == 5  # the start and the design; the probe was cut


def test_the_isotropic_prefix_stops_at_its_sigma_floor():
    def cone(x):
        return float(np.sum(np.abs(x)))

    objective, calls = counted(cone, 10_000)
    result = run_es_phase(
        objective, np.random.RandomState(0), np.zeros(2), BOX2, 10_000, 0
    )
    assert len(calls) < 10_000  # every step from the apex fails: sigma shrinks out
    assert result.sigma <= 1e-12 * 10
    assert result.cost == 0.0


def test_the_isotropic_prefix_accepts_ties_and_spends_its_budget_on_a_plateau():
    """Ties count as successes, so on a flat region sigma grows (documented)."""
    objective, calls = counted(lambda x: 0.0, 400)
    with np.errstate(over="ignore"):
        result = run_es_phase(
            objective, np.random.RandomState(0), np.zeros(2), BOX2, 400, 4
        )
    assert len(calls) == 400
    assert result.cost == 0.0 and result.sigma > 10.0


# ----------------------------------------------------------------- pop_cma ----
@pytest.mark.parametrize(
    ("rule", "left_per_dim", "expected"),
    [
        ("default", 1000, 1),
        ("fixed2", 1000, 2),
        ("cap4", 1000, 1),
        ("steps", 100, 1),
        ("steps", 200, 2),
        ("steps", 700, 4),
        ("ipop4", 1000, 4),
    ],
)
def test_the_first_population_follows_the_lambda_rule(
    monkeypatch, rule, left_per_dim, expected
):
    sizes = []

    def spy(objective, rng, mean, sigma, box, evaluations, population, *args):
        sizes.append(population)
        return EsResult(mean, 0.0, sigma, 0, False)

    monkeypatch.setattr(pop_cma, "run_population_cma", spy)
    dim = 5
    pop_cma.run_population_restarts(
        None,
        np.random.RandomState(0),
        np.zeros(dim),
        1.0,
        1.0,
        BOX5,
        left_per_dim * dim,
        True,
        lambda_rule=rule,
    )
    assert sizes == [expected * pop_cma.default_population(dim)]


@pytest.mark.parametrize("rule", ["default", "cap4", "ipop4", "ipop8", "budget"])
def test_restarted_populations_grow_as_their_rule_says(monkeypatch, rule):
    sizes = []

    def spy(objective, rng, mean, sigma, box, evaluations, population, *args):
        sizes.append(population)
        return EsResult(mean, 0.0, sigma, 10 * population, True)

    monkeypatch.setattr(pop_cma, "run_population_cma", spy)
    pop_cma.run_population_restarts(
        None,
        np.random.RandomState(0),
        np.zeros(5),
        1.0,
        1.0,
        BOX5,
        10**6,
        True,
        lambda_rule=rule,
    )
    base = pop_cma.default_population(5)
    head = [s // base for s in sizes[:4]]
    if rule == "default":
        assert head == [1, 2, 4, 8]
    elif rule == "cap4":
        assert head == [1, 2, 4, 1]
    elif rule == "ipop4":
        assert head == [4, 8, 16, 32]
    elif rule == "ipop8":
        assert head == [8, 16, 32, 64]
    else:  # "budget": the budget left over 35 dim, at least the default
        assert sizes[0] == int(10**6 / (pop_cma.BUDGET_RUN_LENGTH * 5))


def test_a_population_run_starts_from_a_given_covariance():
    objective, calls = counted(rotated_ellipsoid, 2_000)
    cov0 = np.diag([4.0, 1.0, 1.0, 1.0, 0.25])
    run = pop_cma.run_population_cma(
        objective,
        np.random.RandomState(0),
        np.full(5, 2.0),
        0.5,
        BOX5,
        2_000,
        10,
        cov0=cov0,
    )
    assert run.used == len(calls) <= 2_000
    assert run.cost == min(rotated_ellipsoid(c) for c in calls)


def test_a_surrogate_run_stops_cleanly_when_the_objective_runs_out():
    objective, calls = counted(rotated_ellipsoid, 97)
    run = pop_cma.run_population_cma(
        objective,
        np.random.RandomState(0),
        np.full(5, 2.0),
        1.0,
        BOX5,
        1_000,
        10,
        surrogate="lq2",
    )
    assert len(calls) == 97
    assert run.cost == min(rotated_ellipsoid(c) for c in calls)
    assert not run.stalled


# --------------------------------------------------------------- surrogate ----
def test_kendall_tau_of_fewer_than_two_points_is_one():
    assert surrogate.kendall_tau(np.array([1.0]), np.array([2.0])) == 1.0


def test_a_fit_that_cannot_be_solved_reports_failure():
    model = surrogate.QuadraticSurrogate(2)
    for k in range(20):
        model.add(np.array([np.nan, k]), float(k))
    assert model.fit(np.zeros(2), 1.0) is False
    assert model.coef is None


# -------------------------------------------------------------- step_phase ----
def test_step_phase_stops_inside_the_axis_ends_when_the_budget_is_tiny():
    objective, calls = counted(sphere, 100)
    x = np.zeros(5)
    result = step_phase.run_step_phase(objective, x, sphere(x), BOX5, 3)
    assert result.used == len(calls) == 3
    assert not result.stalled


def test_step_phase_stops_when_the_objective_runs_out():
    objective, calls = counted(sphere, 30)
    x = np.zeros(5)
    result = step_phase.run_step_phase(objective, x, sphere(x), BOX5, 1_000)
    assert len(calls) == 30 == result.used
    assert result.cost == min([sphere(x)] + [sphere(c) for c in calls])


def test_step_phase_keeps_a_better_point_found_by_the_failed_check():
    def coupled(x):
        x = np.asarray(x)
        return float(np.sum((x - 1) ** 2) + 3 * x[0] * x[1])

    objective, calls = counted(coupled, 10_000)
    x = np.zeros(2)
    result = step_phase.run_step_phase(objective, x, coupled(x), BOX2, 1_000)
    assert result.stalled
    assert result.cost == min([coupled(x)] + [coupled(c) for c in calls])


def rotated_rastrigin(dim, seed=7):
    q = np.linalg.qr(np.random.RandomState(seed).randn(dim, dim))[0]

    def fun(x):
        z = q @ (np.asarray(x) - 0.3)
        return float(10 * dim + np.sum(z**2 - 10 * np.cos(2 * np.pi * z)))

    return fun


def separable_rastrigin(x):
    z = np.asarray(x) - 0.3
    return float(10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z)))


def test_a_stuck_step_passes_the_axis_check_on_a_rotated_function():
    # From a minimum no probe improves: re-evaluating a stored point of one axis
    # at the unchanged best returns the stored value, so the check never fails.
    fun = rotated_rastrigin(5)
    objective, calls = counted(fun, 10_000)
    x = np.full(5, 0.3)
    result = step_phase.run_step_phase(objective, x, fun(x), BOX5, 3_000)
    assert result.used == len(calls) == 3_000
    assert not result.stalled


def test_the_stall_check_stops_a_stuck_step_on_a_rotated_function():
    fun = rotated_rastrigin(5)
    objective, calls = counted(fun, 10_000)
    x = np.full(5, 0.3)
    result = step_phase.run_step_phase(
        objective, x, fun(x), BOX5, 3_000, pair_when_stuck=True
    )
    assert result.stalled
    assert result.used == len(calls) < 3 * 5 * 5  # within a few sweeps
    assert result.cost == min([fun(x)] + [fun(c) for c in calls])


def test_the_stall_check_passes_on_a_separable_function():
    objective, calls = counted(separable_rastrigin, 10_000)
    x = np.full(5, 0.3)
    result = step_phase.run_step_phase(
        objective, x, separable_rastrigin(x), BOX5, 3_000, pair_when_stuck=True
    )
    assert result.used == len(calls) == 3_000
    assert not result.stalled


def test_step_phase_stops_once_no_interval_can_be_split():
    tiny = Box.make((0.0, 1e-11), 2)
    objective, calls = counted(lambda x: 1.0, 10_000)
    result = step_phase.run_step_phase(objective, np.zeros(2), 1.0, tiny, 10_000)
    assert len(calls) == 2  # the two upper ends; nothing is wider than MIN_WIDTH
    assert result.cost == 1.0 and not result.stalled


def test_step_phase_spends_its_budget_on_a_constant_function():
    """No stored value is above the best, so the check never runs (documented)."""
    objective, calls = counted(lambda x: 1.0, 10_000)
    result = step_phase.run_step_phase(objective, np.zeros(2), 1.0, BOX2, 300)
    assert len(calls) == 300 and not result.stalled


def test_one_interval_difficulty_is_the_vectorised_one_bit_for_bit():
    """``_pair_difficulty`` (inserts) and ``_difficulty`` (re-bases) must agree."""
    rng = np.random.RandomState(3)
    t = np.sort(np.concatenate((rng.uniform(-5, 5, 300), [0.0, 1e-12, 2e-12, 1.0])))
    h = np.abs(rng.standard_cauchy(len(t))) * 10.0 ** rng.randint(-12, 6, len(t))
    h[::17], h[::23] = 0.0, np.inf
    vectorised = step_phase._difficulty(t, h)
    one_by_one = [
        step_phase._pair_difficulty(t[k], h[k], t[k + 1], h[k + 1])
        for k in range(len(t) - 1)
    ]
    assert np.array_equal(vectorised, np.array(one_by_one))
    assert np.isinf(vectorised).any() and np.isfinite(vectorised).any()


@pytest.mark.parametrize("fun", [sphere, rastrigin, rotated_ellipsoid, np.sum])
def test_step_chunks_do_not_change_the_search(monkeypatch, fun):
    """Chunk splits (``_Axis``) must leave every evaluated point as it was."""

    def run(**flags):
        objective, calls = counted(lambda x: float(np.floor(4 * fun(x))), 10_000)
        x = np.full(5, 2.3)
        result = step_phase.run_step_phase(
            objective, x, float(np.floor(4 * fun(x))), BOX5, 3_000, **flags
        )
        return np.array(calls), result.cost, result.stalled

    for flags in ({}, {"parabolic": True, "pair_when_stuck": True, "box_check": True}):
        whole = run(**flags)
        monkeypatch.setattr(step_phase, "CHUNK", 3)
        chunked = run(**flags)
        monkeypatch.undo()
        assert np.array_equal(whole[0], chunked[0]) and whole[1:] == chunked[1:]


# --------------------------------------------------------------- cma_phase ----
def test_the_curvature_probe_hands_rugged_functions_over(monkeypatch):
    monkeypatch.setattr(cma_phase, "RECORD_PROBES", True)
    cma_phase.PROBE_LOG.clear()
    objective, calls = counted(rastrigin, 10_000)
    x = np.full(5, 2.2)
    result = cma_phase.run_cma(
        objective,
        np.random.RandomState(0),
        x,
        rastrigin(x),
        0.5,
        BOX5,
        5_000,
        probe_at=150,
        probe_beta=1.3,
    )
    assert len(cma_phase.PROBE_LOG) == 1
    assert cma_phase.PROBE_LOG[0] < 1.3
    assert result.stalled and result.used == len(calls) < 5_000
    assert result.cost == min([rastrigin(x)] + [rastrigin(c) for c in calls])


def test_the_curvature_probe_keeps_smooth_functions_and_logs_beta(monkeypatch):
    monkeypatch.setattr(cma_phase, "RECORD_PROBES", True)
    cma_phase.PROBE_LOG.clear()
    objective, calls = counted(rotated_ellipsoid, 10_000)
    x = np.full(5, 2.0)
    result = cma_phase.run_cma(
        objective,
        np.random.RandomState(0),
        x,
        rotated_ellipsoid(x),
        0.5,
        BOX5,
        1_000,
        probe_at=150,
        probe_beta=1.3,
    )
    assert len(cma_phase.PROBE_LOG) == 1 and cma_phase.PROBE_LOG[0] >= 1.3
    assert result.used == len(calls) == 1_000


def test_the_probe_log_records_nothing_unless_asked():
    """Review B4: ``PROBE_LOG`` grew on every probe of every run."""
    cma_phase.PROBE_LOG.clear()
    assert not cma_phase.RECORD_PROBES
    objective, _ = counted(rotated_ellipsoid, 10_000)
    x = np.full(5, 2.0)
    cma_phase.run_cma(
        objective,
        np.random.RandomState(0),
        x,
        rotated_ellipsoid(x),
        0.5,
        BOX5,
        1_000,
        probe_at=150,
        probe_beta=1.3,
    )
    assert not cma_phase.PROBE_LOG


def test_a_probe_cut_short_by_the_budget_ends_the_run(monkeypatch):
    monkeypatch.setattr(cma_phase, "RECORD_PROBES", True)
    cma_phase.PROBE_LOG.clear()
    objective, calls = counted(sphere, 160)
    x = np.full(5, 2.0)
    result = cma_phase.run_cma(
        objective,
        np.random.RandomState(0),
        x,
        sphere(x),
        0.5,
        BOX5,
        5_000,
        probe_at=150,
    )
    assert len(calls) == 160 == result.used
    assert not cma_phase.PROBE_LOG
    assert result.cost == min([sphere(x)] + [sphere(c) for c in calls])


def test_the_slsqpdes_prefix_runs_one_descent_then_the_one_plus_one(monkeypatch):
    calls = []
    real = qn_phase.run_slsqp_restarts

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(qn_phase, "run_slsqp_restarts", spy)
    objective, evaluated = counted(rotated_ellipsoid, 50)
    result = run_es_phase(
        objective,
        np.random.RandomState(0),
        np.full(5, 3.0),
        BOX5,
        50,
        5,
        "slsqpdes",
        fd_step=qn_phase.QN_FD_STEP,
    )
    assert len(calls) == 1
    assert calls[0]["max_descents"] == 1 and calls[0]["fd_step"] == qn_phase.QN_FD_STEP
    # the (1+1) reports only its own share as ``used``; the objective counts all
    assert len(evaluated) == 50
    assert result.cost == min(rotated_ellipsoid(c) for c in evaluated)


def two_wells(x):
    """A shallow basin at 2 (value 1) and a deeper one at -3 (value 0)."""
    x = np.asarray(x, dtype=float)
    return float(min(np.sum((x - 2.0) ** 2) + 1.0, 0.5 * np.sum((x + 3.0) ** 2)))


def rotated_ridge(x):
    """f13 as bbob builds it: the sharp ridge, scaled (10**0.5) and rotated."""
    x = np.asarray(x, dtype=float)
    q = np.linalg.qr(np.random.RandomState(3).randn(len(x), len(x)))[0]
    y = 10.0 ** (0.5 * np.arange(len(x)) / (len(x) - 1)) * (q @ (x - 0.7))
    return float(y[0] ** 2 + 100.0 * np.sqrt(np.sum(y[1:] ** 2)))


@pytest.mark.parametrize("budget", [15, 40, 300])
def test_scaled_bfgs_spends_at_most_its_budget(budget):
    objective, calls = counted(rotated_ridge, 10_000)
    x = np.full(5, 3.0)
    result = qn_phase.run_scaled_bfgs(
        objective, np.random.RandomState(0), x, rotated_ridge(x), BOX5, budget
    )
    assert result.used == len(calls) == budget
    assert result.cost == min([rotated_ridge(x)] + [rotated_ridge(c) for c in calls])
    assert all(np.all(np.abs(c) <= 5.0) for c in calls)


def test_scaled_bfgs_stops_when_the_objective_runs_out():
    objective, calls = counted(rotated_ridge, 15)
    x = np.full(5, 3.0)
    result = qn_phase.run_scaled_bfgs(
        objective, np.random.RandomState(0), x, rotated_ridge(x), BOX5, 500
    )
    assert result.used == len(calls) == 15


def test_scaled_bfgs_follows_a_rotated_ridge_to_its_tip():
    x = np.full(5, 3.0)
    objective, _ = counted(rotated_ridge, 10_000)
    result = qn_phase.run_scaled_bfgs(
        objective, np.random.RandomState(0), x, rotated_ridge(x), BOX5, 1_500
    )
    assert result.cost < 1e-8 and np.allclose(result.x, 0.7, atol=1e-6)


def test_one_scaled_bfgs_descent_stops_short_of_its_budget_on_a_sphere():
    objective, calls = counted(sphere, 10_000)
    x = np.full(5, 3.0)
    result = qn_phase.run_scaled_bfgs(
        objective, np.random.RandomState(0), x, sphere(x), BOX5, 2_000, max_descents=1
    )
    assert result.cost < 1e-10 and result.used == len(calls) < 2_000
    assert 1e-8 * 10 <= result.sigma <= 0.2 * 10


@pytest.mark.parametrize("budget", [40, 300, 2_000])
def test_sampled_restarts_spend_at_most_their_budget(budget):
    objective, calls = counted(two_wells, 10_000)
    x = np.full(5, 2.0)
    result = qn_phase.run_sampled_restarts(
        objective, np.random.RandomState(0), x, two_wells(x), BOX5, budget
    )
    assert result.used == len(calls) <= budget
    assert result.cost == min([two_wells(x)] + [two_wells(c) for c in calls])


def test_the_fast_clearance_check_matches_the_norm_scan():
    """``_clear_of`` answers as ``all(norm(point - c) > radius)``, also at the edge."""
    rng = np.random.RandomState(5)
    radius = 1.5
    for _ in range(200):
        point = rng.uniform(-5, 5, 7)
        directions = rng.randn(40, 7)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        # distances well inside, well outside and within a few ulps of the radius
        lengths = radius * (1 + rng.choice([-0.5, 0.5, 1e-16, -1e-16, 0.0], 40))
        claimed = [point + u * r for u, r in zip(directions, lengths, strict=True)]
        rows = np.array(claimed)
        for first in (0, 20, 39):
            for count in (first + 1, 40):
                expected = all(
                    np.linalg.norm(point - c) > radius for c in claimed[first:count]
                )
                found = qn_phase._clear_of(point, claimed[:count], rows, first, radius)
                assert found == expected
    assert qn_phase._clear_of(point, claimed, rows, 40, radius)  # nothing to check


def test_sampled_restarts_leave_the_incumbents_basin_for_a_deeper_one():
    objective, _ = counted(two_wells, 10_000)
    x = np.full(5, 2.0)  # the shallow basin's minimum: SLSQP alone stays here
    result = qn_phase.run_sampled_restarts(
        objective, np.random.RandomState(0), x, two_wells(x), BOX5, 600
    )
    assert result.cost < 1e-8
    assert np.allclose(result.x, -3.0, atol=1e-3)


def test_the_gap_stop_ends_descents_that_cannot_catch_the_incumbent(monkeypatch):
    starts = []
    real = qn_phase.run_sampled_restarts

    def shallow_only(x):  # every descent lands at value 1 unless it starts near -3
        x = np.asarray(x, dtype=float)
        return float(1.0 + 1e-3 * np.sum((x - 2.0) ** 2))

    objective, calls = counted(shallow_only, 10_000)
    x = np.full(5, -3.0)
    result = real(objective, np.random.RandomState(0), x, 0.0, BOX5, 3_000)
    # the incumbent (0.0, passed as its cost) is never beaten, so every descent is
    # above it and stops early: many short descents instead of a few long ones
    assert result.cost == 0.0
    assert len(calls) == result.used <= 3_000
    del starts


def weierstrass(x):
    """Hoelder exponent log 2 / log 3 (about .63) down to 3^-20: fractal at 1e-6."""
    x = np.asarray(x, dtype=float)
    k = np.arange(21)
    terms = 0.5**k * np.cos(2 * np.pi * np.outer(x, 3.0**k))
    return float(np.sum(terms))


@pytest.mark.parametrize(
    ("fun", "fractal"),
    [(sphere, False), (rotated_ellipsoid, False), (weierstrass, True)],
)
def test_micro_slopes_read_fractal_only_where_the_second_difference_is(fun, fractal):
    objective, calls = counted(fun, 10_000)
    x = np.linspace(-1.3, 2.1, 5)
    read = qn_phase.micro_slopes(objective, x, fun(x), BOX5)
    assert read is not None
    verdict, slopes, best_x, best_cost = read
    assert verdict is fractal and len(slopes) == len(qn_phase.MICRO_SCALES) - 1
    if not fractal:
        assert all(abs(s - 2.0) < 0.2 for s in slopes)
    assert len(calls) == 2 * len(qn_phase.MICRO_SCALES) * qn_phase.MICRO_DIRECTIONS
    assert best_cost == min([fun(x)] + [fun(c) for c in calls])
    assert fun(best_x) == best_cost


def test_micro_slopes_stop_when_the_objective_runs_out():
    objective, calls = counted(sphere, 10)
    x = np.full(5, 1.0)
    assert qn_phase.micro_slopes(objective, x, sphere(x), BOX5) is None
    assert len(calls) == 10


def kinked(x):
    """|x - c| summed: a kink at c, so the second difference there is linear in h."""
    return float(np.sum(np.abs(np.asarray(x, dtype=float) - 0.7)))


def stepped(x):
    """A plateau at the micro scales: no second difference at all."""
    return float(np.sum(np.floor(np.asarray(x, dtype=float))))


@pytest.mark.parametrize("fun", [kinked, stepped])
def test_micro_slopes_do_not_read_kinks_or_plateaus_as_fractal(fun):
    objective, _ = counted(fun, 10_000)
    x = np.full(5, 0.7)
    read = qn_phase.micro_slopes(objective, x, fun(x), BOX5)
    assert read is not None and read[0] is False
    if fun is kinked:  # the mean slopes alone would read fractal
        assert all(abs(s - 1.0) < 1e-6 for s in read[1])


def sharp_ridge(x):
    """bbob f13's shape: a quadratic along x0 plus a cone across it."""
    x = np.asarray(x, dtype=float)
    return float(x[0] ** 2 + 100.0 * np.sqrt(np.sum(x[1:] ** 2)))


@pytest.mark.parametrize(("fun", "kink"), [(sharp_ridge, True), (sphere, False)])
def test_kink_check_reads_the_ridge_and_not_a_bowl(fun, kink):
    objective, calls = counted(fun, 10_000)
    x = np.array([0.3, 0.0, 0.0, 0.0, 0.0])
    read = qn_phase.kink_check(objective, x, fun(x), BOX5)
    assert read is not None and read[0] is kink
    assert len(calls) == 2 * qn_phase.KINK_DIRECTIONS
    assert read[3] == min([fun(x)] + [fun(c) for c in calls])


def test_kink_check_stops_when_the_objective_runs_out():
    objective, calls = counted(sphere, 3)
    x = np.full(5, 1.0)
    assert qn_phase.kink_check(objective, x, sphere(x), BOX5) is None
    assert len(calls) == 3


def test_centered_micro_slopes_move_a_point_on_the_bound_inside_first():
    objective, calls = counted(weierstrass, 10_000)
    x = np.array([5.0, 0.3, -1.2, 2.2, -5.0])  # two coordinates on the bounds
    read = qn_phase.micro_slopes(objective, x, weierstrass(x), BOX5, centered=True)
    assert read is not None
    probes = 2 * len(qn_phase.MICRO_SCALES) * qn_phase.MICRO_DIRECTIONS
    assert len(calls) == probes + 1  # the moved centre, then the probes
    margin = qn_phase.MICRO_SCALES[-1] * 10.0
    assert np.allclose(calls[0], np.clip(x, -5.0 + margin, 5.0 - margin))
    assert all(np.all(np.abs(c) < 5.0) for c in calls)  # nothing clipped
    assert read[3] == min([weierstrass(x)] + [weierstrass(c) for c in calls])


def test_centered_micro_slopes_spend_nothing_extra_inside_the_box():
    objective, calls = counted(sphere, 10_000)
    x = np.linspace(-1.3, 2.1, 5)
    read = qn_phase.micro_slopes(objective, x, sphere(x), BOX5, centered=True)
    assert read is not None and read[0] is False
    assert len(calls) == 2 * len(qn_phase.MICRO_SCALES) * qn_phase.MICRO_DIRECTIONS


def separable_ellipsoid(x):
    x = np.asarray(x, dtype=float)
    return float(
        np.sum(10.0 ** (3 * np.arange(len(x)) / (len(x) - 1)) * (x - 0.5) ** 2)
    )


@pytest.mark.parametrize(("smooth_probe", "slsqp"), [(False, 0), (True, 1)])
def test_a_separable_smooth_prefix_reaches_slsqp_only_with_the_cycle38_switch(
    monkeypatch, smooth_probe, slsqp
):
    calls = []

    def spy(objective, rng, x, cost, box, evaluations, *args, **kw):
        calls.append(evaluations)
        return EsResult(x, cost, 0.1, 0)

    monkeypatch.setattr(qn_phase, "run_slsqp_restarts", spy)
    objective, _ = counted(separable_ellipsoid, 1_000)
    run_es_phase(
        objective,
        np.random.RandomState(0),
        np.zeros(5),
        BOX5,
        500,
        5,
        "slsqplq",
        step_share=0.5,
        lq_nonsep_only=True,
        **({"sep_smooth_probe": True} if smooth_probe else {}),
    )
    assert len(calls) == slsqp


def step_sphere(x):
    """Flat steps (f7-like): a plateau at the probe scale."""
    return float(np.sum(np.floor(np.asarray(x, dtype=float) + 0.5) ** 2))


@pytest.mark.parametrize(("fun", "pop"), [(step_sphere, True), (sphere, False)])
def test_the_prefix_plateau_check_hands_plateaus_to_populations(monkeypatch, fun, pop):
    from amm_optimizer import pop_cma

    calls = []
    real = pop_cma.run_population_restarts

    def spy(*args, **kwargs):
        calls.append(args[6])  # the evaluations
        return real(*args, **kwargs)

    monkeypatch.setattr(pop_cma, "run_population_restarts", spy)
    objective, evaluated = counted(fun, 10_000)
    result = run_es_phase(
        objective,
        np.random.RandomState(0),
        np.full(5, 2.3),
        BOX5,
        500,
        5,
        "slsqplq",
        plateau_pop=True,
    )
    assert bool(calls) is pop
    assert len(evaluated) <= 500  # (the SLSQP path's ``used`` undercounts; cycle 33)
    assert result.cost == min(fun(x) for x in evaluated)


@pytest.mark.parametrize("seed", range(20))
def test_box_qp_matches_lbfgsb_on_random_box_qps(seed):
    from scipy.optimize import minimize as scipy_minimize

    rng = np.random.RandomState(seed)
    n = 2 + seed % 9
    q = np.linalg.qr(rng.randn(n, n))[0]
    b = q @ np.diag(10.0 ** rng.uniform(0, 4, n)) @ q.T
    b = 0.5 * (b + b.T)
    g = rng.randn(n) * 10
    lo, hi = -rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    d = qn_phase._box_qp(g, b, lo, hi)
    assert np.all(d >= lo - 1e-12) and np.all(d <= hi + 1e-12)
    model = lambda v: float(g @ v + 0.5 * v @ b @ v)  # noqa: E731
    ref = scipy_minimize(
        model,
        np.zeros(n),
        jac=lambda v: g + b @ v,
        method="L-BFGS-B",
        bounds=list(zip(lo, hi, strict=True)),
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 10_000},
    )
    assert model(d) <= ref.fun + 1e-9 * max(1.0, abs(ref.fun))


def ill_ellipsoid_at_a_face(x):
    """A rotated ellipsoid (condition 1e6) whose optimum sits 0.2 inside the box."""
    x = np.asarray(x, dtype=float)
    q = np.linalg.qr(np.random.RandomState(5).randn(len(x), len(x)))[0]
    y = q @ (x - 4.8)
    return float(np.sum(10.0 ** (6 * np.arange(len(x)) / (len(x) - 1)) * y**2))


@pytest.mark.parametrize("box_qp", [False, True])
def test_scaled_bfgs_with_box_qp_spends_at_most_its_budget(box_qp):
    objective, calls = counted(ill_ellipsoid_at_a_face, 10_000)
    x = np.zeros(5)
    result = qn_phase.run_scaled_bfgs(
        objective,
        np.random.RandomState(0),
        x,
        ill_ellipsoid_at_a_face(x),
        BOX5,
        600,
        box_qp=box_qp,
    )
    assert result.used == len(calls) <= 600
    assert all(np.all(np.abs(c) <= 5.0) for c in calls)


def test_box_qp_solves_an_ill_conditioned_ellipsoid_near_a_face():
    """The engine toy's f10 finding in miniature: at D 10 the clipped direction stalls
    by the box face (H resets), SLSQP's bound handling reaches the optimum."""
    box = Box(np.full(10, -5.0), np.full(10, 5.0))
    finals = {}
    for box_qp in (False, True):
        objective, _ = counted(ill_ellipsoid_at_a_face, 10_000)
        x = np.zeros(10)
        finals[box_qp] = qn_phase.run_scaled_bfgs(
            objective,
            np.random.RandomState(0),
            x,
            ill_ellipsoid_at_a_face(x),
            box,
            1_500,
            box_qp=box_qp,
        ).cost
    assert finals[True] < 1e-8 < 1.0 < finals[False]


@pytest.mark.parametrize(
    ("reading", "flat", "slsqp", "checked"),
    [
        ("rugged", False, False, False),
        ("smooth", False, True, False),
        ("plateau", False, True, True),
        ("plateau", True, False, True),
    ],
)
def test_an_undefined_probe_reading_goes_where_nan_reading_says(
    monkeypatch, reading, flat, slsqp, checked
):
    """Cycle 64: a NaN curvature slope reads rugged unless ``nan_reading`` says
    otherwise; ``"plateau"`` asks ``plateau_check`` first."""
    calls = {"slsqp": 0, "check": 0}

    def nan_beta(objective, rng, x, cost, box, directions=None):
        return float("nan"), x, cost, 0

    def fake_check(objective, rng, x, cost, box):
        calls["check"] += 1
        return flat, 0

    def fake_slsqp(objective, rng, x, cost, box, evaluations, *args, **kwargs):
        calls["slsqp"] += 1
        return EsResult(x, cost, 0.1, 0)

    monkeypatch.setattr(qn_phase, "curvature_beta", nan_beta)
    monkeypatch.setattr(qn_phase, "plateau_check", fake_check)
    monkeypatch.setattr(qn_phase, "run_slsqp_restarts", fake_slsqp)
    objective, _ = counted(sphere, 10_000)
    run_es_phase(
        objective,
        np.random.RandomState(0),
        np.full(5, 3.0),
        BOX5,
        500,
        4,
        "slsqplq",
        rugged_kind="cma",
        **({} if reading == "rugged" else {"nan_reading": reading}),
    )
    assert (calls["slsqp"] > 0) is slsqp
    assert (calls["check"] > 0) is checked
