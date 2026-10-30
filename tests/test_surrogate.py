"""The linear-quadratic surrogate of the population CMA-ES (cycle 10)."""

import numpy as np
import pytest
import scipy.linalg

from amm_optimizer.api import minimize
from amm_optimizer.bounds import Box
from amm_optimizer.config import ATMConfig, Variant
from amm_optimizer.objective import CountedObjective
from amm_optimizer.pop_cma import run_population_cma, run_population_restarts
from amm_optimizer.surrogate import (
    QuadraticSurrogate,
    _solve_pos,
    evaluate_generation,
    kendall_tau,
)


def test_kendall_tau_of_orderings():
    a = np.arange(6.0)
    assert kendall_tau(a, a) == 1.0
    assert kendall_tau(a, -a) == -1.0


@pytest.mark.parametrize("terms", [1, 2, 6, 11, 21, 41, 66, 81, 231])
@pytest.mark.snapshot
def test_solve_pos_is_scipys_solve_bit_for_bit(terms):
    """``dposv(lower=False)`` is what ``scipy.linalg.solve(assume_a="pos")`` computes.

    Re-check after a SciPy upgrade: ``lower=True`` and ``cho_factor`` differ.
    """
    rng = np.random.RandomState(terms)
    for _ in range(10):
        # Rows weighted as lq2 weights them, columns of very different scales.
        phi = rng.randn(2 * terms, terms) * rng.lognormal(0, 2, terms)
        weighted = phi.T * np.linspace(20.0, 1.0, 2 * terms) ** 2
        gram = weighted @ phi
        gram[np.diag_indices_from(gram)] += 1e-10 * np.trace(gram) / terms
        rhs = weighted @ rng.randn(2 * terms)
        got = _solve_pos(gram, rhs)
        want = scipy.linalg.solve(gram, rhs, assume_a="pos")
        assert got.shape == want.shape
        assert got.tobytes() == want.tobytes()


def test_solve_pos_raises_where_scipy_raises():
    with pytest.raises(np.linalg.LinAlgError):  # not positive definite
        _solve_pos(np.array([[1.0, 2.0], [2.0, 1.0]]), np.ones(2))
    for a, b in (
        (np.array([[1.0, 0.0], [np.nan, 1.0]]), np.ones(2)),  # dposv: upper only
        (np.array([[1.0, 0.0], [0.0, np.nan]]), np.ones(2)),
        (np.eye(2), np.array([1.0, np.inf])),
    ):
        with pytest.raises(ValueError):
            _solve_pos(a, b)


def test_the_full_model_recovers_a_rotated_quadratic():
    rng = np.random.RandomState(0)
    dim = 4
    h = rng.randn(dim, dim)
    h = h @ h.T

    def fun(x):
        return float(x @ h @ x + x.sum() + 3.0)

    model = QuadraticSurrogate(dim)
    for x in rng.randn(40, dim):
        model.add(x, fun(x))
    assert model.fit(np.zeros(dim), 1.0)
    assert model.terms == (dim + 1) * (dim + 2) // 2
    test = rng.randn(10, dim)
    assert np.allclose(
        model.predict(test), [fun(x) for x in test], rtol=1e-6, atol=1e-6
    )


def test_the_model_grows_from_linear_and_stops_at_diagonal_above_its_dimension():
    model = QuadraticSurrogate(30)
    assert model.sizes == [31, 61]
    for x in np.random.RandomState(1).randn(50, 30):
        model.add(x, float(x.sum()))
    assert model.fit(np.zeros(30), 1.0)
    assert model.terms == 31


def test_a_good_model_spends_few_evaluations_per_generation():
    rng = np.random.RandomState(2)
    dim = 3
    model = QuadraticSurrogate(dim)
    for x in rng.randn(30, dim):
        model.add(x, float(x @ x))
    calls = []

    def evaluate(x):
        calls.append(x)
        return float(x @ x)

    values, true, used = evaluate_generation(
        evaluate, rng.randn(12, dim), model, np.zeros(dim), 1.0, 100
    )
    assert used == len(calls) < 12
    assert np.isfinite(values).all()
    assert np.isnan(true).sum() == 12 - used


def test_the_plain_run_is_unchanged_by_the_option():
    def sphere(x):
        return float(np.sum((x - 0.5) ** 2))

    box = Box.make((-5, 5), 4)
    runs = [
        run_population_cma(
            CountedObjective(sphere, 400),
            np.random.RandomState(5),
            np.zeros(4),
            1.0,
            box,
            400,
            8,
            **kw,
        )
        for kw in ({}, {"surrogate": "none"})
    ]
    assert runs[0].cost == runs[1].cost and runs[0].used == runs[1].used


def test_the_lq2_prefix_runs_in_budget():
    def fun(x):
        return float(np.sum(x**2) + np.sum(np.cos(3 * x)))

    for extra in ({"es_kind": "lq2"},):
        variant = Variant(
            bounds="fold",
            initial_design="uniform",
            reevaluate_incumbent=False,
            amplitude_control="success_rate_targeted",
            es_max_budget_per_dim=100,
            es_phase="prefix",
            es_kind=extra.get("es_kind", "cma"),
            cma_handoff=0.2,
            **{k: v for k, v in extra.items() if k != "es_kind"},
        )
        for budget in (300, 2000):
            calls = []

            def counted(x, calls=calls):
                calls.append(1)
                return fun(x)

            result = minimize(
                counted,
                np.full(5, 2.0),
                budget,
                seed=1,
                config=ATMConfig(variant=variant),
                bounds=(-5, 5),
            )
            assert len(calls) <= budget
            assert np.isfinite(result.fun)


def test_lq2_solves_a_rotated_ellipsoid():
    dim = 6
    scales = 10.0 ** (2 * np.arange(dim) / (dim - 1))
    rot = np.linalg.qr(np.random.RandomState(6).randn(dim, dim))[0]

    def ellipsoid(x):
        y = rot @ (x - 0.7)
        return float(np.sum((scales * y) ** 2))

    box = Box.make((-5, 5), dim)
    objective = CountedObjective(ellipsoid, 900)
    x0 = np.full(dim, -2.0)
    out = run_population_restarts(
        objective,
        np.random.RandomState(7),
        x0,
        ellipsoid(x0),
        2.0,
        box,
        900,
        True,
        active=True,
        surrogate="lq2",
    )
    assert objective.calls <= 900
    assert out.cost < 1e-8


def test_the_lq_prefix_only_from_its_budget():
    def fun(x):
        return float(np.sum(x**2))

    from amm_optimizer import es_phase

    kinds = []
    real = es_phase.run_es_phase

    def spy(*args, **kwargs):
        kinds.append(args[6] if len(args) > 6 else kwargs.get("kind"))
        return real(*args, **kwargs)

    import amm_optimizer.api as api

    api.run_es_phase, saved = spy, api.run_es_phase
    try:
        variant = Variant(
            bounds="fold",
            initial_design="uniform",
            reevaluate_incumbent=False,
            amplitude_control="success_rate_targeted",
            es_max_budget_per_dim=100,
            es_phase="prefix",
            es_kind="lq2",
            es_lq_min_budget_per_dim=100,
            cma_handoff=0.2,
        )
        for budget in (150, 500):
            minimize(
                fun,
                np.full(5, 2.0),
                budget,
                seed=1,
                config=ATMConfig(variant=variant),
                bounds=(-5, 5),
            )
    finally:
        api.run_es_phase = saved
    assert kinds == ["cma", "lq2"]


def test_restarted_slsqp_solves_an_ellipsoid_in_budget():
    from amm_optimizer.qn_phase import run_slsqp_restarts

    rot = np.linalg.qr(np.random.RandomState(10).randn(6, 6))[0]

    def ellipsoid(x):
        y = rot @ (x - 0.3)
        return float(np.sum(10.0 ** np.arange(6) * y**2))

    objective = CountedObjective(ellipsoid, 600)
    x0 = np.full(6, 2.0)
    out = run_slsqp_restarts(
        objective,
        np.random.RandomState(0),
        x0,
        ellipsoid(x0),
        Box.make((-5, 5), 6),
        600,
    )
    assert objective.calls <= 600 and out.used <= 600
    assert out.cost < 1e-6


def test_curvature_beta_reads_smooth_and_rugged_functions():
    from amm_optimizer.qn_phase import BETA_THRESHOLD, curvature_beta

    box = Box.make((-5, 5), 8)
    x = np.full(8, 1.0)

    def ellipsoid(z):
        return float(np.sum(10.0 ** np.linspace(0, 3, 8) * z**2))

    def rugged(z):
        return float(np.sum(z**2 + 10 * (1 - np.cos(2 * np.pi * 3 * z))))

    for fun, smooth in ((ellipsoid, True), (rugged, False)):
        obj = CountedObjective(fun, 100)
        beta, _, _, used = curvature_beta(obj, np.random.RandomState(0), x, fun(x), box)
        assert used == 24
        assert (beta >= BETA_THRESHOLD) == smooth, (fun.__name__, beta)


def test_the_gated_slsqp_prefix_runs_in_budget():
    def fun(x):
        return float(np.sum(x**2))

    variant = Variant(
        bounds="fold",
        initial_design="uniform",
        reevaluate_incumbent=False,
        amplitude_control="success_rate_targeted",
        es_max_budget_per_dim=100,
        es_phase="prefix",
        es_kind="slsqplq",
        es_lq_nonsep=True,
        sep_probe=True,
        cma_handoff=0.2,
    )
    for budget in (50, 150, 500):
        calls = []

        def counted(x, calls=calls):
            calls.append(1)
            return fun(x)

        result = minimize(
            counted,
            np.full(5, 2.0),
            budget,
            seed=1,
            config=ATMConfig(variant=variant),
            bounds=(-5, 5),
        )
        assert len(calls) <= budget and np.isfinite(result.fun)


def test_the_combined_prefix_routes_by_budget(monkeypatch):
    import amm_optimizer.api as api

    seen = []
    real = api.run_es_phase

    def spy(*args, **kwargs):
        seen.append((args[6], args[12]))  # kind, rugged_kind
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "run_es_phase", spy)
    variant = Variant(
        bounds="fold",
        initial_design="uniform",
        reevaluate_incumbent=False,
        amplitude_control="success_rate_targeted",
        es_max_budget_per_dim=200,
        es_phase="prefix",
        es_kind="slsqplq",
        es_lq_nonsep=True,
        es_lq_min_budget_per_dim=100,
        sep_probe=True,
        cma_handoff=0.2,
    )
    for per_dim in (10, 50, 150):
        minimize(
            lambda x: float(np.sum(x**2)),
            np.full(5, 2.0),
            per_dim * 5,
            seed=1,
            config=ATMConfig(variant=variant),
            bounds=(-5, 5),
        )
    assert seen == [("cma", "cma"), ("slsqplq", "cma"), ("slsqplq", "lq2")]


def test_the_combined_prefix_sends_rugged_functions_to_lq2(monkeypatch):
    import amm_optimizer.pop_cma as pop_cma

    calls = []
    real = pop_cma.run_population_restarts

    def spy(*args, **kwargs):
        calls.append(kwargs.get("surrogate"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pop_cma, "run_population_restarts", spy)
    rot = np.linalg.qr(np.random.RandomState(11).randn(6, 6))[0]

    def rugged(x):
        z = rot @ x
        return float(np.sum(z**2 + 10 * (1 - np.cos(2 * np.pi * z))))

    variant = Variant(
        bounds="fold",
        initial_design="uniform",
        reevaluate_incumbent=False,
        amplitude_control="success_rate_targeted",
        es_max_budget_per_dim=100,
        es_phase="prefix",
        es_kind="slsqplq",
        es_lq_nonsep=True,
        es_lq_min_budget_per_dim=100,
        sep_probe=True,
        cma_handoff=0.2,
    )
    result = minimize(
        rugged,
        np.full(6, 2.0),
        600,
        seed=3,
        config=ATMConfig(variant=variant),
        bounds=(-5, 5),
    )
    assert np.isfinite(result.fun)
    assert calls and calls[0] == "lq2"


def test_parabolic_steps_solve_a_separable_quadratic_faster():
    from amm_optimizer.step_phase import run_step_phase

    dim = 6
    scales = 10.0 ** np.linspace(0, 4, dim)
    opt = np.linspace(-2, 3, dim)

    def ellipsoid(x):
        return float(np.sum(scales * (x - opt) ** 2))

    box = Box.make((-5, 5), dim)
    x0 = np.full(dim, 4.0)
    plain = run_step_phase(
        CountedObjective(ellipsoid, 10**6), x0, ellipsoid(x0), box, 15 * dim
    )
    fast = run_step_phase(
        CountedObjective(ellipsoid, 10**6),
        x0,
        ellipsoid(x0),
        box,
        15 * dim,
        parabolic=True,
    )
    assert fast.used <= 15 * dim
    assert fast.cost < 1e-6 < plain.cost


def test_hill_valley_test_separates_basins_and_ignores_convex_spread():
    from amm_optimizer.qn_phase import Descent, multi_basin

    box = Box.make((-5, 5), 4)

    def bowl(x):
        return float(np.sum(10.0 ** np.arange(4) * x**2))

    def wells(x):
        return float(np.sum((x**2 - 4) ** 2))

    near, far = np.full(4, 0.01), np.array([3.0, 0.0, 0.0, 0.0])
    spread = [Descent(near, bowl(near), 0), Descent(far, bowl(far), 0)]
    assert multi_basin(CountedObjective(bowl, 100), spread, box) is False
    left, right = np.full(4, -2.0), np.full(4, 2.0)
    split = [Descent(left, 0.0, 0), Descent(right, 0.0, 0)]
    assert multi_basin(CountedObjective(wells, 100), split, box) is True
    # more failed descents than converged ones: no verdict, no evaluations
    failing = split + [Descent(left, 0.0, 4)] * 3
    counted = CountedObjective(wells, 100)
    assert multi_basin(counted, failing, box) is False and counted.calls == 0
    assert multi_basin(CountedObjective(wells, 100), split[:1], box) is False


@pytest.mark.parametrize("kind", ["qn7"])
def test_qn7_extends_a_descent_still_running_at_the_cap(monkeypatch, kind):
    import amm_optimizer.qn_phase as qn

    calls = []
    real = qn.run_slsqp_restarts

    def spy(*args, **kwargs):
        calls.append(args[5])
        return real(*args, **kwargs)

    monkeypatch.setattr(qn, "run_slsqp_restarts", spy)
    monkeypatch.setattr(qn, "BETA_THRESHOLD", -1.0)  # read every reading as smooth
    monkeypatch.setattr(qn, "QN_LOCAL_CAP_PER_DIM", 3.0)  # too few for one descent
    rot = np.linalg.qr(np.random.RandomState(17).randn(6, 6))[0]

    def rosen(x):
        z = rot @ x
        return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2))

    variant = Variant(
        bounds="fold",
        initial_design="uniform",
        reevaluate_incumbent=False,
        amplitude_control="success_rate_targeted",
        es_phase="prefix",
        es_kind="cma",
        es_max_budget_per_dim=100,
        cma_handoff=0.2,
        cma_local="then_ipop",
        cma_local_min_dim=2,
        pop_gate_per_dim=100,
        sep_probe=True,
        cma_local_nonsep=kind,
        cma_warm_start=True,
    )
    out = minimize(
        rosen,
        np.full(6, 2.0),
        1000 * 6,
        seed=4,
        config=ATMConfig(variant=variant),
        bounds=(-5, 5),
    )
    assert np.isfinite(out.fun)
    # the capped stage, then two thirds of what was left ("qn7": at most another
    # capped stage); the CMA finisher has the rest
    assert len(calls) == 2 and calls[0] <= 3 * 6
    assert calls[1] <= 3 * 6


def test_one_slsqp_descent_stops_after_its_first_end():
    from amm_optimizer.qn_phase import Descent, run_slsqp_restarts

    def bowl(x):
        return float(np.sum((x - 1.0) ** 2))

    box = Box.make((-5, 5), 2)
    ends: list[Descent] = []
    out = run_slsqp_restarts(
        CountedObjective(bowl, 500),
        np.random.RandomState(0),
        np.zeros(2),
        bowl(np.zeros(2)),
        box,
        500,
        ends,
        max_descents=1,
    )
    assert len(ends) == 1 and ends[0].status == 0
    assert out.used < 100 and out.cost < 1e-10


@pytest.mark.parametrize(
    ("dim", "budget", "expected"),
    [
        (20, 200, "slsqplqc"),
        (20, 600, "slsqplq"),
        (10, 100, "cma"),
        (5, 150, "slsqplq"),
    ],
)
def test_slsqplq10c_uses_the_cheap_probe_only_below_30_d(
    monkeypatch, dim, budget, expected
):
    import amm_optimizer.api as api
    import amm_optimizer.qn_phase as qn

    kinds, directions = [], []
    real_es, real_beta = api.run_es_phase, qn.curvature_beta

    def spy_es(*args, **kwargs):
        kinds.append(args[6])
        return real_es(*args, **kwargs)

    def spy_beta(*args, **kwargs):
        directions.append(args[5] if len(args) > 5 else qn.BETA_DIRECTIONS)
        return real_beta(*args, **kwargs)

    monkeypatch.setattr(api, "run_es_phase", spy_es)
    monkeypatch.setattr(qn, "curvature_beta", spy_beta)
    variant = Variant(
        bounds="fold",
        initial_design="uniform",
        reevaluate_incumbent=False,
        amplitude_control="success_rate_targeted",
        es_phase="prefix",
        es_kind="slsqplq10c",
        es_max_budget_per_dim=100,
        cma_handoff=0.2,
    )
    minimize(
        lambda x: float(np.sum(x**2)),
        np.full(dim, 1.0),
        budget,
        seed=0,
        config=ATMConfig(variant=variant),
        bounds=(-5, 5),
    )
    assert kinds == [expected]
    if expected != "cma":
        assert directions[0] == (2 if expected == "slsqplqc" else 4)


@pytest.mark.parametrize(("budget", "descent"), [(20, True), (60, True), (200, False)])
def test_qnprefix30_descends_only_up_to_30_d(monkeypatch, budget, descent):
    import amm_optimizer.api as api

    kinds = []
    real = api.run_es_phase

    def spy(*args, **kwargs):
        kinds.append(args[6])
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "run_es_phase", spy)
    variant = Variant(
        bounds="fold",
        initial_design="uniform",
        reevaluate_incumbent=False,
        amplitude_control="success_rate_targeted",
        es_phase="prefix",
        es_kind="cma",
        es_max_budget_per_dim=100,
        cma_handoff=0.2,
        nm_max_dim=2,
        nm_stage="qnprefix30",
    )
    minimize(
        lambda x: float(np.sum(x**2)),
        np.full(2, 1.0),
        budget,
        seed=0,
        config=ATMConfig(variant=variant),
        bounds=(-5, 5),
    )
    assert kinds == ["slsqpnm" if descent else "nm"]


def test_qn8_sends_plateaus_to_large_uniform_start_populations(monkeypatch):
    import amm_optimizer.api as api

    calls = []
    real = api.run_population_restarts

    def spy(*args, **kwargs):
        calls.append(kwargs.get("lambda_rule"))
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "run_population_restarts", spy)
    rot = np.linalg.qr(np.random.RandomState(13).randn(6, 6))[0]

    def steps(x):
        z = rot @ (x - 0.5)
        return float(np.sum(10.0 ** np.linspace(0, 2, 6) * np.round(z) ** 2))

    variant = Variant(
        bounds="fold",
        initial_design="uniform",
        reevaluate_incumbent=False,
        amplitude_control="success_rate_targeted",
        es_phase="prefix",
        es_kind="cma",
        es_max_budget_per_dim=100,
        cma_handoff=0.2,
        cma_local="then_ipop",
        cma_local_min_dim=2,
        pop_gate_per_dim=100,
        sep_probe=True,
        cma_local_nonsep="qn8",
        cma_warm_start=True,
        pop_active=True,
    )
    # seed 3: the separable search fails over and the plateau check fires (seeds 1-2
    # solve this function in the separable search already)
    out = minimize(
        steps,
        np.full(6, 3.0),
        300 * 6,
        seed=3,
        config=ATMConfig(variant=variant),
        bounds=(-5, 5),
    )
    assert np.isfinite(out.fun)
    assert "ipop4" in calls


def test_ipop4_starts_at_four_times_the_default_and_doubles():
    from amm_optimizer.pop_cma import default_population, run_population_restarts

    sizes = []
    import amm_optimizer.pop_cma as pc

    real = pc.run_population_cma

    def spy(*args, **kwargs):
        sizes.append(args[6])
        return real(*args, **kwargs)

    pc.run_population_cma = spy
    try:

        def rastrigin(x):
            return float(np.sum(x**2 + 10 * (1 - np.cos(2 * np.pi * x))))

        box = Box.make((-5, 5), 3)
        run_population_restarts(
            CountedObjective(rastrigin, 4000),
            np.random.RandomState(0),
            np.full(3, 2.0),
            np.inf,
            2.0,
            box,
            4000,
            True,
            lambda_rule="ipop4",
        )
    finally:
        pc.run_population_cma = real
    lam = 4 * default_population(3)
    assert sizes[0] == lam and all(
        b == min(2 * a, b) for a, b in zip(sizes, sizes[1:], strict=False)
    )
    assert len(sizes) == 1 or sizes[1] in (2 * lam, 4000 - sum(sizes[:1]))


@pytest.mark.parametrize(("kind", "step"), [("qn8", None), ("qn9", 1e-9)])
def test_qn9_passes_the_small_fd_step_to_every_local_slsqp_call(
    monkeypatch, kind, step
):
    import amm_optimizer.qn_phase as qn

    steps = []
    real = qn.run_slsqp_restarts

    def spy(*args, **kwargs):
        steps.append(kwargs.get("fd_step"))
        return real(*args, **kwargs)

    monkeypatch.setattr(qn, "run_slsqp_restarts", spy)
    monkeypatch.setattr(qn, "BETA_THRESHOLD", -1.0)
    rot = np.linalg.qr(np.random.RandomState(15).randn(6, 6))[0]
    weights = 10.0 ** np.linspace(0, 2, 6)

    def wells(x):
        z = rot @ x
        return float(np.sum(weights * (z**2 - 4) ** 2) + 0.5 * np.sum(z))

    variant = Variant(
        bounds="fold",
        initial_design="uniform",
        reevaluate_incumbent=False,
        amplitude_control="success_rate_targeted",
        es_phase="prefix",
        es_kind="cma",
        es_max_budget_per_dim=100,
        cma_handoff=0.2,
        cma_local="then_ipop",
        cma_local_min_dim=2,
        pop_gate_per_dim=100,
        sep_probe=True,
        cma_local_nonsep=kind,
        cma_warm_start=True,
    )
    minimize(
        wells,
        np.full(6, 2.0),
        6000,
        seed=2,
        config=ATMConfig(variant=variant),
        bounds=(-5, 5),
    )
    assert len(steps) >= 2 and all(s == step for s in steps)


def test_a_smaller_fd_step_lowers_the_slsqp_floor_on_an_ill_conditioned_quadratic():
    from amm_optimizer.qn_phase import QN_FD_STEP, run_slsqp_restarts

    rot = np.linalg.qr(np.random.RandomState(3).randn(10, 10))[0]
    scales = 10.0 ** np.linspace(0, 6, 10)

    def ellipsoid(x):
        z = rot @ (x - 0.7)
        return float(np.sum(scales * z**2))

    box = Box.make((-5, 5), 10)
    x0 = np.zeros(10)
    runs = [
        run_slsqp_restarts(
            CountedObjective(ellipsoid, 3000),
            np.random.RandomState(0),
            x0,
            ellipsoid(x0),
            box,
            3000,
            fd_step=h,
        )
        for h in (None, QN_FD_STEP)
    ]
    assert runs[1].cost < runs[0].cost / 10


def test_stall_monitor_reads_slsqp_iterations_from_the_evaluation_stream():
    from scipy.optimize import minimize as sp_minimize

    from amm_optimizer.qn_phase import SCIPY_FD_STEP, StallMonitor

    rot = np.linalg.qr(np.random.RandomState(5).randn(6, 6))[0]

    def rosen(x):
        z = rot @ x
        return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (1 - z[:-1]) ** 2))

    monitor = StallMonitor(SCIPY_FD_STEP)

    def watched(z):
        value = rosen(z)
        monitor.observe(np.array(z), value)
        return value

    sp_minimize(
        watched,
        np.full(6, -1.5),
        method="SLSQP",
        bounds=[(-5, 5)] * 6,
        options={"maxiter": 60},
    )
    assert len(monitor.records) >= 20
    late = monitor.records[-10:]
    assert np.median([r for r, _ in late]) > 0 and monitor.progressing()


def test_qn12_keeps_the_small_step_for_the_stage_and_scipys_for_multi_basin_restarts(
    monkeypatch,
):
    import amm_optimizer.qn_phase as qn

    steps = []
    real = qn.run_slsqp_restarts

    def spy(*args, **kwargs):
        steps.append(("stage" if len(args) > 6 else "more", kwargs.get("fd_step")))
        return real(*args, **kwargs)

    monkeypatch.setattr(qn, "run_slsqp_restarts", spy)
    monkeypatch.setattr(qn, "BETA_THRESHOLD", -1.0)
    rot = np.linalg.qr(np.random.RandomState(15).randn(6, 6))[0]
    weights = 10.0 ** np.linspace(0, 2, 6)

    def wells(x):
        z = rot @ x
        return float(np.sum(weights * (z**2 - 4) ** 2) + 0.5 * np.sum(z))

    variant = Variant(
        bounds="fold",
        initial_design="uniform",
        reevaluate_incumbent=False,
        amplitude_control="success_rate_targeted",
        es_phase="prefix",
        es_kind="cma",
        es_max_budget_per_dim=100,
        cma_handoff=0.2,
        cma_local="then_ipop",
        cma_local_min_dim=2,
        pop_gate_per_dim=100,
        sep_probe=True,
        cma_local_nonsep="qn12",
        cma_warm_start=True,
    )
    minimize(
        wells,
        np.full(6, 2.0),
        6000,
        seed=2,
        config=ATMConfig(variant=variant),
        bounds=(-5, 5),
    )
    assert steps == [("stage", 1e-9), ("more", None)]
