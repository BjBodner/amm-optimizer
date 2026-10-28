"""The routing between the phases (``atm.api``) and the SLSQP route table.

``_prefix_kind`` replaced a nested conditional expression in ``minimize``; the
expression is kept here verbatim (``legacy_prefix_kind``) and the two must agree on
every option, dimension and budget boundary. ``QN_ROUTES`` replaced eight literal
membership tuples; its rows must keep each route's documented switches.
"""

import ast
import itertools
from dataclasses import replace

import pytest

from amm_optimizer import api, presets, qn_phase
from amm_optimizer.config import Variant
from amm_optimizer.qn_phase import QN_MIN_BUDGET, QN_MIN_BUDGET_PER_DIM

ES_KINDS = (
    "isotropic",
    "cma",
    "lq2",
    "slsqplq",
    "slsqplq10",
    "slsqplq10c",
    "slsqplq10h",
    "slsqplq10hk",
)
NM_STAGES = ("both", "prefix", "qnprefix", "qnprefix30", "qnprefix30g")


def legacy_prefix_kind(variant, dim, budget):
    """The prefix kind as ``minimize`` computed it inline until 2026-10-19 (b9be6af)."""
    return (
        "slsqpnm"
        if dim <= variant.nm_max_dim
        and (
            variant.nm_stage == "qnprefix"
            or (
                variant.nm_stage in ("qnprefix30", "qnprefix30g")
                and budget <= QN_MIN_BUDGET_PER_DIM * dim
            )
        )
        else "slsqpnmg"
        if dim <= variant.nm_max_dim
        and variant.nm_stage == "qnprefix30g"
        and budget >= QN_MIN_BUDGET
        else "nm"
        if dim <= variant.nm_max_dim and variant.nm_stage != "handover"
        else "cma"
        if (
            variant.es_kind in ("lq", "lq2")
            and budget < variant.es_lq_min_budget_per_dim * dim
        )
        or (variant.es_kind == "slsqplq" and budget < QN_MIN_BUDGET_PER_DIM * dim)
        or (
            variant.es_kind in ("slsqplq10", "slsqplq10c", "slsqplq10h")
            and budget < QN_MIN_BUDGET_PER_DIM * dim
            and budget < QN_MIN_BUDGET
        )
        else "ctrk"
        if variant.es_kind == "slsqplq10hk"
        and budget < QN_MIN_BUDGET_PER_DIM * dim
        and budget < QN_MIN_BUDGET
        else "slsqplqc"
        if variant.es_kind in ("slsqplq10c", "slsqplq10h", "slsqplq10hk")
        and budget < QN_MIN_BUDGET_PER_DIM * dim
        else "slsqplq"
        if variant.es_kind in ("slsqplq10", "slsqplq10c", "slsqplq10h", "slsqplq10hk")
        else variant.es_kind
    )


def budgets(dim):
    """Budgets on both sides of every threshold the prefix kind reads."""
    per_dim = (1, 10, 29, 30, 31, 99, 100, 101, 300)
    absolute = (QN_MIN_BUDGET - 1, QN_MIN_BUDGET, QN_MIN_BUDGET + 1)
    return sorted({int(m * dim) for m in per_dim} | set(absolute))


# research-only options (deprecated in 0.6.0)
@pytest.mark.filterwarnings("ignore:Variant:FutureWarning")
@pytest.mark.parametrize(
    ("es_kind", "nm_stage"), list(itertools.product(ES_KINDS, NM_STAGES))
)
def test_prefix_kind_equals_the_legacy_expression(es_kind, nm_stage):
    for nm_max_dim, lq_min in itertools.product((0, 2, 5), (0.0, 100.0)):
        variant = Variant(
            es_kind=es_kind,
            nm_stage=nm_stage,
            nm_max_dim=nm_max_dim,
            es_lq_min_budget_per_dim=lq_min,
        )
        for dim in (1, 2, 3, 5, 10, 20, 40):
            for budget in budgets(dim):
                assert api._prefix_kind(variant, dim, budget) == legacy_prefix_kind(
                    variant, dim, budget
                ), (es_kind, nm_stage, nm_max_dim, lq_min, dim, budget)


@pytest.mark.parametrize(
    ("dim", "budget", "expected"),
    [
        (2, 20, "slsqpnm"),  # 10 x D at D 2: one descent, then the simplex
        (2, 60, "slsqpnm"),  # 30 x D
        (2, 200, "nm"),  # 100 x D: the simplex alone
        (5, 50, "cma"),  # 10 x D, below 200 evaluations: the (1+1)
        (20, 200, "slsqplqc"),  # 10 x D from 200 evaluations: the cheap probe
        (40, 400, "slsqplqc"),
        (5, 150, "slsqplq"),  # from 30 x D: the four-direction probe
        (20, 2000, "slsqplq"),
    ],
)
def test_the_c27_prefix_routes_by_dimension_and_budget(dim, budget, expected):
    variant = presets.get("c27").variant  # its recorded composite (test_presets)
    assert api._prefix_kind(variant, dim, budget) == expected


# research-only options (deprecated in 0.6.0)
@pytest.mark.filterwarnings("ignore:Variant:FutureWarning")
def test_the_prefix_runs_only_up_to_its_budget_limit():
    variant = Variant(
        amplitude_control="success_rate",
        es_phase="prefix",
        es_max_budget_per_dim=100.0,
        es_small_dim=2,
        es_small_max_budget_per_dim=30.0,
    )
    assert api._prefix_runs(variant, 5, 500)
    assert not api._prefix_runs(variant, 5, 501)
    assert api._prefix_runs(variant, 2, 60)
    assert not api._prefix_runs(variant, 2, 61)
    assert not api._prefix_runs(Variant(), 5, 10)


def test_the_route_table_covers_exactly_the_configured_qn_routes():
    with pytest.raises(ValueError, match="cma_local_nonsep") as info:
        Variant(cma_local_nonsep="nope")
    allowed = ast.literal_eval(str(info.value).split("one of ")[1].split(", got")[0])
    assert {n for n in allowed if n.startswith("qn")} == set(qn_phase.QN_ROUTES)


# The routes' switches as their cycles documented them (config.py, RESULTS).
FIELDS = (
    "plateau_check",
    "staged",
    "capped",
    "seeded",
    "keep_warm_shape",
    "extend",
    "basins",
    "rugged_steps",
    "plateau_ipop4",
    "fd_step",
    "extend_in_descent",
    "rugged_grow",
    "basins_scipy_step",
    "failures_to_population",
    "failures_surrogate",
)
_ = False
DOCUMENTED = {
    "qn": (_, _, _, _, True, "none", _, _, _, _, _, _, _, _, "none"),
    "qn2": (True, True, _, _, _, "none", _, _, _, _, _, _, _, _, "none"),
    "qn3": (True, True, True, _, True, "none", _, _, _, _, _, _, _, _, "none"),
    "qn5": (True, True, True, True, True, "none", _, _, _, _, _, _, _, _, "none"),
    "qn6": (True, True, True, _, True, "share", True, True, _, _, _, _, _, _, "none"),
    "qn7": (True, True, True, _, True, "capped", True, True, _, _, _, _, _, _, "none"),
    "qn8": (True, True, True, _, True, "capped", True, True, True)
    + (_, _, _, _, _, "none"),
    "qn9": (True, True, True, _, True, "capped", True, True, True)
    + (True, _, _, _, _, "none"),
    "qn10": (True, True, True, _, True, "none", True, True, True)
    + (True, True, _, _, _, "none"),
    "qn11": (True, True, True, _, True, "capped", True, True, True)
    + (True, _, True, _, _, "none"),
    "qn12": (True, True, True, _, True, "capped", True, True, True)
    + (True, _, _, True, _, "none"),
    "qn13": (True, True, True, _, True, "capped", True, True, True)
    + (True, _, _, True, True, "lq2"),
    "qn13n": (True, True, True, _, True, "capped", True, True, True)
    + (True, _, _, True, True, "none"),
    "qn15": (True, True, True, _, True, "capped", True, True, True)
    + (True, _, _, True, _, "none"),
    "qn17": (True, True, True, _, True, "capped", True, True, True)
    + (True, _, _, True, _, "none"),
    "qn19": (True, True, True, _, True, "capped", True, True, True)
    + (True, True, _, True, _, "none"),
    "qn20": (True, True, True, _, True, "capped", True, True, True)
    + (True, True, _, True, _, "none"),
    "qn29": (True, True, True, _, True, "capped", True, True, True)
    + (True, True, _, True, _, "none"),
    "qn38": (True, True, True, _, True, "capped", True, True, True)
    + (True, True, _, True, _, "none"),
    "qn44": (True, True, True, _, True, "capped", True, True, True)
    + (True, True, _, True, _, "none"),
}


def test_only_qn15_and_qn17_run_populations_on_rugged_readings():
    routes = qn_phase.QN_ROUTES
    rugged = {n for n, r in routes.items() if r.rugged_ipop}
    late = {"qn20", "qn29"}
    late |= {"qn38"}
    late |= {"qn44"}
    assert rugged == {"qn15", "qn17", "qn19"} | late
    assert all(routes[n].rugged_ipop_min_dim == 20 for n in rugged - {"qn15"})


def test_only_qn18_and_qn19_gate_the_stage_once_at_the_cap():
    routes = qn_phase.QN_ROUTES
    gated = {
        "qn19",
        "qn20",
        "qn29",
        "qn38",
        "qn44",
    }
    assert {n for n, r in routes.items() if r.cap_gate} == gated
    assert all(routes[n].extend_in_descent for n in gated)
    second = {n for n, r in routes.items() if r.second_descent_min_left_per_dim}
    assert second == {
        "qn19",
        "qn20",
        "qn29",
        "qn38",
        "qn44",
    }
    assert {n for n, r in routes.items() if r.basins_sampled} == {
        "qn20",
        "qn29",
        "qn38",
        "qn44",
    }
    assert {n for n, r in routes.items() if r.fractal_exempt} == {
        "qn29",
        "qn38",
        "qn44",
    }
    kinked = {n for n, r in routes.items() if r.kink_small_step}
    assert kinked == {"qn38", "qn44"}
    assert all(routes[n].kink_min_dim == 5 for n in kinked)
    tries = {n for n, r in routes.items() if r.rugged_try_per_dim}
    assert tries == {"qn38", "qn44"}
    narrow = {n for n, r in routes.items() if r.rugged_try_max_dim}
    assert narrow == {"qn38", "qn44"}
    assert all(routes[n].rugged_try_max_budget_per_dim == 500 for n in narrow)
    low = {n for n, r in routes.items() if r.rugged_ipop_low_dim}
    assert low == {"qn38", "qn44"}
    smooth_bfgs = {n for n, r in routes.items() if r.smooth_bfgs_max_per_dim}
    assert smooth_bfgs == {"qn44"}
    assert {n for n, r in routes.items() if r.smooth_bfgs_box_qp} == {"qn44"}
    assert {
        n: r.smooth_bfgs_descents for n, r in routes.items() if r.smooth_bfgs_descents
    } == {"qn44": 1}
    assert all(routes[n].rugged_ipop_low_dim == 5 for n in low)


@pytest.mark.parametrize("name", sorted(DOCUMENTED))
def test_each_route_has_its_documented_switches(name):
    route = qn_phase.QN_ROUTES[name]
    assert tuple(getattr(route, f) for f in FIELDS) == DOCUMENTED[name]


def test_a_route_rejects_an_unknown_extension():
    with pytest.raises(ValueError, match="extend"):
        qn_phase.QnRoute(extend="forever")


# research-only options (deprecated in 0.6.0)
@pytest.mark.filterwarnings("ignore:Variant:FutureWarning")
def test_the_cma_options_follow_the_variant():
    variant = Variant(
        cma_model="lq",
        cma_restart_rule="short",
        cma_active=True,
    )
    assert api._cma_options(variant, 5) == {
        "model_kind": "lq",
        "model_full": True,
        "model_sigma": False,
        "restart_rule": "short",
        "active": True,
    }
    assert api._cma_options(Variant(), 5) == {}
    high = Variant(cma_model_high="diagonal")
    assert api._cma_options(high, 20) == {"model_kind": "diagonal"}
    assert api._cma_options(high, 10) == {}


def test_the_population_options_follow_the_variant():
    variant = Variant(pop_lambda="fixed2", pop_active=True)
    assert api._pop_options(variant) == {"lambda_rule": "fixed2", "active": True}
    assert api._pop_options(Variant()) == {}


@pytest.mark.parametrize("dim", [2, 3, 5, 10, 19, 20, 40])
def test_the_cycle33_prefix_kinds_differ_from_their_bases_only_in_their_slot(dim):
    base = presets.get("c33").variant
    q7ed = replace(base, es_kind="slsqplq10hd")
    for budget in budgets(dim):
        old = api._prefix_kind(base, dim, budget)
        # ``slsqplq10hd``: one SLSQP descent where ``slsqplq10h`` skips the probe
        new = api._prefix_kind(q7ed, dim, budget)
        assert new == ("slsqpdes" if old == "cma" and dim > base.nm_max_dim else old)


@pytest.mark.parametrize("dim", [2, 3, 5, 6, 7, 10, 20, 40])
def test_the_cycle58_descent_envelope_moves_only_30_x_dim_below_qn_min_budget(dim):
    q7ed = replace(presets.get("c39").variant, es_kind="slsqplq10hd")
    des30 = replace(q7ed, des_max_per_dim=31.0)
    for budget in budgets(dim):
        old = api._prefix_kind(q7ed, dim, budget)
        new = api._prefix_kind(des30, dim, budget)
        widened = (
            dim > q7ed.nm_max_dim
            and 30 * dim <= budget < 31 * dim
            and budget < qn_phase.QN_MIN_BUDGET
        )
        assert new == ("slsqpdes" if widened else old)
