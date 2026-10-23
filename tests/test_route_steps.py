"""What ``atm.api._qn_local_phase`` does for each route and curvature reading.

The probes, SLSQP and the populations are replaced by spies, so each test checks
one decision of the route sequence (plateau, rugged, smooth stage, extension,
failed descents, basins) for every route of ``QN_ROUTES``, as the cycles that
introduced the routes describe them.
"""

import numpy as np
import pytest

from amm_optimizer import api, qn_phase
from amm_optimizer.bounds import Box
from amm_optimizer.config import Variant
from amm_optimizer.es_phase import EsResult
from amm_optimizer.qn_phase import QN_FD_STEP, QN_ROUTES, Descent

BOX = Box.make((-5.0, 5.0), 2)
POP = {"lambda_rule": "fixed2", "active": True}


class Calls:
    """The objective's call counter, as ``_qn_local_phase`` reads it."""

    calls = 100


def handover():
    return api._Handover(
        variant=Variant(pop_active=True),
        rng=np.random.RandomState(0),
        box=BOX,
        budget=1_000,
        dim=2,
        use_model=False,
        cma_options={},
        pop_options=dict(POP),
        nonseparable=True,
        mlsl=False,
        recorded_x=[],
        recorded_f=[],
        search=None,  # type: ignore[arg-type]
        allocation=np.zeros(4),
        best_x=np.zeros(2),
        best_cost=1.0,
        a0=np.eye(2),
        last_widths=(),
    )


@pytest.fixture
def spies(monkeypatch):
    log = {"slsqp": [], "pop": [], "basins": []}
    reading = {"plateau": False, "beta": 2.0, "ends": [], "basins": False}

    def plateau_check(objective, rng, x, cost, box):
        return reading["plateau"], 4

    def curvature_beta(objective, rng, x, cost, box):
        return reading["beta"], x, cost, 24

    def run_slsqp_restarts(objective, rng, x, cost, box, evaluations, *args, **kw):
        log["slsqp"].append((evaluations, args, kw))
        if kw.get("gate_out") is not None and "gate" in reading:
            kw["gate_out"].append(reading["gate"])
        if len(args) >= 2 and args[1] is not None:  # the stage's ``ends_out``
            args[1].extend(reading["ends"])
        return EsResult(x, cost, 0.25, evaluations)

    def multi_basin(objective, ends, box):
        log["basins"].append(len(ends))
        return reading["basins"]

    def run_population_restarts(*args, **kw):
        log["pop"].append(kw)
        return EsResult(args[2], args[3], 1.0, 0)

    monkeypatch.setattr(qn_phase, "plateau_check", plateau_check)
    monkeypatch.setattr(qn_phase, "curvature_beta", curvature_beta)
    monkeypatch.setattr(qn_phase, "run_slsqp_restarts", run_slsqp_restarts)
    monkeypatch.setattr(qn_phase, "multi_basin", multi_basin)

    def run_sampled_restarts(objective, rng, x, cost, box, evaluations, **kw):
        log.setdefault("sampled", []).append((evaluations, kw))
        return EsResult(x, cost, 0.25, evaluations)

    monkeypatch.setattr(qn_phase, "run_sampled_restarts", run_sampled_restarts)

    def run_scaled_bfgs(objective, rng, x, cost, box, evaluations, **kw):
        log.setdefault("bfgs", []).append((evaluations, kw))
        if "tried" in reading and kw.get("max_descents") == 1:
            used, x_t, cost_t = reading["tried"]
            return EsResult(x_t, cost_t, 0.25, used)
        return EsResult(x, cost, 0.25, evaluations)

    monkeypatch.setattr(qn_phase, "run_scaled_bfgs", run_scaled_bfgs)

    def micro_slopes(objective, x, cost, box, **kw):
        log.setdefault("micro", []).append(len(x))
        log.setdefault("micro_kw", []).append(kw)
        if reading.get("micro") is None and "micro" in reading:
            return None
        fractal = reading.get("micro", False)
        return fractal, (1.0, 1.0, 1.0) if fractal else (2.0, 2.0, 2.0), x, cost

    monkeypatch.setattr(qn_phase, "micro_slopes", micro_slopes)

    def kink_check(objective, x, cost, box):
        log.setdefault("kink", []).append(len(x))
        if reading.get("kink") is None and "kink" in reading:
            return None
        kink = reading.get("kink", False)
        return kink, 1e8 if kink else 1e3, x, cost

    monkeypatch.setattr(qn_phase, "kink_check", kink_check)
    monkeypatch.setattr(api, "run_population_restarts", run_population_restarts)
    return log, reading


CONVERGED = [
    Descent(np.zeros(2), 0.0, 0),
    Descent(np.ones(2), 0.0, 0),
]
FAILED = [
    Descent(np.zeros(2), 0.0, 4),
    Descent(np.ones(2), 0.0, 4),
    Descent(np.full(2, 2.0), 0.0, 0),
]
PLATEAU_IPOP = {
    *("qn8", "qn9", "qn10", "qn11", "qn12", "qn13", "qn13n"),
    *(
        "qn15",
        "qn17",
        "qn19",
        "qn20",
        "qn29",
        "qn38",
        "qn44",
    ),
}
GATED = {"qn19", "qn20"} | {
    "qn29",
    "qn38",
    "qn44",
}
CYCLE16 = {"qn6", "qn7"} | PLATEAU_IPOP


@pytest.mark.parametrize("name", sorted(QN_ROUTES))
def test_a_rugged_reading(spies, name):
    log, reading = spies
    reading["beta"] = 1.0
    h = handover()
    options = api._qn_local_phase(QN_ROUTES[name], h, Calls())
    if name in CYCLE16:
        rule = "stepsgrow" if name == "qn11" else "steps"
        assert options == {**POP, "lambda_rule": rule}
    else:
        assert options is h.pop_options
    assert not log["slsqp"] and not log["pop"]


@pytest.mark.parametrize("name", sorted(QN_ROUTES))
def test_a_plateau_reading(spies, name):
    log, reading = spies
    reading["plateau"] = True
    options = api._qn_local_phase(QN_ROUTES[name], handover(), Calls())
    if name == "qn":  # no plateau check: the curvature decides (smooth here)
        assert options is None and len(log["slsqp"]) == 1
    elif name in PLATEAU_IPOP:
        assert options is None
        assert log["pop"] == [{"lambda_rule": "ipop4", "active": True}]
    elif name in CYCLE16:
        assert options == {**POP, "lambda_rule": "steps"}
    else:
        assert options == POP and not log["pop"]


@pytest.mark.parametrize("name", sorted(QN_ROUTES))
def test_a_smooth_reading_runs_the_stage(spies, name):
    log, reading = spies
    reading["ends"] = CONVERGED
    h = handover()
    options = api._qn_local_phase(QN_ROUTES[name], h, Calls())
    budget, calls, dim = 1_000, 100, 2
    if name == "qn":  # SLSQP for the whole local phase
        assert options is None
        assert log["slsqp"] == [(budget - calls, (), {})]
        return
    evaluations, args, kw = log["slsqp"][0]
    share = (budget - calls) / 3
    assert evaluations == int(share if name == "qn2" else min(share, 100 * dim))
    assert kw["fd_step"] == (QN_FD_STEP if name in PLATEAU_IPOP - {"qn8"} else None)
    if name in {"qn10"} | GATED:
        reserve = max(0.1 * (budget - calls), 20 * dim)
        assert kw["extend_to"] == int(budget - calls - reserve)
    else:
        assert kw["extend_to"] is None
    assert kw.get("gate_once", False) == (name in GATED)
    assert log["basins"] == ([2] if name in CYCLE16 else [])
    assert options is h.pop_options
    assert h.sigma_override == [0.25]
    assert (h.a0 is None) == (name == "qn2")


@pytest.mark.parametrize("name", sorted(CYCLE16 - {"qn10"}))
def test_a_stage_cut_at_the_cap_is_extended(spies, name):
    log, reading = spies
    reading["ends"] = []  # the first descent was still running
    api._qn_local_phase(QN_ROUTES[name], handover(), Calls())
    assert len(log["slsqp"]) == 2
    evaluations, args, kw = log["slsqp"][1]
    left = 1_000 - 100
    more = 2 / 3 * left
    assert evaluations == int(more if name == "qn6" else min(more, 200))
    assert kw == {"fd_step": QN_FD_STEP if name not in ("qn6", "qn7", "qn8") else None}


def test_qn10_extends_inside_the_descent_instead(spies):
    log, reading = spies
    reading["ends"] = []
    api._qn_local_phase(QN_ROUTES["qn10"], handover(), Calls())
    assert len(log["slsqp"]) == 1 and log["basins"] == [0]


@pytest.mark.parametrize("name", ["qn12", "qn13", "qn13n"])
def test_mostly_failed_descents(spies, name):
    log, reading = spies
    reading["ends"] = FAILED
    options = api._qn_local_phase(QN_ROUTES[name], handover(), Calls())
    if name == "qn12":  # the hill-valley test as before
        assert log["basins"] == [3] and not log["pop"] and options is not None
    else:
        assert options is None and not log["basins"]
        surrogate = "lq2" if name == "qn13" else "none"
        assert log["pop"] == [{"active": True, "surrogate": surrogate}]


@pytest.mark.parametrize(
    "name",
    sorted(CYCLE16 - {"qn20", "qn29"} - {"qn38"} - {"qn44"}),
)
def test_separate_basins_keep_restarting_slsqp(spies, name):
    log, reading = spies
    reading["ends"] = CONVERGED
    reading["basins"] = True
    assert api._qn_local_phase(QN_ROUTES[name], handover(), Calls()) is None
    evaluations, args, kw = log["slsqp"][-1]
    assert evaluations == 1_000 - 100
    scipy_step = name in ("qn12", "qn13", "qn13n", "qn15", "qn17", *GATED) or name in (
        "qn6",
        "qn7",
        "qn8",
    )
    assert kw == {"fd_step": None if scipy_step else QN_FD_STEP}


def wide_handover(dim, budget):
    h = handover()
    h.box = Box.make((-5.0, 5.0), dim)
    h.dim, h.budget = dim, budget
    h.best_x = np.zeros(dim)
    return h


@pytest.mark.parametrize(
    ("dim", "left_per_dim", "rule"),
    [(2, 640, "ipop4"), (5, 3000, "ipop4"), (20, 2499, "ipop4"), (20, 2500, "ipop8")],
)
def test_qn15_sends_a_rugged_reading_with_budget_left_to_ipop(
    spies, dim, left_per_dim, rule
):
    log, reading = spies
    reading["beta"] = 1.0
    h = wide_handover(dim, Calls.calls + left_per_dim * dim)
    assert api._qn_local_phase(QN_ROUTES["qn15"], h, Calls()) is None
    assert log["pop"] == [{"lambda_rule": rule, "active": True}]
    assert not log["slsqp"]


def test_qn15_keeps_the_steps_finisher_below_its_budget_floor(spies):
    log, reading = spies
    reading["beta"] = 1.0
    h = wide_handover(5, Calls.calls + 639 * 5)
    options = api._qn_local_phase(QN_ROUTES["qn15"], h, Calls())
    assert options == {**POP, "lambda_rule": "steps"}
    assert not log["pop"]


@pytest.mark.parametrize("beta", [2.0])
def test_qn15_runs_the_smooth_stage_as_qn12(spies, beta):
    log, reading = spies
    reading["beta"] = beta
    calls = []
    for name in ("qn12", "qn15"):
        log["slsqp"].clear()
        h = wide_handover(5, Calls.calls + 3000 * 5)
        api._qn_local_phase(QN_ROUTES[name], h, Calls())
        calls.append(list(log["slsqp"]))
    assert calls[0] == calls[1] and calls[0]


@pytest.mark.parametrize(
    ("dim", "routed"), [(5, False), (10, False), (20, True), (40, True)]
)
def test_qn17_runs_the_rugged_populations_only_from_d20(spies, dim, routed):
    log, reading = spies
    reading["beta"] = 1.0
    h = wide_handover(dim, Calls.calls + 3000 * dim)
    options = api._qn_local_phase(QN_ROUTES["qn17"], h, Calls())
    if routed:
        assert options is None and len(log["pop"]) == 1
    else:
        assert options == {**POP, "lambda_rule": "steps"} and not log["pop"]


ONE_CONVERGED = [Descent(np.ones(10), 0.0, 0)]
ONE_FAILED = [Descent(np.ones(10), 0.0, 4)]


@pytest.mark.parametrize(
    ("gate", "ends", "continued"),
    [
        (False, [], True),  # the gate stopped the descent at the cap: continuation
        (True, ONE_CONVERGED, False),  # extended and converged: straight on
        (True, ONE_FAILED, True),  # extended but failed (f12): the continuation
    ],
)
def test_the_cap_gate_continues_only_a_descent_it_did_not_carry(
    spies, gate, ends, continued
):
    log, reading = spies
    reading["gate"], reading["ends"] = gate, list(ends)
    h = wide_handover(10, Calls.calls + 300 * 10)  # too little for qn19's second
    api._qn_local_phase(QN_ROUTES["qn19"], h, Calls())
    assert len(log["slsqp"]) == (2 if continued else 1)
    if continued:
        assert "gate_once" not in log["slsqp"][1][2]


@pytest.mark.parametrize(("left_per_dim", "second"), [(1000, True), (300, False)])
def test_qn19_adds_one_descent_after_a_gated_converged_one(spies, left_per_dim, second):
    log, reading = spies
    reading["gate"], reading["ends"] = True, list(ONE_CONVERGED)
    h = wide_handover(10, Calls.calls + left_per_dim * 10)
    api._qn_local_phase(QN_ROUTES["qn19"], h, Calls())
    assert len(log["slsqp"]) == (2 if second else 1)
    if second:
        evaluations, args, kw = log["slsqp"][1]
        assert kw["max_descents"] == 1 and kw["fd_step"] == QN_FD_STEP
        left = left_per_dim * 10
        assert evaluations == int(left - max(0.1 * left, 20 * 10))


def test_qn19_adds_nothing_when_the_gate_did_not_pass(spies):
    log, reading = spies
    reading["gate"], reading["ends"] = False, list(ONE_CONVERGED)
    h = wide_handover(10, Calls.calls + 1000 * 10)
    api._qn_local_phase(QN_ROUTES["qn19"], h, Calls())
    assert len(log["slsqp"]) == 1


@pytest.mark.parametrize(
    "name",
    ["qn20", "qn29"]
    + [
        "qn38",
        "qn44",
    ],
)
def test_qn20_runs_the_sampled_restarts_on_separate_basins(spies, name):
    log, reading = spies
    reading["ends"], reading["basins"] = list(CONVERGED), True
    assert api._qn_local_phase(QN_ROUTES[name], handover(), Calls()) is None
    assert len(log["slsqp"]) == 1  # the stage only
    assert log["sampled"] == [(1_000 - 100, {"fd_step": None})]


@pytest.mark.parametrize(
    ("micro", "slsqp", "pop"), [(True, 1, 0), (False, 0, 1), (None, 0, 0)]
)
@pytest.mark.parametrize(
    "name",
    ["qn29"]
    + [
        "qn38",
        "qn44",
    ],
)
def test_qn29_sends_a_fractal_rugged_reading_to_slsqp(spies, micro, slsqp, pop, name):
    log, reading = spies
    reading["beta"], reading["micro"] = 1.0, micro
    h = wide_handover(20, Calls.calls + 3000 * 20)
    assert api._qn_local_phase(QN_ROUTES[name], h, Calls()) is None
    assert log["micro"] == [20]
    assert log["micro_kw"] == [{"centered": True}]
    assert len(log["slsqp"]) == slsqp and len(log["pop"]) == pop
    if slsqp:  # the rest of the budget, SciPy's step, no stage switches
        assert log["slsqp"][0] == (3000 * 20, (), {})


@pytest.mark.parametrize("name", ["qn19", "qn20"])
def test_only_qn29_on_reads_the_micro_slopes(spies, name):
    log, reading = spies
    reading["beta"] = 1.0
    h = wide_handover(20, Calls.calls + 3000 * 20)
    api._qn_local_phase(QN_ROUTES[name], h, Calls())
    assert "micro" not in log and len(log["pop"]) == 1


def test_qn29_reads_nothing_below_d20(spies):
    log, reading = spies
    reading["beta"], reading["micro"] = 1.0, True
    h = wide_handover(10, Calls.calls + 3000 * 10)
    options = api._qn_local_phase(QN_ROUTES["qn29"], h, Calls())
    assert options == {**POP, "lambda_rule": "steps"} and "micro" not in log


@pytest.mark.parametrize(
    ("gate", "read"),
    [
        (False, True),  # the gate refused, no kink: the finisher
        (True, False),  # the gate passed: no check at all
    ],
)
def test_qn38_reads_the_kink_only_where_the_gate_refused(spies, gate, read):
    log, reading = spies
    reading["gate"], reading["kink"] = gate, False
    reading["ends"] = list(ONE_CONVERGED) if gate else []
    h = wide_handover(20, Calls.calls + 1000 * 20)
    options = api._qn_local_phase(QN_ROUTES["qn38"], h, Calls())
    assert ("kink" in log) is read
    assert options is h.pop_options


@pytest.mark.parametrize(
    "name",
    [
        "qn38",
        "qn44",
    ],
)
def test_qn30_runs_the_scaled_bfgs_on_a_kink_the_gate_refused(spies, name):
    log, reading = spies
    reading["gate"], reading["kink"], reading["ends"] = False, True, []
    h = wide_handover(20, Calls.calls + 1000 * 20)
    assert api._qn_local_phase(QN_ROUTES[name], h, Calls()) is None
    assert log["kink"] == [20]
    assert log["bfgs"] == [(1000 * 20, {})]  # uncapped: it restarts itself


@pytest.mark.parametrize(
    ("name", "dim", "left_per_dim", "tried"),
    [
        ("qn38", 5, 300, True),
        ("qn38", 9, 200, True),
        ("qn38", 5, 199, False),  # less than twice the try left
        ("qn29", 5, 300, False),
    ],
)
def test_qn38_tries_one_scaled_bfgs_descent_on_a_rugged_reading(
    spies, name, dim, left_per_dim, tried
):
    log, reading = spies
    reading["beta"] = 1.0
    h = wide_handover(dim, Calls.calls + left_per_dim * dim)
    api._qn_local_phase(QN_ROUTES[name], h, Calls())
    assert log.get("bfgs", []) == ([(100 * dim, {"max_descents": 1})] if tried else [])
    assert not log["slsqp"]


@pytest.mark.parametrize("name", ["qn38", "qn44"])
def test_a_try_that_ended_in_a_minimum_hands_over(spies, name):
    log, reading = spies
    reading["beta"] = 1.0
    reading["tried"] = (40 * 5, np.full(5, 0.5), 0.5)  # ended before the cap
    h = wide_handover(5, Calls.calls + 300 * 5)
    api._qn_local_phase(QN_ROUTES[name], h, Calls())
    assert log["bfgs"] == [(100 * 5, {"max_descents": 1})]
    assert h.best_cost == 0.5


@pytest.mark.parametrize(
    ("name", "dim", "left_per_dim", "ipop"),
    [
        ("qn38", 5, 2000, True),
        ("qn38", 10, 2900, True),
        ("qn38", 5, 1999, False),
        ("qn38", 4, 3000, False),  # below rugged_ipop_low_dim
        ("qn29", 5, 3000, False),
    ],
)
def test_qn38_opens_the_rugged_route_below_d20(spies, name, dim, left_per_dim, ipop):
    log, reading = spies
    reading["beta"], reading["micro"] = 1.0, True  # a fractal reading, if asked
    h = wide_handover(dim, Calls.calls + left_per_dim * dim)
    options = api._qn_local_phase(QN_ROUTES[name], h, Calls())
    if ipop:  # the populations from a uniform point, no fractal probe below D 20
        assert options is None and "micro" not in log
        assert log["pop"] == [{"lambda_rule": "ipop4", "active": True}]
    else:
        assert options == {**POP, "lambda_rule": "steps"} and not log["pop"]


@pytest.mark.parametrize("name", ["qn38"])
def test_qn38_keeps_the_fractal_probe_from_d20(spies, name):
    log, reading = spies
    reading["beta"], reading["micro"] = 1.0, True
    h = wide_handover(20, Calls.calls + 3000 * 20)
    assert api._qn_local_phase(QN_ROUTES[name], h, Calls()) is None
    assert log["micro"] == [20] and len(log["slsqp"]) == 1 and not log["pop"]


@pytest.mark.parametrize(
    ("name", "dim", "read"),
    [("qn38", 5, True), ("qn38", 10, True), ("qn44", 10, True)]
    + [("qn29", 10, False), ("qn38", 4, False)],
)
def test_qn38_reads_the_kink_from_d5(spies, name, dim, read):
    log, reading = spies
    reading["gate"], reading["kink"], reading["ends"] = False, True, []
    h = wide_handover(dim, Calls.calls + 1000 * dim)
    options = api._qn_local_phase(QN_ROUTES[name], h, Calls())
    assert ("kink" in log) is read
    if read:
        assert options is None and log["bfgs"] == [(1000 * dim, {})]
    else:
        assert options is h.pop_options and "bfgs" not in log


@pytest.mark.parametrize(
    ("dim", "budget_per_dim", "tried"),
    [(5, 300, True), (9, 500, True), (10, 300, False), (5, 600, False)],
)
def test_qn38_tries_only_below_d10_and_up_to_500x(spies, dim, budget_per_dim, tried):
    log, reading = spies
    reading["beta"] = 1.0
    h = wide_handover(dim, budget_per_dim * dim)
    Calls.calls, calls = 0, Calls.calls
    try:
        api._qn_local_phase(QN_ROUTES["qn38"], h, Calls())
    finally:
        Calls.calls = calls
    assert ("bfgs" in log) is tried


def test_qn38_leaves_a_smooth_reading_to_the_stage(spies):
    log, reading = spies
    reading["ends"] = CONVERGED
    h = wide_handover(5, Calls.calls + 300 * 5)
    api._qn_local_phase(QN_ROUTES["qn38"], h, Calls())
    assert "bfgs" not in log and log["slsqp"]


@pytest.mark.parametrize("name", ["qn19", "qn20"])
def test_only_the_kink_routes_read_the_kink(spies, name):
    log, reading = spies
    reading["gate"], reading["kink"], reading["ends"] = False, True, []
    h = wide_handover(20, Calls.calls + 1000 * 20)
    api._qn_local_phase(QN_ROUTES[name], h, Calls())
    assert "kink" not in log


def test_qn38_reads_nothing_below_d5(spies):
    log, reading = spies
    reading["gate"], reading["kink"], reading["ends"] = False, True, []
    h = wide_handover(4, Calls.calls + 1000 * 4)
    assert api._qn_local_phase(QN_ROUTES["qn38"], h, Calls()) is h.pop_options
    assert "kink" not in log


def test_qn38_keeps_the_finisher_with_too_little_left(spies):
    log, reading = spies
    reading["gate"], reading["kink"], reading["ends"] = False, True, []
    h = wide_handover(20, Calls.calls + 50 * 20)
    assert api._qn_local_phase(QN_ROUTES["qn38"], h, Calls()) is h.pop_options
    assert log["kink"] == [20]


def test_a_separable_rugged_reading_stays_rugged(spies):
    log, reading = spies
    reading["beta"] = 1.0
    h = wide_handover(5, Calls.calls + 300 * 5)
    h.nonseparable = False
    api._qn_local_phase(QN_ROUTES["qn29"], h, Calls())
    assert not log["slsqp"]


@pytest.mark.parametrize(
    ("dim", "budget_per_dim", "path"),
    [(5, 300, "try"), (9, 500, "try"), (5, 3000, "ipop"), (10, 3000, "ipop")]
    + [(10, 300, "steps"), (5, 1000, "steps")],
)
def test_qn38_routes_a_rugged_reading_by_dim_and_budget(
    spies, dim, budget_per_dim, path
):
    log, reading = spies
    reading["beta"] = 1.0
    h = wide_handover(dim, budget_per_dim * dim)
    Calls.calls, calls = 0, Calls.calls
    try:
        options = api._qn_local_phase(QN_ROUTES["qn38"], h, Calls())
    finally:
        Calls.calls = calls
    assert ("bfgs" in log) is (path == "try")
    assert bool(log["pop"]) is (path == "ipop")
    if path != "ipop":
        assert options == {**POP, "lambda_rule": "steps"}


@pytest.mark.parametrize(
    ("name", "budget_per_dim", "bfgs"),
    [("qn44", 300, True), ("qn44", 200, True), ("qn44", 301, False)]
    + [("qn38", 300, False)],
)
def test_qn44_polishes_a_smooth_reading_with_one_box_qp_descent_then_goes_on_staged(
    spies, name, budget_per_dim, bfgs
):
    log, reading = spies
    reading["ends"] = CONVERGED
    h = wide_handover(5, budget_per_dim * 5)
    Calls.calls, calls = 0, Calls.calls
    try:
        api._qn_local_phase(QN_ROUTES[name], h, Calls())
    finally:
        Calls.calls = calls
    if bfgs:
        assert log["bfgs"][0] == (
            budget_per_dim * 5,
            {"max_descents": 1, "box_qp": True},
        )
    else:
        assert "bfgs" not in log
    assert log["slsqp"]  # the usual staged route, from the BFGS's best if it ran
