"""The opt-in prefix options of a second screen of parked ideas (parked2).

Each option is off by default, so ``c43`` and ``c44`` keep their recorded traces (the
characterisation tests); these tests pin what each option does when it is on.
"""

from dataclasses import replace

import numpy as np
import pytest

from amm_optimizer import api, presets, qn_phase, step_phase
from amm_optimizer.bounds import Box
from amm_optimizer.es_phase import EsResult, _guarded_descents, run_es_phase
from amm_optimizer.objective import CountedObjective

# Pins research-only options: deselected in the default tier (tools/ci.py).
pytestmark = [
    pytest.mark.research,
    # the options it pins are research-only, deprecated in 0.6.0 (FutureWarning)
    pytest.mark.filterwarnings("ignore:Variant:FutureWarning"),
]

BOX5 = Box.make((-5.0, 5.0), 5)
BOX2 = Box.make((-5.0, 5.0), 2)


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def steps(x):
    """A plateau function: constant on unit cells (like bbob f7's rounding)."""
    return float(np.sum(np.floor(np.abs(np.asarray(x)) + 0.5) ** 2))


def counted(fun, allowed):
    return CountedObjective(fun, allowed)


def test_the_flat_gradient_reader_sees_a_zero_first_gradient():
    flat = qn_phase.FlatGradientReader(3)
    for v in (2.0, 2.0, 2.0, 2.0, 5.0):
        flat.after(np.zeros(3), v)
    assert flat.flat and not flat.before(np.zeros(3))
    sloped = qn_phase.FlatGradientReader(3)
    for v in (2.0, 2.0, 2.1, 2.0):
        sloped.after(np.zeros(3), v)
    assert not sloped.flat
    short = qn_phase.FlatGradientReader(3)
    short.after(np.zeros(3), 1.0)
    short.start_descent(1)  # only the first descent is read
    short.after(np.zeros(3), 1.0)
    assert not short.flat


def test_curvature_pairs_report_their_losses_and_clipping():
    pairs: list = []
    probed = qn_phase.curvature_beta(
        counted(sphere, 1_000),
        np.random.RandomState(0),
        np.zeros(5),
        0.0,
        BOX5,
        2,
        pairs_out=pairs,
    )
    assert probed is not None and len(pairs) == 2 * len(qn_phase.BETA_SCALES)
    assert all(loss > 0 and not clipped for loss, clipped in pairs)
    corner: list = []
    qn_phase.curvature_beta(
        counted(sphere, 1_000),
        np.random.RandomState(0),
        np.full(5, 5.0),
        sphere(np.full(5, 5.0)),
        BOX5,
        2,
        pairs_out=corner,
    )
    assert all(clipped for _loss, clipped in corner)


def test_step_ties_count_probes_that_repeat_the_incumbent():
    ties: list = []
    step_phase.run_step_phase(
        counted(steps, 1_000),
        np.full(5, 0.37),
        steps(np.full(5, 0.37)),
        BOX5,
        200,
        ties_out=ties,
    )
    smooth: list = []
    step_phase.run_step_phase(
        counted(sphere, 1_000),
        np.full(5, 0.37),
        sphere(np.full(5, 0.37)),
        BOX5,
        200,
        ties_out=smooth,
    )
    assert len(ties) > 0 and smooth == []


@pytest.mark.parametrize(("again", "slsqp"), [(2.0, True), (1.0, False)])
def test_an_undefined_reading_is_reprobed_inward(monkeypatch, again, slsqp):
    """``nan_reprobe``: the second reading, from the point moved inward, decides."""
    seen = {"slsqp": 0, "points": []}
    readings = iter([float("nan"), again])

    def beta(objective, rng, x, cost, box, directions=None, **kwargs):
        seen["points"].append(np.array(x))
        return next(readings), x, cost, 0

    def fake_slsqp(objective, rng, x, cost, box, evaluations, *args, **kwargs):
        seen["slsqp"] += 1
        return EsResult(x, cost, 0.1, 0)

    monkeypatch.setattr(qn_phase, "curvature_beta", beta)
    monkeypatch.setattr(qn_phase, "plateau_check", lambda *a: (False, 0))
    monkeypatch.setattr(qn_phase, "run_slsqp_restarts", fake_slsqp)
    run_es_phase(
        counted(sphere, 10_000),
        np.random.RandomState(0),
        np.full(5, 5.0),
        BOX5,
        500,
        4,
        "slsqplq",
        rugged_kind="cma",
        nan_reading="plateau",
        nan_reprobe=True,
    )
    margin = qn_phase.BETA_SCALES[-1] * 10.0
    assert len(seen["points"]) == 2
    assert np.all(np.abs(seen["points"][1]) <= 5.0 - margin + 1e-12)
    assert (seen["slsqp"] > 0) is slsqp


def test_the_clip_veto_keeps_a_concave_unclipped_reading_rugged(monkeypatch):
    calls = {"slsqp": 0}

    def beta(objective, rng, x, cost, box, directions=None, pairs_out=None):
        if pairs_out is not None:
            pairs_out.append((-1.0, False))
        return float("nan"), x, cost, 0

    def fake_slsqp(*args, **kwargs):
        calls["slsqp"] += 1
        return EsResult(args[2], args[3], 0.1, 0)

    monkeypatch.setattr(qn_phase, "curvature_beta", beta)
    monkeypatch.setattr(qn_phase, "run_slsqp_restarts", fake_slsqp)
    run_es_phase(
        counted(sphere, 10_000),
        np.random.RandomState(0),
        np.full(5, 3.0),
        BOX5,
        500,
        4,
        "slsqplq",
        rugged_kind="cma",
        nan_reading="smooth",
        nan_clip=True,
    )
    assert calls["slsqp"] == 0


def test_the_plateau_check_waits_for_step_ties(monkeypatch):
    checks = {"n": 0}

    def check(objective, rng, x, cost, box):
        checks["n"] += 1
        return False, 0

    monkeypatch.setattr(qn_phase, "plateau_check", check)
    for ties in (0, 1):
        run_es_phase(
            counted(sphere, 10_000),
            np.random.RandomState(0),
            np.full(5, 3.0),
            BOX5,
            500,
            4,
            "slsqplq",
            step_share=0.5,
            plateau_pop=True,
            plateau_ties=ties,
        )
    assert checks["n"] == 1  # the smooth sphere's STEP never ties: no second check


@pytest.mark.parametrize("fractal", [True, False])
def test_a_fractal_rugged_reading_gets_slsqp(monkeypatch, fractal):
    calls = {"slsqp": 0}

    def fake_slsqp(*args, **kwargs):
        calls["slsqp"] += 1
        return EsResult(args[2], args[3], 0.1, 0)

    monkeypatch.setattr(
        qn_phase, "curvature_beta", lambda o, r, x, c, b, d=None: (0.5, x, c, 0)
    )
    monkeypatch.setattr(
        qn_phase, "micro_slopes", lambda o, x, c, b, centered: (fractal, (), x, c)
    )
    monkeypatch.setattr(qn_phase, "run_slsqp_restarts", fake_slsqp)
    run_es_phase(
        counted(sphere, 10_000),
        np.random.RandomState(0),
        np.full(5, 3.0),
        BOX5,
        500,
        4,
        "slsqplq",
        rugged_kind="cma",
        fractal_route=True,
    )
    assert (calls["slsqp"] > 0) is fractal


@pytest.mark.parametrize(
    ("fun", "x", "descents"),
    [
        (steps, 0.37, 1),  # a flat first gradient (a plateau): no retry
        (sphere, 0.0, 2),  # a descent that ends where it started: one retry
    ],
)
def test_descent_retries_stop_on_a_plateau_and_retry_elsewhere(
    monkeypatch, fun, x, descents
):
    calls = {"n": 0}
    real = qn_phase.run_slsqp_restarts

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(qn_phase, "run_slsqp_restarts", counting)
    start = np.full(2, x)
    _guarded_descents(
        counted(fun, 10_000),
        np.random.RandomState(0),
        start,
        fun(start),
        BOX2,
        200,
        None,
        {},
        5,
    )
    assert calls["n"] == descents


def test_the_parked2_options_reach_the_prefix_only_when_on():
    c44 = presets.with_changes(
        "c43", prefix_plateau_pop=True, prefix_plateau_min_dim=5, small_switch=True
    ).variant
    assert api._parked2_kw(c44, 20, 2_000) == {}
    on = replace(
        c44,
        prefix_nan_clip=True,
        prefix_nan_reprobe=True,
        prefix_plateau_ties=1,
        prefix_fractal_min_dim=20,
        prefix_fractal_min_per_dim=100.0,
        prefix_descent_retries=5,
    )
    assert api._parked2_kw(on, 20, 2_000) == {
        "nan_clip": True,
        "nan_reprobe": True,
        "plateau_ties": 1,
        "fractal_route": True,
        "descent_retries": 5,
    }
    assert "fractal_route" not in api._parked2_kw(on, 20, 600)
    assert "fractal_route" not in api._parked2_kw(on, 10, 1_000)
    with pytest.raises(ValueError):
        replace(c44, prefix_descent_retries=-1)


def ellipsoid(x):
    x = np.asarray(x)
    return float(np.sum(10.0 ** (4 * np.arange(len(x)) / (len(x) - 1)) * x**2))


def test_the_stage_restart_may_pass_the_cap_after_an_early_converged_descent():
    """``extend_restart``: after a first descent that converged before the cap (it
    starts at the optimum), the restart's own descent may run past ``evaluations``
    while it progresses, up to ``extend_to``."""
    used = {}
    for extend in (False, True):
        objective = counted(ellipsoid, 100_000)
        qn_phase.run_slsqp_restarts(
            objective,
            np.random.RandomState(1),
            np.zeros(5),
            0.0,
            BOX5,
            60,
            extend_to=600,
            gate_once=True,
            extend_restart=extend,
        )
        used[extend] = objective.calls
    assert used[False] <= 60 < used[True] <= 600
