"""``atm.basin_phase``: the design readings, MLSL, and the ``basin_route`` option."""

from dataclasses import replace

import numpy as np
import pytest

from amm_optimizer import basin_phase, presets
from amm_optimizer.api import minimize
from amm_optimizer.bounds import Box
from amm_optimizer.objective import CountedObjective

BOX = Box.make((-5.0, 5.0), 2)
CENTRES = np.random.RandomState(11).uniform(-4.5, 4.5, (20, 2))
DEPTHS = np.linspace(1.0, 2.0, 20)


def wells(x):
    """Gallagher-like: 20 smooth Gaussian wells of different depths, the deepest at
    CENTRES[-1] (value -2)."""
    x = np.asarray(x, dtype=float)
    d2 = ((CENTRES - x) ** 2).sum(1)
    return float(-np.max(DEPTHS * np.exp(-d2 / 1.0)))


def sphere(x):
    return float(np.sum((np.asarray(x, dtype=float) - 1.0) ** 2))


def rugged(x):
    """Weierstrass-like: rugged at every scale the probe reads."""
    x = np.asarray(x, dtype=float)
    return float(
        sum(np.sum(0.5**k * np.cos(2 * np.pi * 3**k * (x + 0.5))) for k in range(12))
    )


def steps(x):
    return float(np.sum(np.floor(np.abs(np.asarray(x, dtype=float)))))


def counted(fun, allowed):
    calls = []

    def logged(x):
        calls.append(np.array(x))
        return fun(x)

    return CountedObjective(logged, allowed), calls


@pytest.mark.parametrize(
    ("budget", "dim", "n0"), [(200, 2, 32), (90, 3, 22), (6, 2, 3)]
)
def test_the_design_size(budget, dim, n0):
    assert basin_phase.design_size(budget, dim) == n0


def test_the_rank_quadratic_fit_tells_one_basin_from_many():
    x = np.random.RandomState(0).uniform(-5, 5, (32, 2))
    one = basin_phase.rank_quadratic_r2(x, np.array([sphere(p) for p in x]), BOX)
    many = basin_phase.rank_quadratic_r2(x, np.array([wells(p) for p in x]), BOX)
    assert one > 0.8 and many < basin_phase.GATE_R2Q
    assert basin_phase.rank_quadratic_r2(x[:5], np.zeros(5), BOX) is None  # too few


@pytest.mark.parametrize(
    ("fun", "smooth", "flat"),
    [(sphere, True, 0.0), (rugged, False, 0.0), (steps, True, 1.0)],
)
def test_the_local_reading(fun, smooth, flat):
    objective, calls = counted(fun, 10_000)
    f = basin_phase.BudgetedObjective(objective, 10_000, as_float=True)
    x = np.array([0.3, -1.7])
    lcurv, flat_share = basin_phase.local_reading(f, x, fun(x), BOX)
    assert (lcurv <= basin_phase.GATE_LCURV) is smooth or fun is steps
    assert flat_share == flat
    assert len(calls) == 4 * 2


def test_the_critical_radius_shrinks_with_the_sample():
    radii = [basin_phase.critical_radius(n, BOX) for n in (10, 100, 1000)]
    assert radii[0] > radii[1] > radii[2] > 0


@pytest.mark.parametrize("budget", [30, 400, 2_000])
def test_the_route_spends_at_most_its_budget(budget):
    objective, calls = counted(wells, 10_000)
    routed, x, cost, _ = basin_phase.multi_basin_route(
        objective, np.random.RandomState(0), BOX, budget
    )
    assert len(calls) <= budget
    assert cost == min(wells(c) for c in calls) and wells(x) == cost


def test_the_route_sends_many_smooth_basins_to_mlsl_and_finds_the_deepest():
    objective, _ = counted(wells, 10_000)
    routed, x, cost, readings = basin_phase.multi_basin_route(
        objective, np.random.RandomState(0), BOX, 2_000
    )
    assert routed and readings["multi"] == 1.0
    assert cost < -1.99 and np.allclose(x, CENTRES[-1], atol=1e-3)


@pytest.mark.parametrize("fun", [sphere, rugged])
def test_one_basin_or_a_rugged_reading_is_not_routed(fun):
    objective, calls = counted(fun, 10_000)
    routed, x, cost, readings = basin_phase.multi_basin_route(
        objective, np.random.RandomState(0), BOX, 2_000
    )
    assert not routed and readings["multi"] == 0.0
    assert len(calls) == 32 + 4 * 2 + 1 or len(calls) == 32 + 4 * 2


@pytest.mark.parametrize(
    ("dim", "budget_per_dim"),
    [(2, 99), (4, 300)],  # under the budget, over the dim
)
def test_outside_its_scope_the_option_changes_nothing(dim, budget_per_dim):
    x0 = np.full(dim, 2.0)
    config = presets.with_changes("c38", basin_route=True)
    on = minimize(
        wells if dim == 2 else sphere,
        x0,
        budget_per_dim * dim,
        seed=3,
        config=config,
        bounds=(-5, 5),
    )
    off = minimize(
        wells if dim == 2 else sphere,
        x0,
        budget_per_dim * dim,
        seed=3,
        config=presets.get("c38"),
        bounds=(-5, 5),
    )
    assert on.fun == off.fun and np.array_equal(on.x, off.x)


def test_the_route_inside_minimize(monkeypatch):
    rounds = []
    real = basin_phase._mlsl_rounds

    def spy(*args, **kw):
        rounds.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(basin_phase, "_mlsl_rounds", spy)
    calls = []

    def logged(x):
        calls.append(1)
        return wells(x)

    config = presets.with_changes("c38", basin_route=True)
    result = minimize(
        logged, np.full(2, 2.0), 300 * 2, seed=1, config=config, bounds=(-5, 5)
    )
    assert rounds == [1] and len(calls) <= 600
    assert result.fun < -1.99


@pytest.mark.parametrize("budget_per_dim", [300, 1000])
def test_the_stall_check_acts_only_from_its_budget(budget_per_dim):
    from amm_optimizer import step_phase

    seen = []
    real = step_phase.run_step_phase

    def spy(*args, **kw):
        seen.append(kw.get("pair_when_stuck", False))
        return real(*args, **kw)

    def rotated(x):
        q = np.linalg.qr(np.random.RandomState(4).randn(5, 5))[0]
        return float(np.sum(10.0 ** np.arange(5) * (q @ np.asarray(x)) ** 2))

    import amm_optimizer.step_phase as module

    module.run_step_phase, saved = spy, module.run_step_phase
    try:
        config = presets.with_changes(
            "c38", sep_check="stall", sep_stall_min_per_dim=500.0
        )
        minimize(
            rotated,
            np.full(5, 2.0),
            budget_per_dim * 5,
            seed=1,
            config=config,
            bounds=(-5, 5),
        )
    finally:
        module.run_step_phase = saved
    assert seen and all(flag is (budget_per_dim >= 500) for flag in seen)


def ellipse(x):
    """One smooth basin with condition 1e4, its minimum at (1, -2)."""
    x = np.asarray(x, dtype=float)
    return float((x[0] - 1.0) ** 2 + 1e4 * (x[1] + 2.0) ** 2)


def test_descents_solve_one_smooth_basin_and_hand_back_the_design_point():
    objective, calls = counted(ellipse, 10_000)
    start: list = []
    routed, x, cost, readings = basin_phase.multi_basin_route(
        objective, np.random.RandomState(0), BOX, 600, descents=2, start_out=start
    )
    assert not routed and readings["multi"] == 0.0 and readings["descents"] == 2.0
    # the best seen is the descents' end; the path goes on from the design's best
    assert cost < 1e-8 and np.allclose(x, [1.0, -2.0], atol=1e-4)
    ((x0, f0),) = start
    design = calls[: basin_phase.design_size(600, 2)]
    assert f0 == min(ellipse(c) for c in design) and ellipse(x0) == f0
    assert len(calls) <= 600


@pytest.mark.parametrize("fun", [wells, rugged])
def test_descents_never_run_on_a_multi_basin_or_rugged_reading(fun):
    runs = []
    for kwargs in ({}, {"descents": 2, "start_out": []}):
        objective, calls = counted(fun, 10_000)
        out = basin_phase.multi_basin_route(
            objective, np.random.RandomState(0), BOX, 2_000, **kwargs
        )
        runs.append((out, calls))
    (a, calls_a), (b, calls_b) = runs
    assert a[0] == b[0] and "descents" not in b[3]
    assert all(np.array_equal(p, q) for p, q in zip(calls_a, calls_b, strict=True))


@pytest.mark.parametrize(
    ("budget_per_dim", "on"), [(100, True), (300, True), (301, False)]
)
def test_basin_descents_act_up_to_their_budget(monkeypatch, budget_per_dim, on):
    seen = {}
    real = basin_phase.multi_basin_route

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(basin_phase, "multi_basin_route", spy)
    c41 = presets.get("c41")
    config = replace(c41, variant=replace(c41.variant, basin_descents=2))
    minimize(
        ellipse, np.zeros(2), budget_per_dim * 2, seed=0, config=config, bounds=(-5, 5)
    )
    assert ("descents" in seen) is on


def test_basin_keep_atm_adds_the_descents_to_atms_share(monkeypatch):
    from amm_optimizer.objective import CountedObjective as Counted

    limits: dict = {}
    real = Counted.set_limit

    def spy(self, calls):
        limits.setdefault(key[0], calls)
        return real(self, calls)

    monkeypatch.setattr(Counted, "set_limit", spy)
    c41 = presets.get("c41")
    key = [None]
    for keep in (False, True):
        key[0] = keep
        variant = replace(c41.variant, basin_descents=1, basin_keep_atm=keep)
        minimize(
            ellipse,
            np.zeros(2),
            300 * 2,
            seed=0,
            config=replace(c41, variant=variant),
            bounds=(-5, 5),
        )
    # ATM's hand-over share (60 x D here) grows by exactly the descent's evaluations
    assert limits[True] > limits[False] == 120
