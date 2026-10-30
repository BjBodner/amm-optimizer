"""The restarted (1+1)-Cholesky-CMA-ES phase (variants ``es_kind``, ``cma_handoff``)."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.bounds import Box
from amm_optimizer.cma_phase import cholesky_update, run_cma
from amm_optimizer.config import ATMConfig, Variant
from amm_optimizer.objective import CountedObjective

BOX = (-5.0, 5.0)


def test_cholesky_update_is_the_rank_one_update():
    rng = np.random.RandomState(0)
    a = np.linalg.cholesky(np.cov(rng.randn(5, 40)))
    a_inv = np.linalg.inv(a)
    v = rng.randn(5)
    new, new_inv = cholesky_update(a, a_inv, v, 0.9, 0.1)
    assert new @ new.T == pytest.approx(0.9 * a @ a.T + 0.1 * np.outer(v, v))
    assert new_inv @ new == pytest.approx(np.eye(5), abs=1e-12)


def test_recent_points_are_the_list_windows_bit_for_bit():
    """The ring buffer returns what ``np.array(seen[-k:])`` of the old lists did."""
    from amm_optimizer.cma_phase import _Recent

    rng = np.random.RandomState(1)
    dim, keep = 3, 7
    recent = _Recent(dim, keep)
    xs, fs = [], []
    for step in range(60):
        if step in (0, 23):  # a run starts (or restarts)
            x0, f0 = rng.randn(dim), float(rng.randn())
            recent.reset(x0, f0)
            xs, fs = [x0.copy()], [f0]
        else:
            x, f = rng.randn(dim), float(rng.choice([rng.randn(), np.inf, -0.0]))
            recent.append(x, f)
            xs.append(x)
            fs.append(f)
        assert recent.count == len(fs)
        for k in range(1, keep + 1):
            got_x, got_f = recent.latest(k)
            assert got_x.tobytes() == np.array(xs[-k:]).tobytes()
            assert got_f.tobytes() == np.array(fs[-k:]).tobytes()
            assert got_x.shape == np.array(xs[-k:]).shape


def rotated_ellipsoid(dim, seed=0):
    q, _ = np.linalg.qr(np.random.RandomState(seed).randn(dim, dim))
    scales = 1e6 ** (np.arange(dim) / (dim - 1))
    return lambda x: float(np.sum(scales * (q @ x) ** 2))


def test_learns_a_rotated_ellipsoid_inside_the_box():
    fun = rotated_ellipsoid(8)
    objective = CountedObjective(fun, 8000)
    box = Box.make(BOX, 8)
    x0 = np.full(8, 3.0)
    result = run_cma(objective, np.random.RandomState(1), x0, fun(x0), 2.0, box, 8000)
    assert result.cost < 1e-10
    assert objective.calls <= 8000
    assert (np.abs(result.x) <= 5).all()


def test_restarts_keep_the_best_point_and_the_budget():
    seen = []

    def rastrigin(x):
        seen.append(float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))))
        return seen[-1]

    objective = CountedObjective(rastrigin, 3000)
    x0 = np.full(3, 4.0)
    cost = rastrigin(x0)
    result = run_cma(
        objective, np.random.RandomState(2), x0, cost, 2.0, Box.make(BOX, 3), 3000
    )
    assert objective.calls == 3000
    assert result.cost == min(seen)


def handoff(fraction):
    return ATMConfig(
        variant=Variant(
            amplitude_control="success_rate_targeted",
            bounds="fold",
            pool_rule="budget",
            pool_min=12,
            reevaluate_incumbent=False,
            cma_handoff=fraction,
        )
    )


def test_handoff_spends_the_budget_and_keeps_the_best():
    fun = rotated_ellipsoid(6)
    seen = []

    def counted(x):
        seen.append(fun(x))
        return seen[-1]

    result = atm.api.minimize(
        counted, np.full(6, 2.0), 6000, seed=3, config=handoff(0.3), bounds=BOX
    )
    assert result.nfev == len(seen) == 6000
    assert result.fun == min(seen)
    alone = atm.api.minimize(
        fun, np.full(6, 2.0), 6000, seed=3, config=handoff(0.0), bounds=BOX
    )
    assert result.fun < 1e-3 * alone.fun


def test_handoff_needs_a_success_rate_controller_and_a_box():
    with pytest.raises(ValueError, match="success-rate"):
        Variant(cma_handoff=0.3)
    with pytest.raises(ValueError, match="bounds"):
        atm.api.minimize(lambda x: 0.0, np.zeros(2), 100, seed=0, config=handoff(0.3))


def test_without_restarts_it_returns_at_the_first_stagnation():
    objective = CountedObjective(lambda x: 1.0, 5000)
    result = run_cma(
        objective,
        np.random.RandomState(0),
        np.zeros(2),
        1.0,
        1.0,
        Box.make(BOX, 2),
        5000,
        restarts=False,
    )
    assert result.stalled
    assert result.used == objective.calls < 5000


# research-only options (deprecated in 0.6.0)
def test_quadratic_step_finds_the_minimum_of_a_quadratic():
    from amm_optimizer.cma_phase import quadratic_step

    rng = np.random.RandomState(7)
    q, _ = np.linalg.qr(rng.randn(4, 4))
    centre = np.array([0.3, -0.2, 0.1, 0.4])
    scales = np.array([1.0, 10.0, 100.0, 1000.0])

    def fun(x):
        return float(np.sum(scales * (q @ (x - centre)) ** 2))

    points = rng.randn(40, 4) * 0.5
    values = np.array([fun(p) for p in points])
    step = quadratic_step(points, values, np.zeros(4), 1.0, np.eye(4), np.eye(4))
    assert step is not None
    assert step == pytest.approx(centre, abs=1e-8)


def test_quadratic_step_declines_a_saddle():
    from amm_optimizer.cma_phase import quadratic_step

    rng = np.random.RandomState(8)
    points = rng.randn(30, 2)
    values = points[:, 0] ** 2 - points[:, 1] ** 2
    assert (
        quadratic_step(points, values, np.zeros(2), 1.0, np.eye(2), np.eye(2)) is None
    )


def test_model_steps_speed_up_a_smooth_function():
    fun = rotated_ellipsoid(5)
    box = Box.make(BOX, 5)
    x0 = np.full(5, 3.0)
    reached = {}
    for model in (False, True):
        seen = []

        def counted(x, seen=seen):
            seen.append(fun(x))
            return seen[-1]

        objective = CountedObjective(counted, 3000)
        run_cma(
            objective,
            np.random.RandomState(9),
            x0,
            fun(x0),
            2.0,
            box,
            3000,
            model=model,
        )
        best = np.minimum.accumulate(seen)
        reached[model] = int(np.argmax(best < 1e-8)) if (best < 1e-8).any() else 3000
    assert reached[True] < 0.5 * reached[False]
