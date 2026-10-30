"""``atm.chain_phase``: the additivity readings, the decoupling basis and the
decoupled coordinate search, and the ``chain_route`` / ``sep_route`` options."""

import numpy as np
import pytest

from amm_optimizer import api, chain_phase, presets
from amm_optimizer.api import minimize
from amm_optimizer.bounds import Box
from amm_optimizer.objective import CountedObjective
from amm_optimizer.step_phase import run_step_phase

BOX = Box.make((-5.0, 5.0), 6)
# A unit lower bidiagonal coupling, as in bbob f20: w_{i+1} = x_{i+1} + 0.25 x_i
M = np.eye(6) + np.diag(np.full(5, 0.25), -1)


def separable(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - 1.0) ** 2 + 2.0 * np.sin(3.0 * x)))


def chain(x, coupling=M):
    """Rastrigin-like and separable in w = M x: f20's kind of chain."""
    w = coupling @ (np.asarray(x, dtype=float) - 0.5)
    return float(np.sum(w**2 + 3.0 * (1.0 - np.cos(2.0 * np.pi * w))))


def upper_chain(x):
    """The same chain coupled the other way (M unit upper bidiagonal)."""
    return chain(x, M.T)


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rotated(x):
    q = np.linalg.qr(np.random.RandomState(4).randn(6, 6))[0]
    y = q @ np.asarray(x, dtype=float)
    return float(np.sum(10.0 ** np.arange(6) * y**2))


def counted(fun, allowed):
    calls = []

    def logged(x):
        calls.append(np.array(x))
        return fun(x)

    return CountedObjective(logged, allowed), calls


@pytest.mark.parametrize(
    ("fun", "reading"),
    [
        (separable, "separable"),
        (chain, "chain"),
        (rosenbrock, "chain"),
        (rotated, "nonsep"),
    ],
)
def test_the_pair_test_reads_the_coupling_in_3d_evaluations(fun, reading):
    objective, calls = counted(fun, 10_000)
    x0 = np.full(6, 0.3)
    read = chain_phase.pair_test(objective, np.random.RandomState(0), x0, BOX)
    assert read is not None and read[0] == reading
    assert len(calls) == 3 * 6
    assert read[2] == min(fun(c) for c in calls) and fun(read[1]) == read[2]


def test_the_pair_test_returns_none_when_the_objective_runs_out():
    objective, calls = counted(chain, 5)
    assert (
        chain_phase.pair_test(objective, np.random.RandomState(0), np.zeros(6), BOX)
        is None
    )
    assert len(calls) == 5


@pytest.mark.parametrize(
    ("fun", "linear", "order"),
    [(chain, True, "lower"), (upper_chain, True, "upper"), (rosenbrock, False, None)],
)
def test_the_band_check_tells_a_linear_chain_from_rosenbrock(fun, linear, order):
    objective, calls = counted(fun, 10_000)
    got = chain_phase.band_consistency(
        objective, np.random.RandomState(1), np.full(6, 0.3), BOX
    )
    assert got is not None and len(calls) == 6 * 6
    assert (got[0] < chain_phase.BAND_TOL) is linear
    if linear:
        assert got[1] == order


@pytest.mark.parametrize("coupling", [M, M.T])
def test_the_basis_undoes_a_triangular_coupling_in_its_orientation(coupling):
    order = "lower" if coupling is M else "upper"
    d = np.diag(np.full(6, 120.0))  # equal curvatures: sparsity cannot tell them apart
    basis, positive = chain_phase.decoupling_basis(coupling.T @ d @ coupling, order)
    assert positive
    assert np.allclose(coupling @ basis, np.eye(6), atol=1e-9)


@pytest.mark.parametrize("evaluations", [30, 400, 3_000])
def test_coord_hess_spends_at_most_its_budget(evaluations):
    objective, calls = counted(chain, 10_000)
    result = chain_phase.run_coord_hess(
        objective, np.random.RandomState(0), np.full(6, 3.0), BOX, evaluations
    )
    assert result.used == len(calls) == evaluations
    assert result.cost == min(chain(c) for c in calls)


@pytest.mark.parametrize(("fun", "order"), [(chain, "lower"), (upper_chain, "upper")])
def test_coord_hess_solves_a_multimodal_chain(fun, order):
    objective, _ = counted(fun, 10_000)
    result = chain_phase.run_coord_hess(
        objective, np.random.RandomState(0), np.full(6, 3.0), BOX, 300 * 6, order
    )
    assert result.cost < 1e-8


def test_coord_hess_stops_when_the_objective_runs_out():
    objective, calls = counted(chain, 50)
    result = chain_phase.run_coord_hess(
        objective, np.random.RandomState(0), np.full(6, 3.0), BOX, 3_000
    )
    assert result.used == len(calls) == 50


def test_step_with_its_check_off_does_not_stop_on_a_rotated_function():
    objective, _ = counted(rotated, 10_000)
    x = np.full(6, 2.0)
    on = run_step_phase(objective, x, rotated(x), BOX, 3_000, parabolic=True)
    objective, _ = counted(rotated, 10_000)
    off = run_step_phase(
        objective, x, rotated(x), BOX, 3_000, parabolic=True, check_tol=float("inf")
    )
    assert on.stalled and not off.stalled and off.used > on.used


def _config(**changes):
    return presets.with_changes("c37", **changes)


@pytest.mark.parametrize("fun", [chain, separable, rotated])
def test_below_the_route_budgets_the_options_change_nothing(fun):
    x0, budget = np.full(6, 2.0), 90 * 6  # under sep_route_min_per_dim x dim
    on = minimize(
        fun,
        x0,
        budget,
        seed=3,
        config=_config(chain_route=True, sep_route=True),
        bounds=(-5, 5),
    )
    off = minimize(fun, x0, budget, seed=3, config=presets.get("c37"), bounds=(-5, 5))
    assert on.fun == off.fun and np.array_equal(on.x, off.x)


def test_the_chain_route_solves_a_linear_chain(monkeypatch):
    runs = []
    real = chain_phase.run_coord_hess

    def spy(objective, rng, x, box, evaluations, order=None):
        runs.append((evaluations, order))
        return real(objective, rng, x, box, evaluations, order)

    monkeypatch.setattr(chain_phase, "run_coord_hess", spy)
    budget = 300 * 6
    result = minimize(
        chain,
        np.full(6, 2.0),
        budget,
        seed=1,
        config=_config(chain_route=True),
        bounds=(-5, 5),
    )
    assert runs == [(budget - 3 * 6 - 6 * 6, "lower")]
    assert result.fun < 1e-6  # the global basin; the last digits need budget


def test_rosenbrock_fails_the_band_check_and_keeps_the_usual_path(monkeypatch):
    def never(*args, **kw):
        raise AssertionError("routed")

    monkeypatch.setattr(chain_phase, "run_coord_hess", never)
    result = minimize(
        rosenbrock,
        np.full(6, 2.0),
        300 * 6,
        seed=1,
        config=_config(chain_route=True),
        bounds=(-5, 5),
    )
    assert np.isfinite(result.fun)


def test_the_sep_route_solves_a_separable_multimodal_function():
    def rastrigin(x):
        x = np.asarray(x, dtype=float) - 0.7
        return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))

    budget = 100 * 6
    on = minimize(
        rastrigin,
        np.full(6, 2.0),
        budget,
        seed=1,
        config=_config(sep_route=True),
        bounds=(-5, 5),
    )
    assert on.fun < 1e-8


def test_the_routes_keep_the_budget():
    calls = []

    def logged(x):
        calls.append(1)
        return separable(x)

    minimize(
        logged,
        np.full(6, 2.0),
        600,
        seed=1,
        config=_config(sep_route=True, chain_route=True),
        bounds=(-5, 5),
    )
    assert len(calls) <= 600
    assert api is not None


# research-only options (deprecated in 0.6.0)
