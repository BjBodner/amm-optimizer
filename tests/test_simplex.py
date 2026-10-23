"""The simplex search, the late refinement and the local-method options (E12, E13)."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.bounds import Box
from amm_optimizer.config import Variant
from amm_optimizer.objective import CountedObjective
from amm_optimizer.simplex import run_nelder_mead

BOX2 = Box.make((-5.0, 5.0), 2)
BOX3 = Box.make((-5.0, 5.0), 3)


def sphere(x):
    return float(np.sum((np.asarray(x) - 1.0) ** 2))


def rosenbrock(x):
    x = np.asarray(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def run(fun, box, evaluations, x0=None, edge=0.5, limit=10**6, seed=0):
    dim = len(box.lower)
    x0 = np.full(dim, -3.0) if x0 is None else x0
    objective = CountedObjective(fun, limit)
    result = run_nelder_mead(
        objective,
        np.random.RandomState(seed),
        x0,
        float(fun(x0)),
        edge,
        box,
        evaluations,
    )
    return result, objective


def test_the_simplex_converges_on_a_sphere_and_on_rosenbrock():
    result, objective = run(sphere, BOX3, 400)
    assert result.cost < 1e-12 and objective.calls <= 400
    result, objective = run(rosenbrock, BOX2, 500, x0=np.array([-3.0, 3.0]))
    assert result.cost < 1e-8 and objective.calls <= 500


def test_the_budget_is_never_exceeded_whatever_it_is():
    for budget in range(1, 60):
        for fun, box in ((sphere, BOX3), (rosenbrock, BOX2)):
            result, objective = run(fun, box, budget, x0=np.full(len(box.lower), -3.0))
            assert objective.calls <= budget
            assert result.cost <= fun(np.full(len(box.lower), -3.0))


def test_a_limit_of_the_objective_ends_the_search_at_any_point():
    for limit in range(1, 40):
        result, objective = run(
            rosenbrock, BOX2, 10**6, limit=limit, x0=np.array([-3.0, 3.0])
        )
        assert objective.calls == limit and np.isfinite(result.cost)


def test_every_candidate_is_inside_the_box_and_a_corner_optimum_is_reached():
    seen = []

    def slope(x):
        seen.append(np.array(x))
        return float(-np.sum(x))

    result, _ = run(slope, BOX3, 600, x0=np.zeros(3))
    assert np.abs(np.array(seen[1:])).max() <= 5.0
    np.testing.assert_array_equal(result.x, [5.0, 5.0, 5.0])


def test_the_edge_flips_inward_at_the_upper_wall():
    seen = []

    def spy(x):
        seen.append(np.array(x))
        return sphere(x)

    run(spy, BOX2, 3, x0=np.array([5.0, 5.0]))
    # seen[0] is the start itself; the first two vertices step inward
    assert seen[1][0] < 5.0 and seen[2][1] < 5.0


def test_failed_values_never_become_the_best_point():
    def holes(x):
        x = np.asarray(x)
        return float("nan") if x[0] > 0 else (float("inf") if x[1] > 4 else sphere(x))

    result, _ = run(holes, BOX2, 500)
    assert np.isfinite(result.cost)
    assert result.x[0] <= 0


def test_a_search_that_cannot_improve_restarts_and_then_gives_up():
    x0 = np.zeros(2)

    def needle(x):
        return 0.0 if np.array_equal(x, x0) else 1.0

    result, objective = run(needle, BOX2, 10**6, x0=x0, limit=10**6)
    assert result.cost == 0.0
    assert objective.calls < 200_000  # the edge shrinks to its floor and it stops


def test_every_third_restart_starts_from_a_random_point():
    seen = []

    def spy(x):
        seen.append(np.array(x))
        return 1.0 + float(np.sum(np.abs(np.asarray(x))))

    run(spy, BOX2, 3000, x0=np.zeros(2), edge=1e-9)
    # with a tiny first edge the simplex collapses at once; later restarts draw
    # points far from the origin
    assert max(np.abs(p).max() for p in seen) > 1.0


def test_a_random_restart_that_the_budget_cannot_pay_ends_the_search():
    needle = lambda x: 1.0  # noqa: E731
    for limit in range(1, 80):
        _, objective = run(needle, BOX2, 10**6, limit=limit, edge=1e-9)
        assert objective.calls <= limit


def test_one_dimension_works():
    box = Box.make((-5.0, 5.0), 1)
    result, _ = run(
        lambda x: float((np.asarray(x)[0] - 2) ** 2), box, 200, x0=np.array([-4.0])
    )
    assert result.cost < 1e-12


def test_simplex_parameters_are_validated():
    base = {"amplitude_control": "success_rate"}
    for kwargs, match in [
        ({"local_method": "powell"}, "local_method"),
        ({"simplex_max_dim": 0}, "simplex_max_dim"),
        ({"nm_refine_max_dim": 0}, "nm_refine_max_dim"),
        ({"refine_fraction": 1.0}, "refine_fraction"),
        ({"refine_fraction": -0.1}, "refine_fraction"),
        ({"refine_min_budget_per_dim": -1.0}, "refine_min"),
        ({"refine_max_dim": 0}, "refine_max_dim"),
    ]:
        with pytest.raises(ValueError, match=match):
            Variant(**base, **kwargs)
    with pytest.raises(ValueError, match="amplitude_control"):
        Variant(refine_fraction=0.3)
    assert Variant(amplitude_control="success_rate", refine_fraction=0.3).needs_box


def test_hold_back_and_release_move_the_end_of_the_budget():
    objective = CountedObjective(sphere, 10)
    objective.hold_back(4)
    assert objective.remaining == 6
    for _ in range(6):
        assert objective.evaluate_point(np.zeros(2)) is not None
    assert objective.evaluate_point(np.zeros(2)) is None
    objective.release()
    assert objective.remaining == 4
    assert objective.evaluate_point(np.zeros(2)) is not None


def variant(**extra):
    base = dict(
        amplitude_control="success_rate_targeted",
        bounds="fold",
        pool_rule="budget",
        pool_min=12,
        reevaluate_incumbent=False,
    )
    return atm.ATMConfig(variant=Variant(**{**base, **extra}))


def test_the_late_refinement_spends_the_last_share_on_the_local_search():
    points = []

    def spy(x):
        points.append(np.array(x))
        return rosenbrock(x)

    config = variant(refine_fraction=0.5, local_method="nelder_mead")
    result = atm.minimize(
        spy, np.array([-3.0, 3.0]), 1000, seed=0, config=config, bounds=(-5, 5)
    )
    assert result.nfev == len(points) <= 1000
    plain = atm.minimize(
        rosenbrock,
        np.array([-3.0, 3.0]),
        1000,
        seed=0,
        config=variant(),
        bounds=(-5, 5),
    )
    assert result.fun < plain.fun


def test_the_refinement_uses_the_es_above_the_simplex_dimension():
    config = variant(
        refine_fraction=0.4, local_method="nelder_mead", nm_refine_max_dim=2
    )
    result = atm.minimize(
        sphere, np.full(3, -3.0), 1500, seed=0, config=config, bounds=(-5, 5)
    )
    assert result.nfev == 1500
    assert result.fun < 1e-6


def test_the_refinement_is_skipped_below_its_minimum_budget():
    config = variant(refine_fraction=0.5, refine_min_budget_per_dim=1000.0)
    plain = variant()
    a = atm.minimize(
        sphere, np.full(3, -3.0), 300, seed=1, config=config, bounds=(-5, 5)
    )
    b = atm.minimize(
        sphere, np.full(3, -3.0), 300, seed=1, config=plain, bounds=(-5, 5)
    )
    np.testing.assert_array_equal(a.x, b.x)


def test_the_refinement_is_skipped_above_its_dimension_limit():
    config = variant(refine_fraction=0.5, refine_max_dim=2)
    plain = variant()
    for dim, same in ((3, True), (2, False)):
        a = atm.minimize(
            sphere, np.full(dim, -3.0), 300, seed=1, config=config, bounds=(-5, 5)
        )
        b = atm.minimize(
            sphere, np.full(dim, -3.0), 300, seed=1, config=plain, bounds=(-5, 5)
        )
        assert np.array_equal(a.x, b.x) is same


def test_the_simplex_prefix_beats_the_es_prefix_in_two_dimensions():
    kwargs = dict(es_phase="prefix", es_fraction=0.9, es_max_budget_per_dim=100)
    costs = {}
    for method in ("es", "nelder_mead"):
        runs = []
        for seed in range(4):
            rng = np.random.RandomState(seed)
            result = atm.minimize(
                rosenbrock,
                rng.uniform(-4, 4, 2),
                200,
                rng=rng,
                config=variant(local_method=method, **kwargs),
                bounds=(-5, 5),
            )
            runs.append(np.log10(max(result.fun, 1e-300)))
        costs[method] = np.median(runs)
    assert costs["nelder_mead"] < costs["es"] - 3
