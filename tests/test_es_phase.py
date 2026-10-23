"""``es_phase="prefix"`` (E21): a (1+1)-ES on the first share of the budget."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.bounds import Box
from amm_optimizer.config import Variant
from amm_optimizer.control import LOG_WIDTH_FLOOR, start_log_widths
from amm_optimizer.es_phase import run_es_phase
from amm_optimizer.objective import CountedObjective
from amm_optimizer.profiling import TimingProfiler

BOX = Box.make((-5.0, 5.0), 3)
ES = Variant(
    amplitude_control="success_rate_targeted",
    bounds="fold",
    es_phase="prefix",
    es_fraction=0.3,
)


def sphere(x):
    return float(np.sum((np.asarray(x) - 2) ** 2))


def run(fun, evaluations=400, init_points=4, x0=None, seed=0, limit=10_000):
    objective = CountedObjective(fun, limit)
    x0 = np.full(3, -3.0) if x0 is None else x0
    result = run_es_phase(
        objective, np.random.RandomState(seed), x0, BOX, evaluations, init_points
    )
    return result, objective


def test_the_es_converges_on_a_sphere_and_uses_its_share():
    result, objective = run(sphere, evaluations=600)
    assert objective.calls == 600
    assert result.cost < 1e-6
    assert result.cost == pytest.approx(sphere(result.x))
    assert 0 < result.sigma < 0.1  # the step size shrank with the distance


def test_every_candidate_stays_inside_the_box_and_a_corner_is_reached_exactly():
    seen = []

    def slope(x):
        seen.append(np.array(x))
        return float(-np.sum(x))  # the optimum is the corner (5, 5, 5)

    result, _ = run(slope, evaluations=300)
    assert np.abs(np.array(seen[1:])).max() <= 5.0  # the start itself may be outside
    np.testing.assert_array_equal(result.x, [5.0, 5.0, 5.0])


def test_nan_costs_never_replace_the_incumbent():
    def holes(x):
        return float("nan") if x[0] > 0 else sphere(x)

    result, _ = run(holes, evaluations=300)
    assert np.isfinite(result.cost) and result.x[0] <= 0


def test_a_share_of_one_evaluates_only_the_start():
    result, objective = run(sphere, evaluations=1)
    assert objective.calls == 1
    np.testing.assert_array_equal(result.x, np.full(3, -3.0))


def test_no_design_points_means_the_es_starts_at_the_given_point():
    result, objective = run(sphere, evaluations=5, init_points=0)
    assert objective.calls == 5


def test_a_converged_step_size_ends_the_phase_early():
    x0 = np.zeros(3)

    def needle(x):
        return 0.0 if np.array_equal(x, x0) else 1.0  # nothing ever improves

    _, objective = run(needle, evaluations=10**6, x0=x0, init_points=0, limit=10**6)
    assert objective.calls < 2000


def test_the_budget_can_end_the_phase_before_its_share():
    result, objective = run(sphere, evaluations=500, limit=40)
    assert objective.calls == 40
    assert np.isfinite(result.cost)


def test_evaluate_point_charges_and_respects_the_budget():
    objective = CountedObjective(sphere, 2)
    assert objective.evaluate_point(np.zeros(3)) == sphere(np.zeros(3))
    assert objective.evaluate_point(np.ones(3)) == sphere(np.ones(3))
    assert objective.evaluate_point(np.ones(3)) is None
    assert objective.calls == 2


def test_evaluate_point_is_timed_when_profiling():
    profiler = TimingProfiler()
    objective = CountedObjective(sphere, 5, profiler=profiler)
    objective.evaluate_point(np.zeros(3))
    assert profiler.report().blocks["objective"].calls == 1

    def boom(x):
        raise RuntimeError

    failing = CountedObjective(boom, 5, profiler=profiler)
    with pytest.raises(RuntimeError):
        failing.evaluate_point(np.zeros(3))
    assert failing.calls == 1
    assert profiler.report().blocks["objective"].calls == 2


def test_the_starting_widths_scale_with_the_es_step_size():
    base = start_log_widths(ES)
    scaled = start_log_widths(ES, 0.01)
    np.testing.assert_allclose(np.array(scaled) - np.array(base), np.log(0.01))
    assert min(start_log_widths(ES, 1e-300)) == LOG_WIDTH_FLOOR
    assert max(start_log_widths(ES, 1e6)) == pytest.approx(np.log(ES.sr_max_width))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"es_phase": "suffix"},
        {"es_fraction": 0.0},
        {"es_fraction": 1.0},
        {"es_init_factor": -1.0},
        {"es_max_budget_per_dim": 0.0},
        {"es_small_dim": -1},
        {"es_small_max_budget_per_dim": 0.0},
    ],
)
def test_es_parameters_are_validated(kwargs):
    with pytest.raises(ValueError, match="es_"):
        Variant(**kwargs)


def test_the_es_needs_the_controller_and_the_box():
    with pytest.raises(ValueError, match="amplitude_control"):
        Variant(es_phase="prefix")
    assert Variant(es_phase="prefix", amplitude_control="success_rate").needs_box
    with pytest.raises(ValueError, match="bounds"):
        atm.minimize(
            sphere,
            np.zeros(3),
            200,
            seed=0,
            config=atm.ATMConfig(
                variant=Variant(es_phase="prefix", amplitude_control="success_rate")
            ),
        )


def test_a_run_spends_the_budget_and_beats_atm_alone_on_the_sphere():
    x0 = np.full(5, -3.0)
    budget = 1500
    with_es = atm.minimize(
        sphere, x0, budget, seed=3, config=atm.ATMConfig(variant=ES), bounds=(-5, 5)
    )
    plain = Variant(
        amplitude_control="success_rate_targeted",
        bounds="fold",
        pool_rule="budget",
        pool_min=12,
    )
    without = atm.minimize(
        sphere, x0, budget, seed=3, config=atm.ATMConfig(variant=plain), bounds=(-5, 5)
    )
    assert with_es.nfev == budget
    assert with_es.fun < without.fun


def test_a_tiny_budget_runs_without_error():
    result = atm.minimize(
        sphere, np.zeros(2), 3, seed=0, config=atm.ATMConfig(variant=ES), bounds=(-5, 5)
    )
    assert result.nfev == 3


def test_the_es_is_skipped_above_the_budget_limit():
    limited = Variant(**{**ES.to_dict(), "es_max_budget_per_dim": 50.0})
    x0 = np.full(3, -3.0)

    def first_points(variant, budget):
        points = []

        def spy(x):
            points.append(np.array(x))
            return sphere(x)

        atm.minimize(
            spy,
            x0,
            budget,
            seed=0,
            config=atm.ATMConfig(variant=variant),
            bounds=(-5, 5),
        )
        return points

    inside = first_points(limited, 120)  # 40 x D: the ES runs
    above = first_points(limited, 600)  # 200 x D: ATM alone
    plain = first_points(Variant(**{**ES.to_dict(), "es_phase": "none"}), 600)
    assert len(inside) == 120
    assert all(np.array_equal(a, b) for a, b in zip(above, plain, strict=True))
    assert not np.array_equal(
        inside[1], first_points(Variant(**{**ES.to_dict(), "es_phase": "none"}), 120)[1]
    )


def test_small_dimensions_use_their_own_budget_limit():
    base = Variant(**{**ES.to_dict(), "es_max_budget_per_dim": 1000.0})
    small = Variant(
        **{**base.to_dict(), "es_small_dim": 3, "es_small_max_budget_per_dim": 30.0}
    )

    def points(variant, dim, budget):
        seen = []

        def spy(x):
            seen.append(np.array(x))
            return sphere(x)

        atm.minimize(
            spy,
            np.full(dim, -3.0),
            budget,
            seed=0,
            config=atm.ATMConfig(variant=variant),
            bounds=(-5, 5),
        )
        return seen

    def same(a, b):
        return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))

    no_es = Variant(**{**ES.to_dict(), "es_phase": "none"})
    # 3-D, 100 x D: above the small-dimension limit, so ATM alone ...
    assert same(points(small, 3, 300), points(no_es, 3, 300))
    # ... 3-D, 20 x D: inside it, so the ES runs ...
    assert same(points(small, 3, 60), points(base, 3, 60))
    # ... and 4-D, 100 x D is not a small dimension: the ES runs
    assert same(points(small, 4, 400), points(base, 4, 400))
    assert not same(points(small, 4, 400), points(no_es, 4, 400))


def test_the_design_follows_the_factor_and_has_at_least_four_points(monkeypatch):
    import amm_optimizer as atm
    import amm_optimizer.api

    seen = []
    real = atm.api.run_es_phase

    def spy(*args, **kwargs):  # init_points is the sixth argument
        seen.append(args[5])
        return real(*args, **kwargs)

    monkeypatch.setattr(atm.api, "run_es_phase", spy)

    def design(factor, budget):
        variant = Variant(**{**ES.to_dict(), "es_init_factor": factor})
        atm.minimize(
            sphere,
            np.zeros(3),
            budget,
            seed=0,
            config=atm.ATMConfig(variant=variant),
            bounds=(-5, 5),
        )
        return seen[-1]

    assert design(5.0, 300) == 15  # factor times the dimension
    assert design(0.0, 300) == 4  # never below 4 points


def test_the_uniform_first_pool_is_skipped_after_the_es_but_not_without_it():
    uniform = Variant(**{**ES.to_dict(), "initial_design": "uniform"})
    x0 = np.full(3, -3.0)

    def points(variant, budget):
        seen = []

        def spy(x):
            seen.append(np.array(x))
            return sphere(x)

        atm.minimize(
            spy,
            x0,
            budget,
            seed=0,
            config=atm.ATMConfig(variant=variant),
            bounds=(-5, 5),
        )
        return np.array(seen)

    with_es = points(uniform, 150)
    # after the ES (45 evaluations) ATM's first iteration is not a uniform design:
    # its samples stay near the ES incumbent (sigma is small by then)
    after = with_es[46:70]
    assert np.abs(after - after[after.shape[0] // 2]).max() < 4.0
    far = Variant(**{**uniform.to_dict(), "es_max_budget_per_dim": 1.0})
    without_es = points(far, 150)  # ES skipped: the uniform design is used
    assert np.abs(without_es[1:20] - x0).max() > 3.0


def test_a_corner_optimum_is_reached_in_ten_dimensions_because_pinned_coordinates_stick(
    monkeypatch,
):
    import amm_optimizer.es_phase as es_phase

    box = Box.make((-5.0, 5.0), 10)

    def slope(x):
        return float(-np.sum(x))

    def corner_reached(seed):
        objective = CountedObjective(slope, 10_000)
        result = es_phase.run_es_phase(
            objective, np.random.RandomState(seed), np.zeros(10), box, 1500, 4
        )
        return bool(np.array_equal(result.x, np.full(10, 5.0)))

    assert all(corner_reached(seed) for seed in range(5))
    monkeypatch.setattr(es_phase, "PIN_STICK", 0.0)
    assert not all(corner_reached(seed) for seed in range(5))


@pytest.mark.parametrize(
    ("budget", "least", "whole"),
    [(150, 0.0, True), (155, 0.0, False), (500, 0.0, False), (50, 10.0, False)]
    + [(150, 10.0, True)],
)
def test_the_cycle38_prefix_takes_the_whole_small_budget(
    monkeypatch, budget, least, whole
):
    import amm_optimizer as atm
    import amm_optimizer.api
    from amm_optimizer import presets

    seen = []
    real = atm.api.run_es_phase

    def spy(*args, **kwargs):  # evaluations is the fifth argument
        seen.append(args[4])
        return real(*args, **kwargs)

    monkeypatch.setattr(atm.api, "run_es_phase", spy)
    config = presets.with_changes(
        "c34", prefix_full_per_dim=30.0, prefix_full_min_per_dim=least
    )
    atm.minimize(sphere, np.zeros(5), budget, seed=0, config=config, bounds=(-5, 5))
    assert seen == [budget if whole else int(0.9 * budget)]


@pytest.mark.parametrize(("budget", "passed"), [(150, False), (500, True)])
def test_preplat_runs_only_above_30_x_dim(monkeypatch, budget, passed):
    import amm_optimizer as atm
    import amm_optimizer.api
    from amm_optimizer import presets

    seen = []
    real = atm.api.run_es_phase

    def spy(*args, **kwargs):
        seen.append(kwargs.get("plateau_pop", False))
        return real(*args, **kwargs)

    monkeypatch.setattr(atm.api, "run_es_phase", spy)
    config = presets.with_changes("c36", prefix_plateau_pop=True)
    atm.minimize(sphere, np.zeros(5), budget, seed=0, config=config, bounds=(-5, 5))
    assert seen == [passed]
