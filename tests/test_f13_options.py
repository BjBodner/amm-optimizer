"""The f13 toy's options (``docs/research/toys/f13_cells.md``): the ridge reading of the
prefix's SLSQP descent and the continued rugged try; off by default, so the presets
are unchanged."""

import numpy as np
import pytest

from amm_optimizer import api, cobyqa_phase, es_phase, presets, qn_phase
from amm_optimizer.bounds import Box
from amm_optimizer.objective import CountedObjective

BOX5 = Box.make((-5.0, 5.0), 5)


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def test_the_ridge_reading():
    ridge = [(0.5, 1)] + [(-2.5, 2), (-2.0, 3), (-2.8, 3), (-2.2, 2)]
    assert qn_phase.ridge_reading(ridge)
    assert not qn_phase.ridge_reading([(0.5, 1)])  # the first record is left out
    assert not qn_phase.ridge_reading([(0.5, 1)] + [(0.9, 1)] * 4)  # smooth
    assert not qn_phase.ridge_reading([(0.5, 1)] + [(-9.0, 2)] * 4)  # overshoots more
    assert not qn_phase.ridge_reading([(0.5, 1)] + [(-2.5, 6)] * 4)  # rugged search


@pytest.mark.parametrize("switch", [False, True])
def test_a_ridge_hands_the_prefix_to_cobyqa(monkeypatch, switch):
    """A ridge reading stops the descent and COBYQA spends the rest, also when
    ``small_switch`` reads the same descent first and keeps it."""
    monkeypatch.setattr(qn_phase, "ridge_reading", lambda records: True)
    handed = []
    real = cobyqa_phase.run_cobyqa_restarts

    def spy(*args, **kwargs):
        handed.append(args[4])
        return real(*args, **kwargs)

    monkeypatch.setattr(cobyqa_phase, "run_cobyqa_restarts", spy)
    keep = qn_phase.SmallSwitch(trials=0.0, parallel_cos=0.0, rest=False)
    objective = CountedObjective(rosenbrock, 600)
    out = es_phase.run_es_phase(
        objective,
        np.random.RandomState(0),
        np.full(5, 3.0),
        BOX5,
        600,
        6,
        "slsqplq",
        ridge_after=2,
        **({"small_switch": keep} if switch else {}),
    )
    assert len(handed) == 1 and handed[0] > 0
    assert objective.calls <= 600
    assert out.cost <= rosenbrock(np.full(5, 3.0))


@pytest.mark.parametrize(
    ("after", "max_per_dim", "budget_per_dim", "on"),
    [
        (0, 30.0, 10, False),
        (7, 30.0, 30, True),
        (7, 30.0, 31, False),
        (7, 0.0, 100, True),
    ],
)
def test_the_ridge_reading_takes_only_its_envelope(
    after, max_per_dim, budget_per_dim, on
):
    variant = presets.with_changes(
        "c44", prefix_ridge_after=after, prefix_ridge_max_per_dim=max_per_dim
    ).variant
    kw = api._prefix_ridge_kw(variant, 20, budget_per_dim * 20)
    assert bool(kw) is on


@pytest.mark.parametrize("gate", [None, False, True])
def test_the_rugged_try_goes_on_past_its_cap(gate):
    objective = CountedObjective(rosenbrock, 10_000)
    x = np.full(5, 4.0)
    out = qn_phase.run_scaled_bfgs(
        objective,
        np.random.RandomState(0),
        x,
        rosenbrock(x),
        BOX5,
        12,
        max_descents=1,
        extend_to=300,
        extend_gate=None if gate is None else (lambda x_e, cost_e: gate),
    )
    assert (out.used > 12) is (gate is not False)
    assert out.used <= 300


def test_the_options_are_off_in_the_presets_and_validated():
    for name in ("c43", "c44"):
        variant = presets.get(name).variant
        assert variant.rugged_try_rest == "none"
        assert variant.prefix_ridge_after == 0
        assert variant.prefix_nan_reading == "rugged"
    with pytest.raises(ValueError, match="rugged_try_rest"):
        presets.with_changes("c44", rugged_try_rest="always")
    with pytest.raises(ValueError, match="prefix_ridge"):
        presets.with_changes("c44", prefix_ridge_radius=0.0)
