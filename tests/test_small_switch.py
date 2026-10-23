"""``small_switch`` (the small10 toy): the descent reader, COBYQA from a known point and
the route's envelope; off by default, so the presets are unchanged."""

from dataclasses import replace

import numpy as np
import pytest

from amm_optimizer import cobyqa_phase, es_phase, minimize, presets, qn_phase
from amm_optimizer.bounds import Box
from amm_optimizer.objective import CountedObjective

BOX5 = Box.make((-5.0, 5.0), 5)


def counted(fun, allowed):
    calls = []

    def logged(x):
        calls.append(np.array(x))
        return fun(x)

    return CountedObjective(logged, allowed), calls


def discus(x):
    """Rank-1 dominated: consecutive SLSQP gradients stay (anti)parallel."""
    x = np.asarray(x, dtype=float)
    r = np.ones(len(x)) / np.sqrt(len(x))
    return float(1e6 * (r @ (x - 1.0)) ** 2 + np.sum((x - 1.0) ** 2))


def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def read(fun, rule, budget=50, start=None):
    objective, _ = counted(fun, 10_000)
    reader = qn_phase.SwitchReader(qn_phase.QN_FD_STEP, 5, rule)
    x = np.full(5, 3.0) if start is None else start
    qn_phase.run_slsqp_restarts(
        objective,
        np.random.RandomState(0),
        x,
        fun(x),
        BOX5,
        budget,
        max_descents=1,
        fd_step=qn_phase.QN_FD_STEP,
        reader=reader,
    )
    return reader


@pytest.mark.snapshot
def test_the_reader_keeps_a_parallel_gradient_descent():
    reader = read(discus, qn_phase.SmallSwitch())
    assert reader.stopped_at == 0
    assert len(reader.grads) >= 2
    a, b = reader.grads[1], reader.grads[0]
    assert abs(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) >= 0.99


def test_the_reader_stops_where_the_gradients_turn():
    reader = read(rosenbrock, qn_phase.SmallSwitch(), start=np.full(5, -1.5))
    assert reader.stopped_at == 2 and len(reader.trials) == 2


def test_the_reader_stops_a_struggling_first_line_search():
    reader = read(rastrigin, qn_phase.SmallSwitch(trials=2.0, parallel_cos=0.0))
    assert reader.stopped_at == 1 and reader.trials[0] >= 2


def test_both_checkpoints_off_never_stop():
    rule = qn_phase.SmallSwitch(trials=0.0, parallel_cos=0.0)
    assert read(rosenbrock, rule, start=np.full(5, -1.5)).stopped_at == 0


def test_cobyqa_from_a_known_point_does_not_evaluate_it_again():
    x = np.zeros(5)
    objective, calls = counted(rosenbrock, 10_000)
    out = cobyqa_phase.run_cobyqa_from(
        objective, np.random.RandomState(0), x, rosenbrock(x), BOX5, 40
    )
    assert out.used == len(calls) <= 40
    assert not any(np.array_equal(c, x) for c in calls)
    assert out.cost <= rosenbrock(x)


@pytest.mark.parametrize(
    ("dim", "budget_per_dim", "on"),
    [(5, 10, True), (20, 10, True), (5, 11, False), (40, 10, False), (5, 30, False)],
)
def test_the_switch_takes_only_its_envelope(monkeypatch, dim, budget_per_dim, on):
    seen = []
    real = es_phase.run_es_phase

    def spy(*args, **kwargs):
        seen.append(kwargs.get("small_switch"))
        return real(*args, **kwargs)

    monkeypatch.setattr("amm_optimizer.api.run_es_phase", spy)
    config = presets.with_changes("c43", small_switch=True)
    minimize(
        rosenbrock,
        np.full(dim, 2.0),
        budget_per_dim * dim,
        seed=0,
        config=config,
        bounds=(-5, 5),
    )
    assert (seen[0] is not None) is on


def test_the_switch_is_off_in_c43_and_changes_its_runs_when_on():
    c43 = presets.get("c43")
    assert c43.variant.small_switch is False
    on = replace(
        c43,
        variant=replace(c43.variant, small_switch=True, small_switch_trials=2.0),
    )
    streams = []
    for config in (c43, on):
        seen: list[np.ndarray] = []

        def fun(x, seen=seen):
            seen.append(np.array(x))
            return rastrigin(np.asarray(x) - 1.3)

        out = minimize(fun, np.full(5, 2.0), 50, seed=1, config=config, bounds=(-5, 5))
        assert out.nfev == len(seen) <= 50
        streams.append(np.array(seen))
    assert streams[0].shape != streams[1].shape or not np.array_equal(*streams)


def test_the_rule_is_validated():
    with pytest.raises(ValueError, match="small_switch"):
        presets.with_changes("c43", small_switch_cos=1.5)
