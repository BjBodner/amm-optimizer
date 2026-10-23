"""``reevaluate_incumbent=False`` (E05): known costs, exact relation to legacy."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.config import Variant
from amm_optimizer.objective import CountedObjective
from amm_optimizer.profiling import TimingProfiler
from amm_optimizer.section import SectionRunner
from variant_cases import SHORT, run_case

SKIP = atm.ATMConfig(variant=Variant(reevaluate_incumbent=False))


class Recorder:
    def __init__(self):
        self.points = []

    def __call__(self, x):
        self.points.append(np.array(x))
        return float(np.sum(x))


def columns(n):
    return np.arange(2.0 * n).reshape(2, n)


def test_known_columns_are_neither_called_nor_charged():
    rec = Recorder()
    objective = CountedObjective(rec, 10)
    cost, complete = objective.evaluate_columns(columns(4), {0: 99.0, 2: 77.0})
    assert complete
    np.testing.assert_array_equal(cost, [99.0, 6.0, 77.0, 10.0])
    assert objective.calls == len(rec.points) == 2
    # Only 2 of the 10 evaluations were charged.
    cost, complete = objective.evaluate_columns(columns(9))
    assert not complete
    assert len(cost) == 8


def test_known_columns_before_the_limit_are_reached_when_the_budget_is_spent():
    rec = Recorder()
    objective = CountedObjective(rec, 1)
    cost, complete = objective.evaluate_columns(columns(4), {0: 5.0, 3: 6.0})
    assert not complete
    np.testing.assert_array_equal(cost, [5.0, 6.0])  # stops at the unaffordable one
    assert objective.calls == 1


def test_known_columns_with_profiling_are_timed_per_call():
    profiler = TimingProfiler()
    objective = CountedObjective(Recorder(), 10, profiler=profiler)
    objective.evaluate_columns(columns(3), {0: 1.0})
    assert profiler.report().blocks["objective"].calls == 2


def test_no_known_costs_is_the_ordinary_path():
    rec_a, rec_b = Recorder(), Recorder()
    a = CountedObjective(rec_a, 10).evaluate_columns(columns(3))
    b = CountedObjective(rec_b, 10).evaluate_columns(columns(3), {})
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] is b[1] is True


def test_the_objective_error_still_counts_the_call_with_profiling():
    def boom(x):
        raise RuntimeError

    profiler = TimingProfiler()
    objective = CountedObjective(boom, 10, profiler=profiler)
    with pytest.raises(RuntimeError):
        objective.evaluate_columns(columns(3), {0: 1.0})
    assert objective.calls == 1
    assert profiler.report().blocks["objective"].calls == 1


@pytest.mark.parametrize("scenario", SHORT[:2], ids=lambda s: s.name)
def test_trace_is_the_legacy_trace_with_the_incumbent_evaluations_deleted(
    scenario, monkeypatch
):
    deleted: set[int] = set()
    original = SectionRunner._draw

    def spy(self, state, samplers):
        samples, boundaries = original(self, state, samplers)
        deleted.update(self._objective.calls + int(b) for b in boundaries[:-1])
        return samples, boundaries

    monkeypatch.setattr(SectionRunner, "_draw", spy)
    legacy_costs, legacy_ledger, _ = run_case(atm.ATMConfig(), scenario)
    expected = [c for i, c in enumerate(legacy_costs) if i not in deleted]
    assert len(expected) < len(legacy_costs) - 20  # a real number were deleted

    shorter = type(scenario)(
        scenario.name, scenario.function, scenario.dim, scenario.seed, len(expected)
    )
    costs, ledger, _ = run_case(SKIP, shorter)
    assert costs == expected
    # Same iterations, same draws: nothing but the evaluations differ.
    assert ledger == legacy_ledger


def test_skipping_buys_more_iterations_for_the_same_budget():
    scenario = SHORT[0]
    _, ledger_skip, _ = run_case(SKIP, scenario)
    _, ledger_legacy, _ = run_case(atm.ATMConfig(), scenario)
    assert len(ledger_skip) > len(ledger_legacy)
