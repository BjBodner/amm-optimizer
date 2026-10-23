"""Named-block timing: correct bookkeeping, and no effect on results."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer import profiling
from amm_optimizer.objective import CountedObjective
from amm_optimizer.profiling import NULL_PROFILER, NullProfiler, TimingProfiler, Timings


def sphere(x):
    return float(np.sum((np.asarray(x) - 2) ** 2))


@pytest.fixture
def clock(monkeypatch):
    """A fake perf_counter_ns that advances only when told to."""

    class Clock:
        now = 0

        def __call__(self):
            return self.now

        def tick(self, ns):
            self.now += ns

    fake = Clock()
    monkeypatch.setattr(profiling, "perf_counter_ns", fake)
    return fake


def test_null_profiler_does_nothing():
    assert not NULL_PROFILER.enabled
    with NULL_PROFILER.block("anything") as value:
        assert value is None
    NULL_PROFILER.record("x", 5, 2)
    assert NULL_PROFILER.report() is None
    assert isinstance(NULL_PROFILER, NullProfiler)


def test_nested_blocks_split_inclusive_and_exclusive_time(clock):
    prof = TimingProfiler()
    with prof.block("outer"):
        clock.tick(10)
        with prof.block("inner"):
            clock.tick(30)
        clock.tick(5)
    stats = prof.report().blocks
    assert stats["outer"].total_s == pytest.approx(45e-9)
    assert stats["outer"].self_s == pytest.approx(15e-9)
    assert stats["inner"].total_s == stats["inner"].self_s == pytest.approx(30e-9)


def test_calls_min_and_max_accumulate(clock):
    prof = TimingProfiler()
    for ns in (10, 40, 20):
        with prof.block("b"):
            clock.tick(ns)
    b = prof.report().blocks["b"]
    assert (b.calls, b.min_s, b.max_s) == (
        3,
        pytest.approx(10e-9),
        pytest.approx(40e-9),
    )
    assert b.total_s == pytest.approx(70e-9)


def test_same_name_can_be_entered_from_inside_itself(clock):
    prof = TimingProfiler()
    with prof.block("b"):
        clock.tick(10)
        with prof.block("b"):
            clock.tick(20)
    b = prof.report().blocks["b"]
    assert b.calls == 2
    assert b.total_s == pytest.approx(50e-9)  # inner counted in both entries
    assert b.self_s == pytest.approx(30e-9)  # but only once as exclusive time


def test_record_counts_inside_the_open_block(clock):
    prof = TimingProfiler()
    with prof.block("loop"):
        clock.tick(100)
        prof.record("objective", 60, calls=3)
    blocks = prof.report().blocks
    assert blocks["loop"].self_s == pytest.approx(40e-9)
    assert blocks["objective"].calls == 3
    assert blocks["objective"].min_s == pytest.approx(20e-9)  # per-call mean


def test_record_outside_any_block(clock):
    prof = TimingProfiler()
    prof.record("objective", 7)
    assert prof.report().blocks["objective"].self_s == pytest.approx(7e-9)


def test_block_records_time_when_the_body_raises(clock):
    prof = TimingProfiler()
    with pytest.raises(RuntimeError), prof.block("b"):
        clock.tick(8)
        raise RuntimeError
    assert prof.report().blocks["b"].total_s == pytest.approx(8e-9)


def test_timings_summary(clock):
    prof = TimingProfiler()
    with prof.block("total"):
        clock.tick(20)
        clock.tick(30)  # time spent in the user's function ...
        prof.record("objective", 30)  # ... reported by the wrapper
        clock.tick(50)
    t = prof.report()
    assert t.wall_s == pytest.approx(100e-9)
    assert t.objective_s == pytest.approx(30e-9)
    assert t.overhead_s == pytest.approx(70e-9)
    assert t.share("objective") == pytest.approx(0.3)
    assert t.to_dict()["total"]["share"] == pytest.approx(0.7)
    table = t.table()
    assert "objective" in table and "optimizer overhead" in table


def test_timings_without_total_or_objective():
    stats = profiling.BlockStats(1, 2e-9, 2e-9, 2e-9, 2e-9)
    t = Timings({"only": stats})
    assert t.wall_s == pytest.approx(2e-9)  # falls back to the sum of self times
    assert t.objective_s == 0.0
    empty = Timings({"x": profiling.BlockStats(0, 0.0, 0.0, 0.0, 0.0)})
    assert empty.share("x") == 0.0
    assert "0.0" in empty.table()


def test_resolve_profiler():
    assert profiling.resolve_profiler(False) is NULL_PROFILER
    assert isinstance(profiling.resolve_profiler(True), TimingProfiler)
    mine = TimingProfiler()
    assert profiling.resolve_profiler(mine) is mine


# --- through the optimizer ---------------------------------------------------


def test_no_timings_by_default():
    assert atm.minimize(sphere, np.zeros(3), 300, seed=0).timings is None


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("dim", [1, 4])
def test_profiling_does_not_change_the_run(strict, dim):
    config = atm.ATMConfig(strict_budget=strict)
    trace = []

    def recording(x):
        trace.append(sphere(x))
        return trace[-1]

    plain = atm.minimize(recording, np.zeros(dim), 700, seed=3, config=config)
    plain_trace, trace[:] = list(trace), []
    timed = atm.minimize(
        recording, np.zeros(dim), 700, seed=3, config=config, profile=True
    )
    np.testing.assert_array_equal(plain.x, timed.x)
    assert (plain.fun, plain.nfev, plain.nrestarts) == (
        timed.fun,
        timed.nfev,
        timed.nrestarts,
    )
    assert trace == plain_trace
    assert timed.timings is not None


def test_blocks_cover_the_run_and_add_up():
    result = atm.minimize(sphere, np.zeros(5), 1500, seed=1, profile=True)
    t = result.timings
    expected = {
        "total", "init", "section", "draw", "sample", "sample.crossover",
        "sample.amplitude", "sample.noise", "sample.targeted", "evaluate",
        "objective", "advance", "select", "reallocate", "restart_check",
        "meta_propose", "meta_update",
    }  # fmt: skip
    assert expected <= set(t.blocks)
    assert t.blocks["objective"].calls == result.nfev
    # Exclusive times partition the wall time.
    assert sum(b.self_s for b in t.blocks.values()) == pytest.approx(t.wall_s)
    assert 0 < t.objective_s < t.wall_s
    assert t.overhead_s == pytest.approx(t.wall_s - t.objective_s)


def test_a_shared_profiler_accumulates_over_runs():
    prof = TimingProfiler()
    a = atm.minimize(sphere, np.zeros(3), 400, seed=0, profile=prof)
    b = atm.minimize(sphere, np.zeros(3), 400, seed=1, profile=prof)
    assert b.timings.blocks["total"].calls == 2
    assert a.timings is not None
    assert b.timings.blocks["objective"].calls == a.nfev + b.nfev


@pytest.mark.parametrize("strict", [True, False])
def test_timed_objective_respects_the_budget(strict):
    prof = TimingProfiler()
    obj = CountedObjective(sphere, 5, strict=strict, profiler=prof)
    cost, complete = obj.evaluate_columns(np.zeros((2, 10)))
    assert not complete
    assert len(cost) == (5 if strict else 6)
    assert prof.report().blocks["objective"].calls == len(cost)
    # Nothing left: no time recorded for a batch that evaluates nothing.
    cost, complete = obj.evaluate_columns(np.zeros((2, 3)))
    assert (len(cost), complete) == (0, False)
    assert prof.report().blocks["objective"].calls == (5 if strict else 6)


def test_timed_objective_counts_a_call_that_raises():
    prof = TimingProfiler()

    def failing(x):
        raise ArithmeticError

    obj = CountedObjective(failing, 10, profiler=prof)
    with pytest.raises(ArithmeticError):
        obj.evaluate_columns(np.zeros((2, 3)))
    assert obj.calls == 1
    assert prof.report().blocks["objective"].calls == 1


def test_timed_objective_with_no_columns_records_nothing():
    prof = TimingProfiler()
    obj = CountedObjective(sphere, 10, profiler=prof)
    cost, complete = obj.evaluate_columns(np.zeros((2, 0)))
    assert (len(cost), complete) == (0, True)
    assert prof.report().blocks == {}
