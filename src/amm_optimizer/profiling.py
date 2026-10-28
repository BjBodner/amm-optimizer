"""Named-block timing for runtime attribution.

The optimizer wraps each stage of the algorithm in ``with profiler.block(name)``.
The default :class:`NullProfiler` makes that a no-op, so nothing is measured and
results are unchanged; :class:`TimingProfiler` records, per block name, the number
of calls and the inclusive and exclusive ("self") time. Exclusive times add up to
the wall time of the ``total`` block, so shares are meaningful.

Blocks are placed at iteration granularity (about every 60 objective calls); the
only per-call timer is the one around the user's objective, and it is only used
when profiling is on.

Block names (the tree of timed blocks)::

    total
      init                  samplers, hyperparameter start values, first objective call
      section               one block of iterations with fixed hyperparameters
        draw                candidate generation for all samplers
          sample            one sampler
            sample.crossover | sample.amplitude | sample.noise | sample.targeted
        evaluate            the objective loop (its own overhead)
          objective         time inside the user's function
        advance             bookkeeping after an iteration
          select            winner, best eighth, runner-up
          reallocate        conductor
          restart_check
      meta_propose          next hyperparameters
      meta_update           learn from the section just run
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Protocol

NS = 1e-9


@dataclass(frozen=True)
class BlockStats:
    """Accumulated time of one block name, in seconds."""

    calls: int
    total_s: float  # inclusive of nested blocks
    self_s: float  # exclusive of nested blocks
    min_s: float
    max_s: float


@dataclass(frozen=True)
class Timings:
    """Per-block timing of a run (or of several runs on one profiler)."""

    blocks: dict[str, BlockStats]

    @property
    def wall_s(self) -> float:
        """Time inside the outermost ``total`` block."""
        total = self.blocks.get("total")
        return total.total_s if total else sum(b.self_s for b in self.blocks.values())

    @property
    def objective_s(self) -> float:
        """Time spent in the user's objective function."""
        stats = self.blocks.get("objective")
        return stats.self_s if stats else 0.0

    @property
    def overhead_s(self) -> float:
        """Optimizer time: wall time minus the objective."""
        return self.wall_s - self.objective_s

    def share(self, name: str) -> float:
        """Exclusive time of ``name`` as a fraction of the wall time."""
        wall = self.wall_s
        return self.blocks[name].self_s / wall if wall > 0 else 0.0

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            name: {
                "calls": b.calls,
                "total_s": b.total_s,
                "self_s": b.self_s,
                "min_s": b.min_s,
                "max_s": b.max_s,
                "share": self.share(name),
            }
            for name, b in self.blocks.items()
        }

    def table(self) -> str:
        """Plain-text table, largest exclusive time first."""
        rows = sorted(self.blocks.items(), key=lambda kv: -kv[1].self_s)
        lines = [
            f"{'block':<18}{'calls':>9}{'self ms':>10}{'share':>8}"
            f"{'total ms':>10}{'mean us':>10}"
        ]
        for name, b in rows:
            mean_us = 1e6 * b.total_s / b.calls if b.calls else 0.0
            lines.append(
                f"{name:<18}{b.calls:>9}{1e3 * b.self_s:>10.2f}"
                f"{100 * self.share(name):>7.1f}%{1e3 * b.total_s:>10.2f}"
                f"{mean_us:>10.1f}"
            )
        lines.append(
            f"wall {1e3 * self.wall_s:.2f} ms, "
            f"objective {1e3 * self.objective_s:.2f} ms, "
            f"optimizer overhead {1e3 * self.overhead_s:.2f} ms"
        )
        return "\n".join(lines)


class Profiler(Protocol):
    """What the optimizer needs from a profiler."""

    enabled: bool

    def block(self, name: str) -> AbstractContextManager[Any]:
        """Context manager timing the enclosed code under ``name``."""

    def record(self, name: str, elapsed_ns: int, calls: int = 1) -> None:
        """Add time measured elsewhere (inside the innermost open block)."""

    def report(self) -> Timings | None:
        """Snapshot of what was measured, or ``None`` when nothing is measured."""


class _NullBlock:
    """Shared do-nothing context manager (one instance, no allocation per use)."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


_NULL_BLOCK = _NullBlock()


class NullProfiler:
    """Measures nothing. The default."""

    enabled = False

    def block(self, name: str) -> _NullBlock:
        return _NULL_BLOCK

    def record(self, name: str, elapsed_ns: int, calls: int = 1) -> None:
        return None

    def report(self) -> Timings | None:
        return None


NULL_PROFILER = NullProfiler()


class _Entry:
    """Accumulator of one block name; also its context manager."""

    __slots__ = ("calls", "max_ns", "min_ns", "profiler", "self_ns", "total_ns")

    def __init__(self, profiler: TimingProfiler) -> None:
        self.profiler = profiler
        self.calls = 0
        self.total_ns = 0
        self.self_ns = 0
        self.min_ns = 1 << 62
        self.max_ns = 0

    def add(self, elapsed: int, children: int, calls: int = 1) -> None:
        self.calls += calls
        self.total_ns += elapsed
        self.self_ns += elapsed - children
        per_call = elapsed // calls
        if per_call < self.min_ns:
            self.min_ns = per_call
        if per_call > self.max_ns:
            self.max_ns = per_call

    def __enter__(self) -> None:
        # [start, time spent in nested blocks]; kept on a stack so that the same
        # name may be entered again from inside itself.
        self.profiler._stack.append([perf_counter_ns(), 0])

    def __exit__(self, *exc: object) -> None:
        stack = self.profiler._stack
        start, children = stack.pop()
        elapsed = perf_counter_ns() - start
        self.add(elapsed, children)
        if stack:
            stack[-1][1] += elapsed


class TimingProfiler:
    """Accumulates call counts and inclusive / exclusive time per block name."""

    enabled = True

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._stack: list[list[int]] = []

    def _entry(self, name: str) -> _Entry:
        entry = self._entries.get(name)
        if entry is None:
            entry = self._entries[name] = _Entry(self)
        return entry

    def block(self, name: str) -> _Entry:
        return self._entry(name)

    def record(self, name: str, elapsed_ns: int, calls: int = 1) -> None:
        self._entry(name).add(elapsed_ns, 0, calls)
        if self._stack:
            self._stack[-1][1] += elapsed_ns

    def report(self) -> Timings:
        return Timings(
            {
                name: BlockStats(
                    calls=e.calls,
                    total_s=e.total_ns * NS,
                    self_s=e.self_ns * NS,
                    min_s=e.min_ns * NS if e.calls else 0.0,
                    max_s=e.max_ns * NS,
                )
                for name, e in self._entries.items()
            }
        )


def resolve_profiler(profile: bool | Profiler) -> Profiler:
    """``False`` -> no profiling, ``True`` -> a fresh timer, else use as given."""
    if profile is False:
        return NULL_PROFILER
    if profile is True:
        return TimingProfiler()
    return profile
