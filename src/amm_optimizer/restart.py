"""When to abandon the current search and restart from the best point."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RestartPolicy:
    max_random_growth: float = 30.0
    min_history: int = 100
    improvement_threshold: float = 0.5e-10
    window: int = 30


def stagnation_window(dim: int, pool: int) -> int:
    """Iterations the stagnation test looks back: ``10 + ceil(30 dim / pool)``."""
    return 10 + -(-30 * dim // pool)


def stagnated(
    cost_history: Sequence[float],
    log_widths: Sequence[float],
    window: int,
    tolfun: float,
    min_log_width: float,
) -> bool:
    """Whether a success-rate controlled search has stopped making progress.

    True once ``window`` iterations exist and either the incumbent's cost moved by
    at most ``tolfun`` over the last ``window`` iterations, or every sampler's
    log-width is at or below ``min_log_width``.
    """
    if max(log_widths) <= min_log_width:
        return True
    if len(cost_history) < window:
        return False
    recent = cost_history[-window:]
    return bool(max(recent) - min(recent) <= tolfun)


def check_restart(
    random_growths: Sequence[float],
    cost_history: Sequence[float],
    policy: RestartPolicy | None = None,
) -> tuple[bool, str]:
    """Return ``(should_restart, reason)``.

    ``cost_history`` must hold NumPy floats: the original relies on NumPy's
    division-by-zero semantics (inf/NaN) rather than raising.
    """
    policy = policy or RestartPolicy()
    if min(random_growths) > policy.max_random_growth:
        return True, "Random Search Growth Factor"

    if len(cost_history) <= policy.min_history:
        return False, ""

    recent = cost_history[-1]
    past = cost_history[-policy.window]
    initial = cost_history[0]

    absolute = past - recent
    if absolute < policy.improvement_threshold:
        return True, "Insufficient Absolute Improvement"

    relative = abs(absolute / initial)
    if relative < policy.improvement_threshold:
        return True, "Insufficient Relative Improvement"

    if past == recent:
        return True, "Cost Stagnation"

    return False, ""
