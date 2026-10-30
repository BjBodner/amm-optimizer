"""When to abandon the current search and restart from the best point."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RestartPolicy:
    """When a search of the original ATM restarts (from its best point).

    A search restarts once every sampler's random-search growth counter exceeds
    ``max_random_growth``, or, after ``min_history`` iterations, once the cost
    improved by less than ``improvement_threshold`` (absolutely or relative to the
    first cost) over the last ``window`` iterations, or did not move at all. The
    defaults are the original's.

    Attributes
    ----------
    max_random_growth : float, default 30.0
        The growth counter (iterations since the incumbent last moved
        significantly) at which every sampler gives up.
    min_history : int, default 100
        Iterations before the improvement tests apply.
    improvement_threshold : float, default 5e-11
        The smallest improvement over ``window`` iterations that counts.
    window : int, default 30
        Iterations the improvement tests look back.
    """

    max_random_growth: float = 30.0
    min_history: int = 100
    improvement_threshold: float = 0.5e-10
    window: int = 30


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
