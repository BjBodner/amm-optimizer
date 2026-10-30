"""Decisions that depend on the budget and the dimension, in one place.

``minimize`` asks this module for the configuration it will actually run
(:func:`resolve_config`) and for how many iterations the hyperparameter rounds
may use (:func:`iteration_limit`). The three numbers that make up the pool size
(``initial_population``, ``conductor.total_resources`` and
``evaluations_per_iteration``) are only ever changed here, together.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import N_SAMPLERS, POOL_DIVISOR, POOL_MAX, ATMConfig, Variant


def resolve_config(config: ATMConfig, budget: int, dim: int) -> ATMConfig:
    """The configuration to run for ``budget`` evaluations in ``dim`` dimensions.

    The default configuration is returned itself (not a copy), so that code
    comparing with ``is`` can tell that nothing was resolved.
    """
    variant = config.variant
    if variant.pool_rule == "fixed":
        return config
    return with_pool(config, pool_size(variant, budget))


def pool_size(variant: Variant, budget: int) -> int:
    """Samples per iteration for ``budget`` evaluations under ``pool_rule="budget"``."""
    pool = int(np.floor(budget / POOL_DIVISOR + 0.5))
    pool = N_SAMPLERS * int(np.floor(pool / N_SAMPLERS + 0.5))
    return int(np.clip(pool, variant.pool_min, POOL_MAX))


def with_pool(config: ATMConfig, pool: int) -> ATMConfig:
    """``config`` with the three numbers that make up the pool size set to ``pool``."""
    return replace(
        config,
        initial_population=pool,
        evaluations_per_iteration=pool,
        conductor=replace(config.conductor, total_resources=pool),
    )


def iteration_limit(config: ATMConfig, budget: int) -> int | None:
    """Iterations the hyperparameter rounds may cover; ``None`` means no limit.

    The original derives the limit from the budget assuming
    ``evaluations_per_iteration`` evaluations per iteration, so that the budget
    always runs out first (audit: the margin is a few iterations). A variant
    that spends fewer evaluations per iteration would run out of rounds first
    and silently restart at the incumbent; for variants the search therefore
    ends only by budget or by the restart policy.
    """
    if config.variant != Variant():
        return None
    return int(np.maximum(np.ceil(budget / config.evaluations_per_iteration) - 2, 1))
