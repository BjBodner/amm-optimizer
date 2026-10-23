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

from .bounds import Box
from .config import N_SAMPLERS, ATMConfig, Variant


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
    pool = int(np.floor(budget / variant.pool_divisor + 0.5))
    pool = N_SAMPLERS * int(np.floor(pool / N_SAMPLERS + 0.5))
    return int(np.clip(pool, variant.pool_min, variant.pool_max))


def ipop_pool(config: ATMConfig, variant: Variant, restarts: int) -> int:
    """Pool of the search after ``restarts`` restarts (``restart_mode="ipop"``).

    The resolved pool times ``ipop_factor ** restarts``, rounded to a multiple of
    the sampler count and capped at ``ipop_max_pool``.
    """
    base = config.conductor.total_resources
    pool = base * variant.ipop_factor**restarts
    pool = N_SAMPLERS * int(np.floor(pool / N_SAMPLERS + 0.5))
    return int(min(pool, variant.ipop_max_pool))


def in_box_units(config: ATMConfig, box: Box) -> ATMConfig:
    """``config`` with its widths in absolute units (``width_units="box"``).

    A width of 1 becomes a tenth of the mean box width (1 on the bbob box).
    """
    unit = float(np.mean(box.upper - box.lower)) / 10.0
    variant = config.variant
    return replace(
        config,
        variant=replace(
            variant,
            sr_widths=tuple(w * unit for w in variant.sr_widths),
            sr_max_width=variant.sr_max_width * unit,
        ),
    )


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
