"""The original's random stream, and the draws kept only to stay aligned with it.

A given ``RandomState`` must produce the original's numbers. That fixes not just
the draws the algorithm uses but also draws whose results the original threw
away, and the three always-zero padding rows of the hyperparameter matrix. They
all live here so that the rest of the package reads as the algorithm alone.

The stream contract, in the order the draws happen in one search (checked by
``tests/test_rng_ledger.py``, which compares every draw's name and shape with
the original):

1. Per sampler, four times: five ``rand(n_params)`` vectors; only the fourth is
   used (the adaptive amplitude vector, ``rand - 0.5``). See
   :func:`sampler_init_draws`.
2. ``rand(N_ROWS, 4)`` scaled by 10: the initial hyperparameters, one column
   per sampler. See :func:`draw_initial_hyperparameters`.
3. ``rand(n_params, 64)``: the "directional pool", unused. See
   :func:`burn_directional_pool`. It is drawn again before every section.
4. Each iteration of a section, per sampler, in this order:
   crossover (``rand(n_params, k)`` for two parents, ``randint(0, m, (n_params, k))``
   for the best eighth), then the noise ``rand(n_params, k)``, then, once a
   direction exists, ``rand(k, 1)`` for the targeted step.
5. Between sections, the hyperparameter search draws the same three kinds of
   numbers for its own four candidates.

Sections that run after the budget is gone are skipped here, so this package's
draw sequence is a prefix of the original's.
"""

from __future__ import annotations

import numpy as np

N_HYPERPARAMETERS = 8
# The original allocates 11 rows per hyperparameter vector and fills 8; the other
# three stay [0, 0] after clipping but shape the draws of the initial matrix.
LEGACY_PADDING_ROWS = 3
N_ROWS = N_HYPERPARAMETERS + LEGACY_PADDING_ROWS

# Number of random vectors the original draws per sampler at construction.
_SAMPLER_DRAWS = 5
_ADAPTIVE_DRAW = 3  # zero-based position of the one that is used


def sampler_init_draws(n_params: int, rng: np.random.RandomState) -> np.ndarray:
    """Draw a sampler's five construction vectors; return the adaptive vector.

    The original draws best, second best, change, adaptive and squared-gradient
    vectors. The first, second, third and fifth are overwritten before they are
    read (the fifth only feeds the dead "adaptive amplitude" update, audit B1).
    """
    draws = [rng.rand(n_params) for _ in range(_SAMPLER_DRAWS)]
    return draws[_ADAPTIVE_DRAW] - 0.5


def draw_initial_hyperparameters(
    rng: np.random.RandomState, n_samplers: int
) -> np.ndarray:
    """Unclipped initial hyperparameters, shape ``(N_ROWS, n_samplers)``."""
    return 10 * rng.rand(N_ROWS, n_samplers)


def burn_directional_pool(
    rng: np.random.RandomState, n_params: int, population: int
) -> None:
    """Draw and discard the original's unused "directional pool" (audit B9)."""
    rng.rand(n_params, population)
