"""Hyperparameter bounds and the defaults of the hyperparameter search itself.

ATM tunes the eight sampler hyperparameters online by running the same
two-mode sampler one level up ("meta" level). The original pads each
hyperparameter vector with three always-zero rows (11 instead of 8); the
padding is kept because it shapes the random draws, which the reproduction
contract pins.
"""

from __future__ import annotations

import numpy as np

from .legacy_stream import N_HYPERPARAMETERS, N_ROWS

# (low, high) for: random_growth_rate, random_period, max_random_search,
# targeted_growth_rate, targeted_decay_rate, significant_change, alpha, beta.
_BOUNDS = [
    (0.01, 10),
    (0.1, 10),
    (0.0000001, 10),
    (0.01, 5),
    (0.1, 100),
    (10**-30, 0.01),
    (10**-5, 10**15),
    (0.1, 1),
]


def bounds() -> np.ndarray:
    """Array of shape ``(N_ROWS, 2)``; padding rows are ``[0, 0]``."""
    out = np.zeros((N_ROWS, 2))
    out[:N_HYPERPARAMETERS] = np.array(_BOUNDS)
    return out


def mean_change_value_fit(max_random_search: float, n_params: int) -> float:
    """Polynomial fit of the typical per-step change.

    The constants come from the original author's notebook
    ``Finding_The_Segnificant_Change_Value.ipynb`` (not in this repository).
    """
    fit = [
        -0.007921662,
        0.003755446,
        5.92 * (10 ** (-5)),
        -2.45 * (10 ** (-7)),
        0.250014494,
    ]
    return (
        fit[0] * max_random_search
        + fit[1] * max_random_search**2
        + fit[2] * n_params
        + fit[3] * n_params**2
        + fit[4] * max_random_search * n_params
    )


def meta_hyperparameters() -> np.ndarray:
    """Hyperparameters of the hyperparameter search (same order as ``_BOUNDS``)."""
    meta = np.zeros(N_HYPERPARAMETERS)
    meta[0] = 5.0
    meta[1] = 5.0
    meta[2] = 20.02
    meta[3] = 0.5
    meta[4] = 2.0
    meta[5] = 0.00001 * mean_change_value_fit(meta[2], 8)
    meta[6] = (10**4) / meta[2]
    meta[7] = 0.6
    return meta


def clip_to_bounds(samples: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Clip a ``(N_ROWS, n_samples)`` matrix row-wise to ``allowed``."""
    return np.asarray(np.clip(samples, allowed[:, 0:1], allowed[:, 1:2]))
