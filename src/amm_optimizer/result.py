"""Return type of :func:`amm_optimizer.minimize`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profiling import Timings


@dataclass
class OptimizeResult:
    x: np.ndarray
    fun: float
    nfev: int
    nrestarts: int
    # Per-block time, only when ``minimize(..., profile=...)`` asked for it.
    timings: Timings | None = None
