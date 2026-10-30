"""Every shipped configuration runs in one to three dimensions at every budget.

The characterisation grid starts at two dimensions, so a crash that only a
one-dimensional problem reaches (``np.cov`` returns a 0-d array there) went
unseen. This is a smoke matrix: the run finishes, spends exactly its budget, stays
inside the box and reports the best point it evaluated.
"""

from __future__ import annotations

import numpy as np
import pytest

import amm_optimizer as atm

BOX = (-5.0, 5.0)
DIMS = (1, 2, 3)
BUDGETS = (1, 10, 40, 150, 300)

CONFIGS = {
    "default": lambda: (atm.ATMConfig(), None),
    "composite": lambda: (atm.ATMConfig.composite(), BOX),
    "c46": lambda: (atm.presets.get("c46"), BOX),
}


def _rastrigin(x: np.ndarray) -> float:
    return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize(
    "name",
    list(CONFIGS),
)
def test_a_shipped_config_runs_in_small_dimensions(name: str, dim: int, budget: int):
    config, bounds = CONFIGS[name]()
    seen: list[tuple[np.ndarray, float]] = []

    def fun(x: np.ndarray) -> float:
        value = _rastrigin(np.array(x, dtype=float))
        seen.append((np.array(x, dtype=float), value))
        return value

    x0 = np.full(dim, 1.5)
    result = atm.api.minimize(fun, x0, budget, seed=0, config=config, bounds=bounds)

    assert len(seen) == budget == result.nfev
    assert np.isfinite(result.fun)
    assert result.x.shape == (dim,)
    assert result.fun == min(value for _, value in seen)
    if bounds is not None:
        points = np.array([point for point, _ in seen])
        assert points.min() >= BOX[0] and points.max() <= BOX[1]
