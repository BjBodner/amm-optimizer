"""``box_normalize``: off by default, so the presets are unchanged."""

import numpy as np
import pytest

from amm_optimizer import presets
from amm_optimizer.api import minimize
from amm_optimizer.bounds import Box

BOX5 = Box.make((-5.0, 5.0), 5)


def sphere(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - 1.0) ** 2))


def rotated(x):
    """A non-separable quadratic (every pair of coordinates interacts)."""
    x = np.asarray(x, dtype=float) - 1.0
    return float(np.sum(x**2) + np.sum(x) ** 2)


def test_box_normalize_maps_a_scaled_box_back():
    config = presets.with_changes("c43", box_normalize=True)

    def scaled(x):
        return sphere(np.asarray(x) / 100.0)

    out = minimize(
        scaled, np.full(5, 300.0), 150, seed=2, config=config, bounds=(-500, 500)
    )
    assert np.all(np.abs(out.x) <= 500.0)
    assert out.fun == pytest.approx(scaled(out.x))
    assert out.fun < 1e-6
