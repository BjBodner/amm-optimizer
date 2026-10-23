"""Variant ``cma_placement="mlsl"``: restarts placed by GLOBAL's critical distance."""

import dataclasses

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.bounds import Box
from amm_optimizer.cma_phase import critical_distance, mlsl_start
from amm_optimizer.config import ATMConfig

BOX = (-5.0, 5.0)


@pytest.mark.parametrize(
    ("dim", "n", "expected"), [(2, 2000, 0.27), (5, 5000, 1.77), (20, 20000, 7.90)]
)
def test_critical_distance_matches_the_report_table(dim, n, expected):
    assert critical_distance(n, Box.make(BOX, dim)) == pytest.approx(expected, abs=0.01)


def test_mlsl_start_skips_endpoints_and_points_with_a_better_neighbour():
    # 76 worse filler points, so that the 5% elite covers the four that matter
    filler = np.column_stack([np.linspace(-4, 4, 76), np.full(76, -4.0)])
    points = np.vstack([[[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.05, 3.0]], filler])
    values = np.concatenate([[0.0, 0.5, 1.0, 2.0], np.full(76, 50.0)])
    found = mlsl_start(points, values, [np.zeros(2)], radius=0.5)
    assert found is not None
    x, value, sigma = found
    assert x == pytest.approx([3.0, 3.0])
    assert value == 1.0
    assert sigma == pytest.approx(0.25)
    assert mlsl_start(points, values, [np.zeros(2), np.full(2, 3.0)], 0.5) is None


def test_minimize_with_mlsl_placement():
    seen = []

    def gallagher_like(x):
        x = np.asarray(x)
        peaks = np.array([[3.0, -2.0, 1.0], [-3.0, 2.0, -1.0], [0.5, 0.5, 4.0]])
        heights = np.array([0.0, 1.0, 2.0])
        value = float(np.min(heights + 4 * np.sum((x - peaks) ** 2, axis=1)))
        seen.append(value)
        return value

    variant = dataclasses.replace(
        ATMConfig.improved_wave5().variant,
        es_kind="cma",
        cma_handoff=0.2,
        cma_placement="mlsl",
    )
    config = ATMConfig(iterations_per_hyperparameter_set=1000000, variant=variant)
    result = atm.api.minimize(
        gallagher_like, np.full(3, 4.0), 3000, seed=5, config=config, bounds=BOX
    )
    assert result.nfev == len(seen) == 3000
    assert result.fun == min(seen)
