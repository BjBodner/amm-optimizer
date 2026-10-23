"""The refactored optimizer must reproduce the original draw for draw."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))

from scenarios import FUNCTIONS, SCENARIOS, run_scenario  # noqa: E402

import amm_optimizer as atm  # noqa: E402
from test_golden_reference import (  # noqa: E402
    assert_matches_golden,
    reference_optimize,
)


def new_optimize(fun, x0, budget):
    """Continue the global legacy stream from where the scenario left it."""
    rng = np.random.RandomState()
    rng.set_state(np.random.get_state())
    config = atm.ATMConfig(strict_budget=False)
    return atm.api.minimize(fun, x0, budget, rng=rng, config=config).x


# Scenarios whose golden record is a crash in the original that the new code
# deliberately fixes (see docs/DEVIATIONS.md).
FIXED_CRASHES = {"sphere_d1_crashes", "nan_far_d3_crashes"}
# The original carried on from a NaN start value and crashed later; the new code
# rejects a non-finite start point up front.
REJECTED_AT_START = {"nan_d3_crashes"}


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
@pytest.mark.snapshot
def test_new_matches_golden(scenario):
    if scenario.name in FIXED_CRASHES:
        got = run_scenario(new_optimize, scenario)
        assert "raises" not in got
        assert got["n_evals"] > 1
        return
    if scenario.name in REJECTED_AT_START:
        got = run_scenario(new_optimize, scenario)
        assert got["raises"] == "ValueError"
        assert got["n_evals"] == 1
        return
    assert_matches_golden(new_optimize, scenario)


@pytest.mark.slow
@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("dim", [2, 3, 7, 12])
@pytest.mark.parametrize("function", ["sphere", "ellipsoid", "rastrigin", "rosenbrock"])
def test_new_matches_reference_on_random_cases(seed, dim, function):
    from scenarios import Scenario

    scenario = Scenario(f"{function}_{dim}_{seed}", function, dim, 100 + seed, 700)
    ref = run_scenario(reference_optimize, scenario)
    new = run_scenario(new_optimize, scenario)
    assert new["n_evals"] == ref["n_evals"]
    np.testing.assert_allclose(new["costs"], ref["costs"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(new["x"], ref["x"], rtol=1e-12, atol=1e-300)


def test_functions_registry_is_complete():
    assert {s.function for s in SCENARIOS} <= set(FUNCTIONS)
