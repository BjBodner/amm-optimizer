"""Guards the frozen reference module against the recorded golden masters.

If this fails, either ``reference/`` was edited (it must stay byte-identical to
the original apart from its import line) or the platform's floating point
differs from the recording platform (see golden.json ``_meta``).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "golden"))

# Import at module level on purpose: the reference module calls
# np.random.seed(42) on import (audit item A3), which would otherwise reset the
# stream in the middle of the first scenario.
from reference.atm_original import Orchestra_Optimize_With_Two_Mode_Section  # noqa: E402
from scenarios import SCENARIOS, run_scenario  # noqa: E402

GOLDEN = json.loads((HERE / "golden" / "golden.json").read_text())["scenarios"]


def reference_optimize(fun, x0, budget):
    return Orchestra_Optimize_With_Two_Mode_Section(x0, fun, budget, 0)


def assert_matches_golden(optimize, scenario):
    got = run_scenario(optimize, scenario)
    want = GOLDEN[scenario.name]
    assert got.get("raises") == want.get("raises")
    assert got["n_evals"] == want["n_evals"]
    np.testing.assert_array_equal(got["x0"], want["x0"])
    if "raises" in want:
        return
    np.testing.assert_allclose(got["costs"], want["costs"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(got["x"], want["x"], rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(got["final_cost"], want["final_cost"], rtol=1e-12)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
@pytest.mark.snapshot
def test_reference_matches_golden(scenario):
    assert_matches_golden(reference_optimize, scenario)
