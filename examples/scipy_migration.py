"""From scipy.optimize.minimize to AMM: the same problem, read the same way.

python examples/scipy_migration.py [budget]
"""

import sys

import numpy as np
from scipy.optimize import Bounds, OptimizeResult, minimize, rosen

import amm_optimizer as amm

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
x0 = np.full(5, -1.0)
bounds = Bounds(-5.0, 5.0)  # AMM also takes [(low, high), ...] or (lower, upper)
seen = {"SciPy": 0, "AMM": 0}


def counter(name):
    """A SciPy-style callback: SciPy calls it per iteration, AMM per improvement."""

    def callback(intermediate_result):
        seen[name] += 1

    return callback


options = {"maxfev": budget}  # SciPy's budget is an option; AMM's is an argument
scipy_result = minimize(
    rosen,
    x0,
    method="Powell",
    bounds=bounds,
    options=options,
    callback=counter("SciPy"),
)
amm_result = amm.minimize(rosen, x0, budget, bounds=bounds, callback=counter("AMM"))

for name, r in [("SciPy", scipy_result), ("AMM", amm_result)]:
    assert isinstance(r, OptimizeResult)  # one type: code that reads one reads both
    print(f"{name:5s} success={r.success} fun={r.fun:.3g} nfev={r.nfev}")
    print(f"      x={np.round(r['x'], 4)}, callback calls={seen[name]}")
