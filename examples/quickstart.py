"""Quickstart: minimise a 10-dimensional function over a box with AMM.

python examples/quickstart.py [budget]
"""

import sys

import numpy as np

import amm_optimizer as amm


def rastrigin(x):
    """A multimodal test function; its minimum is 0 at x = 1.2."""
    z = x - 1.2
    return float(10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z)))


budget = int(sys.argv[1]) if len(sys.argv) > 1 else 5000  # evaluations of fun

# The box is all AMM needs: it starts at the centre and spends the whole budget.
result = amm.minimize(rastrigin, bounds=[(-5, 5)] * 10, budget=budget, seed=0)

print(f"best value {result.fun:.4g} after {result.nfev} evaluations")
print("best point", np.round(result.x, 3))
print(f"status {result.status}: {result.message}")
print(f"repeat the run with seed={result.seed}")
