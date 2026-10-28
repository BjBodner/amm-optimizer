"""Stop a run early: at a target value, or when a time limit runs out.

python examples/early_stopping.py [budget]
"""

import sys
import time

import numpy as np

import amm_optimizer as amm

budget = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
bounds = [(-5, 5)] * 8
rotation = np.linalg.qr(np.random.default_rng(1).normal(size=(8, 8)))[0]


def rastrigin(x):
    z = rotation @ (x - 0.5)
    return float(10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z)))


# 1. ftarget: stop as soon as the best value is good enough.
r = amm.minimize(rastrigin, bounds=bounds, budget=budget, ftarget=10.0, seed=0)
print(f"ftarget: {r.message} fun={r.fun:.3g}, {r.nfev} of {budget} evaluations")


# 2. A time limit for a slow objective: the callback runs after each improvement
#    of the best value and raises StopIteration once the time is up.
def slow_rastrigin(x):
    time.sleep(0.002)
    return rastrigin(x)


def time_limit(intermediate_result, seconds=1.0):
    if time.perf_counter() - start > seconds:
        raise StopIteration


start = time.perf_counter()
# anytime=True suits a run you cut short: it does not plan for the whole budget.
r = amm.minimize(slow_rastrigin, bounds=bounds, budget=budget, anytime=True,
                 callback=time_limit, seed=0)  # fmt: skip
print(f"time limit: {r.message} fun={r.fun:.3g} after {r.nfev} evaluations")
