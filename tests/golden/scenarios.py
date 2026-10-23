"""Golden-master scenarios shared by the recorder and the tests.

A scenario fixes the objective, dimension, seed and evaluation budget. The
optimizer under test is any callable ``optimize(fun, x0, budget) -> x`` that
draws its randomness from the *global* NumPy stream, exactly as the original
code does. ``run_scenario`` seeds that stream, draws ``x0`` the same way the
original tests do, and records every objective evaluation.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def sphere(x):
    return np.sum((x - 2) ** 2)


def ellipsoid(x):
    d = len(x)
    w = 10.0 ** (4 * np.arange(d) / max(d - 1, 1))
    return np.sum(w * (x - 1) ** 2)


def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def flat(x):
    return 1.0


def nan_region(x):
    """Returns NaN in part of the domain to exercise the NaN guards."""
    return np.nan if x[0] > 3 else np.sum((x - 2) ** 2)


def nan_far(x):
    """NaN beyond 5.5; every start (|x0| <= 5) is valid, the search leaves it."""
    return np.nan if x[0] > 5.5 else np.sum((x - 2) ** 2)


FUNCTIONS: dict[str, Callable] = {
    "sphere": sphere,
    "ellipsoid": ellipsoid,
    "rosenbrock": rosenbrock,
    "rastrigin": rastrigin,
    "flat": flat,
    "nan_region": nan_region,
    "nan_far": nan_far,
}


@dataclass(frozen=True)
class Scenario:
    name: str
    function: str
    dim: int
    seed: int
    budget: int


SCENARIOS = [
    Scenario("sphere_d2_s0", "sphere", 2, 0, 1000),
    Scenario("sphere_d2_s1", "sphere", 2, 1, 1000),
    Scenario("sphere_d2_s2", "sphere", 2, 2, 1000),
    Scenario("sphere_d2_long", "sphere", 2, 7, 6000),
    Scenario("sphere_d5", "sphere", 5, 3, 1500),
    Scenario("sphere_d10", "sphere", 10, 0, 2000),
    Scenario("sphere_d20", "sphere", 20, 4, 3000),
    Scenario("ellipsoid_d10", "ellipsoid", 10, 1, 2000),
    Scenario("rosenbrock_d5", "rosenbrock", 5, 2, 2000),
    Scenario("rastrigin_d5", "rastrigin", 5, 3, 2000),
    Scenario("flat_d3", "flat", 3, 5, 800),
    # Characterize defect B20 (NaN objective -> empty median -> ValueError).
    Scenario("nan_d3_crashes", "nan_region", 3, 6, 1000),
    Scenario("nan_far_d3_crashes", "nan_far", 3, 6, 1000),
    Scenario("tiny_budget_d2", "sphere", 2, 0, 10),
    # Characterizes known defect B5 (1-D crash); expected to change in phase 6.
    Scenario("sphere_d1_crashes", "sphere", 1, 0, 200),
]


def run_scenario(optimize: Callable, scenario: Scenario) -> dict:
    """Run one scenario and return a JSON-serialisable record."""
    fun = FUNCTIONS[scenario.function]
    np.random.seed(scenario.seed)
    x0 = 10 * (np.random.rand(scenario.dim) - 0.5)

    costs: list[float] = []

    def counted(x):
        c = float(fun(x))
        costs.append(c)
        return c

    record: dict = {"x0": [float(v) for v in x0]}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            x = optimize(counted, x0, scenario.budget)
    except Exception as exc:  # characterise failures too
        record["raises"] = type(exc).__name__
        record["n_evals"] = len(costs)
        return record

    record.update(
        n_evals=len(costs),
        x=[float(v) for v in np.asarray(x)],
        final_cost=float(fun(np.asarray(x))),
        costs=costs,
    )
    return record
