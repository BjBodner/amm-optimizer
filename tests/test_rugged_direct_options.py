"""The opt-in ``rugged_direct_beta`` / ``rugged_direct_min_beta``. Each default is
c44's."""

import numpy as np
import pytest

from amm_optimizer import presets
from amm_optimizer.api import minimize


def separable(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - 1.0) ** 2 + 2.0 * np.sin(3.0 * x)))


def rotated(x):
    c, s = np.cos(0.7), np.sin(0.7)
    y = np.array([[c, -s], [s, c]]) @ np.asarray(x, dtype=float)
    return float(y[0] ** 2 + 100.0 * y[1] ** 2)


def stepped(x):
    """Separable plateaus: every single move from the origin is flat."""
    return float(np.sum(np.floor(np.abs(np.asarray(x, dtype=float)) / 20.0)))


def run(config, fun, dim=2, budget=200, seed=3):
    # x0 from another stream than the optimizer's (as in the harness): with the same
    # seed the pair test's random point t would be x0 itself
    x0 = np.random.RandomState(seed + 100).uniform(-5, 5, dim)
    calls = []

    def logged(x):
        calls.append(np.array(x))
        return fun(x)

    res = minimize(logged, x0, budget, seed=seed, config=config, bounds=(-5.0, 5.0))
    return res, np.array(calls)


@pytest.mark.parametrize(
    "bad",
    [
        {"rugged_direct_beta": -1.0},
        {"rugged_direct_min_beta": -0.5},
    ],
)
def test_bad_values_are_refused(bad):
    with pytest.raises(ValueError):
        presets.with_changes("c44", **bad)


def chain3(x):
    """An f20-like linear chain at D 3 (reads "chain" in the pair test)."""
    m = np.eye(3) + np.diag(np.full(2, 0.25), -1)
    w = m @ (np.asarray(x, dtype=float) - 0.5)
    return float(np.sum(w**2 + 3.0 * (1.0 - np.cos(2.0 * np.pi * w))))


def rugged(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2 + 10.0 * (1.0 - np.cos(2.0 * np.pi * x))))


def test_rugged_direct_runs_and_is_off_outside_its_band():
    base = presets.get("c44")
    on = presets.with_changes("c44", rugged_direct_beta=1.0, rugged_direct_min_beta=0.5)
    never = presets.with_changes(
        "c44", rugged_direct_beta=1.0, rugged_direct_min_beta=0.99
    )
    res, calls = run(on, rugged, dim=5, budget=1500)
    assert len(calls) <= 1500 and np.isfinite(res.fun)
    # a band no reading falls into leaves c44's run unchanged on a smooth function
    _, a = run(never, separable, dim=5, budget=1500)
    _, b = run(base, separable, dim=5, budget=1500)
    assert np.array_equal(a, b)
