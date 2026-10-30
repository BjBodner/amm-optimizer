"""The opt-in ``prefix_descent_retries`` of the one-descent prefixes.

Each option is off by default, so ``c43`` and ``c44`` keep their recorded traces (the
characterisation tests); these tests pin what each option does when it is on.
"""

import numpy as np
import pytest

from amm_optimizer import qn_phase
from amm_optimizer.bounds import Box
from amm_optimizer.es_phase import _guarded_descents
from amm_optimizer.objective import CountedObjective

BOX5 = Box.make((-5.0, 5.0), 5)
BOX2 = Box.make((-5.0, 5.0), 2)


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def steps(x):
    """A plateau function: constant on unit cells (like bbob f7's rounding)."""
    return float(np.sum(np.floor(np.abs(np.asarray(x)) + 0.5) ** 2))


def counted(fun, allowed):
    return CountedObjective(fun, allowed)


def test_the_flat_gradient_reader_sees_a_zero_first_gradient():
    flat = qn_phase.FlatGradientReader(3)
    for v in (2.0, 2.0, 2.0, 2.0, 5.0):
        flat.after(np.zeros(3), v)
    assert flat.flat and not flat.before(np.zeros(3))
    sloped = qn_phase.FlatGradientReader(3)
    for v in (2.0, 2.0, 2.1, 2.0):
        sloped.after(np.zeros(3), v)
    assert not sloped.flat
    short = qn_phase.FlatGradientReader(3)
    short.after(np.zeros(3), 1.0)
    short.start_descent(1)  # only the first descent is read
    short.after(np.zeros(3), 1.0)
    assert not short.flat


@pytest.mark.parametrize(
    ("fun", "x", "descents"),
    [
        (steps, 0.37, 1),  # a flat first gradient (a plateau): no retry
        (sphere, 0.0, 2),  # a descent that ends where it started: one retry
    ],
)
def test_descent_retries_stop_on_a_plateau_and_retry_elsewhere(
    monkeypatch, fun, x, descents
):
    calls = {"n": 0}
    real = qn_phase.run_slsqp_restarts

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(qn_phase, "run_slsqp_restarts", counting)
    start = np.full(2, x)
    _guarded_descents(
        counted(fun, 10_000),
        np.random.RandomState(0),
        start,
        fun(start),
        BOX2,
        200,
        None,
        {},
        5,
    )
    assert calls["n"] == descents


def ellipsoid(x):
    x = np.asarray(x)
    return float(np.sum(10.0 ** (4 * np.arange(len(x)) / (len(x) - 1)) * x**2))
