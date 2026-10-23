"""The order and shape of every random draw is part of the reproduction contract.

Equal costs could hide a swapped pair of draws that happens not to matter. The
ledger records every ``rand`` / ``randint`` call (name and shape), so the draw
sequence of the new package can be compared with the original's directly.

The new code skips the draws of sections that run after the budget is gone (they
evaluate nothing, see docs/DEVIATIONS.md #4), so its ledger must be a *prefix* of
the original's.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "golden"))

from reference.atm_original import Orchestra_Optimize_With_Two_Mode_Section  # noqa: E402
from scenarios import SCENARIOS, Scenario, run_scenario  # noqa: E402

import amm_optimizer as atm  # noqa: E402


def _shape(size):
    if size is None:
        return None
    return tuple(size) if hasattr(size, "__len__") else (size,)


class LoggingRandomState(np.random.RandomState):
    """A RandomState that records the shape of every draw."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.ledger = []

    def rand(self, *shape):
        self.ledger.append(("rand", shape))
        return super().rand(*shape)

    def randint(self, low, high=None, size=None, dtype=int):
        self.ledger.append(("randint", (low, high, _shape(size))))
        return super().randint(low, high, size, dtype)


@pytest.fixture
def global_ledger(monkeypatch):
    """Log the draws made through ``np.random.rand`` / ``np.random.randint``."""
    ledger = []
    rand, randint = np.random.rand, np.random.randint

    def logging_rand(*shape):
        ledger.append(("rand", shape))
        return rand(*shape)

    def logging_randint(low, high=None, size=None, dtype=int):
        ledger.append(("randint", (low, high, _shape(size))))
        return randint(low, high, size, dtype)

    monkeypatch.setattr(np.random, "rand", logging_rand)
    monkeypatch.setattr(np.random, "randint", logging_randint)
    return ledger


def reference_ledger(scenario, ledger):
    def optimize(fun, x0, budget):
        return Orchestra_Optimize_With_Two_Mode_Section(x0, fun, budget, 0)

    ledger.clear()
    run_scenario(optimize, scenario)
    return list(ledger)


def new_ledger(scenario, ledger):
    holder = []

    def optimize(fun, x0, budget):
        rng = LoggingRandomState()
        rng.set_state(np.random.get_state())
        holder.append(rng)
        config = atm.ATMConfig(strict_budget=False)
        return atm.api.minimize(fun, x0, budget, rng=rng, config=config).x

    ledger.clear()
    run_scenario(optimize, scenario)
    return list(ledger) + holder[0].ledger  # the start-point draw, then the run


CASES = [
    s
    for s in SCENARIOS
    if s.name not in {"sphere_d1_crashes", "nan_d3_crashes", "nan_far_d3_crashes"}
] + [
    Scenario("rastrigin_d3_restarts", "rastrigin", 3, 11, 6000),
    Scenario("sphere_d12", "sphere", 12, 12, 1500),
]


@pytest.mark.parametrize("scenario", CASES, ids=lambda s: s.name)
def test_draw_sequence_matches_the_original(scenario, global_ledger):
    want = reference_ledger(scenario, global_ledger)
    got = new_ledger(scenario, global_ledger)
    assert len(got) > 20  # a real run, not an empty comparison
    assert got == want[: len(got)]


def test_ledger_detects_a_swapped_pair_of_draws(global_ledger):
    """The check has teeth: reordering two draws changes the ledger."""
    scenario = CASES[0]
    want = reference_ledger(scenario, global_ledger)
    i = next(k for k in range(1, len(want)) if want[k] != want[k - 1])
    swapped = list(want)
    swapped[i - 1], swapped[i] = swapped[i], swapped[i - 1]
    assert swapped != want
    got = new_ledger(scenario, global_ledger)
    assert got == want[: len(got)]
    assert got != swapped[: len(got)]


def test_logging_random_state_stream_equals_the_plain_one():
    plain, logging = np.random.RandomState(5), LoggingRandomState(5)
    np.testing.assert_array_equal(plain.rand(3, 2), logging.rand(3, 2))
    np.testing.assert_array_equal(
        plain.randint(0, 9, (2, 2)), logging.randint(0, 9, (2, 2))
    )
    np.testing.assert_array_equal(plain.randint(0, 9, 4), logging.randint(0, 9, 4))
    assert logging.ledger == [
        ("rand", (3, 2)),
        ("randint", (0, 9, (2, 2))),
        ("randint", (0, 9, (4,))),
    ]
