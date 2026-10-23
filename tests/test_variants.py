"""Every variant: it does something, stays safe, and is pinned by its own snapshot.

Adding a variant costs one entry in ``variant_cases.VARIANTS`` (and a re-record
with ``tools/record_variants.py``); its specific properties get their own tests.
"""

import json
import warnings

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.config import Variant
from amm_optimizer.search import Search
from variant_cases import SHORT, SNAPSHOTS, VARIANTS, run_case, snapshot, trace_digest

SNAPS = json.loads(SNAPSHOTS.read_text())["variants"]
CASES = [(name, s) for name in VARIANTS for s in SHORT]
IDS = [f"{name}-{s.name}" for name, s in CASES]
# Variants that start further searches by design (not by an accidental restart).
SEGMENTED = {"cma_handoff_alternate"}
SINGLE = [(n, s) for n, s in CASES if n not in SEGMENTED]
SINGLE_IDS = [f"{n}-{s.name}" for n, s in SINGLE]


def config_of(name):
    return atm.ATMConfig(variant=VARIANTS[name])


def test_every_variant_has_a_snapshot():
    assert set(SNAPS) == set(VARIANTS)
    assert all(set(SNAPS[n]) == {s.name for s in SHORT} for n in VARIANTS)


@pytest.mark.parametrize(("name", "scenario"), CASES, ids=IDS)
def test_snapshot(name, scenario):
    assert snapshot(config_of(name), scenario) == SNAPS[name][scenario.name]


@pytest.mark.parametrize("name", VARIANTS)
def test_the_flag_changes_the_search(name):
    """A flag that changes nothing is a typo."""
    differs = [
        trace_digest(run_case(config_of(name), s)[0])
        != trace_digest(run_case(atm.ATMConfig(), s)[0])
        for s in SHORT
    ]
    assert any(differs)


@pytest.mark.parametrize(("name", "scenario"), CASES, ids=IDS)
def test_invariants(name, scenario):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        costs, _, result = run_case(config_of(name), scenario)
    assert result.nfev == len(costs) <= scenario.budget
    assert np.isfinite(result.x).all()
    assert result.fun == min(costs)


@pytest.mark.parametrize(("name", "scenario"), CASES[:3], ids=IDS[:3])
def test_deterministic(name, scenario):
    a = run_case(config_of(name), scenario)
    b = run_case(config_of(name), scenario)
    assert a[0] == b[0]
    assert a[1] == b[1]


@pytest.mark.parametrize(("name", "scenario"), SINGLE, ids=SINGLE_IDS)
def test_the_run_ends_by_budget_not_by_an_accidental_restart(
    name, scenario, monkeypatch
):
    runs = []
    original = Search.run

    def counting(self, *args, **kwargs):
        runs.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Search, "run", counting)
    _, _, result = run_case(config_of(name), scenario)
    assert result.nrestarts == 0
    assert len(runs) == 1


def test_a_variant_that_saves_evaluations_would_otherwise_restart(monkeypatch):
    """The rounds coupling (review R1): with the originals' limit it restarts."""
    import amm_optimizer.api as api

    runs = []
    original = Search.run

    def counting(self, *args, **kwargs):
        runs.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Search, "run", counting)

    def objective(x):
        return float(np.sum((x - 2) ** 2))

    x0 = np.full(10, -3.0)
    config = atm.ATMConfig(variant=Variant(reevaluate_incumbent=False))
    atm.api.minimize(objective, x0, 10_000, seed=0, config=config)
    assert len(runs) == 1

    def originals_limit(config, budget):
        return int(
            np.maximum(np.ceil(budget / config.evaluations_per_iteration) - 2, 1)
        )

    monkeypatch.setattr(api, "iteration_limit", originals_limit)
    runs.clear()
    atm.api.minimize(objective, x0, 10_000, seed=0, config=config)
    assert len(runs) >= 2  # the rounds ran out first: a silent restart


@pytest.mark.parametrize("name", [n for n, v in VARIANTS.items() if v.bounds == "fold"])
@pytest.mark.parametrize("scenario", SHORT[:2], ids=lambda s: s.name)
def test_a_folding_variant_never_evaluates_outside_the_box(name, scenario):
    points: list = []
    run_case(config_of(name), scenario, points)
    inside = np.array(points)
    assert inside.min() >= -5.0
    assert inside.max() <= 5.0
    # the legacy run does leave the box in these scenarios (the test has teeth)
    legacy: list = []
    run_case(atm.ATMConfig(), scenario, legacy)
    assert np.abs(np.array(legacy)).max() > 5.0


@pytest.mark.parametrize("scenario", SHORT, ids=lambda s: s.name)
def test_knowing_the_box_changes_nothing_for_the_default(scenario):
    with_box = run_case(atm.ATMConfig(), scenario)
    rng_costs = with_box[0]
    import variant_cases

    original = variant_cases.BOX
    try:
        variant_cases.BOX = None
        without = run_case(atm.ATMConfig(), scenario)
    finally:
        variant_cases.BOX = original
    assert rng_costs == without[0]
    assert with_box[1] == without[1]


# The merge tier (tools/ci.py --tier merge): subprocess or golden sweeps, not needed
# for a research cycle that leaves ``atm``'s core and the harness alone.
pytestmark = pytest.mark.merge
