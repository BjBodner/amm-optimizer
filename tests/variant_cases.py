"""Shared by the variant tests and ``tools/record_variants.py``.

A variant cannot be compared draw for draw with the original (its trajectory
diverges), so each one is pinned by a self-snapshot: the digest of its own cost
trace and of its own draw ledger on a few short scenarios.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "golden"))

from scenarios import FUNCTIONS, Scenario  # noqa: E402

import amm_optimizer as atm  # noqa: E402
from amm_optimizer.config import Variant  # noqa: E402
from test_rng_ledger import LoggingRandomState  # noqa: E402

SNAPSHOTS = HERE / "golden_variants.json"
# The box the variants that need one are told about (the scenarios start inside).
BOX = (-5.0, 5.0)

# Short enough to run in a second each, long enough to reach the meta level.
SHORT = [
    Scenario("sphere_d2", "sphere", 2, 1, 700),
    Scenario("rastrigin_d5", "rastrigin", 5, 2, 1500),
    Scenario("ellipsoid_d3_tiny", "ellipsoid", 3, 3, 40),
]

# One entry per implemented flag (value other than the default).
VARIANTS: dict[str, Variant] = {
    "no_reeval": Variant(reevaluate_incumbent=False),
    "uniform_start": Variant(initial_design="uniform"),
    "success_rate_targeted": Variant(amplitude_control="success_rate_targeted"),
    "pool_by_budget": Variant(pool_rule="budget"),
    "covariance_rank_mu": Variant(
        amplitude_control="success_rate_targeted",
        covariance="rank_mu",
        cov_rate=3.0,
        cov_select=0.5,
        pool_rule="budget",
        pool_min=12,
    ),
    "es_prefix": Variant(
        amplitude_control="success_rate_targeted",
        bounds="fold",
        pool_rule="budget",
        pool_min=12,
        reevaluate_incumbent=False,
        es_phase="prefix",
    ),
    "es_prefix_cma": Variant(
        amplitude_control="success_rate_targeted",
        bounds="fold",
        pool_rule="budget",
        pool_min=12,
        reevaluate_incumbent=False,
        es_phase="prefix",
        es_kind="cma",
    ),
    "cma_handoff": Variant(
        amplitude_control="success_rate_targeted",
        bounds="fold",
        pool_rule="budget",
        pool_min=12,
        reevaluate_incumbent=False,
        cma_handoff=0.3,
    ),
    "cma_model": Variant(
        amplitude_control="success_rate_targeted",
        bounds="fold",
        pool_rule="budget",
        pool_min=12,
        reevaluate_incumbent=False,
        es_phase="prefix",
        es_kind="cma",
        cma_handoff=0.2,
        cma_model="quadratic",
    ),
    "cma_active": Variant(
        amplitude_control="success_rate_targeted",
        bounds="fold",
        pool_rule="budget",
        pool_min=12,
        reevaluate_incumbent=False,
        es_phase="prefix",
        es_kind="cma",
        cma_handoff=0.2,
        cma_active=True,
    ),
    "cma_warm_start": Variant(
        amplitude_control="success_rate_targeted",
        bounds="fold",
        pool_rule="budget",
        pool_min=12,
        reevaluate_incumbent=False,
        es_phase="prefix",
        es_kind="cma",
        cma_handoff=0.2,
        cma_warm_start=True,
    ),
    "cma_then_ipop": Variant(
        amplitude_control="success_rate_targeted",
        bounds="fold",
        pool_rule="budget",
        pool_min=12,
        reevaluate_incumbent=False,
        es_phase="prefix",
        es_kind="cma",
        cma_handoff=0.2,
        cma_restart_rule="short",
        cma_local="then_ipop",
        cma_local_min_dim=2,
    ),
    "fold": Variant(bounds="fold"),
    "fold_corner": Variant(bounds="fold", corner_clip=0.25),
}


def trace_digest(costs: list[float]) -> str:
    """Digest of a cost trace, rounded to 9 digits so libm ulps cannot matter."""
    text = ",".join(f"{c:.8e}" for c in costs)
    return hashlib.sha256(text.encode()).hexdigest()


def ledger_digest(ledger: list) -> str:
    return hashlib.sha256(repr(ledger).encode()).hexdigest()


def run_case(config: atm.ATMConfig, scenario: Scenario, points: list | None = None):
    """Run ``scenario`` with ``config``; returns (costs, ledger, result).

    ``points``, if given, receives a copy of every evaluated point.
    """
    fun = FUNCTIONS[scenario.function]
    x0 = 10 * (np.random.RandomState(scenario.seed).rand(scenario.dim) - 0.5)
    rng = LoggingRandomState(scenario.seed + 1000)
    costs: list[float] = []

    def counted(x):
        if points is not None:
            points.append(np.array(x))
        costs.append(float(fun(x)))
        return costs[-1]

    result = atm.api.minimize(
        counted, x0, scenario.budget, rng=rng, config=config, bounds=BOX
    )
    return costs, rng.ledger, result


def snapshot(config: atm.ATMConfig, scenario: Scenario) -> dict:
    costs, ledger, result = run_case(config, scenario)
    return {
        "n_evals": len(costs),
        "nfev": result.nfev,
        "trace_sha256": trace_digest(costs),
        "ledger_sha256": ledger_digest(ledger),
    }
