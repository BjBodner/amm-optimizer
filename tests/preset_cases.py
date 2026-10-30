"""Behavioural fingerprints of the presets: cost traces on a fixed grid.

The configurations are spelled out field by field (not through the preset
methods), so the same code can fingerprint older commits where the methods had
other names. A change of any hash means the preset's behaviour changed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import amm_optimizer as atm
from amm_optimizer.config import Variant

GOLDEN = Path(__file__).parent / "golden" / "presets.json"

COMPOSITE = {
    "amplitude_control": "success_rate_targeted",
    "initial_design": "uniform",
    "bounds": "fold",
    "corner_clip": 0.25,
    "pool_rule": "budget",
    "pool_min": 12,
    "reevaluate_incumbent": False,
}
# The composite with the (1+1)-ES prefix and the learned covariance: the starting
# point several tests build their variants from (see ``wave5_variant``).
WAVE5 = {
    **COMPOSITE,
    "es_phase": "prefix",
    "es_fraction": 0.9,
    "es_max_budget_per_dim": 100.0,
    "covariance": "rank_mu",
    "cov_rate": 3.0,
    "cov_select": 0.5,
}
PRESETS = {"default": {}, "composite": COMPOSITE}


def wave5_variant() -> Variant:
    """The composite with the (1+1)-ES prefix and the learned covariance."""
    return Variant(**WAVE5)


def sphere(x):
    return float(np.sum((x - 1.0) ** 2))


def rastrigin(x):
    return float(10 * len(x) + np.sum((x - 1.0) ** 2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


FUNCTIONS = {"sphere": sphere, "rastrigin": rastrigin, "rosenbrock": rosenbrock}
# (dimension, budget multipliers): long runs only where they are cheap.
GRID = [
    (2, (30, 100, 300)),
    (5, (30, 100, 300)),
    (10, (30, 100)),
    (20, (30,)),
    (40, (30,)),
]


def config_of(name: str) -> atm.ATMConfig:
    return atm.ATMConfig(variant=Variant(**PRESETS[name]))


def fingerprint(name: str, function: str, dim: int, multiplier: int) -> str:
    fun = FUNCTIONS[function]
    costs: list[float] = []

    def recording(x):
        costs.append(fun(np.asarray(x)))
        return costs[-1]

    rng = np.random.RandomState(1000 + dim + multiplier)
    x0 = rng.uniform(-4, 4, dim)
    config = config_of(name)
    kwargs = {"bounds": (-5.0, 5.0)} if config.variant.needs_box else {}
    atm.api.minimize(recording, x0, multiplier * dim, rng=rng, config=config, **kwargs)
    digest = hashlib.sha256(np.asarray(costs).tobytes()).hexdigest()[:20]
    return f"{len(costs)}:{digest}"


def all_fingerprints() -> dict[str, dict[str, str]]:
    return {
        name: {
            f"{function}-d{dim}-x{m}": fingerprint(name, function, dim, m)
            for function in FUNCTIONS
            for dim, multipliers in GRID
            for m in multipliers
        }
        for name in PRESETS
    }


def load() -> dict:
    return json.loads(GOLDEN.read_text())["fingerprints"]
