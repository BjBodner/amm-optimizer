"""Bit-exact characterisation of the shipped configurations on COCO bbob problems.

Shared by ``tests/test_characterisation.py`` and ``tools/record_characterisation.py``.
Each case runs ``amm_optimizer.api.minimize`` once on a bbob problem (via ``cocoex``)
with the box ``[-5, 5]`` and records SHA-256 digests of every evaluated point and
value, exactly (float64 bytes, no rounding). A refactor must reproduce them bit for
bit on the recording platform; the tests skip when NumPy, SciPy or the platform
differ from the recording (``_meta``).

The set is ``NAMED``: the shipped configurations, ``ATMConfig()`` (``"default"``),
the presets ``c23``-``c46`` and ``ATMConfig.composite()`` (``"refactor_composite"``),
on a grid of functions, dimensions and budgets (``GRID``), in
``tests/characterisation.json``. Each entry keeps the configuration text it was first
recorded with.

The test tiers (``tools/ci.py --tier``):

* default (``-m "not slow and not merge"``): the whole grid for ``PINNED`` (the
  default configuration and the current composite, ``c46``);
* merge (``-m "not slow"``): also the whole grid for ``RECENT`` and the
  ``SEPARATING`` cases for every named config;
* full: every named config on the whole grid.
"""

from __future__ import annotations

import functools
import hashlib
import json
import platform
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import amm_optimizer as atm
from amm_optimizer import presets

ROOT = Path(__file__).resolve().parents[1]
NAMED_SNAPSHOTS = Path(__file__).parent / "characterisation.json"
BOX = (-5.0, 5.0)


# The presets, oldest first (``presets.get`` reads each one).
COMPOSITES: tuple[str, ...] = tuple(f"c{k}" for k in range(23, 47))
# The shipped configurations by snapshot name.
SHIPPED: dict[str, Callable[[], atm.ATMConfig]] = {
    "default": atm.ATMConfig,
    **{name: functools.partial(presets.get, name) for name in COMPOSITES},
    # ``ATMConfig.composite()``, first recorded as the ``refactor`` line's preset.
    "refactor_composite": atm.ATMConfig.composite,
}


def named_config(name: str) -> atm.ATMConfig:
    """The configuration of the named snapshot ``name``."""
    return SHIPPED[name]()


def text_of(config: atm.ATMConfig) -> str:
    """``config`` as a configuration text: its changes from the defaults, as
    ``variant.<field>=<value>`` items (the form the snapshots record)."""
    base = atm.ATMConfig().to_dict()
    items = []
    for key, value in config.to_dict()["variant"].items():
        if base["variant"][key] != value:
            if isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, (tuple, list)):
                value = ";".join(str(v) for v in value)
            items.append(f"variant.{key}={value}")
    for key, value in config.to_dict().items():
        if key != "variant" and not isinstance(value, dict) and base[key] != value:
            items.append(f"{key}={value}")
    return ",".join(items)


def _value(text: str, default: object) -> object:
    """One item's value, typed like the field's default."""
    if isinstance(default, bool):
        if text not in ("true", "false"):
            raise ValueError(f"not a bool: {text!r}")
        return text == "true"
    if isinstance(default, (int, float)):
        return type(default)(float(text)) if isinstance(default, float) else int(text)
    if isinstance(default, tuple):
        return tuple(float(v) for v in text.split(";"))
    return text


def parse(text: str) -> atm.ATMConfig:
    """The configuration of a config text (``text_of``'s form; a later item for the
    same field wins, as when a composite text adds its changes to its base's)."""
    data = atm.ATMConfig().to_dict()
    for item in filter(None, text.split(",")):
        key, value = item.split("=", 1)
        if key.startswith("variant."):
            name = key.removeprefix("variant.")
            data["variant"][name] = _value(value, data["variant"][name])
        else:
            data[key] = _value(value, data[key])
    return atm.ATMConfig.from_dict(data)


def _recorded_texts() -> dict[str, str]:
    """Each shipped configuration's text as recorded (``text_of`` for a new one)."""
    stored: dict[str, Any] = {}
    if NAMED_SNAPSHOTS.exists():
        stored = json.loads(NAMED_SNAPSHOTS.read_text(encoding="utf-8"))["configs"]
    return {
        name: stored[name]["config"] if name in stored else text_of(make())
        for name, make in SHIPPED.items()
    }


# The named snapshots and the config texts they were recorded with (provenance:
# ``text_of(named_config(name))`` says the same, test_presets.py).
NAMED: dict[str, str] = _recorded_texts()


@dataclass(frozen=True)
class Case:
    function: int
    dim: int
    multiplier: float
    instance: int = 1
    seed: int = 0

    @property
    def name(self) -> str:
        return (
            f"f{self.function}_d{self.dim}_{self.multiplier:g}x"
            f"_i{self.instance}_s{self.seed}"
        )

    @property
    def budget(self) -> int:
        return int(self.multiplier * self.dim)


# Separable (f1, f2), plateau (f7), smooth non-separable (f8, f10, f12: SLSQP's
# status-4 failures), rugged (f15, f17) and multi-basin (f21) functions; budgets
# on both sides of the prefix / hand-over boundary (100 x D) and of the SLSQP
# thresholds (30 x D, 200 evaluations).
GRID_FUNCTIONS = (1, 2, 7, 8, 10, 12, 15, 17, 21)
GRID_DIMS = (2, 5, 10, 20)
GRID_MULTIPLIERS = (10.0, 30.0, 100.0, 300.0, 1000.0)
GRID = [
    Case(f, d, m) for d in GRID_DIMS for m in GRID_MULTIPLIERS for f in GRID_FUNCTIONS
]
# D 40 (the cycles 20-26 routes target it) and 3000 x D (the growing populations).
GRID += [Case(f, 40, m) for m in (10.0, 300.0) for f in (1, 8, 10, 12, 21)]
GRID += [Case(f, 5, 3000.0) for f in (15, 17, 21)]
# D 3 (2026-10-21): ``basin_route`` with ``basin_descents`` (100-300 x D) and the
# ``small_engine`` route (10 x D) of c42 / c43 act at D 2-3 only.
GRID += [Case(f, 3, m) for f in (2, 12, 21) for m in (10.0, 100.0, 300.0)]
# f23 at 100 x D (2026-10-22): c46's low-curvature gate in the ES prefix opens on
# Katsuura's readings from D 10 (Batch 88).
GRID += [Case(23, d, 100.0) for d in (10, 20)]
# One case per route family (the in-process runner check; ``--cases fast``).
FAST = [
    Case(1, 2, 100.0),
    Case(10, 2, 30.0),
    Case(10, 20, 10.0),
    Case(2, 5, 30.0),
    Case(15, 10, 100.0),
    Case(8, 5, 300.0),
    Case(7, 5, 1000.0),
    Case(21, 5, 1000.0),
    Case(12, 10, 300.0),
    Case(10, 10, 1000.0),
    Case(10, 40, 10.0),
]
assert set(FAST) <= set(GRID)

# The test tiers. ``CURRENT`` is the newest composite, the default of ``minimize``.
CURRENT: str = COMPOSITES[-1]
PINNED: tuple[str, ...] = ("default", CURRENT)
RECENT: tuple[str, ...] = (
    *COMPOSITES[-3:],
    "refactor_composite",
)
# Each named config against the one it was built on (consecutive composites).
BASES: dict[str, str] = {
    "default": "c23",
    **{b: a for a, b in zip(COMPOSITES, COMPOSITES[1:], strict=False)},
    "refactor_composite": "default",
}
# Named configs that equal their base on every grid case (none of the shipped ones).
SAME_AS_BASE_ON_THE_GRID: set[str] = set()
# Twelve grid cases on which every other named config differs from its base (a greedy
# cover over the 2026-10-21 snapshots; ``test_the_separating_set_...`` re-checks it).
SEPARATING = [
    Case(21, 20, 300.0),
    Case(10, 2, 100.0),
    Case(12, 40, 10.0),
    Case(12, 10, 1000.0),
    Case(15, 20, 1000.0),
    Case(10, 10, 100.0),
    Case(10, 10, 300.0),
    Case(10, 2, 10.0),
    Case(10, 40, 300.0),
    Case(7, 10, 1000.0),
    Case(12, 5, 30.0),
    Case(23, 10, 100.0),
]
assert set(SEPARATING) <= set(GRID)


def environment() -> dict[str, str]:
    """What a bit-exact comparison depends on."""
    import scipy  # type: ignore[import-untyped]

    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
        "system": f"{platform.system()}-{platform.machine()}",
    }


_SUITES: dict[tuple[int, ...], Any] = {}


def _problem(case: Case) -> Any:
    import cocoex  # type: ignore[import-untyped]

    key = (case.instance,)
    if key not in _SUITES:
        _SUITES[key] = cocoex.Suite(
            "bbob",
            "year: 2019",
            f"dimensions: 2,3,5,10,20,40 instance_indices: {case.instance}",
        )
    return _SUITES[key].get_problem_by_function_dimension_instance(
        case.function, case.dim, case.instance
    )


def run(config: atm.ATMConfig, case: Case) -> dict[str, Any]:
    """Run ``case`` once and return its digests (the snapshot of one case)."""
    problem = _problem(case)
    points = hashlib.sha256()
    values = hashlib.sha256()
    count = 0

    def fun(x: np.ndarray) -> float:
        nonlocal count
        count += 1
        x = np.asarray(x, dtype=float)
        points.update(x.tobytes())
        value = float(problem(x))
        values.update(struct.pack("<d", value))
        return value

    starts = np.random.RandomState(10_000 + case.seed)
    x0 = BOX[0] + (BOX[1] - BOX[0]) * starts.rand(case.dim)
    try:
        result = atm.api.minimize(
            fun,
            x0,
            case.budget,
            rng=np.random.RandomState(case.seed),
            config=config,
            bounds=BOX,
        )
    finally:
        problem.free()
    return {
        "evaluations": count,
        "nfev": int(result.nfev),
        "fun": float(result.fun).hex(),
        "x_sha256": hashlib.sha256(np.asarray(result.x, float).tobytes()).hexdigest(),
        "points_sha256": points.hexdigest(),
        "values_sha256": values.hexdigest(),
    }
