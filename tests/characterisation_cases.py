"""Bit-exact characterisation of the shipped configurations on COCO bbob problems.

Shared by ``tests/test_characterisation.py`` and ``tools/record_characterisation.py``.
Each case runs ``amm_optimizer.api.minimize`` once on a bbob problem (via ``cocoex``)
with the box ``[-5, 5]`` and records SHA-256 digests of every evaluated point and
value, exactly (float64 bytes, no rounding). A refactor must reproduce them bit for
bit on the recording platform; the tests skip when NumPy, SciPy or the platform
differ from the recording (``_meta``).

Two sets:

* ``NAMED``: the shipped configurations, ``ATMConfig()`` (``"default"``), the presets
  ``c23``-``c46`` and ``ATMConfig.composite()`` (``"refactor_composite"``), on a grid
  of functions, dimensions and budgets (``GRID``), in ``tests/characterisation.json``.
  Each entry keeps the configuration text it was first recorded with.
* ``arm_configs()`` (research): every distinct configuration of the recorded research
  arms (``benchmarks/arms/*.json``, keyed by ``override_key``) on four probe problems
  (``ARM_PROBES``), in ``tests/characterisation_arms.json``. It, and everything else
  that reads ``benchmarks/arms`` or ``benchmarks/quick_scan.py`` (``parse``,
  ``composite``, ``arm``), needs the research data (``RESEARCH_DATA_MISSING``); the
  tests that use it carry the ``research`` marker and skip without it.

The test tiers (``tools/ci.py --tier``):

* default (``-m "not slow and not merge and not research"``): the whole grid for
  ``PINNED`` (the default configuration and the current composite, ``c46``);
* merge (``-m "not slow"``): also the whole grid for ``RECENT`` and the
  ``SEPARATING`` cases for every named config;
* full: every named config on the whole grid and every recorded arm.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import platform
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import amm_optimizer as atm
from amm_optimizer import presets

ROOT = Path(__file__).resolve().parents[1]
ARMS = ROOT / "benchmarks" / "arms"
NAMED_SNAPSHOTS = Path(__file__).parent / "characterisation.json"
ARM_SNAPSHOTS = Path(__file__).parent / "characterisation_arms.json"
QUICK_SCAN = ROOT / "benchmarks" / "quick_scan.py"
BOX = (-5.0, 5.0)

# The research data: the recorded arms, the benchmark harness that parses their
# config texts, and the arm snapshots. They leave this repository with the research
# code; the tests that read them skip when they are gone.
_RESEARCH_PATHS = (ARMS, QUICK_SCAN, ARM_SNAPSHOTS)
RESEARCH_DATA_MISSING: str | None = (
    ", ".join(p.relative_to(ROOT).as_posix() for p in _RESEARCH_PATHS if not p.exists())
    or None
)


def _load_quick_scan() -> Any:
    """``benchmarks/quick_scan.py`` as a module (its ``parse_config`` reads arms)."""
    if "quick_scan" in sys.modules:
        return sys.modules["quick_scan"]
    spec = importlib.util.spec_from_file_location("quick_scan", QUICK_SCAN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["quick_scan"] = module
    spec.loader.exec_module(module)
    return module


def composite(name: str) -> str:
    """The config text of ``benchmarks/arms/<name>_config.txt``."""
    return (ARMS / f"{name}_config.txt").read_text(encoding="utf-8").strip()


def arm(file: str, name: str) -> str:
    """The config text of arm ``name`` in ``benchmarks/arms/<file>.json``."""
    arms = json.loads((ARMS / f"{file}.json").read_text(encoding="utf-8"))
    return str(arms[name]["config"])


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
    """``config`` as a ``quick_scan --config`` text of its changes from the defaults."""
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


def _recorded_texts() -> dict[str, str]:
    """Each shipped configuration's text as recorded (``text_of`` for a new one)."""
    stored: dict[str, Any] = {}
    if NAMED_SNAPSHOTS.exists():
        stored = json.loads(NAMED_SNAPSHOTS.read_text(encoding="utf-8"))["configs"]
    return {
        name: stored[name]["config"] if name in stored else text_of(make())
        for name, make in SHIPPED.items()
    }


# The named snapshots and the config texts they were recorded with (provenance: with
# the research data, ``parse(NAMED[name]) == named_config(name)``, test_presets.py).
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
# Probes for every recorded arm (prefix and hand-over, D 2 to 10).
ARM_PROBES = [
    Case(8, 5, 300.0, instance=2),
    Case(15, 10, 1000.0, instance=2),
    Case(21, 2, 100.0, instance=2),
    Case(10, 10, 30.0, instance=2),
]


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


def parse(text: str) -> atm.ATMConfig:
    """A ``quick_scan --config`` text as a configuration (needs the research data)."""
    return _load_quick_scan().parse_config(text)  # type: ignore[no-any-return]


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


def _flat(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in tree.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = list(value) if isinstance(value, tuple) else value
    return out


def override_key(config: atm.ATMConfig) -> str:
    """A fingerprint of the fields that differ from ``ATMConfig()``.

    Unlike ``ATMConfig.config_hash`` it does not change when a field is added with
    its default, so the arm snapshots survive new ``Variant`` options.
    """
    base = _flat(atm.ATMConfig().to_dict())
    changed = {k: v for k, v in _flat(config.to_dict()).items() if base.get(k) != v}
    text = json.dumps(changed, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# Recorded arm configs that use an option the 2026-10-19 merge removed (allocation,
# early_accept; refactor 40222d9). Their contract snapshots stay in the arms file;
# the code at ``pre-merge-2026-10-19/research`` reproduces them.
REMOVED_BY_MERGE = {
    "a407991e5885678a": "wave2b:w2_fixed12_credit",
    "c3f2b2404f877823": "wave2c:w2_early4",
    "2f2f6ac7b0f92362": "wave2c:w2_early6",
    "d7a6f8ddd15ec660": "wave2c:w2_early8",
    "f122eeef1cedd006": "wave2c:w2_early4_p24",
    "0fce88fa32645099": "wave2c:w2_early4_p48",
    "bed36cc9c529477a": "wave2c:w2_early4_p24_t15",
    "4b9b1b197414585c": "wave2c:w2_early4_p24_credit",
    "3ea524adb4b118ac": "wave2c:w2_early4_p24_eta3",
    "470862582c296e18": "wave2d:w2_e4_c100",
    "71d2e69f62f60099": "wave2d:w2_e8_c100",
    "e1e3983365fcf149": "wave2d:w2_e6_c75",
    "c58917f38d3110c7": "wave2d:w2_e6_c100_p16",
    "b6eabc4536f0127c": "wave3:w3_es30_early6",
    # Dead code 4 (2026-10-21): ``race`` and ``pop_first_share`` (Batches 20-22); the
    # code at ``pre-deadcode-2026-10-21`` reproduces them
    "b86fe3e5508023a6": "audit_b20:race",
    "de1b8134feacab16": "audit_b20:race5",
    "0f8aebd8e5012464": "audit_b21:racelate",
    "e1a903d6e02df013": "audit_b22:pf30",
    "c9f0834c08dd42c2": "audit_b22:pf50",
    # Dead code 7: the arm-only routes qn18, qn21, qn24-qn26, qn28, qn34, qn36, qn37
    "f13f1a2774040220": "audit_b58:capgate_hi",
    "5a02b49e7fa36fc0": "audit_b59:fract_hi",
    "3e7a4b8446b7f1e7": "audit_b60:ridge_hi",
    "fc04e41860c42ae2": "audit_b61:fract2_hi",
    "3c1f1f41afd81625": "audit_b63:sepr",
    "c38389dd74af90c8": "audit_fresh246:c36_s",
    "5601ebd24c6e7cfc": "audit_fresh211:c35_a1",
    "49d5f6745dd40a8c": "audit_b65:lowipop",
    "56f66ef05a575d6e": "audit_b65:kinklow",
    "4ce873697e28f442": "audit_b65:tryd5",
    # Dead code 8: the research controls additive_control and basin_control
    "b635e95b51980ba3": "audit_b66:ctrl",
    "3e16437c76a34d8f": "audit_b67:galc",
    # Dead code 6: the arm-only prefix kinds (es_kind lq / slsqp / slsqpb, nm_stage
    # handover / qnprefix30gd) and pop_surrogate
    "d5ba4e4454d4f027": "audit_b17:nm2h",
    "89fce43f2a32ac85": "audit_b17:nm5h",
    "693a34cdcb7dd9cc": "audit_b29:lqloc",
    "c90c35367530ec2d": "audit_b29:lqpop",
    "aaee54edf15cfca9": "audit_b29:lqpre",
    "4860d9df759085b7": "audit_b30:loc2",
    "2d041cee72a7bbb9": "audit_b30:pop2",
    "a1fc2462baaf8610": "audit_b30:pop2b",
    "85b737ecc168e1b1": "audit_b30:sep2",
    "95a54d8c17979a5d": "audit_b31:qnpre",
    "9b26446412cc8f90": "audit_b32:qnb",
    "7fad78fac8845b2c": "audit_b58:leangd_lo",
    # Dead code 5: the hand-over extras of Batches 12-27 (model gate and weights,
    # mirror, ties, pin stick, hand-over sigma, warm shape, switches, reflect bounds)
    "d12a4eaa3805b420": "audit_b12:m_ratio",
    "40f14a76a02b3443": "audit_b12:m_ratio_w",
    "82dadd93b2e25ec5": "audit_b12:mirror",
    "472aa3f206420121": "audit_b12:ties",
    "cd6749b98a8b7c49": "audit_b13:handover",
    "255c298081759663": "audit_b13:pop2_protect",
    "c674e3ad29bf4cce": "audit_b13:pop2_reflect",
    "6873af07482b04b3": "audit_b13:popb_protect",
    "6adb074921d725d9": "audit_b14b:blend",
    "39440f2eb63289b8": "audit_b14b:blend_box",
    "cc05c1043424c600": "audit_b14b:nomodel1000",
    "e4a984fdef8c5e8c": "audit_b19:nipop",
    "571d751be4983fd9": "audit_b19:ties20",
    "60376c2f0b3a5007": "audit_b23:swq",
    "70040cfa8b40ec45": "audit_b23:swq5",
    "5ebf8c6d55207249": "audit_b24:plat20",
    "dde480014971518d": "audit_b27:decay2",
    "2c18814126e6fe48": "audit_b27:decay3",
}


def arm_configs() -> dict[str, dict[str, Any]]:
    """Every distinct ATM config of the recorded arms, keyed by ``override_key``.

    Each entry holds the config text and the arms (``file:name``) that use it. Arms
    whose config no longer parses (one, by design: ``wave2a:w2_maxw3``) are left out.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(ARMS.glob("*.json")):
        arms = json.loads(path.read_text(encoding="utf-8"))
        for name, spec in arms.items():
            if not isinstance(spec, dict) or spec.get("solver", "atm") != "atm":
                continue
            text = spec.get("config", "")
            try:
                key = override_key(parse(text))
            except SystemExit:
                continue
            entry = out.setdefault(key, {"config": text, "arms": []})
            entry["arms"].append(f"{path.stem}:{name}")
    return out
