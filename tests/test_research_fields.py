"""The deprecation of the research-only ``Variant`` fields and values (0.6.0).

``config.RESEARCH_FIELDS`` and ``config.RESEARCH_CHOICES`` are measured by
``tools/research_fields.py``: the fields and values that no shipped configuration
uses. Setting one emits a ``FutureWarning`` that names it and says it is removed in
0.7.0; constructing or running a shipped configuration emits none.
"""

import dataclasses
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import amm_optimizer as amm
from amm_optimizer import config as config_module
from amm_optimizer import presets
from amm_optimizer.config import (
    CHOICES,
    RESEARCH_CHOICES,
    RESEARCH_FIELDS,
    ATMConfig,
    Variant,
)

ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools" / "research_fields.py"
    spec = importlib.util.spec_from_file_location("research_fields_tool", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = load_tool()
DEFAULTS = {f.name: f.default for f in dataclasses.fields(Variant)}


def test_the_committed_lists_are_the_measured_ones():
    fields, choices = tool.measure()
    assert fields == RESEARCH_FIELDS
    assert choices == RESEARCH_CHOICES
    text = Path(config_module.__file__).read_text(encoding="utf8")
    assert tool.current_block(text) == tool.render(fields, choices)


def test_every_string_field_has_its_allowed_values_default_first():
    strings = {n for n, d in DEFAULTS.items() if isinstance(d, str)}
    assert strings == set(CHOICES)
    for name, allowed in CHOICES.items():
        assert allowed[0] == DEFAULTS[name], name
    for name, values in RESEARCH_CHOICES.items():
        assert name not in RESEARCH_FIELDS
        assert set(values) <= set(CHOICES[name][1:]), name


def test_the_changelog_lists_every_deprecation():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf8")
    # the section that announces them: Unreleased until the release, then the version's
    sections = [part.split("\n## [", 1)[0] for part in text.split("\n## [")[1:]]
    section = next(s for s in sections if "### Deprecated" in s and "0.7.0" in s)
    deprecated = section.split("### Deprecated", 1)[1].split("\n### ", 1)[0]
    entry = tool.changelog(*tool.measure())
    assert " ".join(entry.split()) in " ".join(deprecated.split())


@pytest.mark.parametrize("name", list(tool.shipped_configs()))
def test_shipped_configurations_construct_without_a_warning(name):
    config = tool.shipped_configs()[name]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Variant(**config.variant.to_dict())
        ATMConfig.from_dict(config.to_dict())
        dataclasses.replace(config.variant)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"anytime": True},
        {"config": ATMConfig()},
        {"config": ATMConfig.composite()},
    ],
)
def test_shipped_runs_emit_no_future_warning(kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = amm.minimize(
            lambda x: float(np.sum(x**2)),
            bounds=[(-5.0, 5.0)] * 3,
            budget=120,
            seed=0,
            **kwargs,
        )
    assert np.isfinite(result.fun)


# Bases a research value is tried on: some options need others (a success-rate
# controller, a covariance, the hand-over, ...), which the shipped presets set.
BASES = [
    presets.get("c46").variant.to_dict(),
    presets.get("c23").variant.to_dict(),
    ATMConfig.composite().variant.to_dict(),
    {"amplitude_control": "success_rate", "covariance": "rank_mu", "bounds": "fold"},
    {"amplitude_control": "success_rate", "cma_handoff": 0.2, "bounds": "fold"},
    {},
]


def candidates(name):
    default = DEFAULTS[name]
    if isinstance(default, bool):
        return [not default]
    if isinstance(default, str):
        return list(CHOICES[name][1:])
    if isinstance(default, tuple):
        return [tuple(0.5 * v for v in default), tuple(0.9 * v for v in default)]
    numbers = [default + 1, default * 2, default / 2, 0.5, 1, 2, 3, 20]
    if isinstance(default, int):
        numbers = [int(v) for v in numbers]
    return [v for v in numbers if v != default]


def construct(name, value):
    """``Variant`` with ``name=value`` on the first base that accepts it, and the
    research warnings that construction emitted (``None`` if no base accepts it)."""
    for base in BASES:
        data = {**base, name: value}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                Variant(**data)
            except ValueError:
                continue
        return [w for w in caught if issubclass(w.category, FutureWarning)]
    return None


def test_every_research_field_warns_once_naming_itself():
    unreached = []
    for name in RESEARCH_FIELDS:
        for value in candidates(name):
            caught = construct(name, value)
            if caught is not None:
                break
        else:
            unreached.append(name)
            continue
        mine = [w for w in caught if f"Variant.{name} " in str(w.message)]
        assert len(mine) == 1, (name, [str(w.message) for w in caught])
        assert "removed in 0.7.0" in str(mine[0].message)
        assert mine[0].filename == __file__  # points at the caller
    assert not unreached


@pytest.mark.parametrize(
    ("name", "value"),
    [(n, v) for n, values in RESEARCH_CHOICES.items() for v in values],
)
def test_every_research_value_warns_naming_it(name, value):
    caught = construct(name, value)
    assert caught is not None, "no base accepts it"
    messages = [str(w.message) for w in caught]
    mine = [m for m in messages if f"Variant.{name}={value!r}" in m]
    assert len(mine) == 1 and "removed in 0.7.0" in mine[0], messages


def test_research_settings_still_work_with_the_warning():
    with pytest.warns(FutureWarning, match=r"box_steps .*removed in 0\.7\.0"):
        config = presets.with_changes("c46", box_steps=True)
    assert config.variant.box_steps is True
    with pytest.warns(FutureWarning, match="removed in 0.7.0"):
        again = ATMConfig.from_dict(config.to_dict())
    assert again == config and again.config_hash() == config.config_hash()


def test_a_shipped_value_of_a_mixed_field_does_not_warn():
    shipped_values = {
        name: [v for v in CHOICES[name][1:] if v not in values]
        for name, values in RESEARCH_CHOICES.items()
    }
    pairs = [(n, v) for n, vs in shipped_values.items() for v in vs]
    assert pairs
    for name, value in pairs:
        caught = construct(name, value)
        if caught is not None:
            assert not [w for w in caught if f"Variant.{name}" in str(w.message)]
