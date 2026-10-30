"""The ``Variant`` fields removed in 0.7.0 keep every recorded ``config_hash`` valid.

``config.REMOVED_FIELDS`` maps each removed field to the default it had in 0.5.0 and
0.6.0. ``ATMConfig.config_hash`` hashes it at that default, and ``from_dict`` accepts
it only there; any other value raises a ``ValueError`` naming the release.
``tests/test_shipped_hashes.py`` records the hashes themselves.
"""

import dataclasses
import json
import re
from pathlib import Path

import pytest

from amm_optimizer import config as config_module
from amm_optimizer import presets
from amm_optimizer.config import (
    CHOICES,
    HASH_IF_SET,
    REMOVED_CHOICES,
    REMOVED_FIELDS,
    REMOVED_IN,
    ATMConfig,
    Variant,
)
from test_shipped_hashes import RECORDED, shipped

FIELDS = {f.name: f.default for f in dataclasses.fields(Variant)}
ROOT = Path(__file__).resolve().parents[1]


def test_removed_fields_are_gone_from_variant():
    assert REMOVED_IN == "0.7.0"
    assert not set(REMOVED_FIELDS) & set(FIELDS)
    for name, values in REMOVED_CHOICES.items():
        assert name in CHOICES, name
        assert not set(values) & set(CHOICES[name]), name


@pytest.mark.parametrize("name", list(RECORDED))
def test_removing_any_field_at_its_default_keeps_the_hash(name, monkeypatch):
    """The mechanism itself: for every field a shipped configuration leaves at its
    default, dropping it and listing it as removed gives the same hashed text."""
    config = shipped(name)
    data = config.to_dict()
    want = config_module._hash_payload(data)
    for field, default in FIELDS.items():
        if data["variant"][field] != default or field in HASH_IF_SET:
            continue
        trimmed = {**data, "variant": dict(data["variant"])}
        del trimmed["variant"][field]
        monkeypatch.setattr(
            config_module, "REMOVED_FIELDS", {**REMOVED_FIELDS, field: default}
        )
        assert config_module._hash_payload(trimmed) == want, field
    monkeypatch.setattr(config_module, "REMOVED_FIELDS", REMOVED_FIELDS)


@pytest.mark.parametrize("name", list(REMOVED_FIELDS))
def test_a_removed_field_loads_only_at_its_old_default(name):
    default = REMOVED_FIELDS[name]
    data = presets.get("c46").to_dict()
    old = {**data, "variant": {**data["variant"], name: default}}
    old = json.loads(json.dumps(old))
    assert ATMConfig.from_dict(old) == presets.get("c46")
    assert Variant.from_dict({name: default}) == Variant()
    other = (not default) if isinstance(default, bool) else "changed"
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        other = default + 1
    if isinstance(default, tuple):
        other = tuple(2 * v for v in default)
    with pytest.raises(ValueError, match=rf"Variant\.{name} was removed in 0\.7\.0"):
        Variant.from_dict({name: other})
    with pytest.raises(ValueError, match=rf"Variant\.{name} was removed in 0\.7\.0"):
        ATMConfig.from_dict({"variant": {name: other}})


@pytest.mark.parametrize(
    ("name", "value"), [(n, v) for n, vs in REMOVED_CHOICES.items() for v in vs]
)
def test_a_removed_value_names_the_release(name, value):
    with pytest.raises(ValueError, match=rf"{name}={value!r} was removed in 0\.7\.0"):
        Variant.from_dict({name: value})


def test_a_dictionary_of_the_last_release_loads():
    """``to_dict`` of 0.6.0 had every removed field; such a dictionary still loads
    and hashes as recorded."""
    for name, want in RECORDED.items():
        data = shipped(name).to_dict()
        data["variant"].update(REMOVED_FIELDS)
        assert ATMConfig.from_dict(data).config_hash() == want, name


def _deprecated_in_060():
    """(fields, {field: values}) of the 0.6.0 CHANGELOG's Deprecated entry."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf8")
    section = text.split("\n## [0.6.0]", 1)[1].split("\n## [", 1)[0]
    entry = " ".join(section.split("### Deprecated", 1)[1].split("\n### ")[0].split())
    listed = entry.split("Fields: ", 1)[1].split(". Values:")[0]
    fields = re.findall(r"`(\w+)`", listed)
    values: dict[str, list[str]] = {}
    for part in entry.split("Values: ", 1)[1].rstrip(".").split("; "):
        name, *vals = re.findall(r"`\"?(\w+)\"?`", part)
        values[name] = vals
    return fields, values


def test_every_option_that_060_deprecated_is_removed():
    fields, values = _deprecated_in_060()
    assert len(fields) == 92 and sum(map(len, values.values())) == 29
    assert set(fields) == set(REMOVED_FIELDS)
    for name, vals in values.items():
        assert set(vals) <= set(REMOVED_CHOICES[name]), name
    # the values removed with their fields' routes, beyond the 0.6.0 list
    extra = {
        name: set(vals) - set(values.get(name, ()))
        for name, vals in REMOVED_CHOICES.items()
    }
    assert {name: vals for name, vals in extra.items() if vals} == {}


def test_with_changes_names_the_release_for_a_removed_field():
    assert presets.with_changes("c46", sr_eta=1.5) == presets.get("c46")
    with pytest.raises(ValueError, match=r"Variant\.sr_eta was removed in 0\.7\.0"):
        presets.with_changes("c46", sr_eta=2.0)
