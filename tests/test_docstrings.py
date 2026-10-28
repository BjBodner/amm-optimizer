"""The public objects of the user-facing modules have numpydoc docstrings whose
examples run (WP4.5, this branch's part)."""

from __future__ import annotations

import doctest
import inspect

import pytest

import amm_optimizer as amm
from amm_optimizer import default, presets, result, search

MODULES = [result, presets, default]


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_the_examples_in_the_docstrings_run(module):
    outcome = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert outcome.attempted > 0
    assert outcome.failed == 0


@pytest.mark.parametrize(
    ("obj", "sections"),
    [
        (amm.OptimizeResult, ["Attributes", "Examples"]),
        (amm.minimize, ["Parameters", "Returns", "Raises", "Examples"]),
        (presets.names, ["Parameters", "Returns", "Examples"]),
        (presets.get, ["Parameters", "Returns", "Raises", "Examples"]),
        (presets.with_changes, ["Parameters", "Returns", "Examples"]),
        (search.Search, ["Parameters"]),
        (search.Search.run, ["Parameters", "Returns"]),
        (search.SearchOutcome, ["Attributes"]),
    ],
    ids=lambda o: getattr(o, "__qualname__", ""),
)
def test_a_public_object_documents_its_sections(obj, sections):
    text = inspect.getdoc(obj) or ""
    for section in sections:
        assert f"\n{section}\n{'-' * len(section)}\n" in text, section


def test_every_field_of_the_result_is_documented():
    text = inspect.getdoc(amm.OptimizeResult) or ""
    r = amm.minimize(lambda x: float(x @ x), bounds=[(-1, 1)] * 2, budget=20, seed=0)
    for field in r:
        assert f"\n{field} : " in text, field


def test_every_new_argument_of_minimize_is_documented():
    text = inspect.getdoc(amm.minimize) or ""
    for name in inspect.signature(amm.minimize).parameters:
        assert f"\n{name} : " in text, name
