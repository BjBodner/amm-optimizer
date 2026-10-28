"""The version is written once in the package and once in CITATION.cff; they agree.

``pyproject.toml`` reads it from ``src/amm_optimizer/__init__.py`` (``dynamic``).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import amm_optimizer as atm

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_reads_the_version_from_the_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" not in project["project"]
    assert "version" in project["project"]["dynamic"]
    source = project["tool"]["hatch"]["version"]["path"]
    assert (ROOT / source).resolve() == Path(atm.__file__).resolve()


def test_the_citation_version_is_the_package_version():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    found = re.findall(r"^version:\s*[\"']?([^\s\"']+)", text, re.M)
    assert found == [atm.__version__]
    assert re.search(r"^date-released:\s*\"?\d{4}-\d{2}-\d{2}", text, re.M)


def test_the_version_is_pep_440():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.?(a|b|rc|dev)\d+)?", atm.__version__)
