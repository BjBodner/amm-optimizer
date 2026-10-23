"""The version is written in two places; they must agree."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import amm_optimizer as atm

ROOT = Path(__file__).resolve().parents[1]


def test_the_package_version_is_the_project_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert atm.__version__ == project["project"]["version"]


def test_the_version_is_pep_440():
    assert re.fullmatch(r"\d+\.\d+\.\d+(\.?(a|b|rc|dev)\d+)?", atm.__version__)
