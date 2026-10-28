"""The scripts in ``examples/`` run as written and print their results (WP5.2).

Each takes its budget as an optional first argument; the tests pass a small one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
# (script, small budget, a line of what it prints)
CASES = [
    ("quickstart.py", 300, "repeat the run with seed=0"),
    ("scipy_migration.py", 300, "AMM   success=True"),
    ("early_stopping.py", 300, "time limit: The callback raised StopIteration."),
]


def test_every_example_is_tested_and_short():
    assert sorted(p.name for p in EXAMPLES.glob("*.py")) == sorted(c[0] for c in CASES)
    for path in EXAMPLES.glob("*.py"):
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 45, path.name


@pytest.mark.parametrize(("script", "budget", "expected"), CASES)
def test_an_example_runs_and_prints_its_result(script, budget, expected):
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / script), str(budget)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert expected in done.stdout
    assert f"{budget} evaluations" in done.stdout or f"nfev={budget}" in done.stdout
