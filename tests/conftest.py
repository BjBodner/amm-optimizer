"""Test this checkout, and keep bit-exact snapshots to their own environment.

Checkout. The virtual environment's editable install points at the main checkout, so
in a worktree a bare ``pytest`` would otherwise import the main checkout's
``amm_optimizer``. Its ``src`` comes first here, in this process and in every subprocess
a test starts. ``tools/ci.py`` sets the same path for its steps (``step_env``).

Snapshots. Tests marked ``snapshot`` pin exact floating-point traces (the preset
traces, the characterisation grid, a few trajectory-dependent behaviours). They hold
only in the environment that recorded them: the same Python minor version, NumPy, SciPy
and operating system. Measured: Python 3.13 with the same NumPy and SciPy, Linux and
macOS all give different last bits. Elsewhere they are skipped, and the skip is
reported in the test summary (a silent green would hide that the guard is gone).
Everything else, the portable invariants, runs everywhere. ``AMM_SNAPSHOTS=1`` runs
them regardless.
"""

import json
import os
import platform
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parents[1] / "src")

sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [SRC, *filter(None, [os.environ.get("PYTHONPATH")])]
)

RECORDED = Path(__file__).with_name("characterisation.json")


def _minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def environment_mismatch() -> str | None:
    """Why this environment differs from the recorded one, or ``None`` if it matches."""
    if os.environ.get("AMM_SNAPSHOTS") == "1":
        return None
    meta = json.loads(RECORDED.read_text(encoding="utf-8"))["_meta"]
    try:
        import numpy
        import scipy
    except ImportError as error:  # pragma: no cover
        return f"cannot read the environment: {error}"
    here = {
        "python": _minor(platform.python_version()),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "system": f"{platform.system()}-{platform.machine()}",
    }
    want = {
        "python": _minor(meta["python"]),
        "numpy": meta["numpy"],
        "scipy": meta["scipy"],
        "system": meta["system"],
    }
    differ = [f"{k} {here[k]} (recorded {want[k]})" for k in want if here[k] != want[k]]
    return ", ".join(differ) or None


def pytest_collection_modifyitems(config, items):
    reason = environment_mismatch()
    config._snapshot_skipped = (0, reason)
    if reason is None:
        return
    skip = pytest.mark.skip(reason=f"snapshot recorded elsewhere: {reason}")
    count = 0
    for item in items:
        if "snapshot" in item.keywords:
            item.add_marker(skip)
            count += 1
    config._snapshot_skipped = (count, reason)


def pytest_terminal_summary(terminalreporter, config):
    count, reason = getattr(config, "_snapshot_skipped", (0, None))
    if reason is not None and count:
        terminalreporter.write_sep(
            "-",
            f"{count} snapshot tests skipped: this environment differs ({reason}). "
            "AMM_SNAPSHOTS=1 runs them anyway.",
        )
