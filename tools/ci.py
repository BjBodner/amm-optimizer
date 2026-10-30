r"""Run the project's checks on this machine, the same ones CI would run.

    python tools/ci.py                 # default tier: lint, format, types, unit
                                       #   tests, the pinned characterisation
    python tools/ci.py --tier merge    # merge tier: also harness tests, the recent
                                       #   composites and the separating set
    python tools/ci.py --tier full     # everything, with the coverage gate (``--full``)
    python tools/ci.py --list          # show the steps without running them
    python tools/ci.py --keep-going    # do not stop at the first failing step

The tiers are pytest markers (``pyproject.toml``; ``tests/characterisation_cases.py``):
default deselects ``slow`` and ``merge``, merge deselects ``slow``, full runs
everything.

It uses the interpreter that runs it (so use the project's virtual
environment, e.g. ``.venv\Scripts\python tools/ci.py``) and always runs from
the repository root, with that checkout's ``src`` first on ``PYTHONPATH`` (the
virtual environment's editable install points at the main checkout, so a
worktree would otherwise test the main checkout's ``amm_optimizer``). The exit code is
0 only if every step passed. Nothing
here talks to GitHub.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = ["src", "tests", "tools", "benchmarks", "examples"]


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]


TIERS = ("default", "merge", "full")
MARKERS = {
    "default": "not slow and not merge",
    "merge": "not slow",
}
# Branch coverage the full tier must reach, in percent. Measured on 2026-10-29
# (release/0.7.0, the full tier): 95.74%. The gate is floor(measured) - 0.5, below the
# measurement so a harmless change does not trip it.
COVERAGE_GATE = 94.5


def steps(tier: str | bool = "default") -> list[Step]:
    """The steps of a tier (``True`` / ``False`` are the old full / fast flags)."""
    if isinstance(tier, bool):
        tier = "full" if tier else "default"
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    py = sys.executable
    lint = [
        Step("lint (ruff check)", [py, "-m", "ruff", "check", *PATHS]),
        Step(
            "format (ruff format --check)",
            [py, "-m", "ruff", "format", "--check", *PATHS],
        ),
        Step("types (mypy --strict)", [py, "-m", "mypy"]),
    ]
    if tier == "full":
        tests = Step(
            f"tests, all, {COVERAGE_GATE:g}% coverage gate",
            [
                py,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--cov=amm_optimizer",
                "--cov-branch",
                f"--cov-fail-under={COVERAGE_GATE:g}",
            ],
        )
    else:
        tests = Step(
            f"tests, {tier} tier",
            [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", MARKERS[tier]],
        )
    return [*lint, tests]


def step_env() -> dict[str, str]:
    """The environment of every step: this checkout's ``src`` first on PYTHONPATH."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), *filter(None, [env.get("PYTHONPATH")])]
    )
    return env


def run(selected: list[Step], keep_going: bool) -> int:
    results: list[tuple[str, bool, float]] = []
    env = step_env()
    for step in selected:
        print(f"\n=== {step.name}\n$ {' '.join(step.command)}", flush=True)
        started = time.perf_counter()
        code = subprocess.run(step.command, cwd=ROOT, env=env).returncode
        results.append((step.name, code == 0, time.perf_counter() - started))
        if code != 0 and not keep_going:
            break

    print("\n=== summary")
    for name, ok, seconds in results:
        print(f"{'PASS' if ok else 'FAIL'}  {seconds:7.1f}s  {name}")
    skipped = len(selected) - len(results)
    if skipped:
        print(f"      (not run after the first failure: {skipped} step(s))")
    passed = all(ok for _, ok, _ in results) and not skipped
    print("\nALL CHECKS PASSED" if passed else "\nCHECKS FAILED")
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tier", choices=TIERS, default="default")
    parser.add_argument("--full", action="store_true", help="the same as --tier full")
    parser.add_argument("--list", action="store_true", help="print the steps and exit")
    parser.add_argument("--keep-going", action="store_true", help="run every step")
    args = parser.parse_args()
    selected = steps("full" if args.full else args.tier)
    if args.list:
        for step in selected:
            print(f"{step.name}: {' '.join(step.command)}")
        return 0
    return run(selected, args.keep_going)


if __name__ == "__main__":
    raise SystemExit(main())
