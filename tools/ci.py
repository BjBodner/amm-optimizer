r"""Run the project's checks on this machine, the same ones CI would run.

    python tools/ci.py                 # research tier (default): lint, format, types,
                                       #   unit tests, the pinned characterisation
    python tools/ci.py --tier merge    # merge tier: also harness tests, the recent
                                       #   composites and the separating set
    python tools/ci.py --tier full     # everything, 100% coverage gate (``--full``)
    python tools/ci.py --list          # show the steps without running them
    python tools/ci.py --keep-going    # do not stop at the first failing step
    python tools/ci.py --research-check  # only the research-tooling check (< 1 s)

The tiers are pytest markers (``pyproject.toml``; ``tests/characterisation_cases.py``,
``docs/TRIM_REVIEW_2026-10-21.md``): research deselects ``slow`` and ``merge``, merge
deselects ``slow``, full runs everything.

Every tier also runs the research-tooling check (``docs/TRIM_REVIEW_2026-10-21.md``,
"Rules to keep it from growing back"):
* every script at the top level of ``docs/research/audit_2026-10-17/`` is an entry
  point (``AUDIT_ENTRY_POINTS``), is imported by one or by the live research code
  (``RESEARCH_IMPORTERS``), or is listed in ``archive/README.md``; every archived
  script has a line there;
* no folder in the common git dir's ``worktrees/`` lacks its ``gitdir`` file (a
  half-removed worktree makes every commit warn and blocks ``git worktree prune``).

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
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = ["src", "tests", "tools", "benchmarks"]
AUDIT = ROOT / "docs" / "research" / "audit_2026-10-17"
# The audit scripts run from the command line (CYCLE.md, RESULTS.md). A new analysis
# goes into benchmarks/analysis.py or a batch folder, not into a new script here.
AUDIT_ENTRY_POINTS = (
    "screen_v5",
    "fresh_claims",
    "c23_screen",
    "cell_todo",
    "goal_ledger",
    "field_plots",
    "field3000_plots",
    "bbob_ecdf_plots",
    "archive_sota",
    "cycle58_59_plots",
    "cycle60_62_plots",
    "b69_plots",
)
# Live research folders (next to the audit folder) whose imports keep a script live.
RESEARCH_IMPORTERS = ("parked", "router", "toys", "batches")


@dataclass(frozen=True)
class Step:
    name: str
    command: list[str]


TIERS = ("research", "merge", "full")
MARKERS = {"research": "not slow and not merge", "merge": "not slow"}


def steps(tier: str | bool = "research") -> list[Step]:
    """The steps of a tier (``True`` / ``False`` are the old full / fast flags)."""
    if isinstance(tier, bool):
        tier = "full" if tier else "research"
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
    py = sys.executable
    research = Step(
        "research tooling (audit scripts reachable, worktree admin folders)",
        [py, "tools/ci.py", "--research-check"],
    )
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
            "tests, all, 100% coverage gate",
            [
                py,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "--cov=amm_optimizer",
                "--cov-branch",
                "--cov-fail-under=100",
            ],
        )
    else:
        tests = Step(
            f"tests, {tier} tier",
            [py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", MARKERS[tier]],
        )
    return [*lint, research, tests]


IMPORT = re.compile(r"^[ \t]*(?:from[ \t]+(\w+)|import[ \t]+([\w., \t]+))", re.M)


def _imports(path: Path) -> set[str]:
    """The top-level module names a script imports, at any indentation.

    A line scan, not a parse: about 50 files are read per run, and an import that
    sits in a docstring only makes the check more lenient.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return set()
    out: set[str] = set()
    for frm, names in IMPORT.findall(text):
        if frm:
            out.add(frm)
        else:
            out.update(
                n.split()[0].split(".")[0] for n in names.split(",") if n.strip()
            )
    return out


def _archived(readme: Path) -> set[str]:
    """The scripts of archive/README.md's "Scripts" table: rows ``| `name.py` |``."""
    if not readme.exists():
        return set()
    text = readme.read_text(encoding="utf-8")
    scripts = re.split(r"^## ", text, flags=re.M)
    section = next((s for s in scripts if s.startswith("Scripts")), "")
    return set(re.findall(r"^\| `([\w.]+)\.py` \|", section, flags=re.M))


def unreachable_audit_scripts(
    audit: Path = AUDIT,
    entry_points: tuple[str, ...] = AUDIT_ENTRY_POINTS,
    importers: tuple[str, ...] = RESEARCH_IMPORTERS,
) -> list[str]:
    """Problems: audit scripts no entry point reaches, archived scripts with no line."""
    if not audit.is_dir():
        return []  # the public repository has no research archive
    scripts = {p.stem: p for p in audit.glob("*.py")}
    archive = audit / "archive"
    listed = _archived(archive / "README.md")
    problems = [
        f"{audit.name}: entry point {name}.py does not exist (tools/ci.py)"
        for name in entry_points
        if name not in scripts
    ]
    todo = [scripts[name] for name in entry_points if name in scripts]
    todo += [p for d in importers for p in sorted((audit.parent / d).rglob("*.py"))]
    reached = {p.stem for p in todo if p.parent == audit}
    seen: set[Path] = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _imports(path) & scripts.keys():
            if name not in reached:
                reached.add(name)
                todo.append(scripts[name])
    problems += [
        f"{audit.name}/{name}.py: not an entry point (tools/ci.py AUDIT_ENTRY_POINTS), "
        "not imported by one and not in archive/README.md; move it to archive/ with "
        "a README line, or into benchmarks/analysis.py"
        for name in sorted(scripts.keys() - reached - listed)
    ]
    problems += [
        f"{audit.name}/archive/{p.name}: no line in archive/README.md's Scripts table"
        for p in sorted(archive.glob("*.py"))
        if p.stem not in listed
    ]
    return problems


def git_common_dir(root: Path = ROOT) -> Path | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None  # not a git checkout: nothing to check
    common = Path(done.stdout.strip())
    return common if common.is_absolute() else (root / common).resolve()


def stale_worktree_admin_dirs(common: Path | None) -> list[str]:
    """Folders in ``<common git dir>/worktrees`` without their ``gitdir`` file."""
    admin = common / "worktrees" if common else None
    if admin is None or not admin.is_dir():
        return []
    return [
        f"{d} has no gitdir file: a half-removed worktree. With no git process "
        f'running: attrib -R "{d}" /S /D, then git worktree prune -v '
        "(docs/TRIM_REVIEW_2026-10-21.md, Phase 0 item 1)"
        for d in sorted(admin.iterdir())
        if d.is_dir() and not (d / "gitdir").is_file()
    ]


def research_check() -> int:
    problems = unreachable_audit_scripts() + stale_worktree_admin_dirs(git_common_dir())
    for problem in problems:
        print(f"FAIL  {problem}")
    if not problems:
        print("research tooling: every audit script is reachable; worktrees are clean")
    return 1 if problems else 0


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
    parser.add_argument("--tier", choices=TIERS, default="research")
    parser.add_argument("--full", action="store_true", help="the same as --tier full")
    parser.add_argument("--list", action="store_true", help="print the steps and exit")
    parser.add_argument("--keep-going", action="store_true", help="run every step")
    parser.add_argument(
        "--research-check", action="store_true", help="run only the research check"
    )
    args = parser.parse_args()
    if args.research_check:
        return research_check()

    selected = steps("full" if args.full else args.tier)
    if args.list:
        for step in selected:
            print(f"{step.name}: {' '.join(step.command)}")
        return 0
    return run(selected, args.keep_going)


if __name__ == "__main__":
    raise SystemExit(main())
