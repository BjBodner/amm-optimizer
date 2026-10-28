"""Every ``docs/...`` and ``benchmarks/...`` path named in the repository exists.

The Markdown files, the package, the tests and the tools name files by their path from
the repository root. A path that names nothing (a document kept only in the research
archive, a script that was renamed) is a dead end for a reader, so each one must exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# A path from the root: not part of a URL or of a longer path (``main/docs/x``), and
# ending in a name character (so a closing period or backquote is not part of it).
PATH = re.compile(r"(?<![\w./-])((?:docs|benchmarks)/[\w./-]*\w)")
SKIP_DIRS = {".git", ".venv", "venv", "build", "dist", "node_modules", "__pycache__"}
# Known dead ends in files this branch does not edit. Fix the reference, then delete
# its entry here.
PENDING: set[tuple[str, str]] = set()


def scanned_files() -> list[Path]:
    files = [*ROOT.glob("*.md"), *(ROOT / ".github").rglob("*.md")]
    files += (ROOT / "docs").rglob("*.md")
    for folder in ("src", "tests", "tools"):
        files += (ROOT / folder).rglob("*.py")
    here = Path(__file__).resolve()  # its examples and PENDING name missing paths
    return sorted(
        f
        for f in files
        if not SKIP_DIRS & set(f.relative_to(ROOT).parts[:-1]) and f.resolve() != here
    )


def named_paths(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    return [
        (number, match)
        for number, line in enumerate(text.splitlines(), 1)
        for match in PATH.findall(line)
    ]


def test_the_scan_sees_the_files_that_name_paths():
    names = {f.relative_to(ROOT).as_posix() for f in scanned_files()}
    assert {"README.md", "tests/conftest.py", "src/amm_optimizer/api.py"} <= names
    assert any(named_paths(ROOT / "README.md"))


@pytest.mark.parametrize(
    ("line", "found"),
    [
        ("see docs/ANYTIME.md.", ["docs/ANYTIME.md"]),
        ("(``benchmarks/perf.py``)", ["benchmarks/perf.py"]),
        ("https://example.org/main/docs/x.md", []),
        ("subdocs/x.md and ../docs/y.md", []),
    ],
)
def test_the_pattern(line, found):
    assert PATH.findall(line) == found


def test_every_named_path_exists():
    missing = [
        f"{path.relative_to(ROOT).as_posix()}:{number}: {name}"
        for path in scanned_files()
        for number, name in named_paths(path)
        if not (ROOT / name).exists()
        and (path.relative_to(ROOT).as_posix(), name) not in PENDING
    ]
    assert not missing, "paths that name no file:\n" + "\n".join(missing)
