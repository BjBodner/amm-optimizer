"""The source's comments and docstrings carry no research-log ids.

Cycle, batch and composite numbers belong in ``docs/HISTORY.md`` (not checked
here), not in the code. The shipped preset names (``presets.names()``: ``c46`` is
one) and the values of the configuration's choices (``"qn38"`` is one) are public
identifiers, so they may appear; ``presets.py`` defines every preset, the research
ones included, and is not checked.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

from amm_optimizer import presets
from amm_optimizer.config import CHOICES

SRC = Path(__file__).resolve().parents[1] / "src" / "amm_optimizer"
RESEARCH_ID = re.compile(
    r"\b(c\d{2}|qn\d+|cycle \d+|parked\d|batch \d+)\b", re.IGNORECASE
)
EXEMPT_FILES = {"presets.py"}


def _allowed() -> set[str]:
    names = set(presets.names())
    values = {v for choices in CHOICES.values() for v in choices if isinstance(v, str)}
    return names | values


def _texts(path: Path) -> list[tuple[int, str]]:
    """Every comment and docstring of ``path`` with its first line number."""
    source = path.read_text(encoding="utf8")
    out = [
        (tok.start[0], tok.string)
        for tok in tokenize.generate_tokens(io.StringIO(source).readline)
        if tok.type == tokenize.COMMENT
    ]
    for node in ast.walk(ast.parse(source)):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                out.append((node.body[0].lineno, doc))
    return out


def research_ids(path: Path) -> list[str]:
    allowed = _allowed()
    found = []
    for line, text in _texts(path):
        for match in RESEARCH_ID.finditer(text):
            if match.group(0) not in allowed:
                found.append(f"{path.name}:{line}: {match.group(0)}")
    return found


def test_the_source_has_no_research_ids() -> None:
    files = sorted(p for p in SRC.glob("*.py") if p.name not in EXEMPT_FILES)
    assert files
    found = [hit for path in files for hit in research_ids(path)]
    assert found == []


def test_the_check_finds_a_research_id(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        '"""Confirmed in cycle 12."""\n\n# Batch 34 and c31\nX = 1  # qn38 and c46\n',
        encoding="utf8",
    )
    assert [hit.split(": ")[1] for hit in research_ids(sample)] == [
        "Batch 34",
        "c31",
        "cycle 12",
    ]
