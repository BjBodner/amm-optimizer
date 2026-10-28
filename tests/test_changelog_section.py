"""tools/changelog_section.py: a release's notes are its CHANGELOG section."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    path = ROOT / "tools" / "changelog_section.py"
    spec = importlib.util.spec_from_file_location("changelog_section", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = load_tool()

CHANGELOG = """\
# Changelog

Intro text with a [link](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.6.0] - 2026-11-01

### Added

- `callback` (guide: [docs/ANYTIME.md](docs/ANYTIME.md), [site](https://example.org/x)).

### Fixed

- A [section link](#fixed) stays.

## [0.5.0] - 2026-10-25

- Old entry.

## [0.2.0] - 2026-10-22 [YANKED]

- Yanked entry.

[Unreleased]: https://github.com/o/r/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/o/r/compare/v0.5.0...v0.6.0
"""


def test_the_section_runs_from_its_heading_to_the_next():
    notes = tool.section(CHANGELOG, "0.6.0")
    assert notes.startswith("### Added\n")
    assert notes.endswith("- A [section link](#fixed) stays.\n")
    assert "Old entry" not in notes and "2026-11-01" not in notes


def test_a_tag_names_the_same_section():
    assert tool.section(CHANGELOG, "v0.6.0") == tool.section(CHANGELOG, "0.6.0")


def test_the_last_section_leaves_out_the_link_definitions():
    notes = tool.section(CHANGELOG, "0.2.0")
    assert notes == "- Yanked entry.\n"


def test_a_missing_or_empty_section_is_an_error():
    with pytest.raises(tool.SectionError, match="no '## \\[0.9.0\\]'"):
        tool.section(CHANGELOG, "0.9.0")
    with pytest.raises(tool.SectionError, match="empty"):
        tool.section(CHANGELOG, "Unreleased")


def test_a_version_is_matched_whole_not_as_a_prefix():
    with pytest.raises(tool.SectionError):
        tool.section(CHANGELOG, "0.6")


def test_link_base_makes_relative_links_absolute_and_leaves_the_rest():
    base = "https://github.com/o/r/blob/v0.6.0"
    notes = tool.section(CHANGELOG, "0.6.0", link_base=base)
    assert f"[docs/ANYTIME.md]({base}/docs/ANYTIME.md)" in notes
    assert "[site](https://example.org/x)" in notes
    assert "[section link](#fixed)" in notes


def test_the_command_line_prints_the_section_and_fails_on_a_missing_one(tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    script = str(ROOT / "tools" / "changelog_section.py")
    env = {**os.environ, "PYTHONUTF8": "1"}
    ok = subprocess.run(
        [sys.executable, script, "v0.5.0", "--changelog", str(changelog)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert ok.returncode == 0 and ok.stdout == "- Old entry.\n"
    missing = subprocess.run(
        [sys.executable, script, "1.0.0", "--changelog", str(changelog)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert missing.returncode == 1 and "no '## [1.0.0]'" in missing.stderr


def test_every_released_version_of_the_real_changelog_has_notes():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    versions = [
        match["version"]
        for line in text.splitlines()
        if (match := tool.HEADING.match(line)) and match["version"] != "Unreleased"
    ]
    assert "0.5.0" in versions
    for version in versions:
        assert tool.section(text, version).strip()
