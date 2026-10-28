"""Print the CHANGELOG.md section of one version, for the notes of a GitHub Release.

    python tools/changelog_section.py 0.6.0              # the body of "## [0.6.0] ..."
    python tools/changelog_section.py v0.6.0             # a tag works too
    python tools/changelog_section.py 0.6.0 --link-base \
        https://github.com/BjBodner/amm-optimizer/blob/v0.6.0/

The CHANGELOG follows Keep a Changelog: one ``## [version] - date`` heading per release.
The section is everything between that heading and the next ``## `` heading, without
the link reference definitions (``[0.5.0]: https://...``) that close the file. With
``--link-base``, relative link targets (``docs/ANYTIME.md``) become absolute, since a
release page has no repository to resolve them against.

Exit status 1, with a message on stderr, if the version has no section or the section
is empty: a release must not go out with empty notes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]")
LINK_DEFINITION = re.compile(r"^\[[^\]]+\]:\s+\S+")
# An inline Markdown link whose target has no scheme, is not an anchor and not rooted.
RELATIVE_LINK = re.compile(r"\]\((?![a-zA-Z][a-zA-Z0-9+.-]*:|#|/)([^)\s]+)\)")


class SectionError(ValueError):
    """The CHANGELOG has no usable section for the version asked for."""


def normalise(version: str) -> str:
    """``v0.6.0`` and ``0.6.0`` name the same release."""
    version = version.strip()
    return version[1:] if version[:1] in ("v", "V") else version


def section(text: str, version: str, link_base: str | None = None) -> str:
    """The body of ``version``'s section in the Keep a Changelog ``text``."""
    wanted = normalise(version)
    lines = text.splitlines()
    start = None
    for number, line in enumerate(lines):
        match = HEADING.match(line)
        if match and normalise(match["version"]) == wanted:
            start = number + 1
            break
    if start is None:
        raise SectionError(f"CHANGELOG.md has no '## [{wanted}]' section")
    end = len(lines)
    for number in range(start, len(lines)):
        if lines[number].startswith("## "):
            end = number
            break
    body = [line for line in lines[start:end] if not LINK_DEFINITION.match(line)]
    notes = "\n".join(body).strip()
    if not notes:
        raise SectionError(f"the '## [{wanted}]' section of CHANGELOG.md is empty")
    if link_base:
        base = link_base if link_base.endswith("/") else link_base + "/"
        notes = RELATIVE_LINK.sub(lambda m: f"]({base}{m[1]})", notes)
    return notes + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("version", help="the version or tag, e.g. 0.6.0 or v0.6.0")
    parser.add_argument(
        "--changelog", type=Path, default=ROOT / "CHANGELOG.md", help="the file to read"
    )
    parser.add_argument(
        "--link-base", default=None, help="prefix for relative link targets"
    )
    args = parser.parse_args(argv)
    try:
        notes = section(
            args.changelog.read_text(encoding="utf-8"), args.version, args.link_base
        )
    except SectionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
