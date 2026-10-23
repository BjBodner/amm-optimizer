"""Copy the Optimizer Race (code and recorded runs) from the bbob-landscapes repository into public/race/.

    python tools/sync_race.py [--src PATH] [--check]

The race is developed in bbob-landscapes (src/landscape.js, src/race.js, src/race.css,
src/bbob-meta.js, race/index.json, race/data/*.json, recorded by race/record.py). This script
copies those files and writes public/race/SOURCE.json with the source commit and a SHA-256 per
file. --check compares the copies with the source and exits 1 on any difference. The home page
mounts the race from these copies (src/components/RaceHero.astro).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
DEST = SITE / "public" / "race"
DEFAULT_SRC = SITE.parent.parent / "bbob-landscapes"
CODE = ["src/bbob-meta.js", "src/landscape.js", "src/race.js", "src/race.css"]


def files(src: Path) -> list[tuple[Path, Path]]:
    pairs = [(src / c, DEST / Path(c).name) for c in CODE]
    pairs.append((src / "race" / "index.json", DEST / "index.json"))
    pairs += [(p, DEST / "data" / p.name) for p in sorted((src / "race" / "data").glob("f*_i*.json"))]
    return pairs


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(src: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True, check=True).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    pairs = files(a.src)
    if len(pairs) != len(CODE) + 1 + 120:
        print(f"expected 120 races in {a.src / 'race' / 'data'}, found {len(pairs) - len(CODE) - 1}")
        return 1
    if a.check:
        bad = [d for s, d in pairs if not d.exists() or sha(s) != sha(d)]
        for d in bad:
            print(f"differs: {d.relative_to(SITE)}")
        print("public/race matches the source" if not bad else f"{len(bad)} file(s) differ")
        return 1 if bad else 0
    (DEST / "data").mkdir(parents=True, exist_ok=True)
    for s, d in pairs:
        shutil.copyfile(s, d)
    manifest = {
        "source": "bbob-landscapes",
        "commit": git(a.src, "rev-parse", "--short", "HEAD"),
        "uncommitted_changes": bool(git(a.src, "status", "--porcelain", "--", "src", "race")),
        "files": {str(d.relative_to(DEST)).replace("\\", "/"): sha(d) for _, d in pairs},
    }
    (DEST / "SOURCE.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    size = sum(d.stat().st_size for _, d in pairs) / 1e6
    print(f"copied {len(pairs)} files ({size:.1f} MB) from {a.src} at {manifest['commit']}"
          f"{' (with uncommitted changes)' if manifest['uncommitted_changes'] else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
