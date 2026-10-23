"""Check a built site (dist/) for root-relative links that ignore the base path.

Usage: python tools/check_base.py dist /amm-optimizer
Exits 1 and lists the first offenders if any href/src starts with '/' but not with the base.
"""
import pathlib
import re
import sys

dist = pathlib.Path(sys.argv[1])
base = sys.argv[2].rstrip("/")
bad = []
total = 0
for page in dist.rglob("*.html"):
    html = page.read_text(encoding="utf-8")
    for m in re.finditer(r'(?:href|src)="(/[^"]*)"', html):
        link = m.group(1)
        total += 1
        if link.startswith("//") or link == base or link.startswith(base + "/"):
            continue
        bad.append((page.relative_to(dist).as_posix(), link))
print(f"{total} root-relative links checked under base {base!r}; {len(bad)} without the base")
for p, l in bad[:15]:
    print("  ", p, "->", l)
sys.exit(1 if bad else 0)
