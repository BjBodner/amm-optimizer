"""Check that the hero snippet on the home page still prints what the page says it prints.

Extracts the first ```python fence of src/content/docs/index.mdx and the ```text fence that
follows it, runs the snippet verbatim against the repository's own ``src/`` (the package imports as
``amm_optimizer``; the snippet selects AMM with ``config=amm_optimizer.presets.get("c46")``), and
compares the printed output with the text fence.

    python tools/check_snippet.py [--src PATH_TO_SRC]    (default: AMM_SRC, else <repository>/src)

Exit code 0 when the output matches exactly, 1 on a mismatch, 2 when the fences are missing.
BLAS threads are left at the machine default, which is how the page output was produced
(--blas-threads 1 forces one thread, as the benchmark harness does: the last digits of f then
differ, 2.02e-13 instead of 2.46e-13 on the machine that wrote the page, because the BLAS
thread count changes the last bits of SLSQP steps). Use `python`, not python3.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
INDEX = SITE / "src" / "content" / "docs" / "index.mdx"
DEFAULT_SRC = os.environ.get("AMM_SRC", str(SITE.parent / "src"))

def extract(text: str) -> tuple[str, str]:
    """The first python fence and the first text fence after it."""
    py = re.search(r"```python\r?\n(.*?)```", text, re.S)
    if py is None:
        raise ValueError("no ```python fence in index.mdx")
    after = text[py.end() :]
    out = re.search(r"```text\r?\n(.*?)```", after, re.S)
    if out is None:
        raise ValueError("no ```text fence after the python fence in index.mdx")
    return py.group(1), out.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", default=DEFAULT_SRC, help="directory holding the amm_optimizer package")
    parser.add_argument("--index", default=str(INDEX))
    parser.add_argument(
        "--blas-threads", type=int, default=0, help="force this many BLAS threads (0 = machine default)"
    )
    args = parser.parse_args()

    try:
        code, expected = extract(Path(args.index).read_text(encoding="utf-8"))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if "import amm_optimizer" not in code:
        print("error: the snippet has no 'import amm_optimizer' line", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["PYTHONPATH"] = args.src
    if args.blas_threads > 0:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[var] = str(args.blas_threads)
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        run = subprocess.run(
            [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=300
        )
    if run.returncode != 0:
        print("the snippet failed:\n" + run.stderr, file=sys.stderr)
        return 1
    got = run.stdout.replace("\r\n", "\n").strip()
    want = expected.replace("\r\n", "\n").strip()
    if got == want:
        print("OK: the snippet output matches the text fence in index.mdx")
        print(got)
        return 0
    print("MISMATCH")
    print("page says:\n" + want)
    print("snippet printed:\n" + got)
    return 1


if __name__ == "__main__":
    sys.exit(main())
