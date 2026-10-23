"""Smoke test of an *installed* package: run it from outside the source tree.

    python tools/smoke_installed.py [--module amm_optimizer]

Run it in a fresh virtual environment after ``pip install dist/*.whl`` (or the
sdist). It checks that the import resolves to site-packages and not to a source
checkout, that ``py.typed`` shipped, and that the frozen preset runs one short
optimisation with an exact budget. Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="amm_optimizer", help="import name to test")
    parser.add_argument("--dist", default=None, help="distribution name (optional)")
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    where = Path(module.__file__ or "").resolve()
    if "site-packages" not in where.parts and "dist-packages" not in where.parts:
        print(f"FAIL: {args.module} imported from {where}, not an installed copy")
        return 1
    if not (where.parent / "py.typed").exists():
        print("FAIL: py.typed missing from the installed package")
        return 1
    if args.dist:
        installed = importlib.metadata.version(args.dist)
        if installed != module.__version__:
            print(f"FAIL: metadata {installed} != __version__ {module.__version__}")
            return 1

    dim, budget = 5, 200

    def sphere(x: np.ndarray) -> float:
        return float(np.sum((np.asarray(x) - 1.0) ** 2))

    result = module.minimize(
        sphere,
        np.zeros(dim),
        budget,
        seed=0,
        config=module.presets.get("c46"),
        bounds=(-5.0, 5.0),
    )
    if result.nfev != budget or not np.isfinite(result.fun) or result.fun > 1.0:
        print(f"FAIL: nfev={result.nfev} fun={result.fun}")
        return 1
    print(f"OK: {args.module} {module.__version__}: fun={result.fun:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
