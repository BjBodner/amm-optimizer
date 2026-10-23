"""Record golden masters from the frozen reference implementation.

Usage (from the repo root):  python tools/record_golden.py

Baselines are UNVERIFIED characterization data: they pin what the original
code does, not what is correct. Re-record only for a deliberate, documented
behaviour change (see docs/DEVIATIONS.md).
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "golden"))

from reference.atm_original import Orchestra_Optimize_With_Two_Mode_Section  # noqa: E402
from scenarios import SCENARIOS, run_scenario  # noqa: E402


def optimize(fun, x0, budget):
    return Orchestra_Optimize_With_Two_Mode_Section(x0, fun, budget, 0)


def main() -> None:
    out = {
        "_meta": {
            "source": "reference/atm_original.py (BjBodner/atm@69828c6)",
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "status": "UNVERIFIED characterization baseline",
        },
        "scenarios": {s.name: run_scenario(optimize, s) for s in SCENARIOS},
    }
    path = ROOT / "tests" / "golden" / "golden.json"
    path.write_text(json.dumps(out, indent=1))
    for name, rec in out["scenarios"].items():
        outcome = rec.get("raises") or f"cost={rec['final_cost']:.6g}"
        print(f"{name:20s} evals={rec['n_evals']:5d} {outcome}")
    print("wrote", path)


if __name__ == "__main__":
    main()
