"""Where does the time go? Per-block timing of ``atm.minimize`` (docs/PERF.md).

    python benchmarks/block_profile.py                      # default scenarios
    python benchmarks/block_profile.py --dims 10 --functions sphere --budget-per-dim 600
    python benchmarks/block_profile.py --json benchmarks/results/block_profile.json

Each scenario runs ``--seeds`` seeds with ``profile=True`` on one shared
``TimingProfiler``; the table shows exclusive ("self") time per block as a share of
the wall time, so the rows add up to 100 %. The objectives are the cheap NumPy
functions of the golden suite, which makes the optimizer's own cost visible.
Timing a run adds roughly 10 % (see ``benchmarks/profiler_overhead.py``), so use the
shares, not the absolute times, and the un-profiled ``perf.py`` for speed.
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "golden"))

from scenarios import FUNCTIONS  # noqa: E402

import amm_optimizer as atm  # noqa: E402


def profile_scenario(function: str, dim: int, budget: int, seeds: int):
    fun = FUNCTIONS[function]
    prof = atm.TimingProfiler()
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        x0 = 10 * (rng.rand(dim) - 0.5)
        atm.api.minimize(fun, x0, budget, rng=rng, profile=prof)
    return prof.report()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--functions", default="sphere,ellipsoid,rastrigin")
    ap.add_argument("--dims", default="2,10,40")
    ap.add_argument("--budget-per-dim", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", help="also write the numbers to this file")
    args = ap.parse_args()

    record = {}
    for function in args.functions.split(","):
        for dim in (int(d) for d in args.dims.split(",")):
            budget = args.budget_per_dim * dim
            t = profile_scenario(function, dim, budget, args.seeds)
            nfev = t.blocks["objective"].calls
            print(f"\n== {function}, D={dim}, budget {budget}, {args.seeds} seeds ==")
            print(t.table())
            print(
                f"optimizer overhead: {1e6 * t.overhead_s / nfev:.1f} us per evaluation"
            )
            record[f"{function}_d{dim}"] = {
                "budget": budget,
                "seeds": args.seeds,
                "evaluations": nfev,
                "overhead_us_per_eval": 1e6 * t.overhead_s / nfev,
                "blocks": t.to_dict(),
            }
    if args.json:
        Path(args.json).write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
