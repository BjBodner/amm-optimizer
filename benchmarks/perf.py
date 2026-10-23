"""Repeatable speed benchmark: original vs refactored ATM.

    python benchmarks/perf.py --repeats 7 --out benchmarks/results/perf.json

Design for repeatability:

* BLAS/OpenMP threads are pinned to one *before* NumPy is imported.
* Every scenario has a fixed seed; the first (untimed) run records the full
  cost trace and its SHA-256, the timed runs must end on the identical point.
  A timing is only reported for runs whose result matches the recording, and
  the original and the refactor must produce the same trace hash.
* Timings use ``perf_counter_ns`` with the garbage collector paused, after a
  warm-up run, and are summarised by median, min and relative spread (IQR).
* Cheap objectives are used on purpose so that optimizer overhead dominates.
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
import contextlib  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "golden"))

from scenarios import FUNCTIONS  # noqa: E402

# (name, function, dimension, budget, seed)
SCENARIOS = [
    ("sphere_d5", "sphere", 5, 3000, 1),
    ("sphere_d10", "sphere", 10, 6000, 2),
    ("sphere_d40", "sphere", 40, 12000, 3),
    ("ellipsoid_d20", "ellipsoid", 20, 8000, 4),
    ("rastrigin_d10", "rastrigin", 10, 6000, 5),
    ("rosenbrock_d20", "rosenbrock", 20, 8000, 6),
]


def _solver_reference():
    with contextlib.redirect_stdout(io.StringIO()):
        from reference.atm_original import Orchestra_Optimize_With_Two_Mode_Section

    def run(fun, dim, budget, seed):
        np.random.seed(seed)
        x0 = 10 * (np.random.rand(dim) - 0.5)
        return np.asarray(Orchestra_Optimize_With_Two_Mode_Section(x0, fun, budget, 0))

    return run


def _solver_new():
    import amm_optimizer as atm

    def run(fun, dim, budget, seed):
        rng = np.random.RandomState(seed)
        x0 = 10 * (rng.rand(dim) - 0.5)
        config = atm.ATMConfig(strict_budget=False)  # same accounting as the original
        return atm.minimize(fun, x0, budget, rng=rng, config=config).x

    return run


SOLVERS = {"reference": _solver_reference, "new": _solver_new}


def record_trace(solver, function, dim, budget, seed):
    """Untimed run that returns (trace hash, final x, evaluation count)."""
    fun = FUNCTIONS[function]
    costs: list[float] = []

    def recording(x):
        c = float(fun(x))
        costs.append(c)
        return c

    with contextlib.redirect_stdout(io.StringIO()):
        x = solver(recording, dim, budget, seed)
    digest = hashlib.sha256(np.asarray(costs).tobytes()).hexdigest()
    return digest, x, len(costs)


def time_once(solver, function, dim, budget, seed):
    fun = FUNCTIONS[function]
    with contextlib.redirect_stdout(io.StringIO()):
        gc.collect()
        gc.disable()
        try:
            start = time.perf_counter_ns()
            x = solver(fun, dim, budget, seed)
            elapsed = (time.perf_counter_ns() - start) / 1e9
        finally:
            gc.enable()
    return elapsed, x


def summarise(times):
    q = statistics.quantiles(times, n=4) if len(times) >= 4 else [min(times)] * 3
    median = statistics.median(times)
    return {
        "median_s": median,
        "min_s": min(times),
        "iqr_rel": (q[2] - q[0]) / median if median else 0.0,
        "runs": len(times),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--solvers", nargs="+", default=["reference", "new"])
    parser.add_argument("--scenarios", nargs="*", help="subset by name")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    solvers = {name: SOLVERS[name]() for name in args.solvers}
    scenarios = [s for s in SCENARIOS if not args.scenarios or s[0] in args.scenarios]

    report: dict = {
        "meta": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": platform.platform(),
            "repeats": args.repeats,
        },
        "results": {},
    }
    mismatches = 0
    header = f"{'scenario':16s}{'solver':11s}{'evals':>7s}{'median s':>10s}"
    print(header + f"{'min s':>9s}{'iqr/med':>9s}{'us/eval':>9s}  trace")
    for name, function, dim, budget, seed in scenarios:
        hashes = {}
        for solver_name, solver in solvers.items():
            digest, x_ref, n_evals = record_trace(solver, function, dim, budget, seed)
            hashes[solver_name] = digest
            time_once(solver, function, dim, budget, seed)  # warm-up
            times = []
            for _ in range(args.repeats):
                elapsed, x = time_once(solver, function, dim, budget, seed)
                if not np.array_equal(x, x_ref):
                    raise SystemExit(f"{name}/{solver_name}: run is not repeatable")
                times.append(elapsed)
            stats = summarise(times) | {
                "evals": n_evals,
                "us_per_eval": 1e6 * statistics.median(times) / n_evals,
                "trace_sha256": digest,
            }
            report["results"].setdefault(name, {})[solver_name] = stats
            print(
                f"{name:16s}{solver_name:11s}{n_evals:7d}{stats['median_s']:10.4f}"
                f"{stats['min_s']:9.4f}{stats['iqr_rel']:9.1%}"
                f"{stats['us_per_eval']:9.1f}  {digest[:10]}"
            )
        if len(set(hashes.values())) > 1:
            mismatches += 1
            print(f"  !! {name}: solvers produced different traces")
    if "reference" in solvers and "new" in solvers:
        print("\nspeed-up of new over reference (median):")
        for name, by_solver in report["results"].items():
            ratio = by_solver["reference"]["median_s"] / by_solver["new"]["median_s"]
            print(f"  {name:16s}{ratio:6.2f}x")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1))
        print("wrote", args.out)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
