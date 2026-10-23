"""Does a historic ``legacy_atm.py`` solve more bbob problems than ``reference/``?

    python benchmarks/legacy_vs_reference.py --legacy path/to/legacy_atm.py \
        --dims 2 --instances 1-3 --multipliers 1000,10000 --jobs 6

Both modules are driven through the same COCO harness (``run_bbob.solve_problem``:
box penalty, budget cut-off, restart loop) with the global NumPy stream they use,
seeded per problem. The legacy file needs ``scipy`` and ``cma`` only for imports
it never uses; put a stub ``scipy/optimize.py`` on ``PYTHONPATH`` if they are not
installed. Prints, per function, the share of instances that reached the final
target (precision <= 1e-8) at every budget.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import concurrent.futures  # noqa: E402
import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import quick_scan  # noqa: E402

_W: dict = {}


class GlobalStreamSolver:
    """A module with ``Orchestra_Optimize_With_Two_Mode_Section``, global stream."""

    def __init__(self, module, start: str = "uniform") -> None:
        self._optimize = module.Orchestra_Optimize_With_Two_Mode_Section
        self._start = start
        self._first = True

    def begin(self, seed) -> None:
        np.random.seed(int(seed.generate_state(1)[0]) % (2**31))
        self._first = True

    def uniform(self, dim: int) -> np.ndarray:
        first, self._first = self._first, False
        if self._start == "coco2019":
            if first:
                return np.full(dim, 0.5)
            return 0.5 + 0.8 * (np.random.rand(dim) - 0.5)
        return np.random.rand(dim)

    def solve(self, fun, x0, budget: int) -> None:
        self._optimize(x0, fun, budget, 0)


def load(label: str, path: str):
    with contextlib.redirect_stdout(io.StringIO()):
        if label == "reference":
            from reference import atm_original as module
        else:
            spec = importlib.util.spec_from_file_location("legacy_mod", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    return module


def _init(spec: dict) -> None:
    import cocoex
    import matplotlib

    matplotlib.use("Agg")
    _W["spec"] = spec
    _W["suite"] = cocoex.Suite("bbob", "year: 2019", *spec["suite_args"])
    _W["solvers"] = {
        label: GlobalStreamSolver(load(label, spec["legacy"]), spec["start"])
        for label in ("legacy", "reference")
    }


def _work(task):
    label, problem_id, multiplier = task
    spec = _W["spec"]
    return label, quick_scan.run_one(
        _W["suite"],
        _W["solvers"][label],
        spec["fopts"],
        problem_id,
        multiplier,
        0,
        spec["penalty"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--dims", default="2")
    parser.add_argument("--instances", default="1-3")
    parser.add_argument("--functions", default="1-24")
    parser.add_argument("--multipliers", default="1000,10000")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--penalty", type=quick_scan.parse_penalty, default=1.0)
    parser.add_argument("--start", choices=["uniform", "coco2019"], default="uniform")
    args = parser.parse_args()
    import cocoex

    suite_args = (
        f"dimensions: {args.dims} instance_indices: {args.instances} "
        f"function_indices: {args.functions}",
    )
    fopts = quick_scan.fopt_table(suite_args)
    ids = list(cocoex.Suite("bbob", "year: 2019", *suite_args).ids())
    multipliers = [float(m) for m in args.multipliers.split(",")]
    tasks = [
        (lab, p, m) for lab in ("legacy", "reference") for m in multipliers for p in ids
    ]
    tasks.sort(key=lambda t: -t[2])
    spec = {
        "suite_args": suite_args,
        "fopts": fopts,
        "legacy": str(Path(args.legacy).resolve()),
        "start": args.start,
        "penalty": args.penalty,
    }
    results: dict = {}
    ecdf: dict = {}
    with concurrent.futures.ProcessPoolExecutor(
        args.jobs, initializer=_init, initargs=(spec,)
    ) as pool:
        for label, record in pool.map(_work, tasks, chunksize=1):
            results.setdefault(
                (label, record["multiplier"], record["function"]), []
            ).append(record["precision"])
            ecdf.setdefault((label, record["multiplier"]), []).append(
                quick_scan.record_score(record)
            )
    print(
        f"{'':>5}"
        + "".join(
            f"{lab[:3]}@{m:g}".rjust(16)
            for m in multipliers
            for lab in ("legacy", "reference")
        )
    )
    solved = {
        k: 0 for k in [(lab, m) for lab in ("legacy", "reference") for m in multipliers]
    }
    for f in sorted({k[2] for k in results}):
        cells = []
        for m in multipliers:
            for lab in ("legacy", "reference"):
                precisions = results[(lab, m, f)]
                hit = sum(p <= 1e-8 for p in precisions)
                solved[(lab, m)] += hit > 0
                cells.append(
                    f"{hit}/{len(precisions)} med {np.median(precisions):.0e}".rjust(16)
                )
        print(f"f{f:<4}" + "".join(cells))
    print(
        "mean ECDF value: "
        + ", ".join(
            f"{lab} @{m:g}xD {np.mean(v):.3f}"
            for (lab, m), v in sorted(ecdf.items(), key=lambda kv: (kv[0][1], kv[0][0]))
        )
    )
    print(
        "functions with at least one instance solved: "
        + ", ".join(f"{lab} @{m:g}xD: {n}" for (lab, m), n in solved.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
