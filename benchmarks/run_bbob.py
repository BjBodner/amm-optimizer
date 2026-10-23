"""Run ATM on the COCO ``bbob`` suite (2019) and write COCO + summary data.

    python benchmarks/run_bbob.py --solver new --dims 2 5 --instances 1-2 \
        --budget-multiplier 10 --out benchmarks/results/bbob_smoke

Solvers (all see exactly the same objective, start points and random stream):

* ``original``    the frozen original (``reference/``), global NumPy stream
* ``new_legacy``  this package with the original's budget accounting; must be
                  bit-identical to ``original`` (the parity check)
* ``new``         this package with the defaults (strict budget)
* ``new_exclude_winner``  ``new`` with ``second_best="exclude_winner"``

The harness, not the optimizer, enforces the evaluation budget: a call that
would exceed ``budget_multiplier * dimension`` evaluations (or that comes after
the final target was hit) raises ``StopRunError`` inside the objective. The optimizer
core is never modified. Points outside the box [-5, 5]^D are evaluated at their
clipped position (``penalty=None``: raw, unclipped) and the optimizer is shown
``f + penalty * |x - clip(x)|^2``.

Every row of the summary also records ``objective_seconds`` (time inside the COCO
function), so the optimizer's own overhead is ``seconds - objective_seconds`` for
every solver. ``--profile`` (the ``new*`` solvers) additionally times the blocks
of the algorithm (``atm.TimingProfiler``) and writes ``blocks_<folder>.csv``,
summed over all problems of the run.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MAX_RESTARTS = 10_000  # safety net against a solver that returns without working

# The ``clip`` ufunc itself. ``np.clip`` reaches the same loop (so the same bits)
# through three Python wrappers, which cost more than clipping a short vector.
_clip = np._core.umath.clip


class StopRunError(Exception):
    """Raised inside the objective when the budget is spent or the target hit."""


class OriginalSolver:
    name = "original"

    def __init__(self) -> None:
        # Import now: the module seeds the global stream (np.random.seed(42))
        # on import, which must not happen after ``begin`` has seeded it.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            from reference.atm_original import Orchestra_Optimize_With_Two_Mode_Section

        self._optimize = Orchestra_Optimize_With_Two_Mode_Section

    def begin(self, seed: int) -> None:
        np.random.seed(seed)

    def uniform(self, dim: int) -> np.ndarray:
        return np.random.rand(dim)

    def solve(self, fun, x0, budget: int) -> None:
        self._optimize(x0, fun, budget, 0)


class NewSolver:
    def __init__(self, name: str, profile: bool = False, **config) -> None:
        import amm_optimizer as atm

        self.name = name
        self._atm = atm
        self._config = atm.ATMConfig(**config)
        self.profiler = atm.TimingProfiler() if profile else None

    def begin(self, seed: int) -> None:
        self._rng = np.random.RandomState(seed)

    def uniform(self, dim: int) -> np.ndarray:
        return self._rng.rand(dim)

    def solve(self, fun, x0, budget: int) -> None:
        self._atm.api.minimize(
            fun,
            x0,
            budget,
            rng=self._rng,
            config=self._config,
            profile=self.profiler or False,
        )


def make_solver(name: str, profile: bool = False):
    if profile and name == "original":
        raise SystemExit("--profile needs a new* solver (the original has no hooks)")
    if name == "original":
        return OriginalSolver()
    if name == "new_legacy":
        return NewSolver(name, profile, strict_budget=False)
    if name == "new":
        return NewSolver(name, profile)
    if name == "new_exclude_winner":
        return NewSolver(name, profile, second_best="exclude_winner")
    raise SystemExit(f"unknown solver {name!r}")


SOLVERS = ["original", "new_legacy", "new", "new_exclude_winner"]


class BoxedObjective:
    """Budget, target and box handling around one COCO problem."""

    def __init__(self, problem, budget: int, penalty: float) -> None:
        self.problem = problem
        self.budget = budget
        self.penalty = penalty
        self.lower = np.asarray(problem.lower_bounds, dtype=float)
        self.upper = np.asarray(problem.upper_bounds, dtype=float)
        self.values: list[float] = []
        self.objective_ns = 0

    def __call__(self, x: np.ndarray) -> float:
        if self.problem.evaluations >= self.budget or self.problem.final_target_hit:
            raise StopRunError
        x = np.asarray(x, dtype=float)
        # ``penalty=None``: raw COCO, the function is evaluated where asked (the
        # box is only a recommendation for the start, as in the 2019 submission).
        clipped = x if self.penalty is None else _clip(x, self.lower, self.upper)
        start = time.perf_counter_ns()
        value = float(self.problem(clipped))
        self.objective_ns += time.perf_counter_ns() - start
        if self.penalty:
            # A point that was not clipped has a squared excess of exactly +0.0
            # (x - clip(x) is +-0.0 in every coordinate), so the sum is only formed
            # for one that was; the value is the same, bit for bit.
            excess = float(np.sum((x - clipped) ** 2)) if (x != clipped).any() else 0.0
            value += self.penalty * excess
        self.values.append(value)
        return value


def solve_problem(solver, problem, multiplier: float, seed: int, penalty: float):
    dim = problem.dimension
    budget = int(multiplier * dim)
    objective = BoxedObjective(problem, budget, penalty)
    solver.begin(seed)
    restarts = 0
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        while (
            problem.evaluations < budget
            and not problem.final_target_hit
            and restarts < MAX_RESTARTS
        ):
            x0 = objective.lower + (objective.upper - objective.lower) * solver.uniform(
                dim
            )
            with contextlib.suppress(StopRunError):
                solver.solve(objective, x0, budget - problem.evaluations)
            restarts += 1
    trace = np.asarray(objective.values, dtype=float)
    return {
        "evaluations": int(problem.evaluations),
        "best_f": float(problem.best_observed_fvalue1),
        "target_hit": bool(problem.final_target_hit),
        "restarts": restarts - 1,
        "objective_seconds": objective.objective_ns * 1e-9,
        "trace_sha256": hashlib.sha256(trace.tobytes()).hexdigest(),
    }


def write_blocks(path: Path, timings) -> None:
    """One row per algorithm block, summed over all problems of the run."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["block", "calls", "self_s", "total_s", "share_of_wall"])
        for name, b in sorted(timings.blocks.items(), key=lambda kv: -kv[1].self_s):
            writer.writerow(
                [name, b.calls, f"{b.self_s:.6f}", f"{b.total_s:.6f}"]
                + [f"{timings.share(name):.4f}"]
            )
    print(f"block timings -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--solver", choices=SOLVERS, required=True)
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--instances", default="1-2", help='e.g. "1-5" or "1,3"')
    parser.add_argument("--functions", default="1-24", help='e.g. "1-24" or "1,8"')
    parser.add_argument("--budget-multiplier", type=float, default=10.0)
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--postprocess", action="store_true", help="run cocopp")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="new* solvers: time the algorithm blocks, write blocks_<folder>.csv",
    )
    args = parser.parse_args()

    import cocoex

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)  # COCO writes to ./exdata

    solver = make_solver(args.solver, args.profile)
    suite = cocoex.Suite(
        "bbob",
        "year: 2019",
        f"dimensions: {','.join(map(str, args.dims))} "
        f"instance_indices: {args.instances} function_indices: {args.functions}",
    )
    folder = f"{solver.name}_x{args.budget_multiplier:g}"
    observer = cocoex.Observer(
        "bbob",
        f"result_folder: {folder} algorithm_name: ATM-{solver.name} "
        f"algorithm_info: ATM_{solver.name}_budget_x{args.budget_multiplier:g}",
    )

    rows = []
    started = time.perf_counter()
    for index, problem in enumerate(suite):
        problem.observe_with(observer)
        t0 = time.perf_counter()
        seed = args.seed * 100_003 + index
        row = solve_problem(solver, problem, args.budget_multiplier, seed, args.penalty)
        row.update(
            id=problem.id,
            dimension=problem.dimension,
            seed=seed,
            seconds=time.perf_counter() - t0,
        )
        rows.append(row)
        problem.free()
        print(
            f"{row['id']:22s} evals={row['evaluations']:7d} "
            f"best_f={row['best_f']:14.6g} hit={int(row['target_hit'])} "
            f"restarts={row['restarts']}",
            flush=True,
        )

    summary = {
        "solver": solver.name,
        "budget_multiplier": args.budget_multiplier,
        "penalty": args.penalty,
        "seed": args.seed,
        "dims": args.dims,
        "instances": args.instances,
        "functions": args.functions,
        "coco_folder": f"exdata/{folder}",
        "wall_seconds": time.perf_counter() - started,
        "problems": rows,
    }
    (out / f"summary_{folder}.json").write_text(json.dumps(summary, indent=1))
    if args.profile:
        write_blocks(out / f"blocks_{folder}.csv", solver.profiler.report())
    hits = sum(r["target_hit"] for r in rows)
    print(
        f"\n{len(rows)} problems, {hits} reached the final target, "
        f"{summary['wall_seconds']:.1f}s -> {out / f'summary_{folder}.json'}"
    )

    if args.postprocess:
        import cocopp

        cocopp.main(f"exdata/{folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
