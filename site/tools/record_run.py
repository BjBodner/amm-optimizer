"""Record a few short 2-D runs of AMM (preset c46) and of pycma IPOP for the site's player.

Run it with the atm_v2 virtual environment (it has cocoex and cma) and the FROZEN source on
PYTHONPATH; nothing in the frozen worktree is modified:

    cd <repository root>
    PYTHONPATH=src python site/tools/record_run.py

Every evaluation is recorded by an objective wrapper (index = position, x, regret = f - f_opt).
Stage boundaries are taken ONLY from the documented Profiler hook (atm.minimize(profile=...),
the 'init' and 'section' blocks of the ATM sampler, see src/atm/profiling.py), and only when a
second run without the hook gives the identical evaluation sequence. No boundary is inferred
from the trajectory. The frozen code has no hook for the other stages (see data/demo/README.md).

The functions are bbob instance 1, which is in the spent range 1-100, so no new instance is
used. f_opt is read for display only; AMM never sees it. This is an illustration of single
runs, not a benchmark result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parent.parent
FROZEN_WORKTREE = SITE.parent  # the repository root
EXPECTED_COMMIT = "552777b"

DIM = 2
BUDGET = 600  # 300 x D, the cell where the routing map shows the ATM stage at D = 2
SEED = 1
INSTANCE = 1
GRID_N = 81
# (bbob function id, short name, what to look for)
FUNCTIONS = [
    (8, "Rosenbrock", "smooth, a narrow curved valley"),
    (15, "Rastrigin", "rugged, many regularly placed local optima"),
    (21, "Gallagher 101-me", "several smooth basins of different depth"),
]


class OutOfBudget(Exception):
    """Raised on the call after the last allowed evaluation (the call is not recorded)."""


class Recorder:
    """Objective wrapper: counts and stores every evaluation."""

    def __init__(self, problem, fopt: float, budget: int) -> None:
        self.problem, self.fopt, self.budget = problem, fopt, budget
        self.xs: list[list[float]] = []
        self.regret: list[float] = []

    @property
    def n(self) -> int:
        return len(self.xs)

    def __call__(self, x) -> float:
        if self.n >= self.budget:
            raise OutOfBudget
        x = np.asarray(x, dtype=float)
        value = float(self.problem(x))
        self.xs.append([float(v) for v in x])
        self.regret.append(max(value - self.fopt, 0.0))
        return value


class StageProfiler:
    """The documented Profiler protocol; keeps the evaluation count at block entry and exit.

    Only the names 'init' and 'section' (the ATM sampler) are read.
    """

    enabled = True

    def __init__(self, count) -> None:
        self._count = count
        self.events: list[tuple[str, str, int]] = []

    def block(self, name: str):
        outer = self

        class _Block:
            def __enter__(self) -> None:
                outer.events.append(("in", name, outer._count()))

            def __exit__(self, *exc: object) -> None:
                outer.events.append(("out", name, outer._count()))

        return _Block()

    def record(self, name: str, elapsed_ns: int, calls: int = 1) -> None:
        return None

    def report(self):
        return None

    def atm_spans(self) -> list[list[int]]:
        """[start, end) evaluation indices of each ATM-sampler run (init + sections)."""
        spans: list[list[int]] = []
        start = None
        for kind, name, n in self.events:
            if name == "init" and kind == "in":
                start = n
            if name == "section" and kind == "out" and start is not None:
                spans.append([start, n])
                start = None
        return [s for s in spans if s[1] > s[0]]


def fopt_of(cocoex, fid: int) -> float:
    """f_opt of bbob function ``fid``, instance INSTANCE, via a throw-away observer log."""
    here, tmp = os.getcwd(), tempfile.mkdtemp()
    os.chdir(tmp)
    try:
        suite = cocoex.Suite("bbob", "", f"function_indices:{fid} dimensions:{DIM} instance_indices:{INSTANCE}")
        problem = suite[0]
        problem.observe_with(cocoex.Observer("bbob", "result_folder: fopt"))
        problem(np.zeros(DIM))
        problem.free()
        dat = next(next(Path(tmp, "exdata", "fopt").glob("data_f*")).glob(f"*DIM{DIM}.dat"))
        row = [ln for ln in dat.read_text().splitlines() if not ln.startswith("%")][-1].split()
        return round(float(row[3]) - float(row[2]), 2)  # f - (f - fopt); optima are multiples of .01
    finally:
        os.chdir(here)
        shutil.rmtree(tmp, ignore_errors=True)


def make_problem(cocoex, fid: int):
    suite = cocoex.Suite("bbob", "", f"function_indices:{fid} dimensions:{DIM} instance_indices:{INSTANCE}")
    return suite[0]


def run_amm(atm, presets, problem, fopt, x0, profile: bool):
    rec = Recorder(problem, fopt, BUDGET)
    prof = StageProfiler(lambda: rec.n) if profile else False
    try:
        atm.minimize(rec, x0.copy(), BUDGET, seed=SEED, config=presets.get("c46"), bounds=(-5.0, 5.0), profile=prof)
    except OutOfBudget:
        pass
    return rec, (prof if profile else None)


def run_ipop(cma, problem, fopt, x0):
    rec = Recorder(problem, fopt, BUDGET)
    opts = {"bounds": [-5, 5], "seed": SEED, "verbose": -9, "maxfevals": BUDGET}
    try:
        # IPOP: restarts with the population doubled each time (pycma's documented incpopsize).
        cma.fmin(rec, x0.copy(), 2.0, opts, restarts=9, incpopsize=2, bipop=False)
    except OutOfBudget:
        pass
    return rec


def sig(v: float) -> float:
    return float(f"{v:.6g}")


def pack(rec: Recorder) -> dict:
    flat = [round(c, 4) for xy in rec.xs for c in xy]
    return {"n": rec.n, "x": flat, "regret": [sig(r) for r in rec.regret]}


def landscape(problem, fopt: float) -> dict:
    axis = np.linspace(-5.0, 5.0, GRID_N)
    cells = []
    for y in axis:  # row-major: row = x2 (bottom to top), column = x1
        for x in axis:
            cells.append(int(round(10.0 * np.log10(max(float(problem(np.array([x, y]))) - fopt, 1e-8)))))
    return {"n": GRID_N, "lo": -5.0, "hi": 5.0, "unit": "10 x log10(f - f_opt), floor 1e-8", "v": cells}


def poster(path: Path, data: dict) -> None:
    """A static SVG of the finished runs on the landscape (no-JS, print and link previews)."""
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    n = data["grid"]["n"]
    grid = np.array(data["grid"]["v"], dtype=float).reshape(n, n) / 10.0
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.6), constrained_layout=True)
    style = {"amm": ("AMM (c46)", "#111111", "o"), "ipop": ("pycma IPOP", "#009e73", "s")}
    for ax, key in zip(axes, ("amm", "ipop")):
        label, colour, marker = style[key]
        ax.imshow(grid, origin="lower", extent=(-5, 5, -5, 5), cmap="Blues_r", aspect="equal", interpolation="none")
        run = data["runs"][key]
        xy = np.array(run["x"]).reshape(-1, 2)
        ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=0.6, alpha=0.8)
        ax.scatter(xy[:, 0], xy[:, 1], s=5, color=colour, marker=marker, zorder=3)
        best = int(np.argmin(run["regret"]))
        ax.scatter([xy[best, 0]], [xy[best, 1]], s=60, marker="*", color="#d55e00", edgecolor="black", zorder=4)
        ax.set_title(f"{label}: {run['n']} evaluations", fontsize=9)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.tick_params(labelsize=7)
    fig.savefig(path, format="svg", dpi=100)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(SITE / "data" / "demo"))
    ap.add_argument("--publish", action="store_true", help="also copy the JSON and posters to public/demo/")
    args = ap.parse_args()
    t_start = time.process_time()

    import amm_optimizer as atm
    import cma
    import cocoex
    from amm_optimizer import presets

    if not Path(atm.__file__).resolve().is_relative_to((FROZEN_WORKTREE / "src").resolve()):
        sys.exit(f"atm was imported from {atm.__file__}; put {FROZEN_WORKTREE}/src on PYTHONPATH")
    claims = json.loads((SITE / "data" / "claims.json").read_text(encoding="utf-8"))["claims"]
    want_hash = claims["freeze.config_hash"]["value"]
    got_hash = presets.get("c46").config_hash()
    if got_hash != want_hash:
        sys.exit(f"config hash {got_hash} != claims.json {want_hash}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for fid, name, note in FUNCTIONS:
        fopt = fopt_of(cocoex, fid)
        problem = make_problem(cocoex, fid)
        x0 = np.random.RandomState(SEED).uniform(-5.0, 5.0, DIM)

        plain, _ = run_amm(atm, presets, make_problem(cocoex, fid), fopt, x0, profile=False)
        hooked, prof = run_amm(atm, presets, make_problem(cocoex, fid), fopt, x0, profile=True)
        same = plain.xs == hooked.xs
        spans = prof.atm_spans() if same else []
        ipop = run_ipop(cma, make_problem(cocoex, fid), fopt, x0)

        data = {
            "header": {
                "function": f"bbob f{fid} {name}",
                "function_id": fid,
                "note": note,
                "D": DIM,
                "budget": BUDGET,
                "budget_per_dim": BUDGET // DIM,
                "seed": SEED,
                "instance": INSTANCE,
                "x0": [round(float(v), 4) for v in x0],
                "f_opt": fopt,
                "regret": "f(x) - f_opt, 6 significant digits; f_opt is never given to either optimizer",
                "code_commit": commit,
                "preset": "c46",
                "config_hash": got_hash,
                "recorded": dt.date.today().isoformat(),
                "versions": {"cma": cma.__version__, "cocoex": cocoex.__version__,
                             "numpy": np.__version__, "python": sys.version.split()[0]},
                "amm": f"atm.minimize(f, x0, {BUDGET}, seed={SEED}, config=presets.get('c46'), bounds=(-5, 5))",
                "ipop": f"cma.fmin(f, x0, 2.0, {{bounds: [-5, 5], seed: {SEED}, maxfevals: {BUDGET}}}, restarts=9, incpopsize=2); sigma0 2.0",
                "note_ipop": "Demo setting, not the benchmark harness arm. Both runs stop after the last allowed evaluation.",
                "stages": {
                    "source": "atm.Profiler hook: 'init' and 'section' blocks of the ATM sampler (src/atm/profiling.py)",
                    "available": bool(spans),
                    "hook_changes_trajectory": not same,
                    "atm_sampler_spans": spans,
                    "other_stages": "not reported by any hook in the frozen code",
                },
            },
            "grid": landscape(problem, fopt),
            "runs": {"amm": pack(plain), "ipop": pack(ipop)},
        }
        stem = f"f{fid}_d{DIM}_b{BUDGET}_s{SEED}"
        (out / f"{stem}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        poster(out / f"{stem}.svg", data)
        index.append({"id": f"f{fid}", "file": f"{stem}.json", "poster": f"{stem}.svg",
                      "title": f"f{fid} {name}", "note": note, "D": DIM, "budget": BUDGET})
        print(f"f{fid}: AMM {plain.n} evals, regret {min(plain.regret):.3g}; IPOP {ipop.n} evals, "
              f"regret {min(ipop.regret):.3g}; hook ok={same}; ATM spans {spans}")

    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    if args.publish:
        pub = SITE / "public" / "demo"
        pub.mkdir(parents=True, exist_ok=True)
        for f in list(out.glob("*.json")) + list(out.glob("*.svg")):
            shutil.copy2(f, pub / f.name)
    total = sum(f.stat().st_size for f in out.glob("*.json"))
    print(f"JSON payload {total / 1000:.0f} kB; process CPU {time.process_time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
