"""Cost of the profiling hooks: no hooks (older commit) vs default vs timing on.

    python benchmarks/profiler_overhead.py --baseline-ref aba0a7f --rounds 6

Each measurement is a separate process (so the baseline can be a different
checkout of the package); the modes are interleaved round by round so that slow
drifts of the machine hit all modes alike. Every worker checks that its result
equals the baseline's, so a faster or slower mode never changes the numbers.
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
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tarfile  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

WORKER = r"""
import gc, json, sys, time
import numpy as np
import amm_optimizer as atm

mode = sys.argv[1]
dim, budget, repeats = (int(a) for a in sys.argv[2:5])
fun = lambda x: float(np.sum(x ** 2))
kwargs = {"profile": True} if mode == "timing" else {}
x0 = np.full(dim, 3.0)
atm.api.minimize(fun, x0, budget, seed=0, **kwargs)  # warm-up
times = []
for _ in range(repeats):
    gc.disable()
    t = time.perf_counter_ns()
    result = atm.api.minimize(fun, x0, budget, seed=1, **kwargs)
    times.append(time.perf_counter_ns() - t)
    gc.enable()
print(json.dumps({"times": times, "fun": repr(result.fun), "nfev": result.nfev}))
"""


def measure(src: Path, mode: str, dim: int, budget: int, repeats: int) -> dict:
    env = dict(os.environ, PYTHONPATH=str(src))
    out = subprocess.run(
        [sys.executable, "-c", WORKER, mode, str(dim), str(budget), str(repeats)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline-ref", required=True, help="git ref without the hooks")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--repeats", type=int, default=15, help="runs per process")
    ap.add_argument("--rounds", type=int, default=6, help="interleaved rounds")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "base.tar"
        subprocess.run(
            ["git", "archive", args.baseline_ref, "src", "-o", str(archive)],
            cwd=ROOT,
            check=True,
        )
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        sources = {
            "baseline": Path(tmp) / "src",
            "default": ROOT / "src",
            "timing": ROOT / "src",
        }
        times: dict[str, list[int]] = {m: [] for m in sources}
        results = {}
        for _ in range(args.rounds):
            for mode, src in sources.items():
                r = measure(src, mode, args.dim, args.budget, args.repeats)
                times[mode].extend(r["times"])
                results[mode] = (r["fun"], r["nfev"])
    if len(set(results.values())) != 1:
        print("RESULTS DIFFER:", results)
        return 1
    base = statistics.median(times["baseline"])
    print(
        f"D={args.dim}, budget {args.budget}, "
        f"{args.rounds}x{args.repeats} runs per mode"
    )
    print(f"{'mode':<10}{'median ms':>11}{'min ms':>9}{'vs baseline':>13}")
    for mode, ts in times.items():
        med = statistics.median(ts)
        print(
            f"{mode:<10}{med / 1e6:>11.1f}{min(ts) / 1e6:>9.1f}"
            f"{100 * (med / base - 1):>+12.1f}%"
        )
    print("identical results in all modes:", results["baseline"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
