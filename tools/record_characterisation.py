"""Record (or check) the bit-exact characterisation snapshots on bbob.

Usage (from the repo root, ``PYTHONPATH=src``):

    python tools/record_characterisation.py              # both sets, write
    python tools/record_characterisation.py --check      # compare, write nothing
    python tools/record_characterisation.py --set named --only c26,qn6 --jobs 3
    python tools/record_characterisation.py --set named --cases d3 --jobs 4
    python tools/record_characterisation.py --set arms --missing      # new arms only

``named`` writes ``tests/characterisation.json`` (the shipped configurations of
``tests/characterisation_cases.NAMED`` on its ``GRID``; ``--cases`` records a subset:
``fast`` / ``sep`` (``FAST`` / ``SEPARATING``) or ``d3`` (the D 3 cases), merged into
the stored entries case by case); ``arms`` (research) writes
``tests/characterisation_arms.json`` (every recorded arm config on ``ARM_PROBES``;
``--missing`` records only the arm configs the file lacks, as the research harness
does when a batch starts). With
``--only`` the other entries of an existing file are kept; ``--emit PATH`` writes the
snapshots there instead (the tests use it). These pin this package's behaviour, not
a correct one: re-record only for a deliberate behaviour change, in its own commit,
and say why in the message (``docs/DEVIATIONS.md``).

The BLAS libraries run single-threaded, as in ``benchmarks/quick_scan.py``: OpenBLAS
with several threads splits its reductions differently, which changes the last bits
of SLSQP's steps (found in the 2026-10-19 code review). The variables are
set before NumPy and SciPy load, so this script must run in a fresh process.

With ``--jobs N`` every set runs in one process pool (``benchmarks/process_pool.py``),
which replaces itself and reruns the unfinished runs if it breaks: on this machine a
pool worker can die under heavy load with ``WinError 6`` (lesson 23).
"""

from __future__ import annotations

import os

BLAS_THREADS = "1"
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = BLAS_THREADS

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from process_pool import RetryingPool  # noqa: E402

import characterisation_cases as cc  # noqa: E402

Task = tuple[str, str, str, cc.Case]  # (set, key, config text, case)


def _work(task: Task) -> tuple[Task, dict[str, Any]]:
    which, key, text, case = task
    # A shipped configuration is built by the package itself; only an arm's config
    # text needs the research harness's parser.
    config = cc.named_config(key) if which == "named" else cc.parse(text)
    return task, cc.run(config, case)


CASES = {
    "all": cc.GRID,
    "fast": cc.FAST,
    "sep": cc.SEPARATING,
    "d3": [case for case in cc.GRID if case.dim == 3],
    "f23": [case for case in cc.GRID if case.function == 23],
}


def tasks_for(
    which: str, only: set[str], cases_name: str = "all"
) -> tuple[list[Task], dict[str, Any]]:
    """The tasks of one set and the skeleton of its snapshot file."""
    configs: dict[str, Any] = {}
    tasks: list[Task] = []
    if which == "named":
        cases = CASES[cases_name]
        for name, text in cc.NAMED.items():
            if only and name not in only:
                continue
            configs[name] = {"config": text, "cases": {}}
            tasks += [("named", name, text, case) for case in cases]
    else:
        for key, entry in cc.arm_configs().items():
            if only and key not in only:
                continue
            configs[key] = {**entry, "cases": {}}
            tasks += [("arms", key, entry["config"], case) for case in cc.ARM_PROBES]
    return tasks, configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--set", choices=("named", "arms", "all"), default="all")
    parser.add_argument("--only", default="", help="config names or keys")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--cases", choices=tuple(CASES), default="all")
    parser.add_argument(
        "--missing", action="store_true", help="arms: only configs the file lacks"
    )
    parser.add_argument("--emit", type=Path, help="write the snapshots here instead")
    args = parser.parse_args()
    only = set(filter(None, args.only.split(",")))
    sets = ("named", "arms") if args.set == "all" else (args.set,)
    paths = {"named": cc.NAMED_SNAPSHOTS, "arms": cc.ARM_SNAPSHOTS}
    failures = 0
    pool = RetryingPool(args.jobs)  # one pool for every set, started on first use
    for which in sets:
        want = only
        if args.missing and which == "arms":
            path = paths["arms"]
            stored = json.loads(path.read_text(encoding="utf-8"))["configs"]
            want = set(cc.arm_configs()) - set(stored) - set(cc.REMOVED_BY_MERGE)
            if not want:
                print("arms: no missing configs", flush=True)
                continue
        tasks, configs = tasks_for(which, want, args.cases)
        tasks.sort(key=lambda t: -t[3].budget)  # longest first
        print(f"{which}: {len(configs)} configs, {len(tasks)} runs", flush=True)
        started = time.perf_counter()
        done = map(_work, tasks) if args.jobs <= 1 else pool.map(_work, tasks)
        for k, ((_, key, _, case), snap) in enumerate(done, 1):
            configs[key]["cases"][case.name] = snap
            if k % max(len(tasks) // 10, 1) == 0:
                print(f"  {k}/{len(tasks)} {time.perf_counter() - started:.0f}s")
        meta = {
            **cc.environment(),
            "blas_threads": BLAS_THREADS,
            "note": "bit-exact characterisation; tools/record_characterisation.py",
        }
        if args.emit is not None:
            emitted = args.emit.with_name(f"{args.emit.stem}_{which}.json")
            emitted.write_text(
                json.dumps({"_meta": meta, "configs": configs}), encoding="utf-8"
            )
            continue
        path = paths[which]
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if args.check:
            stored = old.get("configs", {})
            for key, entry in configs.items():
                for name, snap in entry["cases"].items():
                    recorded = stored.get(key, {}).get("cases", {}).get(name)
                    if recorded != snap:
                        failures += 1
                        print(f"  MISMATCH {which} {key} {name}")
            print(f"{which}: {failures} mismatches")
            continue
        partial = bool(want) or (which == "named" and args.cases != "all")
        merged = dict(old.get("configs", {})) if partial else {}
        if which == "arms":  # kept for the record (``REMOVED_BY_MERGE``)
            for key, entry in old.get("configs", {}).items():
                if key in cc.REMOVED_BY_MERGE:
                    merged.setdefault(key, entry)
        for key, entry in configs.items():
            if which == "named" and args.cases != "all" and key in merged:
                # a subset of cases: merged into the stored entry case by case
                entry["cases"] = {**merged[key]["cases"], **entry["cases"]}
            entry["cases"] = dict(sorted(entry["cases"].items()))
            merged[key] = entry
        out = {"_meta": meta, "configs": merged}
        path.write_text(
            json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"wrote {path} ({time.perf_counter() - started:.0f}s)")
    pool.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
