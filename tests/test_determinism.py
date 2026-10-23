"""Results must not depend on process state: hash seed, threads, call history."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import amm_optimizer as atm

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = """
import hashlib, json, sys
import numpy as np
import amm_optimizer as atm

def f(x):
    costs.append(float(np.sum((x - 2) ** 2)))
    return costs[-1]

costs = []
rng = np.random.RandomState(11)
x0 = 10 * (rng.rand(6) - 0.5)
res = atm.minimize(f, x0, 1500, rng=rng)
print(json.dumps({
    "sha": hashlib.sha256(np.asarray(costs).tobytes()).hexdigest(),
    "x": res.x.tolist(),
    "nfev": res.nfev,
}))
"""


def run_in_subprocess(**env_overrides):
    env = {**os.environ, **{k: str(v) for k, v in env_overrides.items()}}
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        cwd=ROOT,
    )
    return json.loads(out.stdout)


def test_same_result_across_processes_hash_seeds_and_thread_counts():
    first = run_in_subprocess(
        PYTHONHASHSEED=0, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1
    )
    second = run_in_subprocess(
        PYTHONHASHSEED=12345, OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4
    )
    assert first == second


def test_repeated_runs_in_one_process_are_identical():
    def run():
        costs = []

        def f(x):
            costs.append(float(np.sum((x - 2) ** 2)))
            return costs[-1]

        rng = np.random.RandomState(3)
        res = atm.minimize(f, 10 * (rng.rand(4) - 0.5), 800, rng=rng)
        return hashlib.sha256(np.asarray(costs).tobytes()).hexdigest(), res.x

    (sha_a, x_a), (sha_b, x_b) = run(), run()
    assert sha_a == sha_b
    np.testing.assert_array_equal(x_a, x_b)


def test_interleaved_runs_do_not_share_state():
    """A run depends only on its own rng, not on the global NumPy stream."""

    def solo(seed):
        rng = np.random.RandomState(seed)
        return atm.minimize(lambda x: float(np.sum(x**2)), np.ones(3), 500, rng=rng).x

    expected = [solo(1), solo(2)]
    # Run them again after disturbing the global stream.
    np.random.seed(999)
    np.random.rand(10)
    again = [solo(1), solo(2)]
    for a, b in zip(expected, again, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("solvers", [["reference", "new"]])
def test_perf_harness_smoke(solvers):
    """The benchmark script runs, and both solvers produce identical traces."""
    out = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks" / "perf.py"),
            "--repeats",
            "1",
            "--scenarios",
            "sphere_d5",
            "--solvers",
            *solvers,
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    traces = {
        line.split()[-1]
        for line in out.stdout.splitlines()
        if line.startswith("sphere_d5")
    }
    assert len(traces) == 1
