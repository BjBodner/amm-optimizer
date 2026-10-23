"""The gaps4 toy's opt-in options (``docs/research/toys/gaps4.md``): ``centre_point``,
``smooth_bfgs_second`` and ``prefix_low_beta``. Each default is c45's, which the last
test checks against c45's recorded runs."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import characterisation_cases as cc
from amm_optimizer import minimize, presets
from amm_optimizer.config import Variant


def griewank_rosenbrock_like(x):
    """Low at the box centre (as bbob's f19 is), rugged elsewhere."""
    z = 0.2 * np.asarray(x, dtype=float) + 0.5
    s = 100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2
    return float(10.0 * np.mean(s / 4000.0 - np.cos(s)) + 10.0)


def shifted_sphere(x):
    return float(np.sum((np.asarray(x, dtype=float) - 3.0) ** 2))


def run(config, fun, dim=5, budget=150, seed=301):
    x0 = np.random.RandomState(seed + 100).uniform(-5, 5, dim)
    calls = []

    def logged(x):
        calls.append(np.array(x, dtype=float))
        return fun(x)

    res = minimize(logged, x0, budget, seed=seed, config=config, bounds=(-5.0, 5.0))
    return res, np.array(calls), x0


def test_defaults_are_off():
    v = Variant()
    assert (v.centre_point, v.smooth_bfgs_second, v.prefix_low_beta) == ("off", 0, 0)
    assert presets.get("c45") == presets.with_changes(
        "c45", centre_point="off", smooth_bfgs_second=0.0, prefix_low_beta=0.0
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"centre_point": "yes"},
        {"centre_point_min_dim": 0},
        {"smooth_bfgs_second": -1.0},
        {"prefix_low_beta": -0.1},
        {"prefix_low_beta_min_dim": -1},
    ],
)
def test_bad_values_are_refused(bad):
    with pytest.raises(ValueError):
        presets.with_changes("c45", **bad)


@pytest.mark.parametrize("mode", ["record", "start"])
def test_the_centre_is_the_first_evaluation_and_counts_for_the_best(mode):
    config = presets.with_changes("c45", centre_point=mode)
    for fun in (griewank_rosenbrock_like, shifted_sphere):
        res, calls, _ = run(config, fun)
        assert len(calls) <= 150 and np.array_equal(calls[0], np.zeros(5))
        assert res.fun <= fun(np.zeros(5))
        assert res.fun == min(fun(c) for c in calls)


def test_below_its_min_dim_the_centre_is_not_evaluated():
    on = presets.with_changes("c45", centre_point="record", centre_point_min_dim=5)
    _, a, _ = run(on, shifted_sphere, dim=3, budget=90)
    _, b, _ = run(presets.get("c45"), shifted_sphere, dim=3, budget=90)
    assert np.array_equal(a, b)
    _, c, _ = run(on, shifted_sphere, dim=5, budget=150)
    assert np.array_equal(c[0], np.zeros(5))


def test_record_leaves_the_path_alone_and_start_moves_it():
    base = presets.get("c45")
    record = presets.with_changes("c45", centre_point="record")
    start = presets.with_changes("c45", centre_point="start")
    _, b, x0 = run(base, griewank_rosenbrock_like)
    _, r, _ = run(record, griewank_rosenbrock_like)
    _, s, _ = run(start, griewank_rosenbrock_like)
    # record: the centre, then c45's own evaluations (cut one short at the end)
    assert np.array_equal(r[1:], b[: len(r) - 1])
    # start: the centre, x0, then a run from the centre (it beats x0 here)
    assert np.array_equal(s[1], x0)
    assert griewank_rosenbrock_like(np.zeros(5)) < griewank_rosenbrock_like(x0)
    assert np.array_equal(s[2], np.zeros(5))
    # where x0 beats the centre, start runs as record does
    _, s2, x0b = run(start, shifted_sphere)
    _, r2, _ = run(record, shifted_sphere)
    if shifted_sphere(x0b) <= shifted_sphere(np.zeros(5)):
        assert np.array_equal(s2[2:], r2[1 : len(s2) - 1])


@pytest.mark.skipif(
    json.loads(cc.NAMED_SNAPSHOTS.read_text(encoding="utf-8"))["_meta"]["system"]
    != cc.environment()["system"],
    reason="snapshots recorded on another system",
)
@pytest.mark.snapshot
def test_c45_runs_are_its_recorded_runs(tmp_path):
    """The defaults leave c45 bit-identical: its fast cases, run in a fresh process
    with single-threaded BLAS (``tools/record_characterisation.py``), match the
    snapshots recorded at e529629."""
    pytest.importorskip("cocoex")  # the recorder runs bbob problems
    stored = json.loads(cc.NAMED_SNAPSHOTS.read_text(encoding="utf-8"))
    meta = stored["_meta"]
    here = cc.environment()
    if any(meta.get(k) != here[k] for k in ("numpy", "scipy")):
        pytest.skip("snapshots recorded with other numpy / scipy versions")
    out = tmp_path / "run.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cc.ROOT / "src"), *filter(None, [env.get("PYTHONPATH")])]
    )
    recorder = cc.ROOT / "tools" / "record_characterisation.py"
    subprocess.run(
        [sys.executable, str(recorder), "--set", "named", "--only", "c45"]
        + ["--cases", "fast", "--jobs", "1", "--emit", str(out)],
        env=env,
        check=True,
        capture_output=True,
    )
    got = json.loads(out.with_name("run_named.json").read_text(encoding="utf-8"))
    got_cases = got["configs"]["c45"]["cases"]
    want = stored["configs"]["c45"]["cases"]
    assert got_cases and all(got_cases[c] == want[c] for c in got_cases)


def rosenbrock(x):
    """Rotation-free Rosenbrock: a second minimum near (-1, 1, ..., 1) from D 4."""
    z = 0.4 * np.asarray(x, dtype=float) + 1.0
    return float(np.sum(100.0 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1.0) ** 2))


@pytest.mark.snapshot
def test_second_bfgs_descent_runs_and_is_off_on_rugged_runs():
    base = presets.get("c45")
    on = presets.with_changes("c45", smooth_bfgs_second=1.5)
    # D 5 at 300 x D: the hand-over's smooth route takes the BFGS descent
    res, calls, _ = run(on, rosenbrock, dim=5, budget=1500, seed=302)
    assert len(calls) <= 1500 and np.isfinite(res.fun)
    # seed 306: c45's descent ends at the second minimum (f about 3.9) and stays;
    # the second descent from a uniform point finds the global one
    stuck, _, _ = run(base, rosenbrock, dim=5, budget=1500, seed=306)
    freed, _, _ = run(on, rosenbrock, dim=5, budget=1500, seed=306)
    assert stuck.fun > 3.0 and freed.fun < 1e-8
    # a rugged reading never takes the BFGS descent: the same run as c45
    for seed in (303, 304):
        _, a, _ = run(on, griewank_rosenbrock_like, dim=5, budget=1500, seed=seed)
        _, b, _ = run(base, griewank_rosenbrock_like, dim=5, budget=1500, seed=seed)
        assert np.array_equal(a, b)


def katsuura_like(x):
    """Rough at every scale (as bbob's f23 is)."""
    z = np.asarray(x, dtype=float)
    out = 1.0
    for i, zi in enumerate(z, 1):
        t = sum(abs(2**j * zi - np.round(2**j * zi)) / 2**j for j in range(1, 20))
        out *= (1.0 + i * t) ** (10.0 / len(z) ** 1.2)
    return float(out - 1.0)


def test_low_beta_prefix_runs_and_needs_the_population_budget():
    base = presets.get("c45")
    on = presets.with_changes("c45", prefix_low_beta=0.3)
    res, calls, _ = run(on, katsuura_like, dim=5, budget=500, seed=302)
    assert len(calls) <= 500 and np.isfinite(res.fun)
    _, base_calls, _ = run(base, katsuura_like, dim=5, budget=500, seed=302)
    assert not np.array_equal(calls, base_calls)  # the reading is below 0.3 here
    # below 100 x D the rugged kind is already the (1+1): nothing changes
    _, a, _ = run(on, katsuura_like, dim=5, budget=450)
    _, b, _ = run(base, katsuura_like, dim=5, budget=450)
    assert np.array_equal(a, b)
    # below prefix_low_beta_min_dim: the same run as c45
    floor = presets.with_changes("c45", prefix_low_beta=0.3, prefix_low_beta_min_dim=10)
    _, a, _ = run(floor, katsuura_like, dim=5, budget=500, seed=302)
    assert np.array_equal(a, base_calls)
    # a smooth function never reads rugged: the same run as c45
    _, a, _ = run(on, shifted_sphere, dim=5, budget=500)
    _, b, _ = run(base, shifted_sphere, dim=5, budget=500)
    assert np.array_equal(a, b)
