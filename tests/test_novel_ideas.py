"""The options of a screen of new ideas: face-tangent
probes, the evaluation memo, STEP's early check, box-relative steps and the box
normalisation; all off by default, so the presets are unchanged."""

import numpy as np
import pytest

from amm_optimizer import presets, qn_phase
from amm_optimizer.api import minimize
from amm_optimizer.bounds import Box
from amm_optimizer.objective import CountedObjective
from amm_optimizer.step_phase import run_step_phase

# Pins research-only options: deselected in the default tier (tools/ci.py).
pytestmark = pytest.mark.research

BOX5 = Box.make((-5.0, 5.0), 5)


def sphere(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum((x - 1.0) ** 2))


def rotated(x):
    """A non-separable quadratic (every pair of coordinates interacts)."""
    x = np.asarray(x, dtype=float) - 1.0
    return float(np.sum(x**2) + np.sum(x) ** 2)


def test_face_tangent_keeps_mirrored_pairs_in_the_box():
    rng = np.random.RandomState(0)
    found = 0
    for _ in range(500):
        x = rng.uniform(-5, 5, 5)
        x[: rng.randint(0, 3)] = 5.0
        u = rng.randn(5)
        u /= np.linalg.norm(u)
        v = qn_phase.face_tangent(u, x, BOX5, 3.0)
        if v is None:
            continue
        found += 1
        assert np.linalg.norm(v) == pytest.approx(1.0)
        assert np.all(x + 3.0 * v <= 5.0 + 1e-12)
        assert np.all(x - 3.0 * v >= -5.0 - 1e-12)
        assert np.all(v[x == 5.0] == 0.0)  # a coordinate on a face does not move
    assert found > 400


def test_face_tangent_leaves_an_interior_direction_alone():
    u = np.ones(5) / np.sqrt(5)
    assert np.array_equal(qn_phase.face_tangent(u, np.zeros(5), BOX5, 3.0), u)
    assert qn_phase.face_tangent(u, np.full(5, 5.0), BOX5, 3.0) is None


def test_tangent_probe_reads_a_quadratic_on_a_face():
    """Clipped pairs on a face can read NaN or far from 2; tangent pairs read 2."""
    x = np.array([5.0, 5.0, 4.0, 0.0, -1.0])
    reads = []
    for faces in (False, True):
        objective = CountedObjective(sphere, 1000)
        out = qn_phase.curvature_beta(
            objective, np.random.RandomState(3), x, sphere(x), BOX5, faces=faces
        )
        assert out is not None and out[3] == 24  # the same count
        reads.append(out[0])
    assert reads[1] == pytest.approx(2.0, abs=1e-6)


def test_memo_does_not_charge_a_repeated_point():
    calls = []

    def fun(x):
        calls.append(1)
        return sphere(x)

    objective = CountedObjective(fun, 3, memo=True)
    x = np.zeros(5)
    assert objective.evaluate_point(x) == objective.evaluate_point(x.copy())
    assert objective.calls == 1 and objective.memo_hits == 1 and len(calls) == 1
    costs, complete = objective.evaluate_columns(np.stack([x, x + 1.0], axis=1))
    assert complete and costs[0] == sphere(x) and objective.calls == 2
    assert objective.evaluate_point(x + 2.0) is not None  # the third charge
    assert objective.evaluate_point(x + 3.0) is None  # the budget is spent
    assert objective.evaluate_point(x) == sphere(x)  # but a known point still reads


def test_step_box_check_stops_a_non_separable_function_earlier():
    x = np.full(5, -4.0)
    used = []
    for box_check in (False, True):
        objective = CountedObjective(rotated, 10_000)
        out = run_step_phase(objective, x, rotated(x), BOX5, 1000, box_check=box_check)
        assert out.stalled
        used.append(out.used)
    assert used[1] < used[0]
    objective = CountedObjective(sphere, 10_000)
    out = run_step_phase(objective, x, sphere(x), BOX5, 300, box_check=True)
    assert not out.stalled  # separable: the early check passes


@pytest.mark.parametrize("budget", [50, 300])
def test_new_options_off_leave_c44_unchanged(budget):
    base = presets.with_changes(
        "c43", prefix_plateau_pop=True, prefix_plateau_min_dim=5
    )
    for changes in ({"box_steps": True}, {"box_normalize": True}):
        arm = presets.with_changes(
            "c43", prefix_plateau_pop=True, prefix_plateau_min_dim=5, **changes
        )
        a = minimize(
            sphere, np.full(5, 3.0), budget, seed=1, config=base, bounds=(-5, 5)
        )
        b = minimize(
            sphere, np.full(5, 3.0), budget, seed=1, config=arm, bounds=(-5, 5)
        )
        assert a.fun == b.fun and np.array_equal(a.x, b.x) and a.nfev == b.nfev
    assert qn_phase.X_UNIT == 1.0


def test_box_normalize_maps_a_scaled_box_back():
    config = presets.with_changes("c43", box_normalize=True)

    def scaled(x):
        return sphere(np.asarray(x) / 100.0)

    out = minimize(
        scaled, np.full(5, 300.0), 150, seed=2, config=config, bounds=(-500, 500)
    )
    assert np.all(np.abs(out.x) <= 500.0)
    assert out.fun == pytest.approx(scaled(out.x))
    assert out.fun < 1e-6


def test_box_steps_sets_and_resets_the_unit():
    config = presets.with_changes("c43", box_steps=True)
    minimize(sphere, np.full(5, 3.0), 60, seed=0, config=config, bounds=(-50, 50))
    assert qn_phase.X_UNIT == 10.0
    minimize(
        sphere, np.full(5, 3.0), 60, seed=0, config=presets.get("c43"), bounds=(-50, 50)
    )
    assert qn_phase.X_UNIT == 1.0
