"""Runs in separate threads do not share the step unit (review B5, WP4.9).

``Variant.box_steps`` scales the absolute steps in x (finite differences, COBYQA's
final radius, STEP's smallest interval) by the box width / 10. That unit used to be
the module global ``qn_phase.X_UNIT``, which every ``minimize`` call wrote, so two
threads on different boxes used whichever unit was written last. It is now a
context variable, read through ``qn_phase.x_unit()``.
"""

import threading

import numpy as np
import pytest

from amm_optimizer import presets, qn_phase
from amm_optimizer.api import minimize

# box_steps is a research option (deprecated in 0.6.0): it warns when set
pytestmark = pytest.mark.filterwarnings("ignore:Variant\\.box_steps:FutureWarning")


def config():
    # c43: c46 maps every box onto bbob's (box_normalize), where the unit is 1
    return presets.with_changes("c43", box_steps=True)


def objective(scale, barrier=None):
    def fun(x):
        if barrier is not None and not fun.met:
            # both runs have set their unit before either evaluates more
            fun.met = True
            barrier.wait(timeout=60)
        z = np.asarray(x) / scale
        return float(np.sum((z - 0.3) ** 2) + 0.1 * np.sum(np.abs(z)))

    fun.met = False
    return fun


CASES = [(10.0, (-50.0, 50.0)), (0.1, (-0.5, 0.5))]


def run(scale, box, barrier=None):
    result = minimize(
        objective(scale, barrier),
        np.full(5, 0.6 * box[1]),
        300,
        seed=3,
        config=config(),
        bounds=box,
    )
    return result.x.copy(), result.fun, result.nfev


def test_the_unit_matters_for_these_runs():
    # otherwise the threaded test below could not tell the units apart
    scale, box = CASES[0]
    with_unit = run(scale, box)
    plain = minimize(
        objective(scale),
        np.full(5, 0.6 * box[1]),
        300,
        seed=3,
        config=presets.get("c43"),
        bounds=box,
    )
    assert with_unit[1] != plain.fun or not np.array_equal(with_unit[0], plain.x)


def test_two_threads_on_different_boxes_match_sequential_runs():
    sequential = [run(scale, box) for scale, box in CASES]
    barrier = threading.Barrier(len(CASES))
    threaded: list = [None] * len(CASES)
    errors: list = []

    def work(i):
        try:
            threaded[i] = run(*CASES[i], barrier=barrier)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(CASES))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    for (x_seq, f_seq, n_seq), (x_thr, f_thr, n_thr) in zip(
        sequential, threaded, strict=True
    ):
        assert np.array_equal(x_seq, x_thr)
        assert f_seq == f_thr and n_seq == n_thr


def test_a_unit_set_in_a_thread_stays_in_that_thread():
    qn_phase.set_x_unit(1.0)
    seen = []

    def work():
        seen.append(qn_phase.x_unit())  # a new thread starts at the default
        qn_phase.set_x_unit(7.0)
        seen.append(qn_phase.x_unit())

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert seen == [1.0, 7.0]
    assert qn_phase.x_unit() == 1.0
