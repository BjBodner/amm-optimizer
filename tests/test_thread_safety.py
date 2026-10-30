"""Runs in separate threads on different boxes match the same runs in sequence.

Review B5 (WP4.9): a step unit used to be a module global that every ``minimize``
call wrote; 0.7.0 removed the option that set it, so no run state is shared.
"""

import threading

import numpy as np

from amm_optimizer import presets
from amm_optimizer.api import minimize


def config():
    # c43: c46 maps every box onto bbob's (box_normalize)
    return presets.get("c43")


def objective(scale, barrier=None):
    def fun(x):
        if barrier is not None and not fun.met:
            # both runs have started before either evaluates more
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
