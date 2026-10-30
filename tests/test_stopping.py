"""Stopping a run early: ``callback``, ``ftarget`` and ``catch_interrupt`` (WP4.2, D5).

A stop is raised from inside ``fun`` (``objective._Stop``) and must leave every phase
at once, the way the phases' budget signal does: the tests stop runs inside each
phase family and check that not one evaluation follows.
"""

from __future__ import annotations

import dataclasses
import traceback
import warnings

import numpy as np
import pytest

import amm_optimizer as amm
from amm_optimizer import presets
from amm_optimizer.bounds import Box
from amm_optimizer.config import ATMConfig
from amm_optimizer.objective import CountedObjective, Watch, _Stop
from amm_optimizer.qn_phase import run_lbfgsb_restarts
from preset_cases import wave5_variant

ANYTIME = presets.with_changes("c46", anytime="replay")
CONFIGS = {"default": None, "anytime": ANYTIME, "atm": ATMConfig()}


def sphere(x):
    return float(np.sum((x - 1.0) ** 2))


def rosen(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(x):
    z = x - 0.5
    return float(10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z)))


def chain(x):
    z = x - 0.7
    return float(np.sum(z**2) + np.sum((z[1:] - 0.5 * z[:-1]) ** 2))


def rotated_rastrigin(dim):
    q = np.linalg.qr(np.random.RandomState(0).randn(dim, dim))[0]
    return lambda x: rastrigin(q @ (np.asarray(x) - 0.4) + 0.4)


class Falling:
    """Every call improves on the last one, so the callback sees every evaluation."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return 1e-9 * sphere(x) - self.calls


def run(fun, config, dim=3, budget=600, **kwargs):
    return amm.minimize(
        fun, bounds=[(-5, 5)] * dim, budget=budget, seed=1, config=config, **kwargs
    )


@pytest.mark.parametrize("name", list(CONFIGS))
def test_ftarget_stops_the_run_when_the_target_is_reached(name):
    calls = []

    def f(x):
        calls.append(sphere(x))
        return calls[-1]

    result = run(f, CONFIGS[name], dim=2, budget=4000, ftarget=1e-8)
    assert result.status == 1 and result.success
    assert result.message == "The best value reached ftarget."
    assert result.fun <= 1e-8 < min(calls[:-1])  # the last call reached it
    assert result.nfev == len(calls) < 4000
    assert result.fun == calls[-1] == sphere(result.x)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_a_callback_raising_stop_iteration_at_evaluation_50_spends_50(name):
    def stop_at_50(intermediate_result):
        if intermediate_result.nfev == 50:
            raise StopIteration

    fun = Falling()
    result = run(fun, CONFIGS[name], callback=stop_at_50)
    assert result.nfev == fun.calls == 50
    assert (result.status, result.success) == (2, True)
    assert result.message == "The callback raised StopIteration."
    assert result.fun == 1e-9 * sphere(result.x) - 50
    assert result.nit == 50  # every evaluation improved


@pytest.mark.parametrize("name", list(CONFIGS))
def test_the_callback_sees_each_improvement_once_with_a_copy_of_the_best_point(name):
    seen = []

    def watch(intermediate_result):
        assert isinstance(intermediate_result, amm.OptimizeResult)
        assert set(intermediate_result) == {"x", "fun", "nfev", "nit"}
        seen.append(dict(intermediate_result))
        intermediate_result.x[:] = 99.0  # must not reach the run

    values = []

    def f(x):
        values.append(sphere(x))
        return values[-1]

    result = run(f, CONFIGS[name], callback=watch)
    plain = run(sphere, CONFIGS[name])
    assert (result.fun, result.nfev) == (plain.fun, plain.nfev)
    np.testing.assert_array_equal(result.x, plain.x)
    running = np.minimum.accumulate(values)
    improved = [i for i in range(len(values)) if i == 0 or running[i] < running[i - 1]]
    assert [s["nfev"] for s in seen] == [i + 1 for i in improved]
    assert [s["fun"] for s in seen] == [values[i] for i in improved]
    assert [s["nit"] for s in seen] == list(range(1, len(seen) + 1))
    assert result.nit == len(seen)
    assert seen[-1]["fun"] == result.fun


@pytest.mark.parametrize("name", list(CONFIGS))
def test_a_callback_that_never_stops_changes_nothing(name):
    plain = run(rastrigin, CONFIGS[name], dim=4, budget=800)
    watched = run(
        rastrigin,
        CONFIGS[name],
        dim=4,
        budget=800,
        callback=lambda r: None,
        ftarget=-1.0,  # never reached
    )
    assert (watched.fun, watched.nfev, watched.nit, watched.nrestarts) == (
        plain.fun,
        plain.nfev,
        plain.nit,
        plain.nrestarts,
    )
    np.testing.assert_array_equal(watched.x, plain.x)
    assert watched.status == plain.status == 0


def test_other_exceptions_from_the_callback_propagate():
    def broken(intermediate_result):
        raise RuntimeError("from the callback")

    with pytest.raises(RuntimeError, match="from the callback"):
        run(sphere, None, callback=broken)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"callback": 3}, TypeError, "callback must be callable"),
        ({"ftarget": "0"}, TypeError, "ftarget must be a real number"),
        ({"ftarget": True}, TypeError, "ftarget must be a real number"),
        ({"ftarget": np.zeros(2)}, TypeError, "ftarget must be a real number"),
    ],
)
def test_bad_stopping_arguments_fail_before_any_evaluation(kwargs, error, match):
    fun = Falling()
    with pytest.raises(error, match=match):
        run(fun, None, **kwargs)
    assert fun.calls == 0


def test_ftarget_accepts_numpy_scalars_and_integers():
    for target in (np.float32(1e-6), 0, np.int64(0)):
        result = run(sphere, None, dim=2, budget=4000, ftarget=target)
        assert result.status == 1 and result.fun <= target


class Interrupting:
    def __init__(self, at):
        self.at, self.values = at, []

    def __call__(self, x):
        if len(self.values) + 1 == self.at:
            raise KeyboardInterrupt
        self.values.append(rastrigin(x))
        return self.values[-1]


@pytest.mark.parametrize("name", list(CONFIGS))
def test_catch_interrupt_returns_the_best_point_so_far(name):
    fun = Interrupting(at=120)
    with pytest.warns(RuntimeWarning, match="interrupted"):
        result = run(fun, CONFIGS[name], catch_interrupt=True)
    assert (result.status, result.success) == (3, False)
    assert result.nfev == len(fun.values) == 119  # the interrupted call: not counted
    assert result.fun == min(fun.values) == rastrigin(result.x)


def test_without_catch_interrupt_the_interrupt_propagates():
    with pytest.raises(KeyboardInterrupt):
        run(Interrupting(at=120), None)


def test_an_interrupt_before_any_finite_value_propagates():
    with pytest.raises(KeyboardInterrupt):
        run(Interrupting(at=1), None, catch_interrupt=True)


def test_the_engine_takes_the_same_arguments():
    result = amm.api.minimize(sphere, np.zeros(2), 3000, seed=0, ftarget=1e-6)
    assert result.status == 1 and result.nfev < 3000


# --- a stop inside every phase family ---------------------------------------------


def _hybrid(**changes):
    variant = dataclasses.replace(
        wave5_variant(), es_kind="cma", cma_handoff=0.2, **changes
    )
    return ATMConfig(iterations_per_hyperparameter_set=1000000, variant=variant)


# (frames that must be on the stack, configuration, fun, dim, budget[, which
# improvement inside the phase stops the run, default the second]); each route was
# found by recording the modules on the stack at each improvement
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    PHASES = {
        "ATM sampler": (["search", "section"], ATMConfig(), sphere, 3, 300),
        "composite": (["search", "section"], ATMConfig.composite(), sphere, 3, 300),
        "COBYQA (SciPy)": (["cobyqa_phase", "_cobyqa_py"], None, sphere, 2, 20),
        "basin route, SLSQP": (["basin_phase", "_slsqp_py"], None, sphere, 3, 300),
        "prefix, SLSQP": (["es_phase", "qn_phase", "_slsqp_py"], None, rosen, 5, 150),
        "prefix, STEP": (["es_phase", "step_phase"], None, sphere, 5, 150),
        "prefix, Nelder-Mead (SciPy)": (
            ["nm_phase", "_optimize"],
            None,
            rastrigin,
            2,
            50,
        ),
        "chain route": (["chain_phase"], None, chain, 6, 1800),
        "lq2 populations": (
            ["pop_cma", "surrogate"],
            None,
            rotated_rastrigin(5),
            5,
            500,
        ),
        "CMA-ES finisher": (["cma_phase"], presets.get("c23"), rosen, 5, 2000),
        "anytime stages": (["anytime", "cobyqa_phase"], ANYTIME, sphere, 3, 2000),
        # the hand-over's ATM sampler improves only once on this run
        "anytime hand-over": (["anytime", "search"], ANYTIME, rosen, 2, 30000, 1),
    }


def _on_stack():
    names = set()
    for frame in traceback.extract_stack():
        name = frame.filename.replace("\\", "/").rsplit("/", 1)[-1]
        names.add(name.removesuffix(".py"))
    return names


@pytest.mark.parametrize("phase", list(PHASES))
def test_a_stop_inside_a_phase_leaves_it_at_once(phase):
    frames, config, fun, dim, budget, *nth = PHASES[phase]
    stop_at = nth[0] if nth else 2
    inside = []

    def stop_inside(intermediate_result):
        if set(frames) <= _on_stack():
            inside.append(intermediate_result.nfev)
            if len(inside) == stop_at:  # mid-phase
                raise StopIteration

    calls = []

    def f(x):
        calls.append(fun(x))
        return calls[-1]

    result = run(f, config, dim=dim, budget=budget, callback=stop_inside)
    assert len(inside) == stop_at, f"too few improvements inside {frames}"
    assert result.status == 2
    assert result.nfev == len(calls) == inside[-1]  # no evaluation after the stop
    assert result.fun == min(c for c in calls if np.isfinite(c)) == calls[-1]


def test_a_stop_leaves_scipys_l_bfgs_b():
    box = Box.make((-5.0, 5.0), 4)
    watch = Watch(rosen, ftarget=1.0)
    objective = CountedObjective(watch, 2000)
    x = np.full(4, -2.0)
    with pytest.raises(_Stop) as stop:
        run_lbfgsb_restarts(objective, np.random.RandomState(0), x, rosen(x), box, 2000)
    assert stop.value.status == 1
    assert watch.best_f <= 1.0
    assert objective.calls == watch.calls < 2000
