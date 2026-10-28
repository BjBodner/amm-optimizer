# User guide

This page explains the arguments of [`amm_optimizer.minimize`](api.md#amm_optimizer.minimize):
the box, the budget, the seed, the anytime mode and profiling. The examples run as written, top
to bottom. They share one objective:

```python
import numpy as np
import amm_optimizer as amm

def f(x):
    return float(np.sum((x - 1.0) ** 2))

result = amm.minimize(f, bounds=[(-5, 5)] * 3, budget=300, seed=0)
print(result.fun, result.nfev)
```

## The search box (`bounds`)

AMM reads the box: its probes, designs and step sizes are scaled to it. So `bounds` is required
unless you pass a `config` that does not use the box, such as `ATMConfig()`. The limits must be
finite. Any of these forms describes the same 3-dimensional box:

```python
from scipy.optimize import Bounds

forms = [
    [(-5, 5)] * 3,                          # one (low, high) pair per coordinate, as in SciPy
    np.array([[-5, 5]] * 3),                # the same pairs as an array of shape (D, 2)
    (np.full(3, -5.0), np.full(3, 5.0)),    # (lower, upper), one value per coordinate
    Bounds([-5] * 3, [5] * 3),              # scipy.optimize.Bounds
]
for bounds in forms:
    print(amm.minimize(f, bounds=bounds, budget=300, seed=0).fun)
```

`(lower, upper)` may also use scalars, `bounds=(-5, 5)`, for a box with the same limits on every
coordinate. That form does not say how many variables there are, so pass `x0` with it.

Only one form can be read two ways: a tuple of two length-2 sequences, such as
`((0, 1), (2, 3))`, is both two `(low, high)` pairs and `(lower, upper)` for two coordinates. It
is read as whichever describes a box; when both do, `minimize` raises `ValueError` and asks for a
list of pairs or `scipy.optimize.Bounds`.

**The box is used, not enforced as a constraint.** Samples are folded into the box before `f` is
called, so `f` only sees points inside it. It is not a substitute for other constraints.

**The starting point.** `x0` defaults to the centre of the box. Pass it to start somewhere else:

```python
result = amm.minimize(f, x0=np.full(3, 2.0), bounds=[(-5, 5)] * 3, budget=300, seed=0)
print(result.x)
```

## The budget (`budget`)

`budget` is the maximum number of calls of `f`, and AMM uses all of them: it plans its phases
from the budget, so pass the number you can afford. The default is `100 x D` evaluations, at least
100. A whole float such as `budget=1e4` is accepted; a budget below 1, or one that is not a whole
number, raises `ValueError`.

`result.nfev` is the number of calls actually made. The best point evaluated is returned even if
the budget runs out in the middle of an iteration.

```python
for budget in (30, 300, 3000):
    result = amm.minimize(f, bounds=[(-5, 5)] * 3, budget=budget, seed=0)
    print(budget, result.nfev, result.fun)
```

AMM was developed and tested with budgets of 10 to 1000 x D. If you do not know the budget in
advance, or you will stop the run early, use the [anytime mode](#the-anytime-mode-anytime).

## Seeds and reproducibility (`seed`, `rng`)

A run is deterministic given its seed, the platform and the number of BLAS threads. Without
`seed` every call draws a fresh one. Pass an integer to repeat a run:

```python
first = amm.minimize(f, bounds=[(-5, 5)] * 3, budget=300, seed=42)
again = amm.minimize(f, bounds=[(-5, 5)] * 3, budget=300, seed=42)
print(np.array_equal(first.x, again.x))
```

`seed` also accepts a `numpy.random.SeedSequence`, and `rng=` a `numpy.random.Generator` (the run
draws its seed from it with `rng.integers(2**32)`) or an existing `numpy.random.RandomState` to
continue. Every result records its seed in `result.seed`, so a run without `seed=` can still be
repeated with `seed=result.seed`.

The last bits of a run can differ between operating systems, NumPy or SciPy versions and BLAS
thread counts. To compare runs bit for bit, fix all of these; set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before NumPy is imported.

## The objective (`fun`)

`fun(x)` takes an array of shape `(D,)` and returns a float. It must be deterministic: the probes
compare function values, so noise misleads them. A NaN or infinite value counts as a failed
evaluation and never wins, also at the starting point (a run whose every value fails returns
`fun=inf`). The starting point itself must be finite.

## The result

`minimize` returns an [`OptimizeResult`](api.md#amm_optimizer.OptimizeResult), a subclass of
`scipy.optimize.OptimizeResult`, so both `result.x` and `result["x"]` work. It holds the best
point `x`, its value `fun`, the number of evaluations `nfev`, the number of improvements `nit`,
the number of restarts `nrestarts`, `success`, `status` and `message` (why the run ended), the
`seed`, and `timings` when profiling was on.

## Stopping early (`ftarget`, `callback`)

By default a run spends its whole budget. `ftarget=` ends it as soon as a value at or below the
target is found (`status` 1). `callback(intermediate_result)` is called after every evaluation
that improves the best value, with an `OptimizeResult` of the best point so far; raising
`StopIteration` in it ends the run (`status` 2). Either way `minimize` returns the best point
found.

```python
result = amm.minimize(f, bounds=[(-5, 5)] * 3, budget=3000, seed=0, ftarget=1e-6)
print(result.status, result.nfev < 3000)
```

Neither changes a run that does not stop: with the same seed, the evaluations up to the stop are
the ones a full run makes. `examples/early_stopping.py` stops a run after a time limit.

## The anytime mode (`anytime`)

AMM normally plans every phase from the budget, which it must know when it starts.
`anytime=True` is for runs whose budget is not known in advance or that you will cut short: up to
10 000 evaluations the run never learns `budget`. It plans 10 x D evaluations, doubles the plan
while the run goes on, and answers every point already evaluated from memory.

```python
result = amm.minimize(f, bounds=[(-5, 5)] * 3, budget=600, seed=0, anytime=True)
print(result.fun, result.nfev)
```

It costs something when the budget is known: on COCO/bbob it trails the default by about .05 to
.12 of the target share between 30 x D and 1000 x D evaluations. Leave it off if you know your
budget. [The anytime mode](ANYTIME.md) describes how it works, what it was measured to cost, and
how to cut a run short.

## Profiling (`profile`)

`profile=True` times the named blocks of the algorithm and returns them as `result.timings`.
Profiling does not change the result.

```python
result = amm.minimize(f, bounds=[(-5, 5)] * 3, budget=300, seed=0, profile=True)
timings = result.timings
print(f"wall {timings.wall_s:.3f} s, in f {timings.objective_s:.3f} s")
print(timings.table())
```

[`Timings`](api.md#amm_optimizer.Timings) gives the wall time (`wall_s`), the time inside `f`
(`objective_s`), the optimizer's own time (`overhead_s`), each block's share of the wall time
(`share(name)`), a plain-text table (`table()`) and a dictionary (`to_dict()`).

To add up several runs, pass one [`TimingProfiler`](api.md#amm_optimizer.TimingProfiler) to each
of them; each result's `timings` is then the total so far:

```python
profiler = amm.TimingProfiler()
for seed in range(3):
    result = amm.minimize(f, bounds=[(-5, 5)] * 3, budget=300, seed=seed, profile=profiler)
print(result.timings.blocks["total"].calls)
```

## Choosing a configuration (`config`)

Without `config`, `minimize` runs AMM, the configuration frozen for the paper
(`presets.get("c46")`). [Presets](presets.md) lists the other named configurations, and
[the original ATM sampler](legacy.md) describes `ATMConfig()` and `ATMConfig.composite()`.
