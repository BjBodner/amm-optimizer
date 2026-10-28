# amm-optimizer

**Adaptive Multi Mode (AMM)** is a derivative-free optimizer for continuous black-box problems
with a **known evaluation budget** and a **known box**. It does not run one algorithm. It spends a
small, counted part of the budget on *online probes* of the objective near the incumbent
(curvature, plateau, separability, kink, basin and fractal readings), and uses what they find to
route the rest of the budget among quasi-Newton (SLSQP, L-BFGS-B), trust-region (COBYQA),
coordinate (STEP), multistart (MLSL) and population (CMA-ES family, the ATM sampler) engines.

A probe that does not fire leaves the run on a general path, so a misreading costs evaluations,
not the run. Every rule reads only the dimension, the budget, the box and evaluated function
values.

## Install

```bash
pip install amm-optimizer            # NumPy and SciPy only; Python 3.12 or newer
uv add amm-optimizer                 # or: uv pip install amm-optimizer
```

The import name is `amm_optimizer`.

## Quick start

```python
import numpy as np
import amm_optimizer as amm

def f(x):
    return float(np.sum((x - 1.0) ** 2))

result = amm.minimize(f, bounds=[(-5, 5)] * 10)
print(result.x, result.fun, result.nfev)
```

That is the whole call: the search box is the only thing you must give. Everything else has a
default.

| Argument | Default | Meaning |
|---|---|---|
| `bounds` | required | the search box: a list of `(low, high)` pairs (as in SciPy), `(lower, upper)` with scalars or arrays, or `scipy.optimize.Bounds`; finite limits |
| `x0` | the centre of the box | the starting point |
| `budget` | `100 x D` evaluations (at least 100) | the maximum number of calls of `f`; all are used. Pass what you can afford |
| `seed` | a fresh seed | a run is deterministic given the seed |
| `anytime` | `False` | `True` for a run whose budget is not known or that you will cut short, see [the anytime mode](ANYTIME.md) |
| `config` | AMM (`presets.get("c46")`) | `amm.ATMConfig()` runs the original ATM sampler alone and needs no box, see [the original sampler](legacy.md) |

The fully spelled-out call gives every argument:

```python
result = amm.minimize(
    f,
    x0=np.zeros(10),
    budget=100 * 10,                        # maximum number of function evaluations
    seed=0,                                 # runs are deterministic given the seed
    bounds=(-5, 5),                         # the box (scalars or one value per coordinate)
)
print(result.fun, result.nfev)
```

The [user guide](user-guide.md) explains the arguments in more detail.

## When to use AMM, and when not to

AMM fits **noiseless, box-constrained, continuous problems** where each evaluation costs enough
that you count them, and you can afford **about 10 to 1000 x D evaluations** (D is the number of
variables). It was developed and measured in that range, on COCO/bbob with D from 2 to 40. Use
`anytime=True` if you do not know the budget in advance.

Look elsewhere when:

* **the objective is noisy** (repeated calls at one point differ): AMM assumes a deterministic
  `f`, and its probes read small differences between values, which noise drowns;
* **the problem has no finite box**: AMM needs finite `bounds`;
* **gradients are cheap**: a gradient method such as
  `scipy.optimize.minimize(method="L-BFGS-B", jac=...)` will need far fewer evaluations;
* you need constraints other than the box, discrete variables, or batches of parallel
  evaluations.

The measured comparison with SciPy's local methods, CMA-ES and Nevergrad on COCO/bbob is on the
[project website](https://bjbodner.github.io/amm-optimizer-site/).

## Where to go next

* [User guide](user-guide.md): bounds, budgets, seeds, the anytime mode and profiling.
* [Presets](presets.md): the named configurations and the frozen default.
* [API reference](api.md): every public function and class.
* [Changelog](changelog.md) and [deprecation policy](deprecation-policy.md).
* [Citing AMM](citing.md).

Source code and issues:
[github.com/BjBodner/amm-optimizer](https://github.com/BjBodner/amm-optimizer).
