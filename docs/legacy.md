# The original ATM sampler and the older presets

AMM grew out of the Adaptive Two Mode (ATM) sampler, which is one of its engines. This page
documents the original ATM sampler and the older presets (`ATMConfig()` and `composite()`). They
are kept because every recorded result and test depends on them. None of them is the default of
`amm.minimize`: pass them as `config=`. AMM (`presets.get("c46")`, the default) is the
configuration to use.

This repository is a clean, tested re-implementation of the original ATM that reproduces its
results exactly (see [DEVIATIONS.md](DEVIATIONS.md)).

## The original ATM sampler

```python
import numpy as np
import amm_optimizer as amm

def sphere(x):
    return np.sum((x - 2) ** 2)

result = amm.minimize(
    sphere, x0=np.zeros(10), budget=2000, seed=0,
    config=amm.ATMConfig(),                 # the original ATM sampler; no bounds needed
)
print(result.x, result.fun, result.nfev)
```

`budget` is the maximum number of function evaluations, and ATM uses all of them (the best point
found is kept even if the budget ends mid-iteration). Pass `seed=` (or an
`rng=np.random.RandomState`) for reproducible runs. A starting point where the function is NaN or
infinite raises `ValueError`; NaN or infinite values elsewhere count as failed evaluations and
never win.

`ATMConfig()` reproduces the original ATM exactly, and so does `amm_optimizer.api.minimize` called
without a `config` (see [DEVIATIONS.md](DEVIATIONS.md)); the default of `amm.minimize` is AMM, not
the original. `ATMConfig(strict_budget=False)` also restores the original's looser budget
accounting. Versions up to 0.3.0 ran this sampler when `config` was omitted; see the
[changelog](../CHANGELOG.md).

## The composite preset

`ATMConfig.composite()` is ATM's own sampler plus five improvements (success-rate control of the
search width, a uniform first pool over the box, bounds folded with a 25 % corner clip, a pool
sized from the budget, no re-evaluation of the incumbent). It is kept for reproducibility. If you
know the search box, the composite gets more out of the same budget than the original sampler:

```python
result = amm.minimize(
    sphere,
    x0=np.zeros(10),
    budget=2000,
    seed=0,
    bounds=(-5, 5),                      # scalars or one value per coordinate
    config=amm.ATMConfig.composite(),
)
print(result.fun)
```

The box is used, not enforced as a constraint: samples are folded back into it before the
function is called, so the function only ever sees points inside. Without `bounds=` the preset
raises `ValueError`. The objective must be deterministic. Each option is documented in
[DEVIATIONS.md](DEVIATIONS.md).

## The presets, from simplest to largest

| Preset | What it is | Evidence |
|---|---|---|
| `ATMConfig()` | the original ATM, bit for bit | `tests/golden`, 400-case parity sweep |
| `ATMConfig.composite()` | ATM's sampler plus the five improvements; stays pure ATM | +.101, +.097, +.099 on three untouched holdouts |
| `presets.get("c46")` | AMM, the default of `amm.minimize` | [FREEZE_2026-10-22.md](FREEZE_2026-10-22.md) |

`tests/golden/presets.json` pins the exact cost traces of the first two.
