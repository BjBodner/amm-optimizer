# Adaptive Multi Mode (AMM): budget-aware routing of optimizers by online landscape probes

AMM is a derivative-free optimizer for continuous black-box problems with a **known evaluation
budget** and a **known box**. It does not run one algorithm: it spends a small, counted part of the
budget on *online probes* (curvature, plateau, separability, kink, basin and fractal readings of the
objective near the incumbent) and uses their outcomes to route the rest of the budget among
quasi-Newton (SLSQP, L-BFGS-B), trust-region (COBYQA), coordinate (STEP), multistart (MLSL) and
population (CMA-ES family, the ATM sampler) engines. `amm_optimizer.minimize` runs AMM by default. A probe that does not fire leaves the run on
a general path, so a misreading costs evaluations, not the run. Every rule reads only the
dimension, the budget, the box and evaluated function values.

The method, its development protocol and its evaluation on COCO/bbob are described in the paper
(citation below). The method grew out of the Adaptive Two Mode (ATM) sampler, which is one of its
engines; some internal names (`ATMConfig`, preset codes such as `c46`) still carry that history.

## Install

```bash
pip install amm-optimizer            # NumPy and SciPy only; Python 3.12 or newer
uv add amm-optimizer                 # or: uv pip install amm-optimizer
pip install -e ".[dev]"              # from a checkout; add ",bench" for the COCO/bbob tooling
```

Source, issues and the changelog: <https://github.com/BjBodner/amm-optimizer>. Upgrading from 0.2.0: the import
name changed to `amm_optimizer` (`import amm_optimizer as amm`). Upgrading from 0.3.0: `amm.minimize` now runs
AMM by default and needs `bounds=`; the original ATM sampler is `config=amm.ATMConfig()`. See [CHANGELOG.md](CHANGELOG.md).

## Quick start

```python
import numpy as np
import amm_optimizer as amm

def f(x):
    return float(np.sum((x - 1.0) ** 2))

result = amm.minimize(
    f,
    x0=np.zeros(10),
    budget=100 * 10,                        # maximum number of function evaluations
    seed=0,                                 # runs are deterministic given the seed
    bounds=(-5, 5),                         # required: the box (scalars or one value per coordinate)
)
print(result.x, result.fun, result.nfev)
```

Called without `config=`, `amm.minimize` runs AMM, the configuration frozen for the paper
(`amm.presets.get("c46")`; passing it as `config=` gives a bit-identical run). It needs `bounds=`:
without it, and without a `config`, the call raises `ValueError`. The box is used, not enforced as
a constraint: samples are folded into it before the function is called. The objective must be
deterministic. Tested on noiseless, continuous, box-constrained problems with budgets of 10 to
1000 x the dimension.

**The original ATM sampler** is one engine of AMM and is still available on its own: pass
`config=amm.ATMConfig()` (no `bounds=` needed). `amm_optimizer.api.minimize` is the same engine
without the AMM default, so a call to it with no `config` also runs the original sampler. Versions
up to 0.3.0 ran that sampler when `config` was omitted; see the changelog.

**Reproducing the paper.** The configuration hash of `amm.presets.get("c46").config_hash()` is `2153948ebed51150`; [docs/FREEZE_2026-10-22.md](docs/FREEZE_2026-10-22.md) records what was frozen. The pre-registrations, batch notes and result files of the development are kept in a separate research archive.

**Citation.** See `CITATION.cff` (the paper's reference will be added when it is public).

## The original ATM sampler and the older presets

The sections below document the original ATM sampler and the older presets
(`ATMConfig()` and `composite()`); they are kept because every recorded result and test
depends on them. None of them is the default of `amm.minimize`: pass them as `config=`. This repository is a clean, tested re-implementation of the original ATM that
reproduces its results exactly (see [docs/DEVIATIONS.md](docs/DEVIATIONS.md)).

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

`budget` is the maximum number of function evaluations, and ATM uses all of them (the
best point found is kept even if the budget ends mid-iteration). Pass `seed=` (or an
`rng=np.random.RandomState`) for reproducible runs. A starting point where the function is
NaN or infinite raises `ValueError`; NaN or infinite values elsewhere count as failed
evaluations and never win.

`ATMConfig()` reproduces the original ATM exactly, and so does `amm_optimizer.api.minimize` called
without a `config` (see `docs/DEVIATIONS.md`); the default of `amm.minimize` is AMM, not the original.
`ATMConfig(strict_budget=False)` also restores the original's looser budget accounting.

## The composite preset

`ATMConfig.composite()` is ATM's own sampler plus five improvements (success-rate control of
the search width, a uniform first pool over the box, bounds folded with a 25 % corner clip, a
pool sized from the budget, no re-evaluation of the incumbent). It is kept for reproducibility;
AMM (`c46`, the default) is the configuration to use. If you know the search box, the composite
gets more out of the same budget than the original sampler:

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
function is called, so the function only ever sees points inside. Without `bounds=` the
preset raises `ValueError`. The objective must be deterministic. Each option is documented in
`docs/DEVIATIONS.md`.

Presets, from simplest to largest (`c46` last, as the frozen AMM):

| Preset | What it is | Evidence |
|---|---|---|
| `ATMConfig()` | the original ATM, bit for bit | `tests/golden`, 400-case parity sweep |
| `ATMConfig.composite()` | ATM's sampler plus the five improvements; stays pure ATM | +.101, +.097, +.099 on three untouched holdouts |
| `presets.get("c46")` | AMM, the default of `amm.minimize` | [docs/FREEZE_2026-10-22.md](docs/FREEZE_2026-10-22.md) |

`tests/golden/presets.json` pins the exact cost traces of the first two.

## Development

```bash
pytest -m "not slow"      # fast suite, includes golden-master and parity tests
pytest -m slow            # 400 random cases against the frozen original in reference/
python tools/record_golden.py   # re-record golden masters (only for deliberate changes)
python benchmarks/perf.py       # repeatable speed benchmark, original vs this package
python benchmarks/block_profile.py   # where the time goes, per algorithm block
```

| Path | What it is |
|------|------------|
| `src/amm_optimizer/` | the package (`default.py` is the public `minimize`; `api.py` is the engine, whose default is the original sampler and which the tests call) |
| `reference/` | byte-identical copy of the original module, used only by tests |
| `tests/golden/` | recorded trajectories from the original |
| `docs/DEVIATIONS.md` | every difference from the original |
| `benchmarks/` | speed harness (`perf.py`), per-block timing (`block_profile.py`), COCO/bbob harness (`run_bbob.py`, `compare_bbob.py`, `postprocess_bbob.py`) and stored results |

## Continuous integration

`.github/workflows/ci.yml` runs lint, type checks, the research-tier tests on Windows (Python 3.13), Linux (3.12, 3.14)
and macOS (3.12), and a packaging job (build, `twine check`, an sdist size gate, and a smoke test of the installed wheel
and sdist). The bit-exact snapshot tests only hold in the environment that recorded them and skip themselves elsewhere;
a Windows 3.12 leg pinned to that environment reports but does not gate, because GitHub's runners differ in CPU generation
and so in the last bits. Run the same checks locally with `python tools/ci.py` (fast gate), `python tools/ci.py --tier merge`
or `--tier full`; see [CONTRIBUTING.md](CONTRIBUTING.md). Releases are built in `release.yml` and published with Trusted Publishing.
Security reports: [SECURITY.md](SECURITY.md).

Licensed under Apache-2.0.
