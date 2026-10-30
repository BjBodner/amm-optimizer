# Adaptive Multi Mode (AMM): budget-aware routing of optimizers by online landscape probes

[![PyPI](https://img.shields.io/pypi/v/amm-optimizer)](https://pypi.org/project/amm-optimizer/)
[![Python versions](https://img.shields.io/pypi/pyversions/amm-optimizer)](https://pypi.org/project/amm-optimizer/)
[![CI](https://github.com/BjBodner/amm-optimizer/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/BjBodner/amm-optimizer/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/BjBodner/amm-optimizer/graph/badge.svg?branch=main)](https://codecov.io/gh/BjBodner/amm-optimizer)
[![License: Apache-2.0](https://img.shields.io/pypi/l/amm-optimizer)](LICENSE)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/BjBodner/amm-optimizer/badge)](https://scorecard.dev/viewer/?uri=github.com/BjBodner/amm-optimizer)

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

Source, issues and the changelog: <https://github.com/BjBodner/amm-optimizer>. Results, limits and how it works:
<https://bjbodner.github.io/amm-optimizer-site/> (the website has its own repository,
[BjBodner/amm-optimizer-site](https://github.com/BjBodner/amm-optimizer-site)). Upgrading from 0.2.0: the import
name changed to `amm_optimizer` (`import amm_optimizer as amm`). Upgrading from 0.3.0: `amm.minimize` now runs
AMM by default and needs `bounds=`; the original ATM sampler is `config=amm.ATMConfig()`. See [CHANGELOG.md](CHANGELOG.md).

## Quick start

```python
import numpy as np
import amm_optimizer as amm

def f(x):
    return float(np.sum((x - 1.0) ** 2))

result = amm.minimize(f, bounds=[(-5, 5)] * 10)
print(result.x, result.fun, result.nfev)
```

That is the whole call: the search box is the only thing you must give. Everything else has a default:

| Argument | Default | Meaning |
|---|---|---|
| `bounds` | required | the search box: a list of `(low, high)` pairs (as in SciPy), `(lower, upper)` with scalars or arrays, or `scipy.optimize.Bounds`; finite limits |
| `x0` | the centre of the box | the starting point |
| `budget` | `100 x D` evaluations (at least 100) | the maximum number of calls of `f`; all are used unless `ftarget` or `callback` stops the run. Pass what you can afford |
| `ftarget`, `callback` | none | stop early: at a target value, or when `callback(intermediate_result)` raises `StopIteration` |
| `seed` | a fresh seed, recorded in `result.seed` | a run is deterministic given the seed; `rng=` also takes a NumPy `Generator` |
| `anytime` | `False` | `True` for a run whose budget is not known or that you will cut short, see [docs/ANYTIME.md](docs/ANYTIME.md) |
| `config` | `"default"` (AMM, frozen as `c46`) | a preset name (`presets.names()`) or an `ATMConfig`; `"atm"` runs the original ATM sampler alone and needs no box |

The older, fully spelled-out call still works:

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

Called without `config=`, `amm.minimize` runs AMM, the configuration frozen for the paper
(`amm.presets.get("c46")`; passing it as `config=` gives a bit-identical run). It needs `bounds=`:
without it, and without a `config`, the call raises `ValueError`. The box is used, not enforced as
a constraint: samples are folded into it before the function is called. The objective must be
deterministic. Tested on noiseless, continuous, box-constrained problems with budgets of 10 to
1000 x the dimension.

**The original ATM sampler** and the older presets (`ATMConfig()`, `ATMConfig.composite()`) are
still available as `config=`; see [docs/legacy.md](docs/legacy.md). `amm_optimizer.api.minimize`
is the same engine without the AMM default, so a call to it with no `config` runs the original
sampler.

**Reproducing the paper.** The configuration hash of `amm.presets.get("c46").config_hash()` is `2153948ebed51150`; [docs/FREEZE_2026-10-22.md](docs/FREEZE_2026-10-22.md) records what was frozen. The pre-registrations, batch notes and result files of the development are kept in a separate research archive.

## When to use AMM, and when not to

AMM fits **noiseless, box-constrained, continuous problems** where each evaluation costs enough
that you count them, and you can afford **about 10 to 1000 x D evaluations** (D the number of
variables). That is the range it was developed and measured on (COCO/bbob, D 2 to 40). Use
`anytime=True` if you do not know the budget in advance.

Look elsewhere when:

* **the objective is noisy** (repeated calls at one point differ): AMM assumes a deterministic
  `f`, and its probes read small differences between values, which noise drowns;
* **the problem has no finite box**: AMM needs finite `bounds`;
* **gradients are cheap**: a gradient method such as `scipy.optimize.minimize(method="L-BFGS-B",
  jac=...)` will need far fewer evaluations;
* you need constraints other than the box, discrete variables, or batches of parallel
  evaluations.

Compared with the usual alternatives: SciPy's `minimize` gives single local methods (you pick
one); `cma` gives CMA-ES, a strong population method that is robust on rugged and noisy functions
but does not plan around the budget; Nevergrad offers a portfolio (`NGOpt`) that also covers
noise, discrete variables and parallel evaluation. AMM routes one fixed, counted budget among
local, coordinate, multistart and population engines from probes of the objective. The measured
comparison with these on COCO/bbob is on the [website](https://bjbodner.github.io/amm-optimizer-site/).

## Citation

See [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button reads it); the paper's
reference will be added when it is public. As BibTeX:

```bibtex
@software{bodner_amm_optimizer,
  author  = {Bodner, Benjamin Jacob},
  title   = {Adaptive Multi Mode ({AMM}): budget-aware routing of optimizers by online landscape probes},
  year    = {2026},
  url     = {https://github.com/BjBodner/amm-optimizer},
  license = {Apache-2.0},
  note    = {Python package amm-optimizer}
}
```

## Development

```bash
python tools/ci.py        # lint, types and the default test tier (see CONTRIBUTING.md)
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
| `docs/legacy.md` | the original ATM sampler and the older presets |
| `benchmarks/` | speed harness (`perf.py`, `profiler_overhead.py`), per-block timing (`block_profile.py`), the COCO/bbob harness (`run_bbob.py`), and the process pool of the snapshot recorder (`process_pool.py`) |

## Continuous integration

`.github/workflows/ci.yml` runs lint, type checks, the quick tests on Windows (Python 3.13), Linux (3.12, 3.14)
and macOS (3.12), a floor leg on Linux 3.12 with the oldest supported NumPy and SciPy (2.0 and 1.18.0), and a packaging
job (build, `twine check`, an sdist size gate, and a smoke test of the installed wheel and sdist). The Linux 3.12 leg's
coverage goes to Codecov. The bit-exact snapshot tests only hold in the environment that recorded them and skip themselves elsewhere;
a Windows 3.12 leg pinned to that environment reports but does not gate, because GitHub's runners differ in CPU generation
and so in the last bits. Run the same checks locally with `python tools/ci.py` (the default tier), `python tools/ci.py --tier merge`
or `--tier full`; see [CONTRIBUTING.md](CONTRIBUTING.md). Releases are built in `release.yml` and published with Trusted Publishing;
the tag then gets a GitHub Release with its CHANGELOG section, the wheel, the sdist and a CycloneDX SBOM.
Security reports: [SECURITY.md](SECURITY.md). Contributors follow the [Code of Conduct](CODE_OF_CONDUCT.md).

Licensed under Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
