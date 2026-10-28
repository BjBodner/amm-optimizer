# Changelog

All notable changes to amm-optimizer are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is 0.x, a minor
release may change behaviour, with notice here; a patch release never changes results.
Public API is deprecated, with a warning, for at least one minor release before it is removed
(the deprecation policy will be written out in CONTRIBUTING.md).

## [Unreleased]

Bug fixes. Every run that starts inside the box and gets well-behaved inputs is bit for bit as in 0.5.0 (`c46`'s config hash is still `2153948ebed51150`).

### Added

- A documentation site, for <https://amm-optimizer.readthedocs.io>: getting started, a user
  guide (bounds, budgets, seeds, the anytime mode, profiling), the presets, the API reference
  generated from the docstrings, the changelog, the deprecation policy and how to cite AMM
  (`mkdocs.yml`, `.readthedocs.yaml`, a `docs` extra). CI builds it with `mkdocs build --strict`,
  and the package metadata links it as `Documentation`.
- Each release tag gets a GitHub Release once PyPI has the files: its notes are the version's
  CHANGELOG section, and it carries the wheel, the sdist and a CycloneDX SBOM.
- CI: a floor leg with the oldest supported NumPy and SciPy (2.0 and 1.18.0, Python 3.12), and
  coverage reports on Codecov.

### Changed

- **An `x0` outside `bounds` is clipped onto the box, with a `UserWarning`**, on every box. On the bbob box `[-5, 5]` it used to be evaluated as given; on any other box it was already clipped, silently, so those runs are unchanged. `amm_optimizer.api.minimize`, the engine, does not check `x0`.

### Fixed

- **`fun` gets its own copy of every point**, as in SciPy. It used to receive writeable views of the optimizer's buffers, so an objective that wrote into `x` changed the search, and one that kept `x` saw it change.
- **Bad inputs fail before any evaluation, with plain errors**: a non-finite `x0` (`ValueError`); a `config` that is not an `ATMConfig` (`TypeError` naming `presets.get`); an `rng` that is not a `RandomState` (`TypeError` naming `seed=`); a `seed` that is not an integer (`TypeError`) or is outside `[0, 2**32)` (`ValueError`); a `bool` `budget` (`TypeError`, it was taken as 1). A `fun` that returns an array, a list, a complex number or `None` gets a `TypeError` that says what it returned. A non-finite value at the starting point still raises `ValueError`.
- **The value at the starting point is a `float`** like every other value; a `fun` returning `Fraction` no longer crashes.
- **Warnings raised by `fun` reach the caller** on the COBYQA and basin routes, which used to ignore every warning. SciPy's own overflow warnings from finite differences over a failed evaluation are no longer shown.
- **The diagnostic logs stay bounded.** `anytime.LAST_STAGES` (alias `STAGE_LOG`) holds the stages of the most recent anytime run only; `cma_phase.PROBE_LOG` records only while `cma_phase.RECORD_PROBES` is true.
- The `minimize` docstring has a thread-safety note.

## [0.5.0] - 2026-10-25

A simpler call, and the anytime mode. The default algorithm is unchanged: with `anytime=False`
every configuration runs bit for bit as in 0.4.0, the default and `c46` reproduce their recorded
benchmark runs exactly, and `c46`'s config hash is still `2153948ebed51150`. 0.4.0 was tagged but
never uploaded to PyPI; its changes (below) are part of this release.

### Added

- **Anytime mode** (`minimize(..., anytime=True)`, or `Variant.anytime="replay"`; guide:
  [docs/ANYTIME.md](docs/ANYTIME.md)). For runs whose budget is not known in advance or that you
  cut short: up to 10 000 evaluations the run never learns `budget`; it plans 10 x D evaluations
  and doubles the plan while the run goes on, answering every point already evaluated from
  memory, then hands over to one ordinary run fed those points. Measured on COCO/bbob: level with
  the default at 10^4 and 10^5 x D, and .05 to .12 of the target share below it between 30 x D
  and 1000 x D when the budget is known (so leave it off if you know your budget). The objective
  must be deterministic; `nfev` counts the calls actually made.
- `docs/ANYTIME.md` is in the source distribution.

### Changed

- **Simpler `minimize`.** `x0` and `budget` are now optional, so
  `amm.minimize(f, bounds=[(-5, 5)] * 10)` is a complete call. `x0` defaults to the centre of the
  box and `budget` to `100 x D` evaluations (at least 100). `bounds` also accepts a list of
  `(low, high)` pairs (SciPy's form), a NumPy array of shape `(D, 2)` and
  `scipy.optimize.Bounds`, besides the earlier `(lower, upper)`. A whole float such as
  `budget=1e4` is accepted. Errors are plain: a non-callable `fun`, a `bounds` and `x0` that
  disagree on the number of variables, a `budget` that is not a whole number, and a tuple of two
  length-2 sequences that reads both as pairs and as `(lower, upper)` are refused with a message
  that names the fix. Every call that worked in 0.4.0 runs unchanged.
- **Faster long runs** (same evaluations, bit-identical results): the STEP phase keeps
  incremental difficulties and a chunked axis store, so its cost per evaluation no longer grows
  with the run length, and the start search is incremental.

## [0.4.0] - 2026-10-23

Tagged but never uploaded to PyPI; these changes first shipped in 0.5.0. No algorithmic change:
the engine, `ATMConfig()` and every preset behave exactly as in 0.3.0.

### Added

- `amm_optimizer/default.py` with the public `minimize` (same signature and return type as the
  engine); `amm_optimizer.api` is importable from the package namespace.

### Changed

- **Breaking: `amm_optimizer.minimize` called without `config=` now runs AMM**
  (`presets.get("c46")`, config hash `2153948ebed51150`), not the original ATM sampler. In 0.3.0
  and earlier the omitted `config` meant `ATMConfig()`: on a 5-dimensional Rosenbrock example
  (budget 2000, seed 0) the old default reaches f = 1.7 and AMM reaches f of about 1e-13.
- **Breaking: AMM needs the search box.** A call with neither `config=` nor
  `bounds=(lower, upper)` raises `ValueError` and the message names both ways out. Calls that
  pass a `config` are unchanged, including those that pass `bounds`.
- **To get the old behaviour** pass `config=amm_optimizer.ATMConfig()` (no `bounds` needed), or
  call `amm_optimizer.api.minimize`, the engine, whose default is still the original sampler.
- The author name in the package metadata is "Benjamin Jacob Bodner".
- Docs: the README starts from the default AMM call.

### Removed

- `ATMConfig.improved()`, `improved_wave5()`, `improved_v2()` and `improved_wave6()` (the earlier
  hybrid; AMM is `presets.get("c46")`). `ATMConfig()`, `ATMConfig.composite()` and every preset
  of `presets.names()` are unchanged (same config hashes). To reproduce the earlier hybrid, build
  it from `Variant` fields.

## [0.3.0] - 2026-10-23

Packaging release; no algorithmic change (the default and `c46` reproduce their recorded
benchmark runs bit for bit).

### Added

- `py.typed` for the single package; an installed-package smoke test
  (`tools/smoke_installed.py`); GitHub Actions workflows for CI, release (Trusted Publishing)
  and security checks; `SECURITY.md`, `CONTRIBUTING.md`.
- Tests: a one- to three-dimension smoke matrix over every shipped configuration; tests that
  need `cocoex` skip without it.

### Changed

- **Breaking: the import name is now `amm_optimizer`** (`import amm_optimizer as amm`). 0.2.0
  installed the top-level modules `atm` and `amm`, which collide with the unrelated PyPI
  projects of those names; the `atm` and `amm` modules are gone. `ATMConfig` and the preset
  names (`c46`) are unchanged. Upgrading from 0.2.0: replace `import atm` / `import amm` by
  `import amm_optimizer as amm`; `amm.minimize`, `amm.presets` and `amm.ATMConfig` are
  unchanged. `pip install -U amm-optimizer` needs Python >= 3.12.
- The title and description follow the paper: "Adaptive Multi Mode (AMM): budget-aware routing
  of optimizers by online landscape probes" (the unreleased 0.2.1 metadata changes are folded
  in; 0.2.1 was skipped).
- Requires Python >= 3.12, NumPy >= 2.0, SciPy >= 1.18 (the versions the results were recorded
  with; 0.2.0 declared Python >= 3.10 and numpy >= 1.23, which SciPy 1.16 or newer cannot
  satisfy). Metadata uses a PEP 639 license expression.

### Fixed

- `import` no longer fails on NumPy 1.x (`np._core`).
- The `c46` preset no longer crashes for one-dimensional problems at budgets of 150 or more
  (`np.cov` is 0-d).
- The benchmark harness's process pool replaces a broken pool and reruns unfinished runs
  (fixes `WinError 6` failures on Windows under load).

## [0.2.0] - 2026-10-22 [YANKED]

### Added

- First public release as `amm-optimizer` (import `amm`; the implementation kept its code name
  `atm`). Packaging only: the code frozen on 2026-10-22 (`c46`, config hash
  `2153948ebed51150`), plus the `amm` alias package, `CITATION.cff`, and an AMM README. No
  algorithmic change. Yanked from PyPI: use 0.3.0 or later.

[Unreleased]: https://github.com/BjBodner/amm-optimizer/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/BjBodner/amm-optimizer/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/BjBodner/amm-optimizer/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/BjBodner/amm-optimizer/releases/tag/v0.3.0
[0.2.0]: https://pypi.org/project/amm-optimizer/0.2.0/
