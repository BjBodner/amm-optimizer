# Changelog

All notable changes to amm-optimizer are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is 0.x, a minor
release may change behaviour, with notice here; a patch release never changes results.
Public API is deprecated, with a warning, for at least one minor release before it is removed
(see "Deprecation policy" in
[CONTRIBUTING.md](https://github.com/BjBodner/amm-optimizer/blob/main/CONTRIBUTING.md#deprecation-policy)).

## [Unreleased]

## [0.6.0] - 2026-10-28

Bug fixes, a SciPy-compatible API and early stopping, with the research options deprecated. The default algorithm is unchanged: every configuration that ships runs bit for bit as in 0.5.0 on inputs that worked before, the default and `c46` reproduce the paper's benchmark runs exactly, and `c46`'s config hash is still `2153948ebed51150`.

### Added

- **Early stopping.** `minimize(..., callback=, ftarget=)`: `callback(intermediate_result)` is called after every evaluation that improves the best value, with an `OptimizeResult` holding `x`, `fun`, `nfev` and `nit`; raising `StopIteration` from it ends the run. `ftarget` ends the run once the best value is at or below it. Either way the best point so far is returned, with `status` 2 or 1, and no evaluation follows the stop. `catch_interrupt=True` (opt-in) turns Ctrl-C into a result too (`status` 3, `success=False`, a `RuntimeWarning`). Without these arguments every run is bit for bit as before: the callback path draws no random numbers and changes no evaluation.
- **`rng` accepts a `numpy.random.Generator`** (the run's seed is one draw, `rng.integers(2**32)`), and **`seed` accepts a `numpy.random.SeedSequence`**. A run without a seed draws one from the operating system and records it, so **`result.seed`** repeats any run (`seed=result.seed`); it is `None` only when a `RandomState` was continued.
- **Preset names in `config=`**: `config="default"` (AMM, the default preset, frozen as `c46`), `"c46"`, `"atm"` (`ATMConfig()`), `"composite"` (`ATMConfig.composite()`) or a research preset; `presets.get` takes the same names. Anything else that is not an `ATMConfig` is a `TypeError`.
- `examples/` with three short scripts: `quickstart.py`, `scipy_migration.py` and `early_stopping.py`.
- `docs/HISTORY.md`: how the research composites `c23` to `c46` came about (from the `presets` module docstring, which is now a short list).

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

- **The result is a `scipy.optimize.OptimizeResult`** (`amm_optimizer.OptimizeResult` subclasses it): `r.x` works as before, and `r["x"]`, `r.keys()` and `dict(r)` work as in SciPy. New fields: `nit` (the number of evaluations that improved the best value, the times `callback` is called), `success`, `status` (0 budget used or plan finished, 1 `ftarget` reached, 2 stopped by the callback, 3 interrupted), `message` and `seed`. It is no longer a dataclass, so code that called `dataclasses.replace` or `dataclasses.asdict` on a result needs `dict(r)` or item assignment instead.
- **A non-finite value at the starting point is a failed evaluation**, like a non-finite value anywhere else: the run carries on, and any finite value beats it. It used to raise `ValueError`, so only runs that failed before behave differently; an objective undefined at the centre of the box now works. A run in which every value fails returns `fun=inf` at the starting point.
- **`presets.names()` lists the presets on display**, `("default", "c46", "atm", "composite")`; `presets.names(research=True)` adds the research composites `c23` to `c45`, which all stay loadable with `presets.get`.
- `minimize`'s docstring documents the new arguments and recommends `callback` or `ftarget`, not an exception raised inside `fun`, for cutting an anytime run short.
- **An `x0` outside `bounds` is clipped onto the box, with a `UserWarning`**, on every box. On the bbob box `[-5, 5]` it used to be evaluated as given; on any other box it was already clipped, silently, so those runs are unchanged. `amm_optimizer.api.minimize`, the engine, does not check `x0`.

### Deprecated

- **92 research-only `Variant` fields and 29 research-only values of the other fields** emit a
  `FutureWarning` when set and will be removed in 0.7.0 (`warnings.filterwarnings("ignore",
  "Variant", FutureWarning)` silences them meanwhile). No shipped configuration (`c23`-`c46`,
  `ATMConfig()`, `ATMConfig.composite()`, `anytime=True`) uses them, so none of those warns,
  and their config hashes and results are unchanged. The lists are `config.RESEARCH_FIELDS` and
  `config.RESEARCH_CHOICES`, measured by `tools/research_fields.py`. Fields: `targeted_gate`,
  `targeted_amplitude`, `reset_on_success`, `sr_eta`, `sr_targets`, `sr_widths`,
  `sr_max_width`, `es_init_factor`, `es_small_dim`, `es_small_max_budget_per_dim`,
  `scale_learning`, `scale_rate`, `scale_cap`, `cov_cap`, `cov_steps`, `cov_path`,
  `cov_max_dim`, `local_force`, `fork_seed`, `prefix_plateau_rugged`, `chain_min_per_dim`,
  `sep_route_min_per_dim`, `additive_min_dim`, `basin_max_dim`, `basin_min_per_dim`,
  `basin_descents_max_per_dim`, `small_engine_max_per_dim`, `small_engine_radius`,
  `rugged_try_cap_per_dim`, `rugged_try_max_per_dim`, `prefix_ridge_radius`,
  `small_switch_max_dim`, `small_switch_max_per_dim`, `small_switch_cos`,
  `small_switch_radius`, `small_switch_rest`, `sampled_polish_per_dim`, `qn_false_success`,
  `qn_floor`, `probe_faces`, `eval_memo`, `step_box_check`, `box_steps`,
  `prefix_nan_min_per_dim`, `prefix_nan_clip`, `prefix_nan_reprobe`, `prefix_plateau_ties`,
  `prefix_fractal_min_dim`, `prefix_fractal_min_per_dim`, `stage_extend_restart`,
  `stage_extend_max_per_dim`, `sep_route_low`, `sep_route_low_dim`, `centre_point`,
  `centre_point_min_dim`, `smooth_bfgs_second`, `cma_restart`, `cma_model_max_dim`,
  `cma_lq_max_dim`, `cma_model_sigma`, `cma_handoff_cap_min_dim`,
  `cma_handoff_cap_low_per_dim`, `cma_handoff_cap_low_max_per_dim`, `es_lean_start`,
  `cma_regimes`, `cma_placement`, `cma_placement_max_dim`, `cma_local_nonsep_min_dim`,
  `sep_parabolic`, `sep_parabolic_late`, `sep_parabolic_scope`, `sep_share`,
  `sep_min_budget_per_dim`, `width_units`, `crossover`, `noise`, `skip_starved`, `width_mode`,
  `sr_shared_target`, `restart_mode`, `stag_tolfun`, `stag_tolx`, `ipop_factor`,
  `ipop_max_pool`, `local_method`, `simplex_max_dim`, `nm_refine_max_dim`, `refine_fraction`,
  `refine_min_budget_per_dim`, `refine_max_dim`, `pool_max`, `pool_divisor`. Values:
  `amplitude_control` `"success_rate"`; `es_kind` `"slsqplq10"`, `"slsqplq10hk"`;
  `cma_local_nonsep` `"ipop"`, `"population"`, `"qn"`, `"qn2"`, `"qn5"`, `"qn6"`, `"qn10"`,
  `"qn11"`, `"qn13"`, `"qn13n"`, `"qn15"`; `nm_stage` `"qnprefix"`, `"qnprefix30g"`;
  `cma_model` `"lq"`; `cma_restart_rule` `"economy"`; `cma_local` `"population"`, `"ipop"`;
  `sep_check` `"pair"`; `pop_lambda` `"cap4"`, `"budget"`, `"steps"`, `"ipop4"`, `"stepsgrow"`;
  `pop_switch_start` `"carry"`; `rugged_try_rest` `"cut"`; `prefix_nan_reading` `"smooth"`.

### Fixed

- **`fun` gets its own copy of every point**, as in SciPy. It used to receive writeable views of the optimizer's buffers, so an objective that wrote into `x` changed the search, and one that kept `x` saw it change.
- **Bad inputs fail before any evaluation, with plain errors**: a non-finite `x0` (`ValueError`); a `config` that is neither an `ATMConfig` nor a preset name (`TypeError`), or names no preset (`ValueError`); an `rng` that is not a NumPy `Generator` or `RandomState` (`TypeError` naming `seed=`); a `seed` that is not an integer or a `SeedSequence` (`TypeError`) or is outside `[0, 2**32)` (`ValueError`); a `bool` `budget` (`TypeError`, it was taken as 1). A `fun` that returns an array, a list, a complex number or `None` gets a `TypeError` that says what it returned.
- **The value at the starting point is a `float`** like every other value; a `fun` returning `Fraction` no longer crashes.
- **Warnings raised by `fun` reach the caller** on the COBYQA and basin routes, which used to ignore every warning. SciPy's own overflow warnings from finite differences over a failed evaluation are no longer shown.
- **The diagnostic logs stay bounded.** `anytime.LAST_STAGES` (alias `STAGE_LOG`) holds the stages of the most recent anytime run only; `cma_phase.PROBE_LOG` records only while `cma_phase.RECORD_PROBES` is true.
- **Runs in separate threads no longer share the step unit** of `Variant.box_steps`. It was the module global `qn_phase.X_UNIT`, which every call wrote, so two threads on different boxes used whichever unit was written last; it is now a context variable (`qn_phase.set_x_unit` / `qn_phase.x_unit()`), one per thread. Every run is bit for bit as before.

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

[Unreleased]: https://github.com/BjBodner/amm-optimizer/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/BjBodner/amm-optimizer/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/BjBodner/amm-optimizer/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/BjBodner/amm-optimizer/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/BjBodner/amm-optimizer/releases/tag/v0.3.0
[0.2.0]: https://pypi.org/project/amm-optimizer/0.2.0/
