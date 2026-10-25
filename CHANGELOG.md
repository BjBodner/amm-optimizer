# Changelog

## 0.5.0 (2026-10-25)

A simpler call, and the anytime mode. The default algorithm is unchanged: with `anytime=False` every configuration runs bit for bit as in 0.4.0 (408 recorded runs of `c46` and the default match `tests/characterisation.json`; `c46`'s config hash is still `2153948ebed51150`). 0.4.0 was tagged but never uploaded to PyPI; its changes (AMM as the default of `minimize`, the removal of `ATMConfig.improved*`, the author name "Benjamin Jacob Bodner") are part of this release.

- **Simpler `minimize`.** `x0` and `budget` are now optional, so `amm.minimize(f, bounds=[(-5, 5)] * 10)` is a complete call. `x0` defaults to the centre of the box and `budget` to `100 x D` evaluations (at least 100). `bounds` also accepts a list of `(low, high)` pairs (SciPy's form), a NumPy array of shape `(D, 2)` and `scipy.optimize.Bounds`, besides the earlier `(lower, upper)`. A whole float such as `budget=1e4` is accepted. Errors are plain: a non-callable `fun`, a `bounds` and `x0` that disagree on the number of variables, a `budget` that is not a whole number, and a tuple of two length-2 sequences that reads both as pairs and as `(lower, upper)` are refused with a message that names the fix. Every call that worked in 0.4.0 runs unchanged.
- **Anytime mode** (`minimize(..., anytime=True)`, or `Variant.anytime="replay"`; guide: `docs/ANYTIME.md`). For runs whose budget is not known in advance or that you cut short: up to 10 000 evaluations the run never learns `budget`; it plans 10 x D evaluations and doubles the plan while the run goes on, answering every point already evaluated from memory, then hands over to one ordinary run fed those points. Measured on COCO/bbob: level with the default at 10^4 and 10^5 x D, and .05 to .12 of the target share below it between 30 x D and 1000 x D when the budget is known (so leave it off if you know your budget). The objective must be deterministic; `nfev` counts the calls actually made.
- **Faster long runs** (same evaluations, bit-identical results): the STEP phase keeps incremental difficulties and a chunked axis store, so its cost per evaluation no longer grows with the run length, and the start search is incremental.
- `docs/ANYTIME.md` is in the source distribution.

## 0.4.0 (2026-10-23)

The default of `minimize` is now AMM. No algorithmic change: the engine, `ATMConfig()` and every preset behave exactly as in 0.3.0 (the golden, parity and characterisation tests are untouched).

- **Breaking: `amm_optimizer.minimize` called without `config=` now runs AMM** (`presets.get("c46")`, config hash `2153948ebed51150`), not the original ATM sampler. In 0.3.0 and earlier the omitted `config` meant `ATMConfig()`: on a 5-dimensional Rosenbrock example (budget 2000, seed 0) the old default reaches f = 1.7 and AMM reaches f of about 1e-13.
- **Breaking: AMM needs the search box.** A call with neither `config=` nor `bounds=(lower, upper)` raises `ValueError` and the message names both ways out. Calls that pass a `config` are unchanged, including those that pass `bounds`.
- **To get the old behaviour** pass `config=amm_optimizer.ATMConfig()` (no `bounds` needed), or call `amm_optimizer.api.minimize`, the engine, whose default is still the original sampler and which the test suite uses.
- Added: `amm_optimizer/default.py` with the public `minimize` (same signature and return type as the engine); `amm_optimizer.api` is importable from the package namespace. `tests/test_default_amm.py` pins the new default against the explicit `c46` call and the engine.
- Docs: the README starts from the default AMM call.
- Removed: `ATMConfig.improved()`, `improved_wave5()`, `improved_v2()` and `improved_wave6()` (the earlier hybrid; AMM is `presets.get("c46")`). `ATMConfig()`, `ATMConfig.composite()` and every preset of `presets.names()` are unchanged (same config hashes). To reproduce the earlier hybrid, build it from `Variant` fields.
- The tests, benchmarks and tools that relied on the old default now call `amm_optimizer.api.minimize`; a pure rename.

## 0.3.0 (2026-10-23)

Packaging release; no algorithmic change (c46 is bit-identical: 408 recorded runs of c46 and the default match `tests/characterisation.json`).

- **Breaking: the import name is now `amm_optimizer`** (`import amm_optimizer as amm`). 0.2.0 installed the top-level modules `atm` and `amm`, which collide with the unrelated PyPI projects of those names; the `atm` and `amm` modules are gone. `ATMConfig` and the preset names (`c46`) are unchanged.
- The title and description follow the paper: "Adaptive Multi Mode (AMM): budget-aware routing of optimizers by online landscape probes" (the unreleased 0.2.1 metadata changes are folded in; 0.2.1 was skipped).
- Upgrading from 0.2.0: replace `import atm` / `import amm` by `import amm_optimizer as amm`; `amm.minimize`, `amm.presets` and `amm.ATMConfig` are unchanged. `pip install -U amm-optimizer` needs Python >= 3.12.
- Requires Python >= 3.12, NumPy >= 2.0, SciPy >= 1.18 (the versions the results were recorded with; 0.2.0 declared Python >= 3.10 and numpy >= 1.23, which SciPy 1.16 or newer cannot satisfy). Metadata uses a PEP 639 license expression.
- Fixed: `import` no longer fails on NumPy 1.x (`np._core`); the c46 preset no longer crashes for one-dimensional problems at budgets of 150 or more (`np.cov` is 0-d).
- Added: `py.typed` for the single package; installed-package smoke test (`tools/smoke_installed.py`); GitHub Actions workflows for CI, release (Trusted Publishing) and security checks; `SECURITY.md`, `CONTRIBUTING.md`.
- Tests: a one- to three-dimension smoke matrix over every shipped configuration; tests that need `cocoex` skip without it; the harness process pool replaces a broken pool and reruns unfinished runs (fixes `WinError 6` failures on Windows under load).

## 0.2.0 (release of the freeze)

- Public release as `amm-optimizer` (import `amm`; the implementation keeps its code name `atm`). Packaging only: the code is that of tag `freeze-2026-10-22` (c46, config hash `2153948ebed51150`); added the `amm` alias package, `CITATION.cff`, and an AMM README. No algorithmic change.
