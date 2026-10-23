# Changelog

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
