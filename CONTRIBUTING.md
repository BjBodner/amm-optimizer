# Contributing

Thanks for looking. Bug reports, failing examples and documentation fixes are the most
useful contributions; for a change to the algorithm itself please open an issue first
(see "The algorithm is frozen" below). Everyone taking part follows the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Set up

```bash
git clone https://github.com/BjBodner/amm-optimizer.git
cd amm-optimizer
uv venv
uv pip install -e ".[dev]" coco-experiment pytest-xdist
```

`pip` works the same way (`python -m venv .venv`, activate it, `pip install -e ".[dev]"
coco-experiment pytest-xdist`). The benchmark harness needs Python 3.12 or newer for
`coco-experiment`; the library itself does not need it.

Use one BLAS thread when you run tests, because the thread count changes the last bits of
some runs:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # PowerShell: $env:OPENBLAS_NUM_THREADS = "1"
```

## Checks

```bash
python tools/ci.py                # lint, format, types, default-tier tests
python tools/ci.py --tier merge   # also the harness tests and the recent presets
python tools/ci.py --tier full    # everything incl. the 400-case parity sweep; before a release
ruff check . && ruff format .
mypy
pytest -m "not slow and not merge and not research" -n auto
```

The documentation site (`mkdocs.yml`, pages in `docs/`, the API reference generated from the
docstrings) builds with `pip install -e ".[docs]"` and `mkdocs build --strict`, as CI's lint
job and Read the Docs build it; `mkdocs serve` previews it. A page with Python examples is
listed in `tests/test_docs_examples.py`, which runs them.

Tests are tiered by marker: the default tier is what you run while working, the `merge`
tier is for pull requests that touch presets, and `slow` is the full tier for releases.
Tests marked `research` pin research-only options or read the research harness's data
(`benchmarks/arms`, `benchmarks/quick_scan.py`, `tests/characterisation_arms.json`); the
default tier leaves them out, the merge and full tiers run them, and they skip where that
data is absent. A test that needs `cocoex` skips itself when it is not installed.

Two optional hooks run checks before a push; neither is enabled by default:

* `pre-commit install --hook-type pre-commit --hook-type pre-push` installs the hooks of
  `.pre-commit-config.yaml` (whitespace, YAML and TOML checks, ruff on every commit,
  the quick tests before a push);
* `git config core.hooksPath .githooks` makes `.githooks/pre-push` run
  `python tools/ci.py` before every push (`git config --unset core.hooksPath` turns it
  off; `git push --no-verify` skips it once).

Use one or the other: while `core.hooksPath` is set, git ignores the hooks that
`pre-commit install` wrote.

## The algorithm is frozen

The numbers in the paper come from one frozen preset (`amm_optimizer.presets.get("c46")`), and the
tests pin its behaviour bit for bit in one environment
(`tests/characterisation.json`, `tests/golden/presets.json`). Changes that alter a
trajectory are therefore made on a research branch with a pre-registered evaluation, not in
a pull request. Fixes that leave every trace unchanged (crashes, validation, packaging,
documentation, speed-ups that are bit-identical) are welcome, and `pytest` must show that
no snapshot moved.

## Deprecation policy

amm-optimizer follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html); while
the version is 0.x, the minor number plays the role of the major one.

* **Patch releases never change results.** A run with the same inputs, configuration and
  environment gives the same `x`, `fun` and `nfev` bit for bit, and every shipped
  configuration keeps its `config_hash`. A patch release can fix crashes, validate inputs,
  and improve documentation, packaging or speed, as long as no trace moves.
* **Minor releases may change behaviour**, with an entry in `CHANGELOG.md` that says what
  changes and for which runs.
* **Public API is deprecated before it is removed.** Something deprecated in minor release
  N keeps working throughout N and warns when it is used; it is removed in N+1 at the
  earliest. The CHANGELOG lists it under *Deprecated* in N and under *Removed* in the
  release that removes it, and its warning names the release.
* **The warning:** `DeprecationWarning` for API that code calls (functions, arguments,
  attributes), and `FutureWarning` for configuration values that a user writes, such as
  `Variant` fields, so that people running scripts see it too. Each warning names what is
  deprecated and the release that removes it.
* **What counts as public:** everything documented in `amm_optimizer.__all__`,
  `amm_optimizer.presets`, `ATMConfig` and its methods, and the `Variant` fields that a
  shipped configuration uses (the presets, `ATMConfig()`, `ATMConfig.composite()` and
  `anytime=True`). Modules and names starting with `_`, and the internal phase modules,
  can change in any minor release.
* **Research-only options.** The `Variant` fields and values that no shipped configuration
  uses are listed in `config.RESEARCH_FIELDS` and `config.RESEARCH_CHOICES`, which
  `tools/research_fields.py` measures (a test keeps the lists current). They were
  deprecated in 0.6.0 and are removed in 0.7.0. When an option is removed, its old
  default is kept for `config_hash`, so the hashes of shipped configurations, and the
  results recorded with them, stay valid.
* **Tests of deprecated behaviour** say so with
  `pytest.mark.filterwarnings("ignore:...:FutureWarning")` (or `DeprecationWarning`) on the
  test or the module; `pyproject.toml` makes every other such warning an error, so a shipped
  configuration that starts to warn fails the tests.

## Pull requests

* One change per pull request, with a test that fails before and passes after.
* `ruff`, `mypy` and the default-tier tests are green.
* Describe what a user would notice; add a line to `CHANGELOG.md` under `## [Unreleased]`,
  in its `### Added`, `Changed`, `Deprecated`, `Removed` or `Fixed` group
  ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/)).
* Commits: one logical change each, with a subject line in the imperative or as a
  short label ("Fix ...", "sdist: ..."), at most about 72 characters, and a body that
  says why. Keep refactors and behaviour changes in separate commits.
* Branches: `fix/...`, `feat/...`, `chore/...`, `docs/...` or `refactor/...`, from
  `main`. Pull requests are rebased onto `main` when merged, so keep the history of
  your branch readable.
* Do not commit generated data. Result files belong in a release asset or an archive, not in
  git (`pre-commit` rejects files above 500 kB).

## Security

Report vulnerabilities privately, see [SECURITY.md](SECURITY.md).
