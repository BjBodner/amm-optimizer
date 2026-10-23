# Contributing

Thanks for looking. Bug reports, failing examples and documentation fixes are the most
useful contributions; for a change to the algorithm itself please open an issue first
(see "The algorithm is frozen" below).

## Set up

```bash
git clone <repository-url>
cd <repository>
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
python tools/ci.py                # lint, format, types, research-tier tests
python tools/ci.py --tier merge   # also the harness tests and the recent presets
python tools/ci.py --tier full    # everything incl. the 400-case parity sweep; before a release
ruff check . && ruff format .
mypy
pytest -m "not slow and not merge" -n auto
```

Tests are tiered by marker: the default research tier is what you run while working, the
`merge` tier is for pull requests that touch presets, and `slow` is the full tier for
releases. A test that needs `cocoex` skips itself when it is not installed.

## The algorithm is frozen

The numbers in the paper come from one frozen preset (`amm_optimizer.presets.get("c46")`), and the
tests pin its behaviour bit for bit in one environment
(`tests/characterisation.json`, `tests/golden/presets.json`). Changes that alter a
trajectory are therefore made on a research branch with a pre-registered evaluation, not in
a pull request. Fixes that leave every trace unchanged (crashes, validation, packaging,
documentation, speed-ups that are bit-identical) are welcome, and `pytest` must show that
no snapshot moved.

## Pull requests

* One change per pull request, with a test that fails before and passes after.
* `ruff`, `mypy` and the research-tier tests are green.
* Describe what a user would notice; add a line to `CHANGELOG.md` under "Unreleased".
* Do not commit generated data. Result files belong in a release asset or an archive, not in
  git (`pre-commit` rejects files above 500 kB).

## Security

Report vulnerabilities privately, see [SECURITY.md](SECURITY.md).
