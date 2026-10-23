# Recorded-run demo data

Written by `site/tools/record_run.py` (run with the atm_v2 venv and `PYTHONPATH=<worktree>`; add `--publish` to copy the files to `site/public/demo/`, where the player fetches them). The script takes about 25 s wall time and 8 s of CPU.

## What is recorded

- One run each of AMM (preset c46, commit 552777b, config hash 2153948ebed51150, checked against `data/claims.json`) and of pycma IPOP on bbob f8, f15 and f21, D = 2, instance 1 (spent range 1-100, so no unused instance is touched), budget 600 evaluations (300 x D), seed 1, same start point (uniform in [-5, 5]^2 from `RandomState(1)`).
- Every evaluation through an objective wrapper: index (position), x, regret = f - f_opt (6 significant digits). f_opt is read from a throw-away COCO observer log for display only; neither optimizer sees it.
- A landscape grid, 81 x 81, of 10 x log10(f - f_opt) (floor 1e-8) for the contour bands.
- `header.stages`: see below.
- `*.svg`: the static poster used when JavaScript is off.
- JSON payload: 145 kB for the three runs (limit 300 kB). The SVG posters are about 175 kB each and only load without JavaScript.

IPOP is `cma.fmin(f, x0, 2.0, {bounds: [-5, 5], seed: 1, maxfevals: 600}, restarts=9, incpopsize=2)`. This is a demo setting, not the benchmark harness arm. One seed per function: the runs illustrate behaviour and are not evidence of how often either method wins. The numbers the player shows come from these files and are not site claims (`claims.json` is untouched; proposals are in `claims.proposed.demo.json`).

## Stage boundaries: what could be obtained

The frozen code has no trace of the stage AMM is in. `OptimizeResult` carries only `x, fun, nfev, nrestarts, timings`; `plan.py` and `control.py` hold no routing log.

The one documented hook is the `Profiler` protocol (`atm.minimize(..., profile=...)`, `src/atm/profiling.py`, exported from `atm`). Its blocks cover only the ATM sampler (`total`, `init`, `section`, `draw`, `sample.*`, `evaluate`, `advance`, ...). A custom profiler that logs the objective's evaluation count at the entry and exit of `init` and `section` gives the evaluation span of each ATM-sampler run. The recorder uses exactly that, and only if a second run without the hook gives the identical evaluation sequence (it did in all three recordings, as the docstring "profiling does not change the result" says). Nothing else is inferred. COBYQA, SLSQP, Nelder-Mead, STEP, MLSL, the CMA stages and all probes are not reported by any hook, so the player's timeline strip shows only the ATM-sampler span, and no strip for a run that never enters the ATM stage (f21 is routed away from it).

Open question to check in the rename branch: the routing map plans 120 evaluations for the ATM stage at D = 2, B = 600 (a fifth of the budget), while the profiler spans recorded for f8 and f15 are 80 evaluations long. The spans are what the hook reports; why they are shorter than the plan has not been investigated.

## Hook needed in the post-FT1 rename branch (`src/amm`)

To show every stage, `minimize` needs a stage-event hook. Minimal design, with no effect on the evaluation sequence when unused:

1. An optional `trace` argument to `minimize` (default `None`), a callable `trace(event: str, **info)`.
2. Call it at each decision point that `api.py` already has: entry and exit of each phase (`cobyqa`, `pair_test`, `basin_reading`, `prefix`, `atm`, `step`, `local_tests`, `finisher`), with the number of evaluations spent so far (`CountedObjective`'s counter), and for each probe its name, evaluations used, reading (for example beta, r2q, tie count) and the route chosen. The natural places are `api.py:minimize` (gates, prefix, ATM, hand-over), `_local_phase` / `_qn_local_phase` (local tests and routes), `_finish_gated_then_ipop` (finisher), and the probe functions in `qn_phase.py`, `chain_phase.py`, `basin_phase.py`.
3. Collect the events in `OptimizeResult` (a list) so no callback is needed, off by default (draft FIGURES.md gap 4: the final-test harness did not log routes).
4. Test that a run with the trace on has the same evaluation sequence as one without.

With that, `record_run.py` would write `header.stages.spans` for every engine and the player's strip would label them; the same events would fill the paper's F8 (route firing by function).
