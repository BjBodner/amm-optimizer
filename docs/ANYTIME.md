# The anytime mode (`anytime=True`)

AMM normally plans every phase from the budget it is given, which it must know when it starts. The anytime mode does not need the budget until 10 000 evaluations have been spent, and keeps a small memory instead. This page covers:
* how the mode works and how to switch it on;
* what it was measured to cost and gain;
* when to use it and when not;
* what remains to be done.

The mode is opt-in. With `anytime=False` (the default; `Variant.anytime="off"`) every configuration runs bit for bit as before, and its config hash is unchanged (the frozen `c46` is still `2153948ebed51150`).

## How it works

`variant.anytime = "replay"` (code: `src/amm_optimizer/anytime.py`). The defaults are `anytime_start_per_dim = 10`, `anytime_factor = 2` and `anytime_switch = 10_000`.

1. **Stages up to the switch.** Stage `k` plans `B_k = 10 · 2^k · D` evaluations in total: 10, 20, 40, … × D, for as long as `B_k ≤ 10 000`.
    * Each stage reruns the configuration (with `anytime="off"`) from the same start point and seed, as an ordinary run planned for `B_k`.
    * A point that the previous stage evaluated is answered from memory without a call of `f`. It still counts against the stage's plan, since the planned budget is cumulative.
    * Whatever the larger plan does as the smaller one did is therefore rebuilt exactly and at no evaluation cost: the probes, designs, descents, and ATM's and CMA's states. From the first decision the larger budget changes, the larger plan runs. At the end of stage `k` the run has been exactly the configuration's run at `B_k`, which `tests/test_anytime.py` checks.
    * In this part the run never reads `budget`; it only caps the calls.
2. **The hand-over.** Once the next plan would exceed the switch, one ordinary run of the configuration gets every evaluation left of `budget`: the normal path, which plans for that budget. It starts from the run's own `x0`, which on calibration beat starting at the best point found, by +.013. Every point evaluated before is fed into it:
    * the prefix's design is the archive's best points;
    * ATM's first search skips its uniform first pool;
    * the first CMA run's covariance is learned from the archive (`cma_warm_start`);
    * the restarted SLSQP's sample starts with the archive;
    * a point it asks for again is answered from memory.

   Only the replay part's points are kept, at most about 12 000 of them, which is a few MB at D 40 whatever the budget.
3. **The result** is the best point evaluated in any stage.

The objective must be deterministic: a remembered value stands in for a new call.

```python
import amm_optimizer as amm

result = amm.minimize(f, bounds=[(-5, 5)] * 10, budget=200_000, seed=0, anytime=True)
# the same, spelled out:
config = amm.presets.with_changes("c46", anytime="replay")
result = amm.minimize(f, x0, 200_000, seed=0, bounds=(-5, 5), config=config)
```

`budget` caps the calls of `f`. `result.nfev` counts the calls actually made, which can be a little below the cap, because a replayed point is answered from memory and not called again.

**Stopping a run yourself.** `minimize` returns only when the budget is spent, so to cut an anytime run short, keep the best point in your own wrapper and raise from it when time is up:

```python
class OutOfTime(Exception):
    pass

best = {"f": float("inf"), "x": None}

def watched(x):
    value = f(x)
    if value < best["f"]:
        best["f"], best["x"] = value, x.copy()
    if time_is_up():
        raise OutOfTime
    return value

try:
    amm.minimize(watched, bounds=[(-5, 5)] * 10, budget=10**6, anytime=True)
except OutOfTime:
    pass
# best["x"], best["f"]: the best point seen when the run was cut
```

The benchmark harness takes the mode as `benchmarks/quick_scan.py run --config "preset:c46,variant.anytime=replay"`.

`quick_scan` records for each run the first evaluation that reached each of the 51 targets (`hits`, read with `score_by`), so one run can be scored at every smaller budget.

## What it was measured to do

All of this is post-freeze research on the frozen `c46`. The score is the harness's share of the 51 targets (1e2 … 1e-8) at each budget. Differences are paired by problem, with 95 % cluster-bootstrap CIs. The pre-registrations, results and data of the experiments below (numbers 104 to 106) are kept in the research archive.

**The final mode on the final tests' problems** (experiment 106: ids 101-110, seed 9000, D 2, 5, 10, 20, 40, f1-24, 8400 runs through COCO's observer, one dataset per budget):

| Mean target share | 10 × D | 30 × D | 100 × D | 300 × D | 1000 × D | 10^4 × D | 10^5 × D |
|---|---|---|---|---|---|---|---|
| anytime mode | .171 | .234 | .418 | .571 | .772 | .903 | .940 |
| c46 told its budget | .173 | .288 | .496 | .696 | .820 | .903 | .939 |
| anytime − c46 | −.002 | −.054 | −.078 | −.124 | −.048 | +.001 [−.006, +.007] | +.001 [−.004, +.006] |

* **Per D:** at 300 × D the deficit is −.13 to −.16 for D ≥ 5. At 10^4 and 10^5 × D every D is within ±.01.
* **Against the FT1 field** (IPOP-CMA-ES, BIPOP-CMA-ES, SLSQP, DE, TPE, NGOpt, at 10-1000 × D): above every method at every budget, except SLSQP at 30 × D (−.018 [−.027, −.008]).
* **CPU time:** the mode's own time per evaluation is up to 3.3 × c46's at 10-1000 × D (264 against 81 µs at 1000 × D), and about level at 10^5 × D (81 against 77 µs).
* **Caveat:** this mode was designed after the final test, on other instances, and Experiment 106 reuses the final-test instances only as a descriptive line beside c46 and the field.

**How the design was reached** (fresh ids 850-869 and calibration ids 1-3; experiments 104 and 105):
* **Experiment 104: replay alone, with no switch.** It is −.049 / −.070 / −.105 / −.039 below c46 at 30 / 100 / 300 / 1000 × D, and +.06-.07 above plain restarts (a fresh c46 run from the best point at each stage) at 100 and 1000 × D. The A/A is clean.
* **Experiment 105: two ways of keeping more state; neither beat replay.**
    * An archive warm start at every stage (`continue`) gained only +.01-.03 over restarts. Its stages are small fresh plans, and it trailed replay by .05-.06 at 100 and 1000 × D.
    * Nesting the large stages on the 80 × D prefix stage (`nest`) lost .03-.04 to replay above 100 × D. An ablation put the loss on starting c46's large-budget plan at a good point rather than at a fresh one.
* **Calibration of the hand-over** (ids 1-3): level with c46 at 10^4 × D (+.002 to +.022 per D). At D 20 and 1000 × D, where the hand-over gets about half the budget, it was −.07.

The cause of the deficit below the switch is c46's plan, which changes at almost every doubling:
* the prefix kind at 10, 20 and 30 × D;
* the regime at 100 × D (prefix to ATM and the hand-over phases);
* ATM's pool and share, which grow with the budget.

So the larger plan rebuilds only 0-47 % of each stage from memory (calibration id 1), and at budget E the run mostly holds c46's result for about E / 2.

## When to use it, and when not

**Use it when the budget is not known in advance or the run may be stopped early, with a deterministic objective:**
* **Expensive objectives.** The mode never evaluates a point twice, and its extra cost is the optimizer's own CPU time: 0.2-0.75 ms per evaluation more than c46's own time on these runs (the most at 30 × D). Next to an objective that takes milliseconds or seconds per call, that is negligible.
* **Small budgets you may cut short.** Up to 10 000 evaluations, the run is at every moment the end of a complete smaller plan, plus the best of the earlier ones. Its score at budget E is about c46's at E / 2. At 10 × D it is c46's.
* **Large budgets** (10^4 × D and above, in practice at least about 20 000 evaluations). The hand-over makes it c46 to within .01, at every D measured (2-40).

**Do not use it:**
* **When the budget is known and lies between about 30 × D and 1000 × D.** c46 told its budget is .05-.12 better there, and by .13-.16 at 300 × D for D ≥ 5.
* **When the optimizer's own CPU time matters** (cheap objectives, many runs). Below the switch the mode costs up to 3.3 × c46's own time per evaluation, because every stage reruns c46 from the start.
* **With noisy or non-deterministic objectives.** A remembered value would stand in for a fresh evaluation.
* **When the run may be stopped after the switch.** From then on it is an ordinary run planned for the remaining budget, so it needs a true budget just as the normal mode does. Stopped part-way, it is a c46 run cut short. Where the hand-over gets only part of the budget, it trails c46 told its budget: −.093 at D 20, 1000 × D (20 000 evaluations, about half of them after the switch).

## Limitations, and why the mode recomputes instead of storing state

The mode keeps the run's state by recomputing it, not by storing it. Evaluations of `f` are cached; the optimizer's own work is not. Keeping each engine's state in memory and continuing it when the plan grows is not possible with the current components, for three reasons:
1. **The local searches cannot be paused.** They are SciPy's SLSQP, L-BFGS-B and COBYQA, which run a descent to its end inside one call and expose neither a pause nor their internal state (the quasi-Newton matrix, the trust region).
2. **Every phase plans from the budget when it starts.** It fixes its share of the budget (ATM's fifth, the separable search's half, the stage caps of the local phase) and its route on entry, and discards its state when it returns. Only the best point, a step size and a few readings are passed on.
3. **The route changes with the budget.** The prefix runs below 100 × D and the hand-over phases above. So the state of a smaller plan's engine is often not one the larger plan would use. Experiment 105 measured this: handing state to the larger plan, as a warm start or as a better start point, recovered none of the difference to replay.

## Future work

1. **Resumable engines.**
    * Phases that hold an explicit state object (CMA's mean, step size, covariance and paths; IPOP's population and restart count; the restarted SLSQP's basins and sample; ATM's sampler state; STEP's axes) and continue when given more budget.
    * Local searches that can pause, either implemented in-house or wrapped as coroutines.
    * A planner that re-decides the route at each boundary.

   This would remove the recomputation (CPU like a single run) and let state carry over where routes agree.
2. **Plans that nest by design.** An anytime version of c46's plan whose early decisions do not change with the total: budget-free prefix kinds, a regime switch that keeps the prefix's work, and a budget-independent ATM pool and share. Replay would then rebuild most of each stage from memory. The replay share per boundary (0-47 % today) is the measure to raise.
3. **A large-budget plan that does well from a good start.** c46's large-budget plan does worse from the best point found than from a fresh start (experiment 105), and its separable route's pair test can misread rotated functions from some start points. Fixing that would let the hand-over, and any nesting, build on the replay part's best point.
4. **Tuning.** The factor (2), the first plan (10 × D) and the switch (10^4 evaluations), each on development ids and pre-registered.
5. **Anytime above the switch.** A large-budget part that needs no budget either: for example doubling with nested large plans, or a budget-free large-budget engine. Then the whole run is anytime, and one COCO run per problem gives the full anytime ECDF.
6. **Noisy objectives.** In place of exact-point memory: re-evaluation, or a noise model.
