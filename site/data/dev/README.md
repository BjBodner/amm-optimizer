# Site data, development set (in-sample, ids 61-70)

Every number here is **development, in-sample data on bbob ids 61-70**. Those ids were used during development, so they are not held out. Do not present them as a final result; the held-out test is FT1 (ids 101-110). Label every figure built from these files "development, in-sample, ids 61-70".

Generated 2026-10-22 by a read-only script that reuses the plotting code from worktree `<local-path>` (commit 7689d51, "Field plots for c46 on ids 61-70"). Nothing was run, written or modified outside this folder.

## Files

| File | Content |
|---|---|
| `field_cells.json` | `cells`: 600 records (f1-f24 x D {2,5,10,20,40} x budget {10,30,100,300,1000} x D) with the ECDF value of every method, the best competitor, `delta`, `status`. `records`: the same values flattened to one row per (function, D, budget, method). |
| `ecdf_by_dim.json` | The curves of `c46_field_ecdf_by_dim.png`: mean over f1-f24 per D, method and budget, plus the virtual-best-per-cell curve. Also the five function groups (the `c46_field_ecdf_groups.png` panels). |
| `provenance.json` | Source files with sha256, git commit and branch of each run file, config hashes, seeds, instance ids, recorded package versions, per-file harness argv, plot commands. |

ECDF value: for one run, the fraction of 51 targets (1e2 ... 1e-8) reached within the fixed budget (budget x D evaluations); per cell, the mean over the 10 instances (ids 61-70, one seed each, seed offset 9000); per aggregate curve, the unweighted mean over the 24 functions. Methods: `c46` (the frozen AMM configuration, internal name ATM), `IPOP`, `BIPOP` (pycma), `SLSQP`, `DE` (SciPy), `TPE` (optuna), `NGOpt` (nevergrad), all in the same harness, and `HCMA` (COCO archive levels, its own instances).

## Cell-status rule (as used by the paper heatmap)

From `field_plots.py` main() and `sota_plots.heatmap`: `delta = c46 - max(IPOP, BIPOP, SLSQP, DE, TPE, NGOpt, HCMA)` over the competitors that have a value in that cell (unrounded values).

* `ahead`: delta > +0.02
* `behind`: delta < -0.02
* `level`: |delta| <= 0.02

**Reproduced:** ahead 179, level 313, behind 108, of 600 cells, matching the heatmap title "Ahead in 179 cells, behind in 108 (of 600)". By D (ahead/behind): D2 35/34, D5 38/24, D10 31/16, D20 29/20, D40 46/14. Behind cells are led by SLSQP 40, NGOpt 22, HCMA 18, IPOP 14, TPE 8, DE 4, BIPOP 2.

The same research notes quote "credibly behind 29, ahead of all 99" for c46. That is a different, stricter ledger (goal_ledger / `fresh_claims` style) and is **not** what this file computes. Do not mix the two on the site.

## Caveats

1. **The "c46" series is a composite, not one run.** `field_plots.py` builds it from per-D files (c23 for D=2, then c25 ... c46 overlaid in order). In the final composite, 528 of 600 cells come from the full-grid **c45** run (config hash e39ae91b08a4d3b4) and only 72 cells (D 10, 20, 40 at 100 x D, all 24 functions) come from runs of the literal c46 config (hash 9f570d51aa12852a at run time). c46 = c45 + `prefix_low_beta=0.15` from D>=10, so the other cells are assumed identical under c46 and were not re-run. Each cell carries `c46_source_stems`. Say this on the Reproduce page.
2. **Config hash.** Hash at run time 9f570d51aa12852a; hash at tag `freeze-2026-10-22` (commit 552777b) is 2153948ebed51150 because the `Variant` schema gained an opt-in field. The config text of the runs is identical to `benchmarks/arms/c46_config.txt` at the tag (checked, sha256 in provenance).
3. **Unequal competitor coverage.** HCMA has no D=40 values. TPE ran a capped slice only (D2/5/10 at 10-100 x D, D20 at 10-30, D40 at 10). NGOpt has no D=10 runs. Where a competitor is missing, the "best competitor" is the best of the others, so those cells are easier to win than the 7-way comparison suggests. HCMA is archive data on other instances, not the same harness.
4. One seed per instance, 10 instances: no confidence intervals. A 0.02 margin is a plotting threshold, not a significance test.
5. Run files carry `git dirty: true` for most stems (development worktrees). Run-file sha256s are in provenance.
6. NGOpt D=5, 1000 x D is recorded as run (the plot's footnote notes its budget-dependent choice).
7. In `ecdf_by_dim.json` the virtual best includes c46 and c45 themselves (as in the figure), and c45 is the "previous" curve.

## Not found / gaps

* **Exact command line for the c46 plots** is not recorded in RESULTS.md (it stops at the c43 section). The provenance lists the inferred command (`field_plots.py c46 c25 ... c45`, `bbob_ecdf_plots.py c46 c45`); the generator's composite was verified to reproduce 179/108.
* **optuna and nevergrad versions** are not recorded in the run files. Recorded: Python 3.12.10, NumPy 2.5.4, SciPy 1.18.1, pycma 4.5.0, cocoex 2.8.2, atm 0.2.0.dev0, Windows 11.
* **Hardware** (CPU, RAM) is not recorded in the run files; only platform string and wall seconds.
* Raw run files live in the c61-cobyqa worktree (`benchmarks/results/scan/audit/sota6170/`); the c44 files there are untracked in git, so back them up before that worktree is removed. The `<worktree>` worktrees contain no c44-c46 field files.
* The HCMA levels come from `docs/research/audit_2026-10-17/archive_sota_levels.json` (sha256 in provenance); how it was derived is in `archive_sota.py`, not re-checked here.
