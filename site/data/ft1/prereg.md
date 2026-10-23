# b095_final_ft1: the final test FT1 on bbob ids 101-110

* Cycle: final
* Arms: `c46` (frozen), the field
* Base: the field (pycma IPOP / BIPOP, SLSQP, DE, TPE, NGOpt; HCMA archive levels)
* Verdict: pending
* Composite: `c46` (frozen)
* Commit: (the commit that adds this file)

The maintainer approved FT1 as drafted on 2026-10-22 ("Yes I do, and I approve these both", in answer to "Approve FT1 as drafted?": FT1, and FT3's COCO-format data for AMM only). The pre-registration is `paper/PREREG_final_test.md` on branch `research-paper`; its FT1 entry is copied below verbatim, with the approval and the frozen configuration filled in. Arms file `benchmarks/arms/final_ft1.json`; results in `benchmarks/results/scan/audit/final101/`; the analysis is fixed by the entry.

## Execution notes, written before any run

* Ids 101-110 move from "reserved" to "spent" in `spent_ids.json` in this commit; 111-115 stay reserved.
* The field's arms run first, from 2026-10-22 about 16:15, because they do not depend on our configuration. Their results are not read until `c46`'s runs are complete. `c46`'s runs start after the freeze time (19:48), from this worktree at tag `freeze-2026-10-22`.
* One addition to the ids 61-70 ledger's field: NGOpt also runs at D 10, where that ledger lacked it. Every other arm is the ledger's own (`audit_sota6170*.json`), with `--instance-ids 101-110`.
* The analysis runs once, after every arm has finished, with `final_ft1.py` (written before any result is read).

## The pre-registered entry

```
ID / date / author approval:  FT1 / 2026-10-22 / approved by the maintainer, 2026-10-22 afternoon
Question:  Does the frozen AMM configuration keep its development lead over the classical field on bbob instances that no design decision used?
Claim (one sentence):  On bbob instances 101-110, AMM's target share exceeds pycma IPOP-CMA-ES's at every budget from 10 to 1000 x D, and AMM is credibly behind some competitor in at most a stated number of the 600 function x D x budget cells.
Configuration:  the frozen preset `c46`, tag `freeze-2026-10-22` (main 552777b), config hash `2153948ebed51150` (`ATMConfig.config_hash()`).
Competitors (same harness, same ids, same seed):  pycma IPOP-CMA-ES and BIPOP-CMA-ES, SciPy SLSQP (restarted), SciPy differential evolution, optuna TPE (the capped slice of the ids 61-70 ledger), nevergrad NGOpt; HCMA as COCO archive levels (its own instances, D 2-20), labelled as such. Versions as in the ids 61-70 field runs (the `sota6170` arms).
Data:  bbob (COCO 2.x), functions 1-24, D 2, 5, 10, 20, 40, budgets 10, 30, 100, 300, 1000 x D, instance ids 101-110 (raw instance numbers, `--instance-ids 101-110`), one run per (function, D, instance, budget), seed 9000. Unspent: yes (`spent_ids.json` lists 101-115 as reserved and never used). 111-115 stay reserved.
Metric:  per run, the share of 51 log-spaced targets 1e2 ... 1e-8 reached within the budget (the score of every development batch).
Analysis (fixed now):
  F1  AMM - IPOP per budget: mean over the 120 (function, D) cells with equal weight per function, 95% bootstrap interval over cells (2,000 resamples, seed 0); Holm over the five budgets, one-sided.
  F2  Cells credibly behind: the field ledger's rule (`goal_ledger.py`): a cell counts if some competitor leads by more than .02 with posterior probability above .9 (`eb2`), over the 10 paired runs; union over competitors; reported per D and per budget, and every such cell listed by name.
  F3  Cells credibly ahead of every competitor that ran the cell: same rule, flipped.
  F4  AMM - SLSQP per budget, as F1 (the strongest classical competitor at small budgets).
  Lesson 42: cell-level intervals are about 25% too narrow; F2 / F3 are reported as counts with that caveat, never as significance.
Predicted results (ranges):
  F1  10 x D +.06 to +.17; 30 x D +.12 to +.25; 100 x D +.20 to +.36; 300 x D +.20 to +.36; 1000 x D +.12 to +.27  (ids 61-70, c44: +.122, +.197, +.296, +.302, +.211).
  F2  30 to 75 of 600 cells credibly behind (ids 61-70, c44: 38; c43: 57).
  F3  70 to 120 of 600 cells credibly ahead of all (ids 61-70: 94).
  F4  10 x D -.01 to +.03; 30 x D +.00 to +.05; 100 x D +.05 to +.15; 300 x D +.13 to +.27; 1000 x D +.20 to +.35  (ids 61-70: +.020, +.034, +.110, +.218, +.292).
Falsifiers:  F1 fails if any budget's Holm-adjusted interval reaches 0; the headline "ahead of IPOP at every budget" is then rewritten per budget. F2 above 100 cells, or F3 below 50, means the development ledger was optimistic beyond the stated ranges; the abstract then states the measured numbers and the losses lead section 5.6.
Compute estimate:  AMM on 6,000 runs at 3 workers: about 1-3 h wall (the ids 61-70 runs of c44 at 10 and 100 x D took 1-6 min per D). The field: about 4-8 h wall, dominated by NGOpt and TPE (rough; re-measured on the first function before committing to the rest).
If it fails:  the paper keeps the frozen configuration and reports what the final test shows; the bold sentences T1-T3 in draft_v3 follow the rule written there (weaker form or deleted, never strengthened). If the lead over IPOP collapses, the go/no-go fallback (claim C, a BBSR-primary methodology paper) applies.
```

## Deviation (2026-10-22 ~17:00, before any result was read)

The maintainer: "you can run everything now, I want to get everything done with". `c46`'s arms started at about 17:00 instead of at 19:48. Nothing else changed: the freeze tag, the configuration (`c46`, hash `2153948ebed51150`), the data and the analysis are as above. No result of the field's arms had been read.

## Analysis-script additions (2026-10-22 ~18:30, before any FT1 result was read)

`final_ft1.py` gains `--folder=` (to dry-run it on development data) and `--json=` (machine-readable output for the paper's numbers ledger), and three descriptive tables used only for the paper's wording of T1-T3. They are not part of F1-F4:
* per budget, the cells where AMM has the highest mean (ties included), and otherwise which competitor has it;
* AMM − IPOP per D x budget, bootstrap over functions;
* AMM − HCMA's archive level per budget.

F1-F4 are computed exactly as before. Dry run on ids 61-70 with `c45`'s full grid standing in for `c46` (`--folder=<scratch>`) reproduced `c45`'s development ledger (32 behind, 99 ahead of all).

## NGOpt: wait (the maintainer, 2026-10-22 ~18:00)

Relayed by the release-day status session: "let it run in full, we can prepare everything and just wait for these results to come in". NGOpt runs at D 2-40 in full, with no cut and no deviation from the analysis, which runs once after every arm, NGOpt included, has finished. NGOpt's processes run at below-normal priority so the other arms finish first.

## Deviation for v1 (2026-10-22 ~18:20, before any FT1 result was read)

The maintainer, relayed by the release-day status session: "let's divide this effort into 2, publishing everything all the way without NGOpt and the next version will just add this result to everywhere that's needed." This supersedes the ~18:00 decision to wait.

* **v1, the paper's first version.** The pre-registered analysis runs once, as soon as every arm except NGOpt at D 5-40 has finished, as `final_ft1.py --ngopt-dims=2`. NGOpt enters F2 and F3 at D 2 only; at D 5, 10, 20 and 40 it counts as "not run", as HCMA's archive does outside D 2-20. F1 and F4 (AMM − IPOP, AMM − SLSQP) do not involve NGOpt and are unchanged. Every place in the paper where NGOpt's D 5-40 cells could change a number is marked as a slot pending v2. The figures are drawn with `ft1_plots.py --ngopt-dims=2`.
* **Known effect, from development data only.** With fewer competitors per cell, F2 can only fall and F3 only rise. On ids 61-70 (`c45` standing in for `c46`) this restriction moved F2 from 32 to 27 and F3 from 99 to 110. The paper says so where it prints F2 / F3.
* **v2.** The NGOpt arms keep running at below-normal priority. When they finish, F2 / F3 and the descriptive tables are computed again with every NGOpt dimension (`final_ft1.py` without `--ngopt-dims`), labelled as the full-field count. v2 replaces v1's F2 / F3 wherever they appear, and v1's numbers stay in this record.
* F1-F4's definitions, the data, the prediction ranges and the frozen configuration are unchanged.
