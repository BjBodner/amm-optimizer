# b095_final_ft1: results, v1 (FT1, the final test of the frozen c46 on bbob ids 101-110)

* Cycle: final
* Arms: `c46` (frozen), the field
* Base: the field (pycma IPOP / BIPOP, SLSQP, DE, TPE, NGOpt; HCMA archive levels)
* Verdict: v1 (NGOpt at D 2 only, as logged before any result was read): every F1 and F4 prediction held; F2 and F3 fell just outside their ranges, in AMM's favour
* Composite: `c46` (frozen)
* Commit: (the commit that adds this file)

`final_ft1.py --ngopt-dims=2` (this folder), run once on 2026-10-22 at 19:27, when every non-NGOpt arm had finished. The output is in `ft1_output_v1.txt` and the numbers in `ft1_v1.json`; the figures (`ft1_plots.py --ngopt-dims=2`) are in `figures/`. One run per (function, D, instance, budget), seed 9000: 6,000 runs of `c46`. NGOpt's D 5-40 arms are still running, for v2.

## F1: AMM − pycma IPOP-CMA-ES per budget (120 cells each; Holm over five)

| Budget | Mean [95% CI] | Predicted | |
|---|---|---|---|
| 10·D | **+.121 [+.080, +.168]** | +.06 to +.17 | held |
| 30·D | **+.203 [+.153, +.254]** | +.12 to +.25 | held |
| 100·D | **+.303 [+.244, +.361]** | +.20 to +.36 | held |
| 300·D | **+.312 [+.251, +.375]** | +.20 to +.36 | held |
| 1000·D | **+.222 [+.170, +.280]** | +.12 to +.27 | held |

Every budget is above 0 under Holm (p = .0005 each). Per D x budget, every one of the 25 cells has its interval above 0 (descriptive).

## F4: AMM − restarted SLSQP per budget

| Budget | Mean [95% CI] | Predicted | |
|---|---|---|---|
| 10·D | **+.015 [+.007, +.022]** | −.01 to +.03 | held |
| 30·D | **+.037 [+.020, +.053]** | .00 to +.05 | held |
| 100·D | **+.118 [+.085, +.156]** | +.05 to +.15 | held |
| 300·D | **+.225 [+.183, +.276]** | +.13 to +.27 | held |
| 1000·D | **+.302 [+.246, +.362]** | +.20 to +.35 | held |

## F2 / F3 (v1: NGOpt at D 2 only)

* **F2, credibly behind some competitor:** **29 of 600 cells** (predicted 30-75): just below the range, in AMM's favour. By D: 11 / 4 / 7 / 4 / 3 at D 2 / 5 / 10 / 20 / 40. By budget: 7 / 10 / 3 / 5 / 4 at 10 / 30 / 100 / 300 / 1000·D.
* **F3, credibly ahead of every competitor that ran the cell:** **122** (predicted 70-120): just above the range. By D: 12 / 21 / 21 / 23 / 45.
* **Per competitor (credibly behind / credibly ahead, of the cells it ran):** IPOP 6 / 323; BIPOP 7 / 326; SLSQP 11 / 299; DE 0 / 455; TPE 2 / 149 (of 288); NGOpt 0 / 80 (of 120, D 2 only); HCMA's archive levels 11 / 173 (of 480).
* NGOpt at D 2 only lowers F2 and raises F3 (development data: 32 → 27 and 99 → 110 under the same restriction). v2 recounts with every NGOpt dimension. Lesson 42: cell-level intervals are about 25% too narrow, so F2 and F3 are counts, not significance statements.
* **The 29 cells behind, by type:**
  * small budgets on smooth functions, where restarted SLSQP leads: f2, f8, f10, f11, f12, f13 and f14 at 10-30·D, D 2-10; the largest are f10 D 5 30·D −.31 and f12 D 2 30·D −.26;
  * HCMA's archive levels on f7 (D 10, 100·D −.08; D 20, 300·D −.27) and f19 at D 10 (10-300·D, −.03 to −.05);
  * IPOP / BIPOP on f12 D 2 300·D (−.14), f14 D 2 300·D (−.12), f14 D 20 / D 40 1000·D (−.04), f18 D 2 1000·D (BIPOP −.15), and f17 / f18 at D 20-40, 30·D (−.03);
  * TPE on f17 / f18 at D 5, 30·D (−.06 / −.04);
  * HCMA on f13 D 2 300·D, f13 D 20 10·D and f24 D 2 1000·D (−.03 to −.08).

## Descriptive (for T1-T3; not part of F1-F4)

* **Cells where AMM has the highest mean (ties included), of 120 per budget:**

  | Budget | AMM | Next most frequent leader |
  |---|---|---|
  | 10·D | 64 | HCMA 21 |
  | 30·D | 68 | SLSQP 28 |
  | 100·D | 88 | SLSQP 14 |
  | 300·D | 94 | HCMA 10 |
  | 1000·D | 95 | IPOP 9 |

  409 of 600 in all.
* **AMM − HCMA's archive level (D 2-20, 96 cells per budget):** +.016 / +.120 / +.174 / +.104 / +.103 at 10 / 30 / 100 / 300 / 1000·D.

## Reading

The lead measured in development on ids 61-70 holds on instances no design decision used: F1 and F4 are within about .01 of the development numbers (c44 on ids 61-70) at every budget. The losses are where development had them: small budgets on smooth functions against restarted SLSQP, f7 and f19 against HCMA's levels, and a few D 2 cells against IPOP / BIPOP at 300-1000·D. The paper's T1-T3 (`paper_draft_v1/final_test/pending_claims.md`) follow from these numbers by the rule in `draft_v3.md`: T1, T2 and T3 are kept. T3's margin at 10·D is small (+.016), so the paper should say "ahead from 30·D by +.10 to +.17", or keep "from 10·D" with the number.
