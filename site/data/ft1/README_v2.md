# FT1 v2 record

`ft1_v2.json` and `ft1_output_v2.txt` are the output of the preregistered analysis script
`docs/research/batches/b095_final_ft1/final_ft1.py`, run with no arguments (every NGOpt dimension)
on 2026-10-23 after all 33 arms had finished, at research-final-ft1 commit 91e05e4c. `figures_v2/`
holds `ft1_plots.py --out=...` output (no `--ngopt-dims`). Nothing in the research worktree was written:
the outputs were sent to a scratch folder and copied here. To make them the official record, copy
them into `docs/research/batches/b095_final_ft1/` in the research branch and commit
(ft1_output_v2.txt, ft1_v2.json, figures/), with a results note for v2 beside results.md (v1).
`ft1_v1.json`, `ft1_output_v1.txt` and `results.md` are the v1 record (NGOpt at D 2 only).
