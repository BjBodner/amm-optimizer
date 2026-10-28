# Citing AMM

If you use amm-optimizer in published work, please cite it. The repository's
[CITATION.cff](https://github.com/BjBodner/amm-optimizer/blob/main/CITATION.cff) holds the
citation metadata (GitHub's "Cite this repository" button reads it). The paper's reference will
be added when it is public.

As BibTeX:

```bibtex
@software{bodner_amm_optimizer,
  author  = {Bodner, Benjamin Jacob},
  title   = {Adaptive Multi Mode ({AMM}): budget-aware routing of optimizers by online landscape probes},
  year    = {2026},
  url     = {https://github.com/BjBodner/amm-optimizer},
  license = {Apache-2.0},
  note    = {Python package amm-optimizer}
}
```

To make a result reproducible, report the package version (`amm_optimizer.__version__`) and the
configuration hash of the configuration you ran (`config.config_hash()`; the default is
`presets.get("c46")`, hash `2153948ebed51150`).
