# Presets

A preset is a named, frozen configuration: an [`ATMConfig`](api.md#amm_optimizer.ATMConfig)
that `minimize` takes as `config=`. The module
[`amm_optimizer.presets`](api.md#amm_optimizer.presets) holds them.

```python
import numpy as np
import amm_optimizer as amm

def f(x):
    return float(np.sum((x - 1.0) ** 2))

print(amm.presets.names())
config = amm.presets.get("c46")
result = amm.minimize(f, bounds=[(-5, 5)] * 3, budget=300, seed=0, config=config)
print(result.fun)
```

## The default: `c46`

A call without `config=` runs `c46`, the configuration frozen for the paper on 2026-10-22.
Passing `config=amm.presets.get("c46")` gives a bit-identical run. Its configuration hash
identifies it in recorded results:

```python
print(amm.presets.get("c46").config_hash())   # 2153948ebed51150
```

[The frozen configuration](FREEZE_2026-10-22.md) records what was frozen and in which
environment the bit-exact snapshots were taken.

## The other presets

`c23` to `c45` are the earlier composites of the research cycles that led to `c46`, oldest first.
Each one changes the one before it; the docstring of
[`amm_optimizer.presets`](api.md#amm_optimizer.presets) lists what each step changed and the
measurement that confirmed it. They are kept so that recorded results can be reproduced. For new
work, use the default.

Every preset needs a search box (`bounds=`) and a deterministic objective.

## Changing a preset

`presets.with_changes(name, **changes)` returns preset `name` with some of its variant options
changed. The anytime mode is one of them: `anytime=True` in `minimize` is the same as

```python
config = amm.presets.with_changes("c46", anytime="replay")
print(config.variant.anytime)
```

A changed preset is a different configuration, with a different `config_hash()`, and the results
recorded for the preset do not apply to it.

## The original sampler and the older presets

`ATMConfig()` runs the original Adaptive Two Mode (ATM) sampler alone, and
`ATMConfig.composite()` adds five improvements to it. Neither is a member of `presets.names()`;
both are described in [the original ATM sampler](legacy.md).
