"""Record the self-snapshots of the variants (``tests/golden_variants.json``)
and the default configuration (``tests/config_defaults.json``).

Usage (from the repo root):  python tools/record_variants.py

Unlike ``record_golden.py`` this records this package's own behaviour: variants
have no counterpart in the original. Re-record only for a deliberate change of
a variant, and explain it in the commit message.
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import amm_optimizer as atm  # noqa: E402
from variant_cases import SHORT, SNAPSHOTS, VARIANTS, snapshot  # noqa: E402


def main() -> None:
    out = {
        "_meta": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "note": "self-snapshots of variants; see tools/record_variants.py",
        },
        "variants": {
            name: {s.name: snapshot(atm.ATMConfig(variant=variant), s) for s in SHORT}
            for name, variant in VARIANTS.items()
        },
    }
    SNAPSHOTS.write_text(json.dumps(out, indent=1))
    defaults = ROOT / "tests" / "config_defaults.json"
    defaults.write_text(json.dumps(atm.ATMConfig().to_dict(), indent=1, sort_keys=True))
    for name, scenarios in out["variants"].items():
        for scenario, rec in scenarios.items():
            print(f"{name:20s} {scenario:20s} evals={rec['n_evals']:5d}")
    print("wrote", SNAPSHOTS)


if __name__ == "__main__":
    main()
