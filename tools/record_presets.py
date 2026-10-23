"""Record the preset fingerprints (``tests/golden/presets.json``).

Usage (from the repo root):  python tools/record_presets.py [output.json]

Re-record only for a deliberate change of a preset, and say so in the commit.
Run it on an older commit (the module spells configurations field by field) to
check that a refactor left every preset's behaviour untouched.
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from preset_cases import GOLDEN, all_fingerprints  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN
    data = {
        "_meta": {
            "numpy": np.__version__,
            "python": platform.python_version(),
            "note": "cost-trace fingerprints of the presets; tools/record_presets.py",
        },
        "fingerprints": all_fingerprints(),
    }
    out.write_text(json.dumps(data, indent=1, sort_keys=True))
    print("wrote", out)


if __name__ == "__main__":
    main()
