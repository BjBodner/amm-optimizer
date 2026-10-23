"""Every Python block in the README must run as written, top to bottom."""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


def test_readme_examples_run_in_order_and_print_a_result(capsys):
    assert len(BLOCKS) >= 2
    namespace = {"__name__": "readme"}  # shared: later blocks reuse earlier names
    for number, block in enumerate(BLOCKS, start=1):
        exec(compile(block, f"README.md block {number}", "exec"), namespace)  # noqa: S102
        assert capsys.readouterr().out.strip(), f"block {number} printed nothing"
