"""The Python examples in docs/ run as written, top to bottom, as the README's do."""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = ["legacy.md"]


@pytest.mark.parametrize("page", PAGES)
def test_the_examples_of_a_page_run_in_order_and_print_a_result(page, capsys):
    text = (DOCS / page).read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks, f"{page} has no Python example"
    namespace = {"__name__": "docs"}  # shared: later blocks reuse earlier names
    for number, block in enumerate(blocks, start=1):
        exec(compile(block, f"docs/{page} block {number}", "exec"), namespace)  # noqa: S102
        assert capsys.readouterr().out.strip(), f"{page} block {number} printed nothing"
