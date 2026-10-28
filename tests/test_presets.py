"""``atm.presets`` are the recorded composites exactly, and the harness reads them.

The tests marked ``research`` compare with the research harness's config texts
(``benchmarks/arms``, ``benchmarks/quick_scan.py``) and skip without them.
"""

import pytest

import amm_optimizer as atm
import characterisation_cases as cc
from amm_optimizer import presets

research = pytest.mark.skipif(
    cc.RESEARCH_DATA_MISSING is not None,
    reason=f"research data not in this checkout: {cc.RESEARCH_DATA_MISSING}",
)


@pytest.mark.research
@research
@pytest.mark.parametrize("name", presets.names())
def test_each_preset_is_its_recorded_composite(name):
    recorded = cc.parse(cc.composite(name))
    assert presets.get(name) == recorded
    assert presets.get(name).config_hash() == recorded.config_hash()


def test_the_presets_are_the_composites_in_order():
    assert presets.names() == tuple(f"c{k}" for k in range(23, 47))
    with pytest.raises(ValueError, match="unknown preset 'c99'"):
        presets.get("c99")


@pytest.mark.research
@research
@pytest.mark.parametrize("name", list(cc.NAMED))
def test_each_named_snapshot_is_the_config_text_it_was_recorded_with(name):
    recorded = cc.parse(cc.NAMED[name])
    assert cc.named_config(name) == recorded
    assert cc.named_config(name).config_hash() == recorded.config_hash()


@pytest.mark.research
@research
def test_a_preset_with_changes_is_the_arm_built_on_it():
    qn6 = presets.with_changes("c26", cma_local_nonsep="qn6")
    assert qn6 == cc.parse(cc.arm("audit_b42", "qn6"))
    assert presets.get("c26").variant.cma_local_nonsep == "qn3"  # unchanged


@pytest.mark.research
@research
def test_the_harness_reads_presets_with_overrides():
    assert cc.parse("preset:c26") == presets.get("c26")
    assert cc.parse("preset:c26,variant.cma_local_nonsep=qn6") == cc.parse(
        cc.arm("audit_b42", "qn6")
    )
    with pytest.raises(SystemExit, match="unknown preset"):
        cc.parse("preset:c99")
    with pytest.raises(SystemExit, match="first item"):
        cc.parse("variant.bounds=fold,preset:c26")


@pytest.mark.research
@research
def test_a_preset_run_is_the_recorded_run():
    pytest.importorskip("cocoex")  # the case runs a bbob problem
    case = cc.FAST[4]
    want = cc.run(cc.parse(cc.composite("c26")), case)
    assert cc.run(presets.get("c26"), case) == want


def test_the_package_exposes_the_presets():
    assert atm.presets is presets
