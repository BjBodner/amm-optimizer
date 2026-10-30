"""``atm.presets`` are the recorded composites exactly."""

import pytest

import amm_optimizer as atm
import characterisation_cases as cc
from amm_optimizer import presets

COMPOSITES = [name for name in presets.names(research=True) if name[1:].isdigit()]


def test_the_presets_on_display_and_the_research_composites_in_order():
    assert presets.names() == ("default", "c46", "atm", "composite")
    assert presets.names(research=True) == (
        "default",
        "c46",
        "atm",
        "composite",
        *(f"c{k}" for k in range(23, 46)),
    )
    assert sorted(COMPOSITES) == [f"c{k}" for k in range(23, 47)]
    with pytest.raises(ValueError, match="unknown preset 'c99'"):
        presets.get("c99")


def test_every_research_preset_stays_loadable():
    for name in presets.names(research=True):
        assert isinstance(presets.get(name), atm.ATMConfig)
    assert presets.get("c23").variant.es_kind == "cma"


def test_the_aliases():
    assert presets.get("default") == presets.get("c46")
    assert presets.get("default").config_hash() == "2153948ebed51150"
    assert presets.get("atm") == atm.ATMConfig()
    assert presets.get("composite") == atm.ATMConfig.composite()
    assert presets.with_changes("default", anytime="replay") == presets.with_changes(
        "c46", anytime="replay"
    )
    assert presets.DEFAULT == atm.default.DEFAULT_PRESET == "c46"


@pytest.mark.parametrize("name", [None, 46, "C46", ""])
def test_get_refuses_what_is_not_a_preset_name(name):
    with pytest.raises(ValueError, match="unknown preset"):
        presets.get(name)


@pytest.mark.parametrize("name", list(cc.NAMED))
def test_each_named_snapshot_is_the_config_text_it_was_recorded_with(name):
    recorded = cc.parse(cc.NAMED[name])
    assert cc.named_config(name) == recorded
    assert cc.named_config(name).config_hash() == recorded.config_hash()


def test_the_package_exposes_the_presets():
    assert atm.presets is presets
