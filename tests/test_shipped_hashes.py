"""The ``config_hash`` of every shipped configuration, recorded.

The shipped configurations are the presets ``c23``-``c46``, ``ATMConfig()``,
``ATMConfig.composite()`` and the ``anytime=True`` form of ``c46``. Their hashes tie
results to configurations, so no change to ``config.py`` (a new field, a deprecation,
a removal in 0.7.0) may move one of them.
"""

import dataclasses

import pytest

from amm_optimizer import presets
from amm_optimizer.config import ATMConfig

RECORDED = {
    "c23": "6c2f8eb648e973cb",
    "c24": "2e68223cd450be1a",
    "c25": "c1bf22dba741f33f",
    "c26": "6ce636d02b13326d",
    "c27": "c0d66132c38d8739",
    "c28": "03f429c8da0e6fa9",
    "c29": "27f3ffdae2d7798a",
    "c30": "03d49de84b13c57c",
    "c31": "24749d11670c4998",
    "c32": "8d0fe23a0e3a68a9",
    "c33": "c064b293d962cc1f",
    "c34": "491ad57dcf16f12b",
    "c35": "4d6b0e2a46fbb50a",
    "c36": "3ed0449967a861f5",
    "c37": "70cf85556a44e0dd",
    "c38": "26918a66f574b07a",
    "c39": "ae9e37fd20900a22",
    "c40": "a0e460176cca0f1a",
    "c41": "02d0cb22eb4b7f35",
    "c42": "9bc94fdded1b87ac",
    "c43": "5970e726ac4bd33c",
    "c44": "e953da1bb0fee3fe",
    "c45": "fac9eef89213261a",
    "c46": "2153948ebed51150",
    "ATMConfig()": "f8fc4662f473b680",
    "composite": "7f02e0c2f5a96883",
    "c46 anytime": "2fffb7471bd6cdb2",
}


def shipped(name: str) -> ATMConfig:
    if name == "ATMConfig()":
        return ATMConfig()
    if name == "composite":
        return ATMConfig.composite()
    if name == "c46 anytime":
        config = presets.get("c46")
        return dataclasses.replace(
            config, variant=dataclasses.replace(config.variant, anytime="replay")
        )
    return presets.get(name)


@pytest.mark.parametrize("name", list(RECORDED))
def test_shipped_config_hash_is_unchanged(name):
    assert shipped(name).config_hash() == RECORDED[name]


@pytest.mark.parametrize("name", list(RECORDED))
def test_shipped_config_survives_a_dict_round_trip(name):
    config = shipped(name)
    again = ATMConfig.from_dict(config.to_dict())
    assert again == config and again.config_hash() == RECORDED[name]


def test_every_listed_preset_is_recorded():
    # aliases such as a "default" name resolve to a recorded configuration
    for name in presets.names():
        assert presets.get(name).config_hash() in RECORDED.values(), name
