"""Every preset still takes exactly the steps it took when it was recorded."""

import pytest

from preset_cases import FUNCTIONS, GRID, PRESETS, fingerprint, load

CASES = [
    (name, function, dim, m)
    for name in PRESETS
    for function in FUNCTIONS
    for dim, multipliers in GRID
    for m in multipliers
]


@pytest.mark.parametrize(
    ("name", "function", "dim", "m"),
    CASES,
    ids=[f"{n}-{f}-d{d}-x{m}" for n, f, d, m in CASES],
)
@pytest.mark.snapshot
def test_preset_trace_is_unchanged(name, function, dim, m):
    assert (
        fingerprint(name, function, dim, m) == load()[name][f"{function}-d{dim}-x{m}"]
    )


def test_the_named_presets_are_the_recorded_ones():
    import amm_optimizer as atm
    from preset_cases import config_of

    assert atm.ATMConfig.composite() == config_of("composite")
    assert atm.ATMConfig.improved() == config_of("v2")
    assert atm.ATMConfig.improved_v2() == atm.ATMConfig.improved()
    assert atm.ATMConfig.improved_wave6() == config_of("wave6")
    assert atm.ATMConfig() == config_of("default")
