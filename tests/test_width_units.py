"""Variant ``width_units``: widths relative to the box."""

import dataclasses

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.bounds import Box
from amm_optimizer.config import ATMConfig
from amm_optimizer.plan import in_box_units
from preset_cases import wave5_variant


def with_changes(**changes):
    return ATMConfig(variant=dataclasses.replace(wave5_variant(), **changes))


def rastrigin(x):
    return float(10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def test_box_units_are_exact_on_the_bbob_box():
    runs = [
        atm.api.minimize(
            rastrigin,
            np.full(4, 2.0),
            4000,
            seed=5,
            config=with_changes(width_units=units),
            bounds=(-5, 5),
        )
        for units in ("absolute", "box")
    ]
    assert runs[0].fun == runs[1].fun
    assert runs[0].nfev == runs[1].nfev


def test_box_units_follow_the_box():
    config = in_box_units(with_changes(width_units="box"), Box.make((0.0, 1.0), 3))
    assert config.variant.sr_widths == pytest.approx((0.05, 0.1, 0.2, 0.4))
    assert config.variant.sr_max_width == pytest.approx(0.5)


def test_box_units_need_a_box():
    with pytest.raises(ValueError, match="bounds"):
        atm.api.minimize(
            rastrigin,
            np.zeros(2),
            100,
            seed=0,
            config=ATMConfig(
                variant=dataclasses.replace(
                    wave5_variant(),
                    es_phase="none",
                    width_units="box",
                )
            ),
        )
