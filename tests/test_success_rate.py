"""``amplitude_control`` (E02): success-rate control of the random-search width."""

import warnings

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.config import N_SAMPLERS, Variant
from amm_optimizer.control import LOG_WIDTH_FLOOR, start_log_widths, update_log_widths
from amm_optimizer.sampler import SampleOptions, sample_options, two_mode_samples
from amm_optimizer.state import SamplerState
from test_sampler_options import draw, params
from variant_cases import SHORT, run_case

# research-only options, deprecated in 0.6.0 (they warn when set)
pytestmark = pytest.mark.filterwarnings("ignore:Variant:FutureWarning")

# research-only options (deprecated in 0.6.0): constructing them warns
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "Variant", FutureWarning)
    SR = Variant(amplitude_control="success_rate")
SR2 = Variant(amplitude_control="success_rate_targeted")


def samplers_at(log_widths):
    out = []
    for w in log_widths:
        s = SamplerState(2, np.random.RandomState(0))
        s.random_growth = w
        out.append(s)
    return out


def boundaries_of(sizes):
    return np.cumsum([0, *sizes])


def test_start_widths_are_the_logs_of_the_configured_widths():
    np.testing.assert_allclose(
        start_log_widths(SR), np.log([0.5, 1.0, 2.0, 4.0]), rtol=1e-12
    )


def test_all_failures_shrink_every_width_and_all_successes_grow_them():
    samplers = samplers_at([0.0] * N_SAMPLERS)
    bounds = boundaries_of([10] * N_SAMPLERS)
    update_log_widths(samplers, np.ones(40), bounds, SR)
    shrunk = [s.random_growth for s in samplers]
    np.testing.assert_allclose(shrunk, -1.5 * np.array(SR.sr_targets))
    update_log_widths(samplers, -np.ones(40), bounds, SR)
    grown = [s.random_growth for s in samplers]
    np.testing.assert_allclose(
        grown, shrunk + 1.5 * (1 - np.array(SR.sr_targets)), rtol=1e-12
    )


def test_a_width_is_stationary_at_its_target_rate():
    # 1 success in 20 samples is exactly the first sampler's target 0.05
    samplers = samplers_at([0.3] * N_SAMPLERS)
    probe = np.ones(80)
    probe[0] = -1.0
    update_log_widths(samplers, probe, boundaries_of([20] * N_SAMPLERS), SR)
    assert samplers[0].random_growth == pytest.approx(0.3)
    assert samplers[1].random_growth < 0.3  # target 0.10 > rate 0.05: shrink


def test_ties_and_the_incumbent_column_count_as_failures():
    samplers = samplers_at([0.0] * N_SAMPLERS)
    update_log_widths(samplers, np.zeros(8), boundaries_of([2] * N_SAMPLERS), SR)
    assert all(s.random_growth < 0 for s in samplers)


def test_the_width_is_capped_and_floored():
    samplers = samplers_at([np.log(5.0) - 0.01, LOG_WIDTH_FLOOR + 0.01, 0.0, 0.0])
    update_log_widths(samplers, -np.ones(8), boundaries_of([2] * N_SAMPLERS), SR)
    assert samplers[0].random_growth == pytest.approx(np.log(5.0))
    samplers = samplers_at([LOG_WIDTH_FLOOR + 0.01] * N_SAMPLERS)
    update_log_widths(samplers, np.ones(8), boundaries_of([2] * N_SAMPLERS), SR)
    assert samplers[0].random_growth == pytest.approx(LOG_WIDTH_FLOOR)


def test_the_update_is_linear_in_eta():
    one, two = samplers_at([0.0] * 4), samplers_at([0.0] * 4)
    probe = np.array([-1.0, 1.0] * 4)
    bounds = boundaries_of([2] * 4)
    update_log_widths(one, probe, bounds, Variant(amplitude_control="success_rate"))
    update_log_widths(
        two, probe, bounds, Variant(amplitude_control="success_rate", sr_eta=3.0)
    )
    for a, b in zip(one, two, strict=True):
        assert b.random_growth == pytest.approx(2 * a.random_growth)


def test_the_variants_map_to_sampler_options():
    assert sample_options(SR) == SampleOptions(scalar_width=True, no_targeted=True)
    assert sample_options(SR2) == SampleOptions(
        scalar_width=True, targeted_gate_any=True, targeted_loguniform=True
    )


def test_scalar_width_samples_use_the_width_and_ignore_the_random_vector():
    options = SampleOptions(scalar_width=True, no_targeted=True)
    width = 0.7
    (a, growth, _), ledger = draw(
        [1.0, -1.0, 2.0], options, random_growth=np.log(width)
    )
    assert growth == np.log(width)  # nothing but the controller changes it
    assert ("rand", (6, 1)) not in ledger  # no targeted draw
    # a different "adaptive" vector gives the same samples: it is not read
    rng = np.random.RandomState(3)
    rng2 = np.random.RandomState(3)
    args = (np.zeros(3), np.zeros(3), 6, np.array([1.0, -1.0, 2.0]))
    same = [
        two_mode_samples(
            *args, np.log(width), 1.0, params(), np.full((3, 6), k), r, options=options
        )[0]
        for k, r in ((0.1, rng), (0.4, rng2))
    ]
    np.testing.assert_array_equal(same[0], same[1])
    half_widths = np.abs(a[:, 1:])
    assert half_widths.max() <= width + 1e-12  # U(-1, 1) * width around the seed 0


def test_scalar_width_does_not_touch_the_growth_factors():
    options = SampleOptions(scalar_width=True, no_targeted=True)
    (_, growth, targeted), _ = draw(
        [5.0, 5.0, 5.0], options, random_growth=0.25, targeted_growth=3.0
    )
    assert (growth, targeted) == (0.25, 3.0)


def test_the_targeted_variant_steps_along_any_move_by_a_bounded_multiple():
    options = sample_options(SR2)
    change = np.array([-1.0, -2.0, 0.0])  # no positive coordinate
    (steps, _, _), ledger = draw(change, options, n=500, random_growth=np.log(1e-9))
    assert ("rand", (500, 1)) in ledger
    ratio = steps[:2, 1:] / change[:2, None]
    assert ratio.min() >= 0.1 * (1 - 1e-6) and ratio.max() <= 10 * (1 + 1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"targeted_gate": "any"},
        {"targeted_amplitude": "loguniform"},
        {"reset_on_success": False},
    ],
)
def test_the_original_samplers_flags_conflict_with_the_controller(kwargs):
    with pytest.raises(ValueError, match="replaces it"):
        Variant(amplitude_control="success_rate", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"amplitude_control": "pid"},
        {"sr_targets": (0.1, 0.2, 0.3)},
        {"sr_widths": (1.0,) * 5},
        {"sr_targets": (0.0, 0.1, 0.2, 0.3)},
        {"sr_targets": (0.1, 0.2, 0.3, 1.0)},
        {"sr_eta": 0.0},
        {"sr_max_width": -1.0},
        {"sr_widths": (0.5, 1.0, 2.0, 9.0)},
        {"sr_widths": (0.0, 1.0, 2.0, 4.0)},
    ],
)
def test_controller_parameters_are_validated(kwargs):
    with pytest.raises(ValueError, match="sr_|amplitude_control"):
        Variant(**kwargs)


def test_parameter_lists_from_json_become_tuples():
    variant = Variant.from_dict({"sr_targets": [0.1, 0.2, 0.3, 0.4]})
    assert variant.sr_targets == (0.1, 0.2, 0.3, 0.4)
    assert Variant.from_dict(variant.to_dict()) == variant


@pytest.mark.parametrize("scenario", SHORT, ids=lambda s: s.name)
def test_the_controller_parameters_do_nothing_for_the_original_sampler(scenario):
    tuned = atm.ATMConfig(
        variant=Variant(sr_eta=9.0, sr_targets=(0.5,) * 4, sr_widths=(1.0,) * 4)
    )
    assert run_case(tuned, scenario)[0] == run_case(atm.ATMConfig(), scenario)[0]


def test_the_controller_converges_on_a_sphere_where_the_original_does_not():
    def sphere(x):
        return float(np.sum((x - 2) ** 2))

    x0 = np.full(5, -3.0)
    for variant in (SR, SR2):
        config = atm.ATMConfig(variant=variant)
        results = [
            atm.api.minimize(sphere, x0, 3000, seed=s, config=config).fun
            for s in range(3)
        ]
        assert max(results) < 1e-8
    legacy = atm.api.minimize(sphere, x0, 3000, seed=0).fun
    assert legacy > 1e-8


def test_a_changed_target_changes_the_search():
    base = run_case(atm.ATMConfig(variant=SR), SHORT[0])[0]
    other = run_case(
        atm.ATMConfig(variant=Variant(amplitude_control="success_rate", sr_eta=0.5)),
        SHORT[0],
    )[0]
    assert base != other
