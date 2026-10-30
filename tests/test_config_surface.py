"""The configuration surface: defaults are the original's, and round trips."""

import dataclasses
import json
from pathlib import Path

import pytest

from amm_optimizer.conductor import ConductorConfig
from amm_optimizer.config import N_SAMPLERS, ATMConfig, Variant
from amm_optimizer.plan import iteration_limit, resolve_config
from variant_cases import SHORT, run_case

DEFAULTS = Path(__file__).parent / "config_defaults.json"

# The original's behaviour for every field of ``Variant``. Adding a field
# without declaring its legacy value here fails ``test_every_field_is_declared``.
LEGACY_VALUES = {
    "reevaluate_incumbent": True,
    "initial_design": "legacy",
    "bounds": "none",
    "corner_clip": 0.0,
    "amplitude_control": "legacy",
    "covariance": "none",
    "cov_rate": 1.0,
    "cov_select": 0.25,
    "es_phase": "none",
    "es_fraction": 0.3,
    "es_max_budget_per_dim": 1e9,
    "es_kind": "isotropic",
    "es_lq_min_budget_per_dim": 0.0,
    "es_lq_nonsep": False,
    "prefix_sep_probe": False,
    "prefix_plateau_pop": False,
    "prefix_plateau_min_dim": 0,
    "chain_route": False,
    "sep_route": False,
    "basin_route": False,
    "basin_descents": 0,
    "basin_keep_atm": False,
    "small_engine": "none",
    "small_engine_max_dim": 2,
    "rugged_try_rest": "none",
    "prefix_ridge_after": 0,
    "prefix_ridge_max_per_dim": 0.0,
    "sep_stall_min_per_dim": 0.0,
    "prefix_full_per_dim": 0.0,
    "prefix_full_min_per_dim": 0.0,
    "des_max_per_dim": 30.0,
    "small_switch": False,
    "small_switch_trials": 0.0,
    "small_switch_keep_unit": False,
    "prefix_nan_reading": "rugged",
    "box_normalize": False,
    "prefix_descent_retries": 0,
    "rugged_direct_beta": 0.0,
    "rugged_direct_min_beta": 0.0,
    "prefix_low_beta": 0.0,
    "prefix_low_beta_min_dim": 0,
    "cma_handoff": 0.0,
    "cma_model": "none",
    "cma_restart_rule": "legacy",
    "cma_active": False,
    "cma_warm_start": False,
    "cma_handoff_cap_per_dim": 0.0,
    "cma_local": "one_plus_one",
    "cma_local_min_dim": 10,
    "cma_model_high": "none",
    "pop_active": False,
    "pop_lambda": "default",
    "pop_gate_per_dim": 0.0,
    "pop_protect": False,
    "cma_local_nonsep": "none",
    "pop_switch_per_dim": 0.0,
    "pop_switch_start": "uniform",
    "pop_switch_min_dim": 1,
    "cma_probe_at_per_dim": 0.0,
    "cma_probe_beta": 0.0,
    "cma_probe_min_dim": 1,
    "nm_max_dim": 0,
    "nm_stage": "both",
    "sep_probe": False,
    "sep_check": "axis",
    "qn_fallback": "none",
    "pool_rule": "fixed",
    "pool_min": 24,
    "anytime": "off",
    "anytime_start_per_dim": 10.0,
    "anytime_factor": 2.0,
    "anytime_switch": 10_000,
}


def test_every_field_is_declared():
    assert set(LEGACY_VALUES) == {f.name for f in dataclasses.fields(Variant)}


def test_defaults_are_the_legacy_values():
    assert Variant() == Variant(**LEGACY_VALUES)
    assert dataclasses.asdict(Variant()) == LEGACY_VALUES


def test_default_config_snapshot():
    """A default can only change deliberately, together with this file."""
    want = json.loads(DEFAULTS.read_text())
    assert json.loads(json.dumps(ATMConfig().to_dict())) == want


@pytest.mark.parametrize("scenario", SHORT, ids=lambda s: s.name)
def test_spelled_out_defaults_run_the_same_trace(scenario):
    explicit = ATMConfig(variant=Variant(**LEGACY_VALUES))
    assert run_case(explicit, scenario)[0] == run_case(ATMConfig(), scenario)[0]
    legacy = ATMConfig(strict_budget=False)
    spelled = ATMConfig(strict_budget=False, variant=Variant(**LEGACY_VALUES))
    assert run_case(spelled, scenario)[0] == run_case(legacy, scenario)[0]


def test_round_trip_through_a_dict_and_json():
    config = ATMConfig(
        initial_population=24,
        variant=Variant(reevaluate_incumbent=False),
        conductor=ConductorConfig(total_resources=24),
    )
    data = json.loads(json.dumps(config.to_dict()))
    assert ATMConfig.from_dict(data) == config
    assert Variant.from_dict(config.variant.to_dict()) == config.variant
    assert ATMConfig.from_dict({}) == ATMConfig()


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        ({"initial_populaton": 3}, "ATMConfig"),
        ({"conductor": {"mass_": 1}}, "ConductorConfig"),
        ({"restart": {"windows": 1}}, "RestartPolicy"),
        ({"variant": {"reevaluate": False}}, "Variant"),
    ],
)
def test_unknown_keys_are_rejected_with_the_valid_ones(data, kind):
    with pytest.raises(ValueError, match=kind):
        ATMConfig.from_dict(data)


def test_hash_is_stable_and_changes_with_any_field():
    base = ATMConfig()
    assert base.config_hash() == ATMConfig().config_hash()
    assert len(base.config_hash()) == 16
    changed = [
        ATMConfig(initial_population=48),
        ATMConfig(variant=Variant(reevaluate_incumbent=False)),
        ATMConfig(strict_budget=False),
    ]
    hashes = {c.config_hash() for c in changed} | {base.config_hash()}
    assert len(hashes) == 4


def test_a_variant_needs_the_strict_budget():
    with pytest.raises(ValueError, match="strict_budget"):
        ATMConfig(strict_budget=False, variant=Variant(reevaluate_incumbent=False))
    ATMConfig(strict_budget=False, variant=Variant())  # the default is fine


def test_skipping_the_incumbent_needs_a_pool_with_other_columns():
    variant = Variant(reevaluate_incumbent=False)
    with pytest.raises(ValueError, match="samples per iteration"):
        ATMConfig(variant=variant, initial_population=N_SAMPLERS)
    with pytest.raises(ValueError, match="samples per iteration"):
        ATMConfig(
            variant=variant,
            conductor=ConductorConfig(total_resources=4),
        )
    ATMConfig(variant=variant, initial_population=2 * N_SAMPLERS)


@pytest.mark.parametrize("field", ["reevaluate_incumbent"])
def test_flags_must_be_real_bools(field):
    with pytest.raises(ValueError, match=field):
        Variant(**{field: 0})


@pytest.mark.parametrize(("budget", "dim"), [(1, 1), (20, 2), (500, 5), (10**5, 40)])
def test_resolve_config_returns_the_default_itself(budget, dim):
    config = ATMConfig()
    assert resolve_config(config, budget, dim) is config


@pytest.mark.parametrize("budget", [1, 59, 60, 61, 500, 10_000])
def test_the_iteration_limit_is_the_originals_formula_for_the_default(budget):
    import numpy as np

    want = int(np.maximum(np.ceil(budget / 60) - 2, 1))
    assert iteration_limit(ATMConfig(), budget) == want


def test_a_variant_has_no_iteration_limit():
    config = ATMConfig(variant=Variant(reevaluate_incumbent=False))
    assert iteration_limit(config, 10_000) is None


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("initial_design", "sobol"),
        ("bounds", "clip"),
        ("corner_clip", 1.5),
        ("corner_clip", -0.1),
        ("pool_rule", "adaptive"),
    ],
)
def test_choices_are_validated(field, bad):
    with pytest.raises(ValueError, match=field):
        Variant(**{field: bad})


def test_corner_clip_needs_the_fold():
    with pytest.raises(ValueError, match="corner_clip"):
        Variant(corner_clip=0.25)
    Variant(bounds="fold", corner_clip=0.25)


def test_which_variants_need_the_box():
    assert not Variant().needs_box
    assert not Variant(reevaluate_incumbent=False).needs_box
    assert Variant(bounds="fold").needs_box
    assert Variant(initial_design="uniform").needs_box


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pool_min": 4},
        {"pool_min": 70},
    ],
)
def test_pool_parameters_are_validated(kwargs):
    with pytest.raises(ValueError, match="pool_"):
        Variant(**kwargs)
