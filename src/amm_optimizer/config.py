"""User-facing configuration: :class:`ATMConfig` and its :class:`Variant`.

The presets (``amm_optimizer.presets``) are ``ATMConfig`` values. The research-only
``Variant`` options, deprecated in 0.6.0, were removed in 0.7.0: :data:`REMOVED_FIELDS`
and :data:`REMOVED_CHOICES` list them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from .conductor import ConductorConfig
from .restart import RestartPolicy

# The algorithm is built around four samplers; the conductor and the
# hyperparameter search are shaped accordingly.
N_SAMPLERS = 4
# ``pool_rule="budget"``: the pool is ``budget / POOL_DIVISOR``, at most POOL_MAX.
POOL_MAX = 64
POOL_DIVISOR = 30


def _reject_unknown(kind: type, data: dict[str, Any]) -> None:
    valid = {f.name for f in fields(kind)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(
            f"unknown {kind.__name__} field(s) {unknown}; valid: {sorted(valid)}"
        )


def _choose(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value in REMOVED_CHOICES.get(name, ()):
        raise ValueError(
            f"{name}={value!r} was removed in {REMOVED_IN} (it was a research "
            f"option); amm-optimizer 0.6.x still has it. Allowed: {allowed}"
        )
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


# ``Variant`` fields added after the freeze of 2026-10-22: ``ATMConfig.config_hash``
# reads them only when set, so the configurations recorded before keep their hashes.
HASH_IF_SET = ("anytime", "anytime_start_per_dim", "anytime_factor", "anytime_switch")

# The release that removed the fields of :data:`REMOVED_FIELDS`.
REMOVED_IN = "0.7.0"
# ``Variant`` fields removed in :data:`REMOVED_IN`, each with the default it had in
# 0.5.0 and 0.6.0. ``ATMConfig.config_hash`` hashes every one of them at that default,
# so a hash recorded before the removal stays valid; ``from_dict`` accepts one only at
# that default (a dictionary written by 0.6.0 still loads).
REMOVED_FIELDS: dict[str, Any] = {
    # the in-package Nelder-Mead simplex and the late refinement
    "local_method": "es",
    "simplex_max_dim": 4,
    "nm_refine_max_dim": 10,
    "refine_fraction": 0.0,
    "refine_min_budget_per_dim": 0.0,
    "refine_max_dim": 1_000_000,
    # BIPOP-style restart regimes of the hand-over
    "cma_regimes": "none",
    # the lq-CMA-ES model steps and the MLSL restart placement
    "cma_lq_max_dim": 20,
    "cma_model_sigma": False,
    "cma_placement": "uniform",
    "cma_placement_max_dim": 5,
    # face-tangent curvature probes and the sampled restarts' central-difference polish
    "probe_faces": "clip",
    "sampled_polish_per_dim": 0.0,
    # the hand-over's ATM restarts between CMA runs
    "cma_restart": "uniform",
    # research switches without a behavioural test: forced readings, forked streams,
    # lean starts, parabolic STEP steps, the low hand-over cap, try overrides, ...
    "es_small_dim": 0,
    "es_small_max_budget_per_dim": 30.0,
    "local_force": "none",
    "fork_seed": -1,
    "prefix_plateau_rugged": False,
    "rugged_try_cap_per_dim": 0.0,
    "rugged_try_max_per_dim": 0.0,
    "prefix_nan_min_per_dim": 0.0,
    "prefix_plateau_ties": 0,
    "smooth_bfgs_second": 0.0,
    "cma_handoff_cap_low_per_dim": 0.0,
    "cma_handoff_cap_low_max_per_dim": 300.0,
    "es_lean_start": False,
    "sep_parabolic": False,
    "sep_parabolic_late": False,
    "sep_parabolic_scope": "both",
    # ATM sampler and controller variants: the targeted step's gate and length, the
    # sawtooth reset, learned per-coordinate scales, covariance noise steps and path,
    # the memo, box width units, crossover and noise forms, shared widths and the
    # stagnation restarts
    "targeted_gate": "legacy",
    "targeted_amplitude": "legacy",
    "reset_on_success": True,
    "scale_learning": "none",
    "scale_rate": 0.15,
    "scale_cap": 100.0,
    "cov_steps": "displacement",
    "cov_path": "none",
    "eval_memo": False,
    "width_units": "absolute",
    "crossover": "legacy",
    "noise": "uniform",
    "skip_starved": False,
    "width_mode": "per_sampler",
    "sr_shared_target": 0.15,
    "restart_mode": "legacy",
    "stag_tolfun": 1e-11,
    "stag_tolx": 1e-12,
    "ipop_factor": 2.0,
    "ipop_max_pool": 1024,
    # the small scattered research switches: SLSQP's false-success and floor rules,
    # the stage's extended restart, the NaN reading's clip test and reprobe, the
    # prefix's fractal route, the centre point, the low separable route, box-scaled
    # steps and STEP's box check
    "qn_false_success": False,
    "qn_floor": "none",
    "step_box_check": False,
    "box_steps": False,
    "prefix_nan_clip": False,
    "prefix_nan_reprobe": False,
    "prefix_fractal_min_dim": 0,
    "prefix_fractal_min_per_dim": 0.0,
    "stage_extend_restart": False,
    "stage_extend_max_per_dim": 0.0,
    "sep_route_low": "off",
    "sep_route_low_dim": 2,
    "centre_point": "off",
    "centre_point_min_dim": 2,
    # the constant knobs: no shipped configuration changed them, and they are module
    # constants now (``control.SR_*``, ``COV_CAP``, ``search.COV_MAX_DIM``,
    # ``POOL_MAX``, ``POOL_DIVISOR`` and the routing thresholds of ``api``)
    "sr_eta": 1.5,
    "sr_targets": (0.05, 0.10, 0.20, 0.30),
    "sr_widths": (0.5, 1.0, 2.0, 4.0),
    "sr_max_width": 5.0,
    "es_init_factor": 1.0,
    "cov_cap": 1e6,
    "cov_max_dim": 10,
    "chain_min_per_dim": 300.0,
    "sep_route_min_per_dim": 100.0,
    "additive_min_dim": 5,
    "basin_max_dim": 3,
    "basin_min_per_dim": 100.0,
    "basin_descents_max_per_dim": 300.0,
    "small_engine_max_per_dim": 10.0,
    "small_engine_radius": 0.25,
    "prefix_ridge_radius": 0.1,
    "small_switch_max_dim": 20,
    "small_switch_max_per_dim": 10.0,
    "small_switch_cos": 0.99,
    "small_switch_radius": 0.25,
    "small_switch_rest": True,
    "cma_model_max_dim": 10,
    "cma_handoff_cap_min_dim": 10,
    "cma_local_nonsep_min_dim": 5,
    "sep_share": 0.5,
    "sep_min_budget_per_dim": 30.0,
    "pool_max": 64,
    "pool_divisor": 30,
}
# String values removed in :data:`REMOVED_IN` from fields that remain.
REMOVED_CHOICES: dict[str, tuple[str, ...]] = {
    # the other values no preset used
    "amplitude_control": ("success_rate",),
    "nm_stage": ("qnprefix", "qnprefix30g"),
    "cma_restart_rule": ("economy",),
    "sep_check": ("pair",),
    "pop_lambda": ("cap4", "budget", "steps", "ipop4", "stepsgrow"),
    "pop_switch_start": ("carry",),
    "rugged_try_rest": ("cut",),
    "prefix_nan_reading": ("smooth",),
    # the plain population finishers of the hand-over
    "cma_local": ("population", "ipop"),
    "cma_local_nonsep": (
        "ipop",
        "population",
        # the SLSQP routes of the local phase that no preset took
        "qn",
        "qn2",
        "qn5",
        "qn6",
        "qn10",
        "qn11",
        "qn13",
        "qn13n",
        "qn15",
    ),
    # the start-point (kappa) probe prefix and the ungated-step prefix
    "es_kind": ("slsqplq10", "slsqplq10hk"),
    # the lq-CMA-ES model steps of the CMA phase
    "cma_model": ("lq",),
}


def _same_value(value: object, default: object) -> bool:
    """Whether ``value`` is ``default`` (a list matches a tuple, as JSON gives it)."""
    if isinstance(default, tuple) and isinstance(value, list):
        value = tuple(value)
    if isinstance(default, bool) or isinstance(value, bool):
        return value is default
    return bool(
        type(value) is type(default)
        or {type(value), type(default)}
        <= {
            int,
            float,
        }
    ) and bool(value == default)


def _drop_removed(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` (``Variant`` fields) without the removed fields at their old defaults.

    Raises
    ------
    ValueError
        If ``data`` sets a removed field to anything but its old default.
    """
    removed = [name for name in data if name in REMOVED_FIELDS]
    for name in removed:
        if not _same_value(data[name], REMOVED_FIELDS[name]):
            raise ValueError(
                f"Variant.{name} was removed in {REMOVED_IN} (it was a research "
                f"option); only its old default {REMOVED_FIELDS[name]!r} is accepted, "
                f"got {data[name]!r}. amm-optimizer 0.6.x still has it."
            )
    return {name: value for name, value in data.items() if name not in removed}


def _hash_payload(data: dict[str, Any]) -> str:
    """The text that ``ATMConfig.config_hash`` digests, from a ``to_dict`` result.

    Every removed field is put back at its old default and the late fields of
    :data:`HASH_IF_SET` are left out at their defaults, so the text is the one the
    release that recorded a hash wrote.
    """
    variant = dict(data["variant"])
    for name in HASH_IF_SET:
        if name in variant and variant[name] == _FIELD_DEFAULTS[name]:
            del variant[name]
    for name, default in REMOVED_FIELDS.items():
        variant[name] = list(default) if isinstance(default, tuple) else default
    return json.dumps({**data, "variant": variant}, sort_keys=True)


# Validation of ``Variant`` is table driven. Each row is checked in order.
_BOOLS = ("reevaluate_incumbent",)
_CHOICES = {
    "initial_design": ("legacy", "uniform"),
    "bounds": ("none", "fold"),
    "pool_rule": ("fixed", "budget"),
    "amplitude_control": ("legacy", "success_rate_targeted"),
    "es_phase": ("none", "prefix"),
    "covariance": ("none", "rank_mu"),
}
# The other string-valued fields, checked later in ``__post_init__``.
_LATE_CHOICES: dict[str, tuple[str, ...]] = {
    "es_kind": (
        "isotropic",
        "cma",
        "lq2",
        "slsqplq",
        "slsqplq10c",
        "slsqplq10h",
        "slsqplq10hd",
    ),
    "cma_local_nonsep": (
        "none",
        "qn3",
        "qn7",
        "qn8",
        "qn9",
        "qn12",
        "qn17",
        "qn19",
        "qn20",
        "qn29",
        "qn38",
        "qn44",
    ),
    "nm_stage": ("both", "prefix", "qnprefix30"),
    "cma_model": ("none", "quadratic"),
    "cma_restart_rule": ("legacy", "short"),
    "cma_local": ("one_plus_one", "then_ipop"),
    "cma_model_high": ("none", "diagonal"),
    "sep_check": ("axis", "stall"),
    "qn_fallback": ("none", "lbfgsb"),
    "pop_lambda": ("default", "fixed2"),
    "pop_switch_start": ("uniform", "incumbent"),
    "small_engine": ("none", "cobyqa"),
    "rugged_try_rest": ("none", "cutkink"),
    "prefix_nan_reading": ("rugged", "plateau"),
    "anytime": ("off", "replay"),
}
# Allowed values of every string-valued ``Variant`` field.
CHOICES: dict[str, tuple[str, ...]] = {**_CHOICES, **_LATE_CHOICES}
# (field, is the value valid?, message; ``{v}`` is the value)
_RANGES: tuple[tuple[str, Callable[[Any], bool], str], ...] = (
    ("cov_rate", lambda v: v > 0, "cov_rate must be positive, got {v}"),
    ("cov_select", lambda v: 0 < v <= 1, "cov_select must be in (0, 1], got {v}"),
    ("es_fraction", lambda v: 0 < v < 1, "es_fraction must be in (0, 1), got {v}"),
    (
        "es_max_budget_per_dim",
        lambda v: v > 0,
        "es_max_budget_per_dim must be positive",
    ),
    ("corner_clip", lambda v: 0 <= v <= 1, "corner_clip must be in [0, 1], got {v}"),
)
# Options that only make sense with a success-rate amplitude control.
_NEEDS_SUCCESS_RATE: dict[str, Callable[[Any], bool]] = {
    "es_phase": lambda v: v != "none",
}


@dataclass(frozen=True)
class Variant:
    """Switchable departures from the original ATM algorithm.

    Every default is the original's behaviour, so ``Variant()`` reproduces it bit
    for bit, and the presets (``presets.get``) are ``Variant`` values with some
    fields changed. A variant other than the default needs
    ``ATMConfig.strict_budget`` (the original's accounting is only reproduced, not
    extended). ``docs/DEVIATIONS.md`` describes the departures from the original.

    The attributes below are the fields that a shipped configuration (a preset,
    ``ATMConfig.composite()`` or the anytime mode) sets; "the presets" are the
    values they use, and the default preset is ``c46``. Distances "x dim" are
    evaluations per dimension: ``100 x dim`` in 5 dimensions is 500 evaluations.

    The research options that no shipped configuration used were removed in 0.7.0
    (:data:`REMOVED_FIELDS`, :data:`REMOVED_CHOICES`); ``from_dict`` still accepts a
    removed field at its old default.

    Attributes
    ----------
    reevaluate_incumbent : bool, default True
        Whether every iteration evaluates the incumbent again (each sampler's first
        sample repeats it). ``False`` reuses its known cost without charging an
        evaluation, so the budget lasts longer; the objective must be
        deterministic. The presets use ``False``.
    initial_design : {"legacy", "uniform"}, default "legacy"
        The first iteration of a search. ``"legacy"`` samples a small ball (radius
        about 0.03) around the start point; ``"uniform"`` draws the other samples
        uniformly from the box. ``"uniform"`` needs a box.
    bounds : {"none", "fold"}, default "none"
        Samples outside the box: ``"none"`` evaluates them as they are; ``"fold"``
        reflects them at the walls first and stores the repaired point. ``"fold"``
        needs a box.
    corner_clip : float, default 0.0
        With ``bounds="fold"``: the probability that a sample outside the box is
        clipped onto it instead of reflected, which keeps optima in a corner
        reachable. The presets use 0.25.
    amplitude_control : {"legacy", "success_rate_targeted"}, default "legacy"
        The width of ATM's random search. ``"legacy"``: the original's open-loop
        sawtooth, driven by the sampler hyperparameters. ``"success_rate_targeted"``:
        one width per sampler, adapted to its success rate, plus a targeted step of
        a log-uniform multiple (0.1 to 10) of the last move.
    es_phase : {"none", "prefix"}, default "none"
        ``"prefix"``: a local search (``es_kind``) takes the first ``es_fraction``
        of the budget, after a uniform design that picks its start, and ATM
        continues from its best point. Needs a box and
        ``amplitude_control="success_rate_targeted"``.
    es_fraction : float, default 0.3
        The prefix's share of the budget, in (0, 1). The presets use 0.9.
    es_max_budget_per_dim : float, default 1e9
        The prefix runs for budgets up to this many x dim; larger budgets use the
        hand-over (``cma_handoff``) instead. The presets use 100.
    covariance : {"none", "rank_mu"}, default "none"
        ``"rank_mu"``: ATM's random search is Gaussian with a full covariance
        matrix, learned CMA-style from the best ``cov_select`` share of each
        iteration's samples (in up to 10 dimensions). Needs a success-rate
        ``amplitude_control``.
    cov_rate : float, default 1.0
        Multiplier of the covariance's rank-mu learning rate. The presets use 3.
    cov_select : float, default 0.25
        The share of each iteration's samples the covariance learns from, in
        (0, 1]. The presets use 0.5.
    es_kind : str, default "isotropic"
        The local search of the prefix. ``"isotropic"``: a (1+1)-ES with the
        1/5 rule. ``"cma"``: the restarted (1+1)-Cholesky-CMA-ES. ``"lq2"``: IPOP
        population CMA-ES ranked by a linear-quadratic surrogate (``"cma"`` below
        ``es_lq_min_budget_per_dim``). ``"slsqplq"``: a curvature probe decides:
        restarted SLSQP where the function reads as smooth, the surrogate
        populations where it reads as rugged (the (1+1)-ES below
        ``es_lq_min_budget_per_dim``), and the (1+1)-ES below 30 x dim.
        ``"slsqplq10c"``: the same, also below 30 x dim once the budget reaches
        200 evaluations, with a cheaper probe there. ``"slsqplq10h"``:
        ``"slsqplq10c"`` with SLSQP's finite-difference step at 1e-9.
        ``"slsqplq10hd"`` (``c46``): ``"slsqplq10h"``, and where it skips the
        probe, one SLSQP descent from the design's best point (below
        ``des_max_per_dim``), then the (1+1)-ES.
    es_lq_min_budget_per_dim : float, default 0.0
        The surrogate populations run from this many x dim. The presets use 100.
    es_lq_nonsep : bool, default False
        The surrogate populations only where the separable search (``sep_probe``)
        finds the function non-separable.
    prefix_sep_probe : bool, default False
        With ``es_lq_nonsep``: a function found separable no longer forces the
        (1+1)-ES on the ``"slsqplq"`` prefixes; the curvature probe decides.
    prefix_plateau_pop : bool, default False
        Above 30 x dim the prefix checks for a plateau after its design and its
        separable search; on a plateau, restarted CMA-ES populations take the rest
        of the prefix.
    prefix_plateau_min_dim : int, default 0
        No plateau check below this dimension. The presets use 5.
    chain_route : bool, default False
        From 5 dimensions, with a box, a pair test at ``x0`` (3 x dim evaluations)
        reads the function as separable, as a chain (only neighbouring coordinates
        interact) or as neither. A chain that also passes a band check gets a
        decoupled coordinate search for the whole budget, from 300 x dim.
    sep_route : bool, default False
        With the same pair test: a separable reading gets the restarted separable
        search (STEP) for the whole budget, from 100 x dim.
    basin_route : bool, default False
        In 2 and 3 dimensions, from 100 x dim: a Sobol design and a curvature probe
        at its best point. A smooth reading with several basins gets MLSL for the
        budget; any other reading continues from the design's best point.
    basin_descents : int, default 0
        On a smooth reading with one basin, at budgets up to 300 x dim: up to this
        many SLSQP descents before the usual path. Their best point is the result
        unless the rest of the run beats it. The presets use 1.
    basin_keep_atm : bool, default False
        ATM's hand-over share is counted after those descents, so that they do not
        shorten ATM's sample.
    small_engine : {"none", "cobyqa"}, default "none"
        ``"cobyqa"``: budgets up to 10 x dim in up to ``small_engine_max_dim``
        dimensions run SciPy's COBYQA with restarts instead of every other route.
    small_engine_max_dim : int, default 2
        The largest dimension of ``small_engine``. The presets use 3.
    rugged_try_rest : {"none", "cutkink"}, default "none"
        What follows the local phase's capped descent on a rugged reading (5 to 9
        dimensions). ``"none"``: the finisher's CMA-ES populations;
        ``"cutkink"``: where the descent is still making progress at its cap and its
        best point reads a kink (6 evaluations, as across a sharp ridge), the same
        descent continues to the end of the budget.
    prefix_ridge_after : int, default 0
        Greater than 0: the prefix's SLSQP descent is checked once, after that many
        iterations; a ridge reading (every full step overshoots by a similar factor)
        hands the rest of the prefix to restarted COBYQA. 0: off. The presets
        use 7.
    prefix_ridge_max_per_dim : float, default 0.0
        The ridge check only at budgets up to this many x dim (0: at any budget).
        The presets use 30.
    prefix_full_per_dim : float, default 0.0
        At budgets up to this many x dim, and above ``prefix_full_min_per_dim``,
        the prefix gets the whole budget (no ATM after it). The presets use 100.
    prefix_full_min_per_dim : float, default 0.0
        The lower end of ``prefix_full_per_dim``. ``c46`` uses 0.
    des_max_per_dim : float, default 30.0
        With ``es_kind="slsqplq10hd"``: the one-descent prefix runs below this
        many x dim. The presets use 31.
    small_switch : bool, default False
        At budgets up to 10 x dim, in up to 20 dimensions: the prefix's first
        SLSQP descent is read at two checkpoints and hands the rest of the prefix to
        restarted COBYQA when it struggles: at the first when its line search took
        at least ``small_switch_trials`` trials, at the second when its first two
        gradients are not parallel (``|cos| < 0.99``).
    small_switch_trials : float, default 0.0
        The trial count of the first checkpoint (0: that checkpoint is off). The
        presets use 6.
    small_switch_keep_unit : bool, default False
        No hand-over at the second checkpoint when the second line search
        accepted its first trial.
    prefix_nan_reading : {"rugged", "plateau"}, default "rugged"
        Where an undefined curvature reading of the prefix goes (fewer than two
        usable probe pairs, mostly next to a box face): ``"rugged"``, or
        ``"plateau"``: smooth unless a plateau check then reads a plateau.
    box_normalize : bool, default False
        ``minimize`` maps any box onto ``[-5, 5]`` in every coordinate and runs
        there, so the search does not depend on the scale of ``x`` (on that box
        nothing changes).
    prefix_descent_retries : int, default 0
        The one-descent prefixes retry a descent that did not move, from uniform
        points, up to this many times (not when its first gradient was exactly
        zero). The presets use 5.
    rugged_direct_beta : float, default 0.0
        On a hand-over curvature reading below this value in the SLSQP routes'
        rugged branch, the CMA-ES populations start at once, without the capped
        descent and the first (1+1)-CMA-ES (0: off). The presets use 1.
    rugged_direct_min_beta : float, default 0.0
        ``rugged_direct_beta`` applies only to readings at or above this value.
        The presets use 0.5.
    prefix_low_beta : float, default 0.0
        In the prefix, a rugged curvature reading below this value takes the
        (1+1)-CMA-ES where the surrogate populations would run (0: off). ``c46``
        uses 0.15.
    prefix_low_beta_min_dim : int, default 0
        ``prefix_low_beta`` only from this dimension. ``c46`` uses 10.
    cma_handoff : float, default 0.0
        For budgets where the prefix does not run: ATM runs this share of the
        budget, in [0, 1), then the restarted (1+1)-Cholesky-CMA-ES continues from
        ATM's best point (0: ATM to the end). Needs a box and a success-rate
        ``amplitude_control``. The presets use 0.2.
    cma_model : {"none", "quadratic"}, default "none"
        ``"quadratic"``: every other candidate of the CMA phase minimises a
        least-squares quadratic fitted to the run's recent points (in up to
        10 dimensions).
    cma_restart_rule : {"legacy", "short"}, default "legacy"
        When a CMA run gives up: after no progress over ``100 + 50 dim``
        evaluations (``"legacy"``) or ``10 + 30 dim`` (``"short"``).
    cma_active : bool, default False
        The active (1+1) covariance update of Arnold and Hansen (2010).
    cma_warm_start : bool, default False
        The hand-over's CMA phase starts with the shape of the covariance of
        ATM's best evaluations.
    cma_handoff_cap_per_dim : float, default 0.0
        From 10 dimensions, ATM's share of the hand-over is at most this many
        x dim evaluations (0: no cap). The presets use 60.
    cma_local : {"one_plus_one", "then_ipop"}, default "one_plus_one"
        The hand-over's local phase from ``cma_local_min_dim`` dimensions.
        ``"one_plus_one"``: the restarted (1+1)-CMA-ES. ``"then_ipop"``: the
        (1+1)-CMA-ES from ATM's best point until it stalls, then IPOP population
        CMA-ES runs from uniform points.
    cma_local_min_dim : int, default 10
        The presets use 2.
    cma_model_high : {"none", "diagonal"}, default "none"
        Above 10 dimensions: ``"diagonal"`` fits a diagonal quadratic model
        (``2 dim + 1`` parameters) where ``cma_model`` would fit a full one.
    pop_active : bool, default False
        The active (negative) covariance update in the population runs, with
        pycma's default weights.
    pop_lambda : {"default", "fixed2"}, default "default"
        The population size of the population runs: the default (doubling, for
        IPOP) or twice the default.
    pop_gate_per_dim : float, default 0.0
        With ``cma_local="then_ipop"``: keep restarting the (1+1)-CMA-ES and launch
        a population run only at a stall with at least this many x dim evaluations
        left, so smaller budgets run as without it. The presets use 100.
    pop_protect : bool, default False
        Stop launching population runs once one ends without beating the best
        point found; the (1+1) restarts take the rest.
    cma_local_nonsep : str, default "none"
        With ``sep_probe``, from 5 dimensions: the local phase where the separable
        search finds the function non-separable (``"none"``: ``cma_local``). The
        ``"qn..."`` values name the routes of ``qn_phase.QN_ROUTES``: a curvature
        probe sends smooth functions to restarted SLSQP and rugged or flat ones to
        CMA-ES populations, with checks for separate basins, stalls and failing
        descents. The presets use ``"qn3"``, ``"qn7"`` to ``"qn9"``, ``"qn12"``,
        ``"qn17"``, ``"qn19"``, ``"qn20"``, ``"qn29"``, ``"qn38"`` and ``"qn44"``
        (``c46``).
    pop_switch_per_dim : float, default 0.0
        With ``pop_gate_per_dim``: the first (1+1) run also hands over to a
        population run after this many x dim evaluations without a stall (0: only
        at a stall). The presets use 150.
    pop_switch_start : {"uniform", "incumbent"}, default "uniform"
        Where that population run starts: a uniform point, or the best point at
        twice the (1+1)'s width.
    pop_switch_min_dim : int, default 1
        ``pop_switch_per_dim`` only from this dimension. The presets use 10.
    cma_probe_at_per_dim : float, default 0.0
        With the gate: after this many x dim evaluations of the first (1+1) run,
        probe the local curvature (24 evaluations) and hand over to a population
        run where it is below ``cma_probe_beta`` (0: no probe). The presets use 30.
    cma_probe_beta : float, default 0.0
        The presets use 1.3.
    cma_probe_min_dim : int, default 1
        The probe only from this dimension. The presets use 5.
    nm_max_dim : int, default 0
        Up to this dimension (0: off), the local search is SciPy's Nelder-Mead
        with restarts, where ``nm_stage`` says. The presets use 2.
    nm_stage : {"both", "prefix", "qnprefix30"}, default "both"
        ``"both"``: in the prefix and at the hand-over; ``"prefix"``: in the prefix
        only; ``"qnprefix30"``: in the prefix only, after one SLSQP descent (the
        descent only at budgets up to 30 x dim).
    sep_probe : bool, default False
        Before the local phase, the separable coordinate search (STEP) runs on at
        most half of the remaining budget, at budgets from 30 x dim; it stops
        after about 3 x dim evaluations on a non-separable function.
    sep_check : {"axis", "stall"}, default "axis"
        The separable search's check: ``"axis"`` re-evaluates a stored point of
        one axis; ``"stall"`` adds a pair test (two axes moved at once, testing
        additivity) after two sweeps without improvement.
    sep_stall_min_per_dim : float, default 0.0
        ``"stall"`` acts only from this many x dim. The presets use 1000.
    qn_fallback : {"none", "lbfgsb"}, default "none"
        ``"lbfgsb"``: a restarted-SLSQP call hands its remaining evaluations to
        restarted L-BFGS-B once its failed descents outnumber the converged ones.
    pool_rule : {"fixed", "budget"}, default "fixed"
        Samples per iteration. ``"fixed"``: 64 at the start and 60 afterwards.
        ``"budget"``: ``budget / 30``, clipped to ``[pool_min, 64]`` and rounded to
        a multiple of 4, so small budgets get more iterations.
    pool_min : int, default 24
        The smallest pool of ``pool_rule="budget"``. The presets use 12.
    anytime : {"off", "replay"}, default "off"
        ``"replay"`` is the anytime mode, which ``minimize(..., anytime=True)``
        turns on (``docs/ANYTIME.md``): up to 10 000 evaluations the run never
        learns its budget. The objective must be deterministic.
    """

    # False: the incumbent that every sampler's first column repeats is not
    # evaluated again; its known cost is used and not charged. The search path
    # is the original's with those evaluations deleted, so the budget lasts
    # longer. Assumes a deterministic objective.
    reevaluate_incumbent: bool = True

    # The very first iteration of a search samples a ball of radius ~0.03 around
    # the start (the sawtooth begins at its minimum), so the whole budget below
    # about 10 x dim is spent there. ``"uniform"``: the non-incumbent columns of
    # the first iteration are drawn uniformly from the box. Needs ``bounds``.
    initial_design: str = "legacy"
    # Samples outside the box are evaluated as they are (the caller clips and
    # penalises) and the incumbent can drift out. ``"fold"``: they are reflected
    # at the walls before evaluation, and the repaired point is stored. Needs
    # ``bounds``.
    bounds: str = "none"
    # With ``bounds="fold"``: probability that a sample with a coordinate outside
    # the box is clipped onto the box as a whole instead of reflected (keeps
    # corner optima reachable).
    corner_clip: float = 0.0
    # Width of the random search. ``"legacy"``: an open-loop sawtooth in an
    # iteration counter scaled by hyperparameters, times a fixed random vector.
    # ``"success_rate_targeted"``: one width per sampler, adapted by feedback (see
    # ``control``), with no use of the sampler hyperparameters, plus a targeted step
    # of a log-uniform multiple (0.1 to 10) of the last move, taken for any
    # non-zero move.
    amplitude_control: str = "legacy"
    # Controller: ``log_width += sr_eta * (success_rate - target)`` per sampler,
    # starting from ``sr_widths`` and capped at ``sr_max_width``.
    # A (1+1)-ES with the 1/5 rule on the first ``es_fraction`` of the budget
    # (after a uniform design of ``es_init_factor * dim`` points, at least 4, to
    # pick its start), then ATM from the ES incumbent with its width ladder scaled
    # to the ES step size. ``"prefix"`` needs ``bounds`` and a success-rate
    # controller; when it runs, the ES does its own design and ATM's uniform first
    # pool (``initial_design``) is skipped for that first search.
    es_phase: str = "none"
    es_fraction: float = 0.3
    # The ES phase is only used for budgets of at most this many evaluations per
    # dimension; above it the search is ATM alone (the ES prefix helps up to about
    # 100 x dim and hurts beyond, where ATM's pool beats a greedy ES on multimodal
    # and ill-conditioned functions).
    es_max_budget_per_dim: float = 1e9
    # Correlated search directions. ``"rank_mu"``: the random search is Gaussian
    # with a full covariance matrix (unit geometric mean, so the widths keep their
    # meaning) learned CMA-style from the best ``cov_select`` share of every
    # iteration's evaluated samples, weighted by log rank, with the rank-mu learning
    # rate times ``cov_rate``; the eigenvalue ratio is limited to ``cov_cap``. This
    # is the one variant with a DxD object. Needs a success-rate
    # ``amplitude_control``.
    covariance: str = "none"
    cov_rate: float = 1.0
    cov_select: float = 0.25
    # Only used up to this dimension: the covariance is learned from a few samples
    # per iteration, which is too slow beyond about 10 dimensions (at 20 it costs
    # .03 at 300 x dim and gains nothing).
    # The local search of the ES prefix after its uniform design: ``"isotropic"``:
    # the (1+1)-ES with the 1/5 rule; ``"cma"``: the restarted (1+1)-Cholesky-
    # CMA-ES of ``cma_phase``; ``"lq2"``: IPOP population CMA-ES runs ranked by
    # the linear-quadratic surrogate of ``surrogate`` (active update, pycma's
    # style), and ``"cma"`` for budgets below ``es_lq_min_budget_per_dim`` times dim;
    # ``"slsqplq"``: restarted SLSQP (``qn_phase``) where a curvature probe reads
    # the function as smooth (cycles 11-12), lq2
    # where rugged (budgets >= ``es_lq_min_budget_per_dim``), else the (1+1), and the
    # (1+1) below 30 dim evaluations (cycle 12); ``"slsqplq10c"``: the same, except
    # that it also runs below 30 dim evaluations once the budget reaches 200 (cycle 17;
    # ``qn_phase.QN_MIN_BUDGET``), with a two-direction (12-evaluation) probe there;
    # ``"slsqplq10h"``:
    # ``"slsqplq10c"`` with SLSQP's finite-difference step 1e-9 (cycle 20);
    # ``"slsqplq10hd"``: ``"slsqplq10h"``, and where that skips the probe one SLSQP
    # descent from the design's best, then the (1+1) (cycle 33, ``q7ed``).
    es_kind: str = "isotropic"
    es_lq_min_budget_per_dim: float = 0.0
    # The surrogate prefix only where the separable search finds the function
    # non-separable (cycle 10: ``pre2`` lost f3 at D = 5).
    es_lq_nonsep: bool = False
    # Cycle 38 (small budgets; c34 trails restarted SLSQP at 10-30 x D on f10-f14
    # by .05-.06): ``prefix_sep_probe``: with ``es_lq_nonsep``, a passed separability
    # check no longer forces the (1+1) on the ``"slsqplq"`` prefix; the curvature probe
    # decides (SLSQP where smooth). ``prefix_full_per_dim``: at budgets up to that many
    # x dim the ES prefix gets the whole budget (no ATM tail), above
    # ``prefix_full_min_per_dim`` x dim (Batch 62: at 10 x D the tail's global samples
    # still help multimodal functions a little).
    prefix_sep_probe: bool = False
    # Cycle 42 (``preplat``): above 30 x dim the ES prefix checks for a plateau after
    # its design and separable search; on one (f7) restarted CMA-ES populations take the
    # prefix's rest (c36 trails pycma IPOP on f7 at D 5, 100 x D by .33).
    prefix_plateau_pop: bool = False
    # ``prefix_plateau_min_dim``: no plateau check below that dim (cycle 43).
    prefix_plateau_min_dim: int = 0
    # Cycle 50 (the f20 toy study, ``chain_phase``): from ``additive_min_dim`` and
    # with a box, a pair test at x0 (3 x dim evaluations) reads the function as
    # separable, a chain (only neighbouring coordinates interact) or neither.
    # ``chain_route``: a chain that also passes the band check (6 x dim; f20, not
    # Rosenbrock) gets the decoupled coordinate search for the whole budget, from
    # ``chain_min_per_dim`` x dim. ``sep_route``: a separable reading gets restarted
    # STEP with its check off for the whole budget, from ``sep_route_min_per_dim``.
    chain_route: bool = False
    sep_route: bool = False
    # Cycle 53 (``basin_route``, the D 2 toy): at dim 2 up to ``basin_max_dim`` with at
    # least ``basin_min_per_dim`` x dim, a Sobol design (min(16 dim, budget / 4)) and a
    # curvature probe at its best point (4 dim); a smooth multi-basin reading
    # (Gallagher) gets MLSL for the budget, any other continues from the design's best
    # point.
    # Cycle 60 (``basin_descents``, the D 2 thread): on a smooth reading that is not
    # multi-basin, up to that many SLSQP descents (the design's best point, then the
    # best point a quarter of the box away) up to ``basin_descents_max_per_dim`` x dim;
    # the usual path then continues from the design's best point, the descents' best
    # stays the result if nothing beats it (restarted SLSQP led `c41` at D 2 on f6 /
    # f9 / f12 / f13, 100-300 x D).
    basin_route: bool = False
    basin_descents: int = 0
    # ``basin_keep_atm`` (cycle 60, Batch 72): ATM's hand-over share is counted after
    # the descents, so they do not shorten ATM's sample, which seeds the CMA warm start
    # (f4 at D 2-3, 300 x D lost −.19 / −.21 when they did)
    basin_keep_atm: bool = False
    # Cycle 61 (``small_engine``, the small-budget engines toy): ``"cobyqa"`` spends
    # budgets up to ``small_engine_max_per_dim`` x dim at dim up to
    # ``small_engine_max_dim``
    # on restarted SciPy COBYQA (``cobyqa_phase``), initial trust radius
    # ``small_engine_radius`` x box width, instead of every other route.
    small_engine: str = "none"
    small_engine_max_dim: int = 2
    # From a study of bbob f13 (sharp ridge): what follows the local
    # phase's capped rugged try (``QnRoute.rugged_try_per_dim``, D 5-9). ``"none"``:
    # the finisher's populations; ``"cutkink"``: a try still descending at its cap
    # goes on, the same scaled-FD BFGS descent, to the end of the budget where its
    # best point at the cap reads a kink (``qn_phase.kink_check``, 6 evaluations;
    # f13's ridge).
    rugged_try_rest: str = "none"
    # The f13 toy: ``prefix_ridge_after`` > 0 reads the gated prefix's SLSQP descent
    # (``"slsqplq"`` / ``"slsqplqc"`` on a smooth reading) once, when its
    # ``StallMonitor`` has recorded that many iterations: a ridge reading
    # (``qn_phase.ridge_reading``: every full step overshoots by a similar factor and
    # one backtrack recovers it, as across f13's sharp ridge) hands the rest of the
    # prefix to restarted COBYQA (initial trust radius ``prefix_ridge_radius`` x box
    # width). 0: off.
    prefix_ridge_after: int = 0
    # ``prefix_ridge_max_per_dim`` > 0: the ridge reading only at budgets up to that
    # many x dim (at 100 x D the SLSQP descent beats COBYQA on f13: -.108 per run at
    # D 20 / 40, exploration seeds). 0: no limit.
    prefix_ridge_max_per_dim: float = 0.0
    prefix_full_per_dim: float = 0.0
    prefix_full_min_per_dim: float = 0.0
    # Cycle 58 (``des30``): ``es_kind="slsqplq10hd"`` runs its one-descent prefix
    # (``"slsqpdes"``) below ``des_max_per_dim`` x dim (and ``QN_MIN_BUDGET``); the
    # default 30 is ``q7ed``'s 10 x D envelope.
    des_max_per_dim: float = 30.0
    # Research (``small_switch``, from a study of budgets up to 10 x D):
    # at budgets up to ``small_switch_max_per_dim`` x dim and dim up to
    # ``small_switch_max_dim``, the prefix's first SLSQP descent is read at two
    # checkpoints (the first probe after its first / second line search) and hands
    # the rest to restarted COBYQA (initial trust radius ``small_switch_radius`` x box
    # width): at checkpoint 1 when that line search took at least
    # ``small_switch_trials`` trials (0: off), at checkpoint 2 when the two gradients
    # are not parallel, |cos| < ``small_switch_cos`` (0: off), unless
    # ``small_switch_keep_unit`` and the second line search accepted its first trial.
    # ``small_switch_rest``: COBYQA also instead of the prefix's (1+1) (after a descent
    # that ended early, or after the curvature probe's rugged reading).
    small_switch: bool = False
    small_switch_trials: float = 0.0
    small_switch_keep_unit: bool = False
    # Cycle 64 (from a study of the cells where SLSQP leads):
    # ``prefix_nan_reading`` says where an undefined curvature reading of the
    # ``slsqplq`` / ``slsqplqc`` prefix goes (NaN beta: fewer than two positive
    # mirrored losses, mostly probe pairs clipped at a box face): ``"rugged"`` (c43),
    # ``"smooth"``, or ``"plateau"`` (smooth unless one ``plateau_check`` then reads
    # a plateau, f7). On ids 1-3, 14 of f13's 24 readings at D 40, 100 x D were NaN.
    prefix_nan_reading: str = "rugged"
    # ``box_normalize`` (the novel-ideas toy): ``minimize`` maps any box onto bbob's
    # [-5, 5]^D coordinate by coordinate and runs there (bbob's box: unchanged).
    box_normalize: bool = False
    # ``prefix_descent_retries`` > 0: the one-descent prefixes (``"slsqpdes"`` outside
    # ``small_switch``, ``"slsqpnm"``) retry a descent that did not move from uniform
    # points, unless its first gradient was exactly zero (the toy's ``retry5``).
    prefix_descent_retries: int = 0
    # Research (gaps3 toy): on a hand-over curvature reading below
    # ``rugged_direct_beta`` (0: off) in the SLSQP routes' rugged branch, no capped
    # BFGS try and no first (1+1) in the finisher: its populations start at once (from
    # a uniform point, as after the (1+1)'s own rugged probe). On calibration ids f13
    # reads 1.24-1.60 (p10-p90) at D 5, 300 x D; f15, f17-f19, f23, f24 below 1.2.
    rugged_direct_beta: float = 0.0
    # ``rugged_direct_min_beta`` (0: none): only readings at or above it (rough
    # readings near 0, f19 / f23 / f24, keep the (1+1) first, which gained there).
    rugged_direct_min_beta: float = 0.0
    # Research (gaps4 toy): ``prefix_low_beta`` (0: off): in the ES prefix, a finite
    # rugged curvature reading below it takes the (1+1)-CMA-ES where the rugged kind
    # would be the ``lq2`` populations (from ``es_lq_min_budget_per_dim``, 100 x dim);
    # on calibration ids f23 reads -0.3 to 0.3 at 100 x dim, f19 mostly 0.1-0.8.
    prefix_low_beta: float = 0.0
    # ``prefix_low_beta_min_dim``: only from this dim (at D 3-5 f17 / f18 / f24 lost
    # about as much as f23 gained on calibration ids).
    prefix_low_beta_min_dim: int = 0
    # For budgets where the ES prefix does not run: ATM runs the first
    # ``cma_handoff`` share of the budget, then the restarted (1+1)-Cholesky-CMA-ES
    # of ``cma_phase`` takes over from ATM's best point, starting at the
    # geometric mean of ATM's final widths (0: ATM to the end). Needs ``bounds``.
    cma_handoff: float = 0.0
    # Quadratic model steps in the CMA phase (prefix and hand-over): every other
    # candidate is the minimiser of a least-squares quadratic fitted to the run's
    # recent points (``cma_phase.quadratic_step``), for dim up to
    # ``cma_model_max_dim``. ``"none"``: only random steps.
    cma_model: str = "none"
    # When a CMA run gives up (``cma_phase.run_cma``): ``"legacy"``: no progress
    # over 100 + 50 dim evaluations; ``"short"``: over 10 + 30 dim.
    cma_restart_rule: str = "legacy"
    # The active (1+1) covariance update of Arnold and Hansen (2010) in the CMA
    # phase.
    cma_active: bool = False
    # At the hand-over, start the CMA phase with the shape (unit determinant) of the
    # regularised covariance of ATM's best max(2 dim, 1/8 of its) evaluations.
    cma_warm_start: bool = False
    # For dim >= ``cma_handoff_cap_min_dim``: ATM's share of the hand-over is at most
    # ``cma_handoff_cap_per_dim * dim`` evaluations (0: no cap).
    cma_handoff_cap_per_dim: float = 0.0
    # The hand-over's local phase for dim >= ``cma_local_min_dim``:
    # ``"one_plus_one"``: the restarted (1+1)-Cholesky-CMA-ES of ``cma_phase``;
    # ``"then_ipop"``:
    # the (1+1)-CMA's first run from ATM's best point until it stalls, then IPOP
    # population runs from uniform points.
    cma_local: str = "one_plus_one"
    cma_local_min_dim: int = 10
    # Batch 12: ``cma_model_high="diagonal"``: above ``cma_model_max_dim``, a
    # ratio-gated, rank-weighted diagonal quadratic model (2 dim + 1 parameters,
    # fitted to the last 3 (2 dim + 1) points; ``cma_phase.model_step``).
    cma_model_high: str = "none"
    # The population runs of ``cma_local``: ``pop_active`` adds the negative
    # (active) covariance update with pycma's default weights; ``pop_lambda`` is
    # ``"default"`` (doubling as the mode says) or ``"fixed2"`` (twice the default);
    # ``pop_gate_per_dim > 0`` (with ``then_ipop``) keeps the restarted (1+1) and
    # launches a population run only at a stall with at least that many times dim
    # evaluations left, so smaller budgets run exactly as without it.
    pop_active: bool = False
    pop_lambda: str = "default"
    pop_gate_per_dim: float = 0.0
    # Population runs (Batch 13): ``pop_protect`` stops launching population runs
    # once one ends without beating the best point found, so the (1+1) restarts
    # take the rest.
    pop_protect: bool = False
    # With ``sep_probe``, for dim >= ``cma_local_nonsep_min_dim``: when the separable
    # search's check finds the function non-separable, the local phase is a route
    # of ``qn_phase.QN_ROUTES`` instead of ``cma_local``. ``"qn3"``: a plateau check
    # first (a plateau skips the curvature probe), then on a smooth reading SLSQP for
    # a third of the local budget, at most 100 dim evaluations, then ``cma_local``
    # from its best. ``"qn7"``: as ``"qn3"``, but rugged or plateau readings get
    # ``steps`` populations; smooth functions whose converged SLSQP descents end in
    # separate basins (hill-valley test) keep restarting SLSQP instead of the CMA
    # finisher, and a stage that ends mid-descent continues for 2/3 of what is left,
    # at most another 100 dim evaluations, before the finisher. ``"qn8"``: as
    # ``"qn7"``, but a plateau reading (f7) launches IPOP populations from a uniform
    # point with 4x the default lambda instead (cycle 18). ``"qn9"``: ``"qn8"`` with
    # SLSQP's finite-difference step 1e-9 instead of SciPy's 1.49e-8 (cycle 20,
    # ``qn_phase.QN_FD_STEP``). ``"qn12"``: ``"qn9"`` with SciPy's default step in the
    # multi-basin route's restarts (cycle 24: the 1e-9 step makes each Gallagher
    # descent dearer, so fewer restarts fit).
    # The later routes are described in ``qn_phase.QnRoute``.
    cma_local_nonsep: str = "none"
    # With ``pop_gate_per_dim``: the first (1+1) run also hands over to the
    # population after this many times dim evaluations without a stall (0: only at
    # a stall); a protected population run that fails returns to the (1+1) at
    # the best point with the run's last width scale.
    pop_switch_per_dim: float = 0.0
    # Where a population run launched at the switch cap starts: ``"uniform"``, or
    # ``"incumbent"``: the best point, at twice the (1+1)'s width scale.
    pop_switch_start: str = "uniform"
    pop_switch_min_dim: int = 1
    # With the gate: after this many times dim evaluations of the first (1+1)
    # run, probe the local curvature (``cma_phase``, 24 evaluations) and hand
    # over to the population (uniform start) if beta < ``cma_probe_beta`` (0: only
    # log beta). 0: no probe. Only for dim >= ``cma_probe_min_dim``.
    cma_probe_at_per_dim: float = 0.0
    cma_probe_beta: float = 0.0
    cma_probe_min_dim: int = 1
    # ``sep_probe``: before the local phase (after the ES prefix's design, or after
    # ATM at the hand-over) run the separable coordinate search of
    # ``step_phase`` on at most ``sep_share`` of the remaining budget; it stops
    # after about 3 dim evaluations on non-separable functions. Only for budgets of
    # at least ``sep_min_budget_per_dim`` times dim.
    # For dim <= ``nm_max_dim`` (0: off) the local search is the restarted
    # Nelder-Mead of ``nm_phase``: in the prefix, at the hand-over or both
    # (``nm_stage``); ``"qnprefix30"``: in the prefix only, after one SLSQP descent
    # at budgets up to 30 dim (at D = 2, restarted SLSQP led on f1, f2 and f10-f14
    # at 10-30 x D).
    nm_max_dim: int = 0
    nm_stage: str = "both"
    sep_probe: bool = False
    # The separable search's check: ``"axis"`` re-evaluates a stored point of one
    # axis; ``"stall"``: the axis check, and a pair test (two axes moved at once,
    # testing additivity) only after two sweeps without improvement, when
    # re-evaluating a stored point at an unchanged best can no longer fail.
    sep_check: str = "axis"
    # ``sep_stall_min_per_dim`` (cycle 54, the late-router study): ``"stall"`` acts
    # only from this many x dim (below it the axis check alone, as ``"axis"``).
    sep_stall_min_per_dim: float = 0.0
    # ``"lbfgsb"``: every restarted-SLSQP call (prefix and local phase) hands its
    # remaining evaluations to restarted L-BFGS-B once failed descents outnumber
    # converged ones, and a local stage that ends that way gives L-BFGS-B the rest of
    # the budget (f12, bent cigar; cycle 28, ``qn_phase.run_lbfgsb_restarts``).
    qn_fallback: str = "none"
    # Samples per iteration. ``"fixed"``: 64 at the start and 60 afterwards
    # (``initial_population`` and the conductor's total), whatever the budget.
    # ``"budget"``: ``clip(budget / pool_divisor, pool_min, pool_max)`` rounded to
    # a multiple of 4, for the start, the conductor and the iteration count
    # together (they are three separate numbers in the original). Small budgets
    # then get more iterations out of their few evaluations.
    pool_rule: str = "fixed"
    pool_min: int = 24
    # The anytime mode (``docs/ANYTIME.md``; ``minimize(..., anytime=True)``).
    # ``"off"`` plans the run for ``budget`` (every configuration recorded so far).
    # ``"replay"`` (``anytime.py``): up to
    # ``anytime_switch`` evaluations the run never learns its budget. It plans
    # ``anytime_start_per_dim`` x dim evaluations, and each time those are spent it
    # plans ``anytime_factor`` times as many in total, rerunning the configuration
    # from the same start and seed with every earlier point answered from memory.
    # Once the next plan would exceed ``anytime_switch``, one ordinary run gets
    # everything left of ``budget``, fed the points evaluated so far. Deterministic
    # objectives only. Left out of ``config_hash`` at their defaults.
    anytime: str = "off"
    anytime_start_per_dim: float = 10.0
    anytime_factor: float = 2.0
    anytime_switch: int = 10_000

    def __post_init__(self) -> None:
        """Validate the fields."""
        for name in _BOOLS:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        for name, allowed in _CHOICES.items():
            _choose(name, getattr(self, name), allowed)
        for name, valid, message in _RANGES:
            value = getattr(self, name)
            if not valid(value):
                raise ValueError(message.format(v=value))
        if not 2 * N_SAMPLERS <= self.pool_min <= POOL_MAX:
            raise ValueError(
                f"need {2 * N_SAMPLERS} <= pool_min <= {POOL_MAX}, got {self.pool_min}"
            )
        for name, active in _NEEDS_SUCCESS_RATE.items():
            if active(getattr(self, name)) and self.amplitude_control == "legacy":
                raise ValueError(f"{name} needs a success-rate amplitude_control")
        _choose("es_kind", self.es_kind, _LATE_CHOICES["es_kind"])
        _choose(
            "cma_local_nonsep", self.cma_local_nonsep, _LATE_CHOICES["cma_local_nonsep"]
        )
        _choose("nm_stage", self.nm_stage, _LATE_CHOICES["nm_stage"])
        if self.nm_max_dim < 0:
            raise ValueError("nm_max_dim must be >= 0")
        _choose("cma_model", self.cma_model, _LATE_CHOICES["cma_model"])
        _choose(
            "cma_restart_rule", self.cma_restart_rule, _LATE_CHOICES["cma_restart_rule"]
        )
        _choose("cma_local", self.cma_local, _LATE_CHOICES["cma_local"])
        if self.cma_local_min_dim < 1:
            raise ValueError("cma_local_min_dim must be at least 1")
        _choose("cma_model_high", self.cma_model_high, _LATE_CHOICES["cma_model_high"])
        _choose("sep_check", self.sep_check, _LATE_CHOICES["sep_check"])
        _choose("qn_fallback", self.qn_fallback, _LATE_CHOICES["qn_fallback"])
        _choose("pop_lambda", self.pop_lambda, _LATE_CHOICES["pop_lambda"])
        _choose(
            "pop_switch_start", self.pop_switch_start, _LATE_CHOICES["pop_switch_start"]
        )
        if self.pop_switch_per_dim < 0:
            raise ValueError("pop_switch_per_dim must be >= 0")
        if self.pop_gate_per_dim < 0:
            raise ValueError("pop_gate_per_dim must be >= 0")
        if self.pop_gate_per_dim and self.cma_local != "then_ipop":
            raise ValueError("pop_gate_per_dim needs cma_local='then_ipop'")
        for name in (
            "cma_active",
            "cma_warm_start",
            "pop_active",
            "pop_protect",
            "es_lq_nonsep",
            "prefix_sep_probe",
            "prefix_plateau_pop",
            "chain_route",
            "sep_route",
            "basin_route",
            "sep_probe",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.sep_stall_min_per_dim < 0:
            raise ValueError("sep_stall_min_per_dim must be >= 0")
        _choose("small_engine", self.small_engine, _LATE_CHOICES["small_engine"])
        _choose(
            "rugged_try_rest", self.rugged_try_rest, _LATE_CHOICES["rugged_try_rest"]
        )
        _choose(
            "prefix_nan_reading",
            self.prefix_nan_reading,
            _LATE_CHOICES["prefix_nan_reading"],
        )
        if self.prefix_ridge_after < 0 or self.prefix_ridge_max_per_dim < 0:
            raise ValueError(
                "need prefix_ridge_after >= 0 and prefix_ridge_max_per_dim >= 0"
            )
        if not isinstance(self.box_normalize, bool):
            raise ValueError("box_normalize must be a bool")
        if self.prefix_descent_retries < 0:
            raise ValueError("the parked2 prefix options must not be negative")
        if self.rugged_direct_beta < 0 or self.rugged_direct_min_beta < 0:
            raise ValueError("rugged_direct_beta and _min_beta must be >= 0")
        _choose("anytime", self.anytime, _LATE_CHOICES["anytime"])
        if (
            not self.anytime_start_per_dim > 0
            or not self.anytime_factor > 1
            or self.anytime_switch < 1
        ):
            raise ValueError(
                "anytime_start_per_dim must be > 0, anytime_factor > 1 and "
                "anytime_switch >= 1"
            )
        if self.prefix_low_beta < 0 or self.prefix_low_beta_min_dim < 0:
            raise ValueError("prefix_low_beta and prefix_low_beta_min_dim must be >= 0")
        if self.small_engine_max_dim < 1:
            raise ValueError("small_engine_max_dim must be >= 1")
        if self.basin_descents < 0:
            raise ValueError("basin_descents must be >= 0")
        if self.small_switch_trials < 0:
            raise ValueError("small_switch_trials must be >= 0")
        if (
            min(self.prefix_full_per_dim, self.prefix_full_min_per_dim) < 0
            or self.des_max_per_dim < 0
        ):
            raise ValueError("prefix_full_*_per_dim and des_max_per_dim must be >= 0")
        if self.cma_handoff_cap_per_dim < 0:
            raise ValueError("cma_handoff_cap_per_dim must be >= 0")
        if not 0 <= self.cma_handoff < 1:
            raise ValueError(f"cma_handoff must be in [0, 1), got {self.cma_handoff}")
        if self.cma_handoff and self.amplitude_control == "legacy":
            raise ValueError("cma_handoff needs a success-rate amplitude_control")
        if self.covariance != "none" and self.amplitude_control == "legacy":
            raise ValueError("covariance needs a success-rate amplitude_control")
        if self.corner_clip and self.bounds != "fold":
            raise ValueError("corner_clip needs bounds='fold'")

    @property
    def needs_box(self) -> bool:
        """Whether ``minimize`` must be given ``bounds=(lower, upper)``."""
        return (
            self.initial_design != "legacy"
            or self.bounds != "none"
            or self.es_phase != "none"
            or self.cma_handoff > 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Every field, defaults included, as a ``dict``.

        Returns
        -------
        dict
            Field name to value; :meth:`from_dict` inverts it.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Variant:
        """Build a ``Variant`` from a :meth:`to_dict` result (or a subset of it).

        Parameters
        ----------
        data : dict
            Field names and values; a missing field keeps its default, and lists
            are accepted for the tuple fields (a JSON round trip gives lists).

        Returns
        -------
        Variant
            The variant with those fields.

        Raises
        ------
        ValueError
            If ``data`` names an unknown field or a value is invalid, or sets a
            field of :data:`REMOVED_FIELDS` to anything but its old default.
        """
        data = _drop_removed(data)
        _reject_unknown(cls, data)
        return cls(**data)


_FIELD_DEFAULTS: dict[str, Any] = {f.name: f.default for f in fields(Variant)}


@dataclass(frozen=True)
class ATMConfig:
    """The configuration of a run: ATM's structure and its ``Variant``.

    ``ATMConfig()`` is the original ATM sampler (no box needed), the presets
    (``presets.get("c46")``, the default of ``minimize``) are ``ATMConfig`` values,
    and :meth:`composite` is the best configuration within ATM's own sampler. The
    structural defaults are the original's; ``strict_budget=False`` also restores
    the original's budget accounting. Instances are frozen; use
    ``dataclasses.replace`` or ``presets.with_changes`` for a changed copy.

    Attributes
    ----------
    iterations_per_hyperparameter_set : int, default 5
        Iterations run with one set of sampler hyperparameters before the meta
        level proposes the next set. The presets use 1 000 000, so one set lasts
        the whole search.
    second_best : {"legacy", "exclude_winner"}, default "legacy"
        How the crossover partner (the "second best" point) is chosen.
        ``"legacy"`` is the original's rule, which picks the best point itself
        when the costs are negative; ``"exclude_winner"`` picks the real runner-up
        (it changes the search path of almost every run).
    strict_budget : bool, default True
        ``True``: at most ``budget`` objective calls, every one counted, and the
        best point of an iteration cut short by the budget is kept. ``False``
        reproduces the original's accounting (``budget + 2`` calls, the calls at
        the start of a search not counted, a cut-short iteration discarded); it
        needs the default ``variant``.
    initial_population : int, default 64
        Samples per iteration at the start of every search (4 samplers x 16).
    evaluations_per_iteration : int, default 60
        The evaluations per iteration that the iteration count is derived from
        (the conductor's total, not the population).
    conductor : ConductorConfig
        How the samples of an iteration are shared between the four samplers.
    restart : RestartPolicy
        When a search restarts.
    variant : Variant
        The departures from the original algorithm; the presets differ only here.

    Examples
    --------
    >>> from amm_optimizer import presets
    >>> from amm_optimizer.config import ATMConfig
    >>> presets.get("c46").config_hash()
    '2153948ebed51150'
    >>> ATMConfig.from_dict(ATMConfig().to_dict()) == ATMConfig()
    True
    """

    iterations_per_hyperparameter_set: int = 5
    # How the crossover partner ("second best") is chosen. ``"legacy"`` is the
    # original's trick, which degenerates to the best point itself for negative
    # costs (audit B21); ``"exclude_winner"`` picks the real runner-up. Opt-in
    # because it changes the search path of almost every run.
    second_best: str = "legacy"
    # True: at most ``budget`` objective calls, every one of them counted, and the
    # best point of a batch cut short by the budget is kept. False reproduces the
    # original accounting (``budget + 2`` calls, start-of-search calls free, a
    # cut-short batch discarded), which the parity tests rely on.
    strict_budget: bool = True
    # Samples per iteration at the start of every search (4 x 16).
    initial_population: int = 64
    # The iteration count is derived from the budget assuming this many
    # evaluations per iteration (the conductor's total, not the population).
    evaluations_per_iteration: int = 60
    conductor: ConductorConfig = field(default_factory=ConductorConfig)
    restart: RestartPolicy = field(default_factory=RestartPolicy)
    variant: Variant = field(default_factory=Variant)

    def __post_init__(self) -> None:
        """Check that the fields fit together."""
        if self.second_best not in ("legacy", "exclude_winner"):
            raise ValueError(
                "second_best must be 'legacy' or 'exclude_winner', "
                f"got {self.second_best!r}"
            )
        if self.variant != Variant() and not self.strict_budget:
            raise ValueError(
                "a non-default variant needs strict_budget=True; "
                "strict_budget=False only reproduces the original"
            )
        if not self.variant.reevaluate_incumbent and (
            min(self.initial_population, self.conductor.total_resources)
            < 2 * N_SAMPLERS
        ):
            # Every sampler's first column is the incumbent, which is then free;
            # fewer than two columns per sampler could leave nothing to evaluate.
            raise ValueError(
                "reevaluate_incumbent=False needs at least "
                f"{2 * N_SAMPLERS} samples per iteration"
            )

    @classmethod
    def composite(cls) -> ATMConfig:
        """Return the five-option composite, confirmed on three holdouts.

        Success-rate amplitude control with a targeted step, a uniform first
        pool, bounds folded into the box with a 25 % corner clip, a pool sized
        from the budget (at least 12) and no re-evaluation of the incumbent. It
        stays within ATM's own sampler. Gains over the defaults on untouched
        bbob instances: +.101, +.097 and +.099 on three held-out sets of COCO
        instances; the gain shrinks with dimension (+.033 to +.038 at D = 40).

        Returns
        -------
        ATMConfig
            It uses the search box, so call ``minimize(..., bounds=...)``.
        """
        return cls(
            variant=Variant(
                amplitude_control="success_rate_targeted",
                initial_design="uniform",
                bounds="fold",
                corner_clip=0.25,
                pool_rule="budget",
                pool_min=12,
                reevaluate_incumbent=False,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Every field, defaults included, as plain Python containers.

        Returns
        -------
        dict
            Field name to value, with ``conductor``, ``restart`` and ``variant`` as
            nested ``dict``; JSON-serialisable, and :meth:`from_dict` inverts it.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ATMConfig:
        """Build an ``ATMConfig`` from a :meth:`to_dict` result (or a subset of it).

        Parameters
        ----------
        data : dict
            Field names and values, with ``conductor``, ``restart`` and
            ``variant`` as nested ``dict``; a missing field keeps its default.

        Returns
        -------
        ATMConfig
            The configuration with those fields.

        Raises
        ------
        ValueError
            If ``data`` names an unknown field, at any level, or a value is
            invalid, or sets a field of :data:`REMOVED_FIELDS` to anything but its
            old default.
        """
        data = dict(data)
        _reject_unknown(cls, data)
        nested: dict[str, Any] = {
            "conductor": ConductorConfig,
            "restart": RestartPolicy,
            "variant": Variant,
        }
        if "variant" in data:
            data["variant"] = _drop_removed(data["variant"])
        for name, kind in nested.items():
            if name in data:
                _reject_unknown(kind, data[name])
                data[name] = kind(**data[name])
        return cls(**data)

    def config_hash(self) -> str:
        """Return a short, stable fingerprint of the configuration (provenance).

        Two configurations have the same hash exactly when their :meth:`to_dict`
        results are equal, except that the ``Variant`` fields of
        :data:`HASH_IF_SET` count only when they differ from their defaults: they
        were added later, and every hash recorded before them stays valid. The
        fields removed since (:data:`REMOVED_FIELDS`) are hashed at the defaults
        they had, for the same reason. The hash does not depend on the platform or
        the package version (``c46`` is ``2153948ebed51150``).

        Returns
        -------
        str
            16 hexadecimal digits (the start of a SHA-256 digest).
        """
        text = _hash_payload(self.to_dict())
        return hashlib.sha256(text.encode()).hexdigest()[:16]
