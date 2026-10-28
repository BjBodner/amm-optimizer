"""User-facing configuration."""

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


def _reject_unknown(kind: type, data: dict[str, Any]) -> None:
    valid = {f.name for f in fields(kind)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(
            f"unknown {kind.__name__} field(s) {unknown}; valid: {sorted(valid)}"
        )


def _choose(name: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


# ``Variant`` fields added after the freeze of 2026-10-22: ``ATMConfig.config_hash``
# reads them only when set, so the configurations recorded before keep their hashes.
HASH_IF_SET = ("anytime", "anytime_start_per_dim", "anytime_factor", "anytime_switch")

# Validation of ``Variant`` is table driven. Each row is checked in order.
_BOOLS = ("reevaluate_incumbent", "reset_on_success")
_CHOICES = {
    "targeted_gate": ("legacy", "any"),
    "targeted_amplitude": ("legacy", "loguniform"),
    "initial_design": ("legacy", "uniform"),
    "bounds": ("none", "fold"),
    "pool_rule": ("fixed", "budget"),
    "amplitude_control": ("legacy", "success_rate", "success_rate_targeted"),
    "es_phase": ("none", "prefix"),
    "local_method": ("es", "nelder_mead"),
    "scale_learning": ("none", "diagonal"),
    "covariance": ("none", "rank_mu"),
}
# (field, is the value valid?, message; ``{v}`` is the value)
_RANGES: tuple[tuple[str, Callable[[Any], bool], str], ...] = (
    (
        "refine_fraction",
        lambda v: 0 <= v < 1,
        "refine_fraction must be in [0, 1), got {v}",
    ),
    (
        "refine_min_budget_per_dim",
        lambda v: v >= 0,
        "refine_min_budget_per_dim must not be negative",
    ),
    ("refine_max_dim", lambda v: v >= 1, "refine_max_dim must be at least 1"),
    ("cov_rate", lambda v: v > 0, "cov_rate must be positive, got {v}"),
    ("cov_select", lambda v: 0 < v <= 1, "cov_select must be in (0, 1], got {v}"),
    ("cov_max_dim", lambda v: v >= 1, "cov_max_dim must be >= 1, got {v}"),
    ("cov_cap", lambda v: v >= 1, "cov_cap must be at least 1, got {v}"),
    ("scale_rate", lambda v: 0 < v <= 1, "scale_rate must be in (0, 1], got {v}"),
    ("scale_cap", lambda v: v >= 1, "scale_cap must be at least 1, got {v}"),
    ("es_fraction", lambda v: 0 < v < 1, "es_fraction must be in (0, 1), got {v}"),
    (
        "es_max_budget_per_dim",
        lambda v: v > 0,
        "es_max_budget_per_dim must be positive",
    ),
    ("es_small_dim", lambda v: v >= 0, "es_small_dim must not be negative"),
    (
        "es_small_max_budget_per_dim",
        lambda v: v > 0,
        "es_small_max_budget_per_dim must be positive",
    ),
    ("es_init_factor", lambda v: v >= 0, "es_init_factor must not be negative"),
    ("pool_divisor", lambda v: v >= 1, "pool_divisor must be positive, got {v}"),
    ("corner_clip", lambda v: 0 <= v <= 1, "corner_clip must be in [0, 1], got {v}"),
)
# Options that only make sense with a success-rate amplitude control.
_NEEDS_SUCCESS_RATE: dict[str, Callable[[Any], bool]] = {
    "refine_fraction": lambda v: bool(v),
    "scale_learning": lambda v: v != "none",
    "es_phase": lambda v: v != "none",
}


@dataclass(frozen=True)
class Variant:
    """Switchable departures from the original algorithm.

    Every default is the original's behaviour, so ``Variant()`` reproduces it
    bit for bit; each field is documented in ``docs/DEVIATIONS.md``. A variant
    other than the default needs ``ATMConfig.strict_budget`` (the original's
    accounting is only reproduced, not extended).
    """

    # False: the incumbent that every sampler's first column repeats is not
    # evaluated again; its known cost is used and not charged. The search path
    # is the original's with those evaluations deleted, so the budget lasts
    # longer. Assumes a deterministic objective.
    reevaluate_incumbent: bool = True

    # How the last move decides whether a targeted step is taken at all.
    # ``"legacy"``: only if some coordinate of the move is positive (a move
    # with no positive coordinate, e.g. toward a negative orthant, loses its
    # targeted step); ``"any"``: if any coordinate is non-zero.
    targeted_gate: str = "legacy"
    # Length of the targeted step, as a multiple of the last move. ``"legacy"``:
    # ``max_random_search * exp(..) * U(0, 1) * exp(growth * rate - decay)``, which
    # is unbounded and driven by four hyperparameters; ``"loguniform"``:
    # ``exp(U(ln 0.1, ln 10))``, bounded and hyperparameter free.
    targeted_amplitude: str = "legacy"
    # The sawtooth that scales the random search restarts (a counter is reset to
    # zero) whenever the incumbent moved by more than ``significant_change``.
    # ``False``: the counter runs on, a pure scale scan; note that the restart
    # policy then fires after ``restart.max_random_growth`` iterations.
    reset_on_success: bool = True
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
    # ``"success_rate"``: one width per sampler, adapted by feedback (see
    # ``control``), no targeted step, and no use of the sampler
    # hyperparameters. ``"success_rate_targeted"``: the same plus a targeted step
    # of a log-uniform multiple (0.1 to 10) of the last move, taken for any
    # non-zero move. The two targeted-step flags above must stay default.
    amplitude_control: str = "legacy"
    # Controller: ``log_width += sr_eta * (success_rate - target)`` per sampler,
    # starting from ``sr_widths`` and capped at ``sr_max_width``.
    sr_eta: float = 1.5
    sr_targets: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    sr_widths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    sr_max_width: float = 5.0
    # A (1+1)-ES with the 1/5 rule on the first ``es_fraction`` of the budget
    # (after a uniform design of ``es_init_factor * dim`` points, at least 4, to
    # pick its start), then ATM from the ES incumbent with its width ladder scaled
    # to the ES step size. ``"prefix"`` needs ``bounds`` and a success-rate
    # controller; when it runs, the ES does its own design and ATM's uniform first
    # pool (``initial_design``) is skipped for that first search.
    es_phase: str = "none"
    es_fraction: float = 0.3
    es_init_factor: float = 1.0
    # The ES phase is only used for budgets of at most this many evaluations per
    # dimension; above it the search is ATM alone (the ES prefix helps up to about
    # 100 x dim and hurts beyond, where ATM's pool beats a greedy ES on multimodal
    # and ill-conditioned functions).
    es_max_budget_per_dim: float = 1e9
    # In dimensions up to ``es_small_dim`` (0: off) the limit is
    # ``es_small_max_budget_per_dim`` instead: there ATM's pool converges and
    # explores within a few dozen evaluations per dimension, and a local prefix
    # that takes 90% of a larger budget costs the multimodal functions (f15-f19 at
    # 100 x dim) more than it gains.
    es_small_dim: int = 0
    es_small_max_budget_per_dim: float = 30.0
    # Per-coordinate scales for the random search. ``"none"``: every coordinate
    # gets the same width (with a success-rate ``amplitude_control``). ``"diagonal"``:
    # the width of coordinate i is multiplied by a scale learned from the accepted
    # moves of the incumbent: a moving average (``scale_rate``) of the squared
    # relative displacement, normalised to geometric mean 1 and kept within
    # ``[1/scale_cap, scale_cap]``. Needs a success-rate ``amplitude_control``.
    scale_learning: str = "none"
    scale_rate: float = 0.15
    scale_cap: float = 100.0
    # Correlated search directions. ``"rank_mu"``: the random search is Gaussian
    # with a full covariance matrix (unit geometric mean, so the widths keep their
    # meaning) learned CMA-style from the best ``cov_select`` share of every
    # iteration's evaluated samples, weighted by log rank, with the rank-mu learning
    # rate times ``cov_rate``; the eigenvalue ratio is limited to ``cov_cap``. This
    # is the one variant with a DxD object. Needs a success-rate
    # ``amplitude_control`` and excludes ``scale_learning``.
    covariance: str = "none"
    cov_rate: float = 1.0
    cov_select: float = 0.25
    cov_cap: float = 1e6
    # What the covariance learns from. ``"displacement"``: each sample's offset from
    # the incumbent over its sampler's width (crossover and targeted step
    # included); ``"noise"``: the Gaussian noise actually drawn for it.
    cov_steps: str = "displacement"
    # A rank-one update with an evolution path (CMA's ``c1`` term), fed on the
    # iterations that improve the incumbent: ``"noise"``: by the winner's drawn
    # noise; ``"move"``: by the accepted move over the winner's width. ``"none"``:
    # rank-mu only.
    cov_path: str = "none"
    # Only used up to this dimension: the covariance is learned from a few samples
    # per iteration, which is too slow beyond about 10 dimensions (at 20 it costs
    # .03 at 300 x dim and gains nothing).
    cov_max_dim: int = 10
    # The local search of the ES prefix after its uniform design: ``"isotropic"``:
    # the (1+1)-ES with the 1/5 rule; ``"cma"``: the restarted (1+1)-Cholesky-
    # CMA-ES of ``cma_phase``; ``"lq2"``: IPOP population CMA-ES runs ranked by
    # the linear-quadratic surrogate of ``surrogate`` (active update, pycma's
    # style), and ``"cma"`` for budgets below ``es_lq_min_budget_per_dim`` times dim;
    # ``"slsqplq"``: restarted SLSQP (``qn_phase``) where a curvature probe reads
    # the function as smooth (cycles 11-12), lq2
    # where rugged (budgets >= ``es_lq_min_budget_per_dim``), else the (1+1), and the
    # (1+1) below 30 dim evaluations (cycle 12); ``"slsqplq10"``: the same, except
    # that it also runs below 30 dim evaluations once the budget reaches 200 (cycle 17;
    # ``qn_phase.QN_MIN_BUDGET``); ``"slsqplq10c"``: as ``"slsqplq10"`` with a
    # two-direction (12-evaluation) probe below 30 dim evaluations; ``"slsqplq10hk"``:
    # ``"slsqplq10h"``, and where that skips the probe (below 30 dim and below 200
    # evaluations) the median two-scale probe at the start point routes smooth
    # functions to SLSQP from there (cycle 26); ``"slsqplq10h"``:
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
    # Research only (cycle 40, the landscape router's labels): ``"smooth"`` /
    # ``"rugged"`` forces the ``cma_local_nonsep`` route's reading after its probe and
    # opens the route at every dim and separability verdict, so one run can be
    # finished both ways from the same hand-over state.
    local_force: str = "none"
    # ``fork_seed`` (research, cycle 52, the router dataset): >= 0 reseeds the random
    # stream at the hand-over, so one run is finished several ways from the same
    # hand-over state (with ``local_force``) on independent continuations.
    fork_seed: int = -1
    # Cycle 42 (``preplat``): above 30 x dim the ES prefix checks for a plateau after
    # its design and separable search; on one (f7) restarted CMA-ES populations take the
    # prefix's rest (c36 trails pycma IPOP on f7 at D 5, 100 x D by .33).
    prefix_plateau_pop: bool = False
    # Cycle 43 (``preplat2``): ``prefix_plateau_rugged`` runs the check only after the
    # prefix's curvature probe reads rugged (Batch 64's check cost f10-f14 a little);
    # ``prefix_plateau_min_dim``: no check below that dim.
    prefix_plateau_rugged: bool = False
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
    chain_min_per_dim: float = 300.0
    sep_route_min_per_dim: float = 100.0
    additive_min_dim: int = 5
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
    basin_max_dim: int = 3
    basin_min_per_dim: float = 100.0
    basin_descents: int = 0
    basin_descents_max_per_dim: float = 300.0
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
    small_engine_max_per_dim: float = 10.0
    small_engine_radius: float = 0.25
    # From a study of bbob f13 (sharp ridge): what follows the local
    # phase's capped rugged try (``QnRoute.rugged_try_per_dim``, D 5-9). ``"none"``:
    # the finisher's populations; ``"cut"``: a try still descending at its cap goes
    # on, the same scaled-FD BFGS descent, to the end of the budget; ``"cutkink"``:
    # only where its best point at the cap reads a kink (``qn_phase.kink_check``, 6
    # evaluations; f13's ridge).
    rugged_try_rest: str = "none"
    # The f13 toy: overrides of the route's try cap (``rugged_try_cap_per_dim`` x dim)
    # and of its budget envelope (``rugged_try_max_per_dim`` x dim); 0: the route's.
    rugged_try_cap_per_dim: float = 0.0
    rugged_try_max_per_dim: float = 0.0
    # The f13 toy: ``prefix_ridge_after`` > 0 reads the gated prefix's SLSQP descent
    # (``"slsqplq"`` / ``"slsqplqc"`` on a smooth reading) once, when its
    # ``StallMonitor`` has recorded that many iterations: a ridge reading
    # (``qn_phase.ridge_reading``: every full step overshoots by a similar factor and
    # one backtrack recovers it, as across f13's sharp ridge) hands the rest of the
    # prefix to restarted COBYQA (initial trust radius ``prefix_ridge_radius`` x box
    # width). 0: off.
    prefix_ridge_after: int = 0
    prefix_ridge_radius: float = 0.1
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
    small_switch_max_dim: int = 20
    small_switch_max_per_dim: float = 10.0
    small_switch_trials: float = 0.0
    small_switch_cos: float = 0.99
    small_switch_keep_unit: bool = False
    small_switch_radius: float = 0.25
    small_switch_rest: bool = True
    # Cycle 64 (from a study of the cells where SLSQP leads):
    # ``prefix_nan_reading`` says where an undefined curvature reading of the
    # ``slsqplq`` / ``slsqplqc`` prefix goes (NaN beta: fewer than two positive
    # mirrored losses, mostly probe pairs clipped at a box face): ``"rugged"`` (c43),
    # ``"smooth"``, or ``"plateau"`` (smooth unless one ``plateau_check`` then reads
    # a plateau, f7). On ids 1-3, 14 of f13's 24 readings at D 40, 100 x D were NaN.
    prefix_nan_reading: str = "rugged"
    # Research (from a screen of 132 ideas; 0 / False:
    # off, the composites unchanged):
    # * ``sampled_polish_per_dim`` (idea #33): in the multi-basin route's sampled
    #   restarts (``qn20``), a descent that ends normally below every earlier end is
    #   polished by SLSQP with central differences at 1e-4 x the box width, for at
    #   most this many x dim evaluations (``qn_phase.run_sampled_restarts``).
    # * ``qn_false_success`` (idea #23): an SLSQP descent that exits with status 0
    #   after at most ``FALSE_SUCCESS_MAX_ITER`` iterations without improving on its
    #   start point counts as failed (status -1) wherever statuses are read (the
    #   L-BFGS-B fallback's failure majority, the hill-valley test, the stage's
    #   extension and second-descent checks).
    # * ``qn_floor`` (ideas #24 / #32 / #55; ``"none"``: off): the first descent of
    #   every restarted-SLSQP call ends once it sits on its floor (its last
    #   10 (dim + 1) evaluations gained at most 1e-6 of its decrease); ``"bfgs"``: one
    #   scaled-FD BFGS descent (``box_qp``, steps tied to |f| and curvature),
    #   ``"cd"``: SLSQP with central differences at 1e-4 x the box width, continues
    #   from the best point within that call's evaluations.
    sampled_polish_per_dim: float = 0.0
    qn_false_success: bool = False
    qn_floor: str = "none"
    # Research (from a screen of new ideas):
    # ``probe_faces`` says how ``qn_phase.curvature_beta`` treats a probe centre near a
    # box face. ``"clip"`` (c44): probe points are clipped onto the box, so a pair is
    # no longer mirrored. ``"tangent"``: each random direction is first made feasible
    # at the largest probe scale (components that would leave the box are cut to the
    # room left, the others rescaled to unit length), so every pair stays mirrored;
    # the same draws, no extra evaluations. ``eval_memo``: an exactly repeated point
    # returns its stored value and is not charged (deterministic objectives only).
    probe_faces: str = "clip"
    eval_memo: bool = False
    # ``step_box_check`` (the novel-ideas toy): the separable search (STEP) runs its
    # separability check once right after probing the box ends, when the best point
    # has already moved along an axis other than the first (``run_step_phase``).
    step_box_check: bool = False
    # ``box_steps`` (the novel-ideas toy): the absolute steps in x (finite
    # differences, COBYQA's final radius, STEP's smallest interval, the scaled BFGS's
    # step range) scale with the box width / 10 (``qn_phase.X_UNIT``); on bbob's box
    # [-5, 5] nothing changes.
    box_steps: bool = False
    # ``box_normalize`` (the novel-ideas toy): ``minimize`` maps any box onto bbob's
    # [-5, 5]^D coordinate by coordinate and runs there (bbob's box: unchanged).
    box_normalize: bool = False
    # Research (from a second screen of parked ideas; all opt-in):
    # ``prefix_nan_min_per_dim``: ``prefix_nan_reading`` applies only from that budget
    # per dim (100: where the prefix's rugged kind is lq2 populations, not the (1+1)).
    # ``prefix_nan_clip``: an undefined reading goes that way only when clipping at the
    # box explains it (no unclipped probe pair with a negative loss).
    # ``prefix_nan_reprobe``: an undefined reading is taken again from the point moved
    # inward by the probe's largest scale; only a smooth second reading goes that way.
    # ``prefix_plateau_ties`` > 0: the prefix's plateau checks run only when its
    # separable search's probes tied the incumbent's value that often (f7's steps).
    # ``prefix_fractal_min_dim`` > 0: from that dim and ``prefix_fractal_min_per_dim``
    # x dim, a rugged prefix reading takes the centred fractal test and a fractal one
    # gets restarted SLSQP (the SLSQP-cells toy's ``fract20``).
    # ``prefix_descent_retries`` > 0: the one-descent prefixes (``"slsqpdes"`` outside
    # ``small_switch``, ``"slsqpnm"``) retry a descent that did not move from uniform
    # points, unless its first gradient was exactly zero (the toy's ``retry5``).
    # ``stage_extend_restart``: the hand-over's SLSQP stage lets the restart after an
    # early converged first descent run past the cap while it progresses (``stagerx``).
    prefix_nan_min_per_dim: float = 0.0
    prefix_nan_clip: bool = False
    prefix_nan_reprobe: bool = False
    prefix_plateau_ties: int = 0
    prefix_fractal_min_dim: int = 0
    prefix_fractal_min_per_dim: float = 0.0
    prefix_descent_retries: int = 0
    stage_extend_restart: bool = False
    # ``stage_extend_max_per_dim`` (0: no limit): ``stage_extend_restart`` only at
    # budgets up to that many x dim (Batch 89: at 1000 x D the extended restart took
    # the finisher's evaluations on f9 / f14 / f17 / f18 runs at D 10).
    stage_extend_max_per_dim: float = 0.0
    # Research (from a study of the cells AMM trailed): ``sep_route_low`` also
    # takes the pair test below ``additive_min_dim``, at dim >= ``sep_route_low_dim``
    # from ``sep_route_min_per_dim`` x dim: ``"route"`` sends a separable reading to
    # ``sep_route``'s restarted STEP (HCMA's STEP leads f3 at D 2); ``"strict"`` only
    # after a second separable reading at another random point (3 dim + 1 more
    # evaluations; at D 2 the test has one pair, which f7's plateaus can pass) and
    # never on a test whose pairs were all flat; ``"probe"`` reads and ignores it
    # (the perturbation control); never the chain branch there.
    sep_route_low: str = "off"
    sep_route_low_dim: int = 2
    # Research (gaps3 toy): on a hand-over curvature reading below
    # ``rugged_direct_beta`` (0: off) in the SLSQP routes' rugged branch, no capped
    # BFGS try and no first (1+1) in the finisher: its populations start at once (from
    # a uniform point, as after the (1+1)'s own rugged probe). On calibration ids f13
    # reads 1.24-1.60 (p10-p90) at D 5, 300 x D; f15, f17-f19, f23, f24 below 1.2.
    rugged_direct_beta: float = 0.0
    # ``rugged_direct_min_beta`` (0: none): only readings at or above it (rough
    # readings near 0, f19 / f23 / f24, keep the (1+1) first, which gained there).
    rugged_direct_min_beta: float = 0.0
    # Research (from a study of the cells AMM trailed): ``centre_point`` spends
    # the run's first evaluation on the centre of the box (a design-of-experiments
    # centre point; COCO's ``initial_solution``). ``"record"``: the rest of the run is
    # unchanged (one evaluation fewer) and the centre only counts as a candidate for
    # the best point; ``"start"``: also the start point of the run when it is better
    # than ``x0``. Off by default. Only from ``centre_point_min_dim`` (at D 2, 30 x dim
    # the evaluation it takes costs the smooth functions' last descent step).
    centre_point: str = "off"
    centre_point_min_dim: int = 2
    # Research (gaps4 toy): ``smooth_bfgs_second`` (0: off) in the SLSQP routes'
    # one-descent scaled-FD BFGS on a smooth hand-over reading (``qn44``, up to 300 x
    # dim): when that descent ended on its own (its line search found no Armijo point,
    # e.g. at Rosenbrock's second minimum) and at least this factor x its evaluations
    # are left, one more descent of the same engine from a uniform point, for at most
    # that many evaluations; the staged route then continues from the better end.
    smooth_bfgs_second: float = 0.0
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
    # What the hand-over's CMA phase does on stagnation. ``"uniform"``: restart
    # from a uniform point of the box; ``"atm"``: ATM resumes from the best point
    # (a new search with a uniform first pool) for max(the evaluations the CMA run
    # used, 50 dim), then the CMA phase restarts from ATM's best point.
    cma_restart: str = "uniform"
    # Quadratic model steps in the CMA phase (prefix and hand-over): every other
    # candidate is the minimiser of a least-squares quadratic fitted to the run's
    # recent points (``cma_phase.quadratic_step``), for dim up to
    # ``cma_model_max_dim``. ``"none"``: only random steps.
    cma_model: str = "none"
    cma_model_max_dim: int = 10
    # ``cma_model="lq"``: the lq-CMA-ES style model of ``cma_phase.lq_step``
    # (linear / diagonal / full rungs, rank-weighted fit of the best 75%, a
    # Kendall-tau gate), its full rung up to ``cma_model_max_dim`` and its diagonal
    # rung up to ``cma_lq_max_dim``; ``cma_model_sigma`` rescales sigma toward the
    # length of an accepted model step.
    cma_lq_max_dim: int = 20
    cma_model_sigma: bool = False
    # When a CMA run gives up (``cma_phase.run_cma``): ``"legacy"``: no progress
    # over 100 + 50 dim evaluations; ``"short"``: over 10 + 30 dim; ``"economy"``:
    # a run not holding the best point stops after 10 + 30 dim without progress or
    # once its sigma fell a thousandfold, the run holding it keeps the legacy rule.
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
    cma_handoff_cap_min_dim: int = 10
    # Research (idea #46, the 132-ideas stream's ``atm20`` at 300 x D): at budgets up
    # to ``cma_handoff_cap_low_max_per_dim`` x dim the cap is at most
    # ``cma_handoff_cap_low_per_dim`` x dim instead (0: off), from the same min_dim.
    cma_handoff_cap_low_per_dim: float = 0.0
    cma_handoff_cap_low_max_per_dim: float = 300.0
    # For dim >= 10, the ES prefix uses a design of 4 points (not dim) and its first
    # run starts at sigma0 times sqrt(10 / dim).
    es_lean_start: bool = False
    # Restart regimes of the hand-over's CMA phase (``regimes``): ``"none"``:
    # uniform (1+1) restarts; ``"bipop"``: after the first run, small (1+1) runs and
    # large non-elitist (mu/mu_w, lambda)-CMA-ES runs with doubling lambda, the
    # regime with fewer evaluations going next (Hansen 2009); ``"nbipop"``: the
    # regime holding the best point counts its evaluations at half (Loshchilov et
    # al. 2012). Needs ``cma_handoff`` and ``cma_restart="uniform"``.
    cma_regimes: str = "none"
    # Where the hand-over's CMA phase restarts: ``"uniform"``: a uniform point;
    # ``"mlsl"``: for dim up to ``cma_placement_max_dim``, the best evaluated point
    # (ATM's and CMA's) with no better point and no earlier run's endpoint within
    # GLOBAL's critical distance, at half that distance as sigma (uniform if none).
    cma_placement: str = "uniform"
    cma_placement_max_dim: int = 5
    # The hand-over's local phase for dim >= ``cma_local_min_dim``:
    # ``"one_plus_one"``: the restarted (1+1)-Cholesky-CMA-ES of ``cma_phase``;
    # ``"population"``: restarted (mu/mu_w, lambda)-CMA-ES runs of ``pop_cma``
    # (first from ATM's best point, then from uniform points, default lambda);
    # ``"ipop"``: the same with lambda doubled at every restart; ``"then_ipop"``:
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
    # ``"default"`` (doubling as the mode says), ``"cap4"`` (1, 2, 4 times the
    # default, then again from 1) or ``"fixed2"`` (twice the default);
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
    # search's check finds the function non-separable, the local phase is this
    # population kind (``"ipop"``: IPOP runs from the incumbent with the default
    # lambda) instead of ``cma_local`` (cycle 10: the surrogate local phase without
    # its losses on separable f4). ``"qn"``: probe the curvature
    # (``qn_phase.curvature_beta``) and run restarted SLSQP for the whole local
    # phase if beta >= 1.45, else ``cma_local`` (cycle 13). ``"qn2"``: a plateau check
    # first, SLSQP for a third of the local budget, then ``cma_local`` from its best.
    # ``"qn3"``: as ``"qn2"`` with the SLSQP stage capped at 100 dim evaluations and
    # the warm-start shape kept for the finisher. ``"qn5"``: as ``"qn3"`` with SLSQP's
    # restarts seeded from the best 5% of the global phase's archive (MLSL-style).
    # ``"qn6"``: as ``"qn3"``, but rugged or plateau readings get ``steps``
    # populations; smooth functions whose converged SLSQP descents end in separate
    # basins (hill-valley test) keep restarting SLSQP instead of the CMA finisher,
    # and a stage that ends mid-descent continues for 2/3 of what is left before
    # the finisher (cycle 16). ``"qn7"``: as ``"qn6"`` with that continuation
    # capped at another 100 dim evaluations. ``"qn8"``: as ``"qn7"``, but a plateau
    # reading (f7) launches IPOP populations from a uniform point with 4x the default
    # lambda instead (cycle 18). ``"qn9"``: ``"qn8"`` with SLSQP's finite-difference
    # step 1e-9 instead of SciPy's 1.49e-8 (cycle 20, ``qn_phase.QN_FD_STEP``).
    # ``"qn10"``: ``"qn9"``, but instead of a second capped stage the first descent goes
    # on past the cap while ``qn_phase.StallMonitor`` reads it as progressing,
    # keeping a reserve for the CMA finisher (cycle 21). ``"qn11"``: ``"qn9"`` whose
    # rugged-route populations double from 4x per restart while at least 2000 dim
    # evaluations are left (``lambda_rule="stepsgrow"``, cycle 22). ``"qn12"``:
    # ``"qn9"`` with SciPy's default step in the multi-basin route's restarts (cycle
    # 24: the 1e-9 step makes each Gallagher descent dearer, so fewer restarts fit).
    # ``"qn13"`` / ``"qn13n"``: ``"qn12"``, but a stage where most SLSQP descents fail
    # (f12's signature) hands over to population CMA-ES restarts, ranked by the lq2
    # surrogate (``"qn13"``) or plain (``"qn13n"``) (cycle 25).
    cma_local_nonsep: str = "none"
    cma_local_nonsep_min_dim: int = 5
    # With ``pop_gate_per_dim``: the first (1+1) run also hands over to the
    # population after this many times dim evaluations without a stall (0: only at
    # a stall); a protected population run that fails returns to the (1+1) at
    # the best point with the run's last width scale.
    pop_switch_per_dim: float = 0.0
    # Where a population run launched at the switch cap starts: ``"uniform"``, or
    # ``"incumbent"``: the best point, at twice the (1+1)'s width scale.
    pop_switch_start: str = "uniform"
    pop_switch_min_dim: int = 1
    # ``pop_switch_start="carry"``: as "incumbent", with the (1+1)'s covariance shape
    # and its width scale (not twice it).
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
    # (``nm_stage``); ``"qnprefix"``: in the prefix only, after one SLSQP descent
    # (cycle 17: at D = 2, restarted SLSQP led on f1, f2 and f10-f14 at 10-30 x D);
    # ``"qnprefix30"``: the descent only for budgets up to 30 dim (Batch 43);
    # ``"qnprefix30g"``: as ``"qnprefix30"``, and from 200 evaluations the two-direction
    # curvature probe sends smooth functions to restarted SLSQP (cycle 19).
    nm_max_dim: int = 0
    nm_stage: str = "both"
    sep_probe: bool = False
    # ``sep_parabolic``: the separable search alternates parabolic steps with STEP's
    # midpoints (``step_phase``, cycle 14; f2 at 30 x D).
    sep_parabolic: bool = False
    # With ``sep_parabolic``: parabolic steps only after the separability check has
    # passed once (cycle 14, Batch 38).
    sep_parabolic_late: bool = False
    # ``"prefix"``: parabolic steps only in the prefix's separable search (Batch 38's
    # losses were in the hand-over's).
    sep_parabolic_scope: str = "both"
    # The separable search's check: ``"axis"`` re-evaluates a stored point of one
    # axis; ``"pair"`` moves two axes at once and tests additivity, which also fails
    # when STEP is stuck (cycle 23, ``step_phase``); ``"stall"``: the axis check,
    # and the pair test only after two sweeps without improvement, when re-evaluating
    # a stored point at an unchanged best can no longer fail (cycle 29).
    sep_check: str = "axis"
    # ``sep_stall_min_per_dim`` (cycle 54, the late-router study): ``"stall"`` acts
    # only from this many x dim (below it the axis check alone, as ``"axis"``).
    sep_stall_min_per_dim: float = 0.0
    # ``"lbfgsb"``: every restarted-SLSQP call (prefix and local phase) hands its
    # remaining evaluations to restarted L-BFGS-B once failed descents outnumber
    # converged ones, and a local stage that ends that way gives L-BFGS-B the rest of
    # the budget (f12, bent cigar; cycle 28, ``qn_phase.run_lbfgsb_restarts``).
    qn_fallback: str = "none"
    sep_share: float = 0.5
    sep_min_budget_per_dim: float = 30.0
    # Units of ``sr_widths`` and ``sr_max_width``. ``"absolute"``: as given (they
    # were tuned on the bbob box [-5, 5]); ``"box"``: multiples of a tenth of the
    # mean box width, so that the search does not depend on the scale of x.
    # ``"box"`` needs ``bounds``.
    width_units: str = "absolute"
    # Where a sample's random part is centred. ``"legacy"``: a coordinate-wise
    # crossover of the incumbent with the second best or the best eighth of the
    # last batch; ``"none"``: the incumbent itself.
    crossover: str = "legacy"
    # Distribution of the random part. ``"uniform"``: uniform in the cube of the
    # width; ``"gaussian"``: Gaussian with the same standard deviation (width over
    # sqrt 3). ``covariance`` is always Gaussian.
    noise: str = "uniform"
    # With a success-rate ``amplitude_control``: do not move the width of a sampler
    # that had no sample of its own this iteration (only the incumbent's column),
    # whose success rate is then zero by construction.
    skip_starved: bool = False
    # With a success-rate ``amplitude_control``: ``"per_sampler"``: four widths,
    # each moved toward its own target; ``"shared"``: one width for all samplers,
    # moved by the success rate of the whole pool toward ``sr_shared_target``;
    # ``"ladder"``: one such width sigma, sampler k searching at sigma times
    # (1/4, 1/2, 1, 2) (``control.LADDER``), starting at the original widths.
    width_mode: str = "per_sampler"
    sr_shared_target: float = 0.15
    # What ends a search early and where the next one starts. ``"legacy"``: the
    # original policy only (``ATMConfig.restart``), next search at the incumbent;
    # under a success-rate controller it practically never fires. The other modes
    # add a stagnation test (``restart.stagnated``: the incumbent's cost moved
    # by at most ``stag_tolfun`` over ``10 + ceil(30 dim / pool)`` iterations, or
    # every width is at most ``stag_tolx`` times the mean box width) and restart
    # ``"incumbent"``: at the incumbent; ``"uniform"``: at a uniform point of the
    # box (the best point found is kept); ``"ipop"``: like ``"uniform"`` with the
    # pool multiplied by ``ipop_factor`` at every restart, up to ``ipop_max_pool``.
    # Needs a success-rate ``amplitude_control``; ``"uniform"`` and ``"ipop"``
    # need ``bounds``.
    restart_mode: str = "legacy"
    stag_tolfun: float = 1e-11
    stag_tolx: float = 1e-12
    ipop_factor: float = 2.0
    ipop_max_pool: int = 1024
    # The local search of the ES phase and of the late refinement (``refactor`` line,
    # E12-E13): ``"es"`` is the (1+1)-ES, ``"nelder_mead"`` the restarted in-package
    # simplex search of ``simplex`` for dimensions up to ``simplex_max_dim`` in
    # the ES phase and up to ``nm_refine_max_dim`` in the late refinement (the ES
    # above them). The simplex is a poor start in more than about 4 dimensions but
    # refines ATM's best point well up to 10. ``simplex_max_dim`` was called
    # ``nm_max_dim`` on ``refactor`` until the 2026-10-19 merge, where that name was
    # already the research line's SciPy Nelder-Mead limit (below).
    local_method: str = "es"
    simplex_max_dim: int = 4
    nm_refine_max_dim: int = 10
    # Late refinement: the last ``refine_fraction`` of the budget (0: off) goes to a
    # (1+1)-ES from the best point ATM found, with its step size set from ATM's
    # final widths (a final-precision phase; the 2019 paper's search has none).
    # Used for budgets of at least ``refine_min_budget_per_dim`` x dim. Needs
    # ``bounds`` and a success-rate ``amplitude_control``.
    refine_fraction: float = 0.0
    refine_min_budget_per_dim: float = 0.0
    # Only used up to this dimension (the (1+1)-ES tail above ``nm_refine_max_dim``
    # takes half of the budget from ATM and does not make it up).
    refine_max_dim: int = 1_000_000
    # Samples per iteration. ``"fixed"``: 64 at the start and 60 afterwards
    # (``initial_population`` and the conductor's total), whatever the budget.
    # ``"budget"``: ``clip(budget / pool_divisor, pool_min, pool_max)`` rounded to
    # a multiple of 4, for the start, the conductor and the iteration count
    # together (they are three separate numbers in the original). Small budgets
    # then get more iterations out of their few evaluations.
    pool_rule: str = "fixed"
    pool_min: int = 24
    pool_max: int = 64
    pool_divisor: int = 30
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
        for name in _BOOLS:
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        for name, allowed in _CHOICES.items():
            _choose(name, getattr(self, name), allowed)
        self._check_amplitude_control()
        for name, valid, message in _RANGES:
            value = getattr(self, name)
            if not valid(value):
                raise ValueError(message.format(v=value))
        if self.simplex_max_dim < 1 or self.nm_refine_max_dim < 1:
            raise ValueError("simplex_max_dim and nm_refine_max_dim must be at least 1")
        if self.refine_fraction and self.cma_handoff:
            # Two different ends of the budget: the late refinement keeps it back,
            # the hand-over spends it on the CMA phase.
            raise ValueError("refine_fraction and cma_handoff exclude each other")
        if not 2 * N_SAMPLERS <= self.pool_min <= self.pool_max:
            raise ValueError(
                f"need {2 * N_SAMPLERS} <= pool_min <= pool_max, "
                f"got {self.pool_min}, {self.pool_max}"
            )
        for name, active in _NEEDS_SUCCESS_RATE.items():
            if active(getattr(self, name)) and self.amplitude_control == "legacy":
                raise ValueError(f"{name} needs a success-rate amplitude_control")
        _choose(
            "restart_mode",
            self.restart_mode,
            ("legacy", "incumbent", "uniform", "ipop"),
        )
        _choose("cov_steps", self.cov_steps, ("displacement", "noise"))
        _choose("cov_path", self.cov_path, ("none", "noise", "move"))
        if self.covariance == "none" and (
            self.cov_steps != "displacement" or self.cov_path != "none"
        ):
            raise ValueError("cov_steps and cov_path need covariance='rank_mu'")
        _choose(
            "es_kind",
            self.es_kind,
            (
                "isotropic",
                "cma",
                "lq2",
                "slsqplq",
                "slsqplq10",
                "slsqplq10c",
                "slsqplq10h",
                "slsqplq10hk",
                "slsqplq10hd",
            ),
        )
        _choose(
            "cma_local_nonsep",
            self.cma_local_nonsep,
            (
                "none",
                "ipop",
                "population",
                "qn",
                "qn2",
                "qn3",
                "qn5",
                "qn6",
                "qn7",
                "qn8",
                "qn9",
                "qn10",
                "qn11",
                "qn12",
                "qn13",
                "qn13n",
                "qn15",
                "qn17",
                "qn19",
                "qn20",
                "qn29",
                "qn38",
                "qn44",
            ),
        )
        _choose(
            "nm_stage",
            self.nm_stage,
            (
                "both",
                "prefix",
                "qnprefix",
                "qnprefix30",
                "qnprefix30g",
            ),
        )
        if self.nm_max_dim < 0:
            raise ValueError("nm_max_dim must be >= 0")
        _choose("cma_restart", self.cma_restart, ("uniform", "atm"))
        _choose("cma_model", self.cma_model, ("none", "quadratic", "lq"))
        if self.cma_model_max_dim < 1 or self.cma_lq_max_dim < 1:
            raise ValueError("cma_model_max_dim and cma_lq_max_dim must be >= 1")
        _choose(
            "cma_restart_rule", self.cma_restart_rule, ("legacy", "short", "economy")
        )
        _choose("cma_regimes", self.cma_regimes, ("none", "bipop", "nbipop"))
        _choose("cma_placement", self.cma_placement, ("uniform", "mlsl"))
        _choose(
            "cma_local",
            self.cma_local,
            ("one_plus_one", "population", "ipop", "then_ipop"),
        )
        if self.cma_local_min_dim < 1:
            raise ValueError("cma_local_min_dim must be at least 1")
        _choose("cma_model_high", self.cma_model_high, ("none", "diagonal"))
        _choose("sep_parabolic_scope", self.sep_parabolic_scope, ("both", "prefix"))
        _choose("sep_check", self.sep_check, ("axis", "pair", "stall"))
        _choose("qn_fallback", self.qn_fallback, ("none", "lbfgsb"))
        _choose("local_force", self.local_force, ("none", "smooth", "rugged"))
        _choose(
            "pop_lambda",
            self.pop_lambda,
            ("default", "cap4", "fixed2", "budget", "steps", "ipop4", "stepsgrow"),
        )
        if not 0 < self.sep_share < 1:
            raise ValueError("sep_share must lie in (0, 1)")
        _choose(
            "pop_switch_start", self.pop_switch_start, ("uniform", "incumbent", "carry")
        )
        if self.pop_switch_per_dim < 0:
            raise ValueError("pop_switch_per_dim must be >= 0")
        if self.pop_gate_per_dim < 0:
            raise ValueError("pop_gate_per_dim must be >= 0")
        if self.pop_gate_per_dim and self.cma_local != "then_ipop":
            raise ValueError("pop_gate_per_dim needs cma_local='then_ipop'")
        if self.cma_placement_max_dim < 1:
            raise ValueError("cma_placement_max_dim must be at least 1")
        if self.cma_regimes != "none" and (
            not self.cma_handoff or self.cma_restart != "uniform"
        ):
            raise ValueError("cma_regimes needs cma_handoff and cma_restart='uniform'")
        for name in (
            "cma_model_sigma",
            "cma_active",
            "cma_warm_start",
            "es_lean_start",
            "pop_active",
            "pop_protect",
            "es_lq_nonsep",
            "prefix_sep_probe",
            "prefix_plateau_pop",
            "prefix_plateau_rugged",
            "prefix_nan_clip",
            "prefix_nan_reprobe",
            "stage_extend_restart",
            "chain_route",
            "sep_route",
            "basin_route",
            "sep_parabolic_late",
            "sep_parabolic",
            "sep_probe",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if self.sep_stall_min_per_dim < 0:
            raise ValueError("sep_stall_min_per_dim must be >= 0")
        if self.basin_max_dim < 2 or self.basin_min_per_dim < 0:
            raise ValueError("need basin_max_dim >= 2 and basin_min_per_dim >= 0")
        _choose("small_engine", self.small_engine, ("none", "cobyqa"))
        _choose("rugged_try_rest", self.rugged_try_rest, ("none", "cut", "cutkink"))
        _choose(
            "prefix_nan_reading",
            self.prefix_nan_reading,
            ("rugged", "smooth", "plateau"),
        )
        if self.rugged_try_cap_per_dim < 0 or self.rugged_try_max_per_dim < 0:
            raise ValueError("the rugged try overrides must not be negative")
        if (
            self.prefix_ridge_after < 0
            or self.prefix_ridge_max_per_dim < 0
            or not (0 < self.prefix_ridge_radius <= 1)
        ):
            raise ValueError(
                "need prefix_ridge_after >= 0, prefix_ridge_max_per_dim >= 0 and "
                "prefix_ridge_radius in (0, 1]"
            )
        _choose("probe_faces", self.probe_faces, ("clip", "tangent"))
        for name in ("eval_memo", "step_box_check", "box_steps", "box_normalize"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if (
            min(self.prefix_nan_min_per_dim, self.prefix_fractal_min_per_dim) < 0
            or min(
                self.prefix_plateau_ties,
                self.prefix_fractal_min_dim,
                self.prefix_descent_retries,
            )
            < 0
        ):
            raise ValueError("the parked2 prefix options must not be negative")
        _choose(
            "sep_route_low", self.sep_route_low, ("off", "route", "strict", "probe")
        )
        if self.sep_route_low_dim < 2:
            raise ValueError("sep_route_low_dim must be >= 2")
        if self.rugged_direct_beta < 0 or self.rugged_direct_min_beta < 0:
            raise ValueError("rugged_direct_beta and _min_beta must be >= 0")
        _choose("anytime", self.anytime, ("off", "replay"))
        if (
            not self.anytime_start_per_dim > 0
            or not self.anytime_factor > 1
            or self.anytime_switch < 1
        ):
            raise ValueError(
                "anytime_start_per_dim must be > 0, anytime_factor > 1 and "
                "anytime_switch >= 1"
            )
        _choose("centre_point", self.centre_point, ("off", "record", "start"))
        if self.centre_point_min_dim < 1:
            raise ValueError("centre_point_min_dim must be >= 1")
        if (
            self.smooth_bfgs_second < 0
            or self.prefix_low_beta < 0
            or self.prefix_low_beta_min_dim < 0
            or self.stage_extend_max_per_dim < 0
        ):
            raise ValueError(
                "smooth_bfgs_second, prefix_low_beta* and stage_extend_max_per_dim "
                "must be >= 0"
            )
        if (
            self.small_engine_max_dim < 1
            or self.small_engine_max_per_dim < 0
            or not (0 < self.small_engine_radius <= 1)
        ):
            raise ValueError(
                "need small_engine_max_dim >= 1, small_engine_max_per_dim >= 0 and "
                "small_engine_radius in (0, 1]"
            )
        if self.basin_descents < 0 or self.basin_descents_max_per_dim < 0:
            raise ValueError(
                "need basin_descents >= 0 and basin_descents_max_per_dim >= 0"
            )
        if (
            self.small_switch_max_dim < 1
            or self.small_switch_max_per_dim < 0
            or self.small_switch_trials < 0
            or not (0 <= self.small_switch_cos <= 1)
            or not (0 < self.small_switch_radius <= 1)
        ):
            raise ValueError(
                "need small_switch_max_dim >= 1, small_switch_max_per_dim >= 0, "
                "small_switch_trials >= 0, small_switch_cos in [0, 1] and "
                "small_switch_radius in (0, 1]"
            )
        if self.fork_seed < -1:
            raise ValueError(f"fork_seed must be >= -1, got {self.fork_seed}")
        if self.sampled_polish_per_dim < 0:
            raise ValueError("sampled_polish_per_dim must be >= 0")
        if not isinstance(self.qn_false_success, bool):
            raise ValueError("qn_false_success must be a bool")
        _choose("qn_floor", self.qn_floor, ("none", "bfgs", "cd"))
        if (
            min(self.chain_min_per_dim, self.sep_route_min_per_dim) < 0
            or self.additive_min_dim < 3
        ):
            raise ValueError("need *_min_per_dim >= 0 and additive_min_dim >= 3")
        if (
            min(self.prefix_full_per_dim, self.prefix_full_min_per_dim) < 0
            or self.des_max_per_dim < 0
        ):
            raise ValueError("prefix_full_*_per_dim and des_max_per_dim must be >= 0")
        if self.cma_handoff_cap_per_dim < 0 or self.cma_handoff_cap_min_dim < 1:
            raise ValueError("need cma_handoff_cap_per_dim >= 0 and min_dim >= 1")
        if (
            min(self.cma_handoff_cap_low_per_dim, self.cma_handoff_cap_low_max_per_dim)
            < 0
        ):
            raise ValueError("cma_handoff_cap_low_* must be >= 0")
        if not 0 <= self.cma_handoff < 1:
            raise ValueError(f"cma_handoff must be in [0, 1), got {self.cma_handoff}")
        if self.cma_handoff and self.amplitude_control == "legacy":
            raise ValueError("cma_handoff needs a success-rate amplitude_control")
        _choose("width_units", self.width_units, ("absolute", "box"))
        _choose("crossover", self.crossover, ("legacy", "none"))
        _choose("noise", self.noise, ("uniform", "gaussian"))
        _choose("width_mode", self.width_mode, ("per_sampler", "shared", "ladder"))
        if not isinstance(self.skip_starved, bool):
            raise ValueError("skip_starved must be a bool")
        if not 0 < self.sr_shared_target < 1:
            raise ValueError("sr_shared_target must lie in (0, 1)")
        if self.amplitude_control == "legacy" and (
            self.skip_starved or self.width_mode != "per_sampler"
        ):
            raise ValueError(
                "skip_starved and width_mode need a success-rate amplitude_control"
            )
        if self.restart_mode != "legacy" and self.amplitude_control == "legacy":
            raise ValueError("restart_mode needs a success-rate amplitude_control")
        if self.stag_tolfun < 0 or self.stag_tolx <= 0:
            raise ValueError("stag_tolfun must be >= 0 and stag_tolx > 0")
        if self.ipop_factor < 1 or self.ipop_max_pool < 2 * N_SAMPLERS:
            raise ValueError(
                f"need ipop_factor >= 1 and ipop_max_pool >= {2 * N_SAMPLERS}"
            )
        if self.covariance != "none" and (
            self.amplitude_control == "legacy" or self.scale_learning != "none"
        ):
            raise ValueError(
                "covariance needs a success-rate amplitude_control "
                "and no scale_learning"
            )
        if self.corner_clip and self.bounds != "fold":
            raise ValueError("corner_clip needs bounds='fold'")

    def _check_amplitude_control(self) -> None:
        # JSON round trips give lists; the fields are tuples.
        for name in ("sr_targets", "sr_widths"):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))
        if len(self.sr_targets) != N_SAMPLERS or len(self.sr_widths) != N_SAMPLERS:
            raise ValueError(f"sr_targets and sr_widths need {N_SAMPLERS} values each")
        if not all(0 < t < 1 for t in self.sr_targets):
            raise ValueError("sr_targets must lie in (0, 1)")
        if self.sr_eta <= 0 or self.sr_max_width <= 0:
            raise ValueError("sr_eta and sr_max_width must be positive")
        if not all(0 < w <= self.sr_max_width for w in self.sr_widths):
            raise ValueError("sr_widths must lie in (0, sr_max_width]")
        if self.amplitude_control != "legacy" and (
            self.targeted_gate != "legacy"
            or self.targeted_amplitude != "legacy"
            or not self.reset_on_success
        ):
            raise ValueError(
                "targeted_gate, targeted_amplitude and reset_on_success belong to "
                "the original sampler; amplitude_control replaces it"
            )

    @property
    def needs_box(self) -> bool:
        """Whether ``minimize`` must be given ``bounds=(lower, upper)``."""
        return (
            self.initial_design != "legacy"
            or self.bounds != "none"
            or self.es_phase != "none"
            or self.restart_mode in ("uniform", "ipop")
            or self.cma_handoff > 0
            or self.width_units == "box"
            or self.refine_fraction > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Variant:
        _reject_unknown(cls, data)
        return cls(**data)


@dataclass(frozen=True)
class ATMConfig:
    """Tunable structure of the optimizer.

    The structural defaults are the original's; ``strict_budget=False`` also
    restores the original's budget accounting.
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
        """The five-option composite, confirmed on three independent holdouts.

        Success-rate amplitude control with a targeted step, a uniform first
        pool, bounds folded into the box with a 25 % corner clip, a pool sized
        from the budget (at least 12) and no re-evaluation of the incumbent. It
        stays within ATM's own sampler. Gains over the defaults on untouched
        bbob instances: +.101, +.097 and +.099 on three held-out sets of COCO
        instances; the gain shrinks
        with dimension (+.033 to +.038 at D = 40).

        It uses the search box, so call ``minimize(..., bounds=(lower, upper))``.
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
        """Every field, defaults included, as plain Python containers."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ATMConfig:
        """Inverse of :meth:`to_dict`; unknown keys raise ``ValueError``."""
        data = dict(data)
        _reject_unknown(cls, data)
        nested: dict[str, Any] = {
            "conductor": ConductorConfig,
            "restart": RestartPolicy,
            "variant": Variant,
        }
        for name, kind in nested.items():
            if name in data:
                _reject_unknown(kind, data[name])
                data[name] = kind(**data[name])
        return cls(**data)

    def config_hash(self) -> str:
        """Short stable fingerprint of :meth:`to_dict` (for result provenance).

        The ``Variant`` fields of ``HASH_IF_SET`` count only when they differ from
        their defaults, so adding them left every recorded hash as it was (the
        frozen ``c46`` is still ``2153948ebed51150``).
        """
        data = self.to_dict()
        defaults = Variant()
        for name in HASH_IF_SET:
            if getattr(self.variant, name) == getattr(defaults, name):
                del data["variant"][name]
        text = json.dumps(data, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16]
