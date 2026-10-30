"""Public entry point, and the routing between the phases of the composites.

:func:`minimize` runs, in this order (each step only when its ``Variant`` options
ask for it; the default ``ATMConfig()`` runs ATM alone, as the original):

1. **The prefix** (``es_phase="prefix"``, budgets up to ``es_max_budget_per_dim``):
   ``es_phase.run_es_phase`` with the local search of :func:`_prefix_kind`
   (the (1+1)-CMA-ES, SLSQP where the curvature probe reads smooth, lq2
   populations, Nelder-Mead at very low D, ...). ATM then continues from its best.
   With ``qn_fallback="lbfgsb"`` an SLSQP call whose failed descents
   outnumber its converged ones hands the rest to restarted L-BFGS-B.
2. **ATM** (``search.Search``), restarted by its restart policy; with
   ``cma_handoff`` only for its share of the budget.
3. **The hand-over** (``cma_handoff``, when the prefix did not run): the separable
   search (``sep_probe``), then :func:`_local_phase`.
4. **The local phase** (:func:`_local_phase`): Nelder-Mead at very low D, the
   SLSQP routes of ``cma_local_nonsep`` (:func:`_qn_local_phase`, configured by
   ``qn_phase.QN_ROUTES``) on non-separable functions: a smooth reading runs an
   SLSQP stage (L-BFGS-B for the rest when most descents failed), a plateau
   IPOP populations (``"qn8"`` on), a rugged reading the finisher's ``steps``
   populations, or from D 20 with enough budget unprotected IPOP from a uniform
   point (``"qn17"`` on). Then one finisher:
   :func:`_finish_gated_then_ipop` (the earliest research presets),
   :func:`_finish_then_ipop` or :func:`_finish_one_plus_one`.

The hand-over state that these functions share is a :class:`_Handover`. The phase
functions are looked up as this module's globals at call time (``run_cma``,
``run_population_restarts``, ``run_es_phase``) and ``qn_phase``'s functions and
constants as attributes of that module, so tests can monkeypatch either.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import basin_phase, chain_phase, cobyqa_phase, qn_phase
from .anytime import Warm
from .bounds import Box
from .cma_phase import run_cma, warm_factor
from .conductor import ConductorConfig, ResourceConductor
from .config import N_SAMPLERS, ATMConfig, Variant
from .es_phase import (
    INITIAL_SIGMA_SHARE,
    EsResult,
    run_es_phase,
)
from .objective import CountedObjective, Watch, _Stop, restarts_on_stop
from .plan import iteration_limit, resolve_config
from .pop_cma import run_population_restarts
from .profiling import Profiler, resolve_profiler
from .qn_phase import (
    QN_FD_STEP,
    QN_MIN_BUDGET,
    QN_MIN_BUDGET_PER_DIM,
    Descent,
    QnRoute,
)
from .result import BUDGET, INTERRUPTED, MESSAGES, OptimizeResult
from .search import Search

# Routing thresholds and fixed settings of the phases (no shipped configuration
# changed them; until 0.7.0 they were ``Variant`` fields of the same name).
# The prefix's uniform design: ``ES_INIT_FACTOR`` x dim points, at least 4.
ES_INIT_FACTOR = 1.0
# The pair test's routes: from this dim, ``chain_route`` from CHAIN_MIN_PER_DIM x dim
# and ``sep_route`` from SEP_ROUTE_MIN_PER_DIM x dim.
ADDITIVE_MIN_DIM = 5
CHAIN_MIN_PER_DIM = 300.0
SEP_ROUTE_MIN_PER_DIM = 100.0
# ``basin_route`` runs in dim 2 to BASIN_MAX_DIM from BASIN_MIN_PER_DIM x dim; its
# ``basin_descents`` only up to BASIN_DESCENTS_MAX_PER_DIM x dim.
BASIN_MAX_DIM = 3
BASIN_MIN_PER_DIM = 100.0
BASIN_DESCENTS_MAX_PER_DIM = 300.0
# ``small_engine``: COBYQA for budgets up to this many x dim, initial trust radius
# SMALL_ENGINE_RADIUS x box width.
SMALL_ENGINE_MAX_PER_DIM = 10.0
SMALL_ENGINE_RADIUS = 0.25
# ``prefix_ridge_after``: the restarted COBYQA's initial trust radius x box width.
PREFIX_RIDGE_RADIUS = 0.1
# ``small_switch``: in up to SMALL_SWITCH_MAX_DIM dims, at budgets up to
# SMALL_SWITCH_MAX_PER_DIM x dim; the gradients' parallel test |cos| <
# SMALL_SWITCH_COS; COBYQA's radius; COBYQA also instead of the prefix's (1+1).
SMALL_SWITCH_MAX_DIM = 20
SMALL_SWITCH_MAX_PER_DIM = 10.0
SMALL_SWITCH_COS = 0.99
SMALL_SWITCH_RADIUS = 0.25
SMALL_SWITCH_REST = True
# The CMA phase's full quadratic model up to this dim (``cma_model_high`` above it).
CMA_MODEL_MAX_DIM = 10
# ``cma_handoff_cap_per_dim`` caps ATM's share of the hand-over from this dim.
CMA_HANDOFF_CAP_MIN_DIM = 10
# ``cma_local_nonsep`` acts from this dim.
CMA_LOCAL_NONSEP_MIN_DIM = 5
# ``sep_probe``: the separable search's share of the remaining budget, and the
# smallest budget (x dim) it runs at.
SEP_SHARE = 0.5
SEP_MIN_BUDGET_PER_DIM = 30.0


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    budget: int,
    *,
    seed: int | np.random.SeedSequence | None = None,
    rng: np.random.RandomState | np.random.Generator | None = None,
    config: ATMConfig | str | None = None,
    profile: bool | Profiler = False,
    bounds: tuple[object, object] | None = None,
    callback: Callable[[OptimizeResult], object] | None = None,
    ftarget: float | None = None,
    catch_interrupt: bool = False,
) -> OptimizeResult:
    """Minimise ``fun`` starting from ``x0`` with about ``budget`` evaluations.

    The engine behind :func:`amm_optimizer.minimize`, which documents every
    argument. It takes ``x0`` and ``budget`` explicitly, reads ``bounds`` only as
    ``(lower, upper)`` and does not check ``x0`` against the box (the public
    function clips it onto the box, with a ``UserWarning``).

    Pass ``seed`` for a reproducible run, or an existing ``RandomState`` as
    ``rng`` to continue its stream (this is how results are compared against
    the original code, which used NumPy's global state); a ``Generator`` as
    ``rng`` gives the run a seed drawn from it.

    ``bounds=(lower, upper)`` (scalars or one value per coordinate) tells the
    optimizer the search box. It is not enforced and changes nothing by itself;
    only variants that use the box (``initial_design``, ``bounds``) read it, and
    they raise ``ValueError`` without it.

    ``config`` defaults to the original algorithm (``ATMConfig()``), unlike the
    public function; a string is a preset name (``presets.get``). The module
    docstring lists the phases a composite runs. With ``config.variant.anytime``
    set, the run never learns ``budget``, which then only caps the calls
    (:mod:`.anytime`).
    """
    if isinstance(config, str):
        from . import presets

        config = presets.get(config)
    seed, rng = _resolve_seed(seed, rng)
    profiler = resolve_profiler(profile)
    watch = Watch(fun, callback, ftarget, keep=catch_interrupt)
    try:
        result = _minimize(
            watch,
            x0,
            budget,
            seed=seed,
            rng=rng,
            config=config,
            profile=profiler,
            bounds=bounds,
            warm=None,
        )
        status = BUDGET
        if not watch.improvements:  # every value failed: no point beats another
            result.fun = float("inf")
    except _Stop as stop:
        if stop.watch is not watch:  # pragma: no cover  (another run's stop)
            raise
        result = _stopped(watch, stop.nrestarts, profiler)
        status = stop.status
    except KeyboardInterrupt:
        if not catch_interrupt or watch.best_x is None:
            raise
        warnings.warn(
            "minimize was interrupted; returning the best point so far",
            RuntimeWarning,
            stacklevel=2,
        )
        result = _stopped(watch, 0, profiler)
        status = INTERRUPTED
    result.update(
        nit=watch.improvements,
        success=status != INTERRUPTED,
        status=status,
        message=MESSAGES[status],
        seed=seed,
    )
    return result


def _stopped(watch: Watch, nrestarts: int, profiler: Profiler) -> OptimizeResult:
    """The result of a run that ended early: the best point ``watch`` saw."""
    assert watch.best_x is not None  # a stop follows an improvement
    return OptimizeResult(
        x=watch.best_x,
        fun=watch.best_f,
        nfev=watch.calls,
        nrestarts=nrestarts,
        timings=profiler.report(),
    )


def _resolve_seed(
    seed: int | np.random.SeedSequence | None,
    rng: np.random.RandomState | np.random.Generator | None,
) -> tuple[int | None, np.random.RandomState | None]:
    """``(seed, rng)`` for :func:`_minimize`: an integer seed in ``[0, 2**32)`` for
    every run that does not continue a ``RandomState``.

    A ``Generator`` gives ``rng.integers(2**32)`` (one draw from it, an independent
    stream); a ``SeedSequence`` its first 32-bit word of state; ``None`` a fresh
    seed from the operating system's entropy, so every run can be repeated from
    ``result.seed``.
    """
    if isinstance(rng, np.random.RandomState):
        return None, rng
    if isinstance(rng, np.random.Generator):
        return int(rng.integers(2**32)), None
    if isinstance(seed, np.random.SeedSequence):
        return int(seed.generate_state(1, dtype=np.uint32)[0]), None
    if seed is None:
        entropy = np.random.SeedSequence().entropy
        assert isinstance(entropy, int)
        return entropy % 2**32, None
    return operator.index(seed), None


def _minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    budget: int,
    *,
    seed: int | None,
    rng: np.random.RandomState | None,
    config: ATMConfig | None,
    profile: bool | Profiler,
    bounds: tuple[object, object] | None,
    warm: Warm | None,
) -> OptimizeResult:
    """:func:`minimize`; ``warm`` (only the anytime mode's hand-over passes one) is
    what the replay part evaluated, for the phases that would otherwise draw fresh
    points. Without it (every other run) nothing below reads it."""
    if config is not None and config.variant.anytime != "off":
        from .anytime import minimize_anytime

        return minimize_anytime(
            fun,
            x0,
            budget,
            seed=seed,
            rng=rng,
            config=config,
            profile=profile,
            bounds=bounds,
        )
    if rng is None:
        rng = np.random.RandomState(seed)
    x0 = np.asarray(x0, dtype=float)
    normalized = _box_normalized(fun, x0, budget, rng, config, profile, bounds, warm)
    if normalized is not None:
        return normalized
    config = resolve_config(config or ATMConfig(), budget, len(x0))
    total_iterations = iteration_limit(config, budget)
    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")
    if bounds is None and config.variant.needs_box:
        raise ValueError("this variant needs bounds=(lower, upper)")
    box = None if bounds is None else Box.make(bounds, len(x0))
    profiler = resolve_profiler(profile)
    record = _Recording(config.variant.cma_warm_start, warm)
    objective = CountedObjective(
        record.wrap(fun),
        budget,
        strict=config.strict_budget,
        profiler=profiler,
    )

    start = ResourceConductor(
        ConductorConfig(total_resources=config.initial_population)
    )
    allocation = start.step(start.rest, np.zeros(N_SAMPLERS))

    search = Search(objective, rng, config, total_iterations, profiler, box)
    nrestarts = 0
    known_cost: float | None = None
    width_scale: float | None = None
    with (
        np.errstate(invalid="ignore"),
        profiler.block("total"),
        restarts_on_stop(lambda: nrestarts),
    ):
        variant = config.variant
        dim = len(x0)
        use_model = variant.cma_model == "quadratic" and dim <= CMA_MODEL_MAX_DIM
        cma_options = _cma_options(variant, dim)
        sep = (
            variant.sep_probe
            and box is not None
            and budget >= SEP_MIN_BUDGET_PER_DIM * dim
        )
        early = _EarlyRoutes(x0)
        if box is not None:
            early = _early_routes(objective, rng, box, budget, variant, x0)
        best_x, best_cost, x_start = early.best_x, early.best_cost, early.x_start
        explored, out_of_budget = early.done, early.done
        if box is not None and not out_of_budget and _prefix_runs(variant, dim, budget):
            es = _run_prefix(
                objective,
                rng,
                box,
                budget,
                variant,
                early.x_start,
                early.spent,
                warm,
                use_model,
                cma_options,
                sep,
            )
            x_start, known_cost, width_scale = es.x, es.cost, es.sigma
            explored = True
        handoff = box is not None and variant.cma_handoff > 0 and not explored
        if handoff:
            objective.set_limit(_handoff_limit(variant, budget, dim, early))
            record.on = variant.cma_warm_start
        last_widths: tuple[float, ...] = ()
        # the anytime hand-over: the first search skips the uniform first pool, which
        # the replay part's points stand in for (and starts with its value known
        # when ``x0`` is the best point so far)
        fresh = warm is None
        if (
            warm is not None
            and known_cost is None
            and np.array_equal(x_start, warm.best_x)
        ):
            known_cost = warm.best_f
        while not out_of_budget:
            outcome = search.run(
                x_start, allocation, known_cost, width_scale, not explored and fresh
            )
            fresh = True
            width_scale = None
            explored = False
            out_of_budget = outcome.out_of_budget
            last_widths = outcome.widths
            nrestarts += int(outcome.restarted)
            x_start = outcome.x
            if config.strict_budget:
                known_cost = outcome.cost
            if outcome.cost <= best_cost:
                best_cost = outcome.cost
                best_x = outcome.x
        if handoff and objective.calls < budget:
            assert box is not None
            best_x, best_cost = _hand_over(
                objective,
                rng,
                box,
                budget,
                variant,
                (best_x, best_cost),
                (use_model, cma_options, sep),
                record,
                last_widths,
                warm,
            )

    return OptimizeResult(
        x=np.array(best_x),
        fun=float(best_cost),
        nfev=objective.calls,
        nrestarts=nrestarts,
        timings=profiler.report(),
    )


def _box_normalized(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    budget: int,
    rng: np.random.RandomState,
    config: ATMConfig | None,
    profile: bool | Profiler,
    bounds: tuple[object, object] | None,
    warm: Warm | None,
) -> OptimizeResult | None:
    """``box_normalize``: the run on the box mapped onto bbob's [-5, 5]^D, coordinate
    by coordinate, so every share of the box width and every absolute step means what
    it means on bbob; None on bbob's own box (the run is exactly the unnormalised one)
    and without the option."""
    if config is None or not config.variant.box_normalize or bounds is None:
        return None
    if budget < 1:
        return None
    norm = Box.make(bounds, len(x0))
    if np.all(norm.lower == -5.0) and np.all(norm.upper == 5.0):
        return None
    lo, span = norm.lower, (norm.upper - norm.lower) / 10.0

    def mapped(u: np.ndarray) -> float:
        return fun(lo + (np.asarray(u, dtype=float) + 5.0) * span)

    inner = _minimize(
        mapped,
        np.clip((x0 - lo) / span - 5.0, -5.0, 5.0),
        budget,
        seed=None,
        rng=rng,
        config=config,
        profile=profile,
        bounds=(-5.0, 5.0),
        warm=None if warm is None else warm.mapped(lo, span),
    )
    inner.x = lo + (np.asarray(inner.x, dtype=float) + 5.0) * span
    return inner


class _Recording:
    """``cma_warm_start``: the evaluations ATM makes while ``on``, for the first CMA
    run's shape (with the anytime hand-over, every point the earlier stages
    evaluated too)."""

    def __init__(self, enabled: bool, warm: Warm | None) -> None:
        self.enabled = enabled
        self.on = False
        self.x: list[np.ndarray] = []
        self.f: list[float] = []
        if warm is not None and enabled:
            self.x.extend(warm.x)
            self.f.extend(warm.f.tolist())

    def wrap(self, fun: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
        """``fun``, recording while ``on`` (``fun`` itself when not enabled)."""
        if not self.enabled:
            return fun

        def evaluated(x: np.ndarray) -> float:
            kept = np.array(x, dtype=float)  # before ``fun``, which may write into x
            value = fun(x)
            if self.on:
                self.x.append(kept)
                self.f.append(float(value))
            return value

        return evaluated


@dataclass
class _EarlyRoutes:
    """What the routes before the prefix left: the best point, where the run goes on
    from, the evaluations they spent (``descent_calls``: the basin descents', for
    ``basin_keep_atm``) and whether they spent the budget."""

    x_start: np.ndarray
    best_x: np.ndarray = field(init=False)
    best_cost: float = 10**100
    spent: int = 0
    descent_calls: int = 0
    done: bool = False

    def __post_init__(self) -> None:
        self.best_x = self.x_start.copy()


def _early_routes(
    objective: CountedObjective,
    rng: np.random.RandomState,
    box: Box,
    budget: int,
    variant: Variant,
    x0: np.ndarray,
) -> _EarlyRoutes:
    """The routes that run before the prefix, in order: ``small_engine``, the pair
    test (``chain_route`` / ``sep_route``) and ``basin_route``."""
    dim = len(x0)
    out = _EarlyRoutes(x0)
    if (
        variant.small_engine == "cobyqa"
        and dim <= variant.small_engine_max_dim
        and budget <= SMALL_ENGINE_MAX_PER_DIM * dim
    ):
        # ``small_engine``: restarted COBYQA spends the whole budget
        cobyqa = cobyqa_phase.run_cobyqa_restarts(
            objective, rng, x0, box, budget, SMALL_ENGINE_RADIUS
        )
        out.best_x, out.best_cost = cobyqa.x, cobyqa.cost
        out.spent = objective.calls
        out.done = True
    if (
        not out.done
        and dim >= ADDITIVE_MIN_DIM
        and (
            (variant.chain_route and budget >= CHAIN_MIN_PER_DIM * dim)
            or (variant.sep_route and budget >= SEP_ROUTE_MIN_PER_DIM * dim)
        )
    ):
        routed, out.best_x, out.best_cost = _additive_route(
            objective, rng, box, budget, variant, x0
        )
        out.spent = objective.calls
        out.done = routed  # ``chain_route`` / ``sep_route``: the budget is spent
    if (
        not out.done
        and variant.basin_route
        and 2 <= dim <= BASIN_MAX_DIM
        and budget >= BASIN_MIN_PER_DIM * dim
    ):
        _basin_route(objective, rng, box, budget, variant, out)
    return out


def _basin_route(
    objective: CountedObjective,
    rng: np.random.RandomState,
    box: Box,
    budget: int,
    variant: Variant,
    out: _EarlyRoutes,
) -> None:
    """``basin_route``: a design reading; on a smooth
    multi-basin reading MLSL spends the budget, otherwise the usual path continues
    from the design's best point. ``basin_descents`` are passed only when
    on, so the other configurations' calls stay exactly as recorded."""
    dim = len(out.x_start)
    basin_start: list[tuple[np.ndarray, float]] = []
    descents: dict[str, Any] = (
        {"descents": variant.basin_descents, "start_out": basin_start}
        if variant.basin_descents and budget <= BASIN_DESCENTS_MAX_PER_DIM * dim
        else {}
    )
    routed, out.best_x, out.best_cost, readings = basin_phase.multi_basin_route(
        objective,
        rng,
        box,
        budget - objective.calls,
        **descents,
    )
    out.spent = objective.calls
    if variant.basin_keep_atm:
        out.descent_calls = int(readings.get("descent_calls", 0.0))
    if routed or objective.calls >= budget:
        out.done = True
    else:
        # after descents the usual path starts from the design's best point, while
        # their best stays ``best_cost`` unless something beats it
        out.x_start = np.array(
            basin_start[0][0] if basin_start else out.best_x, dtype=float
        )


def _run_prefix(
    objective: CountedObjective,
    rng: np.random.RandomState,
    box: Box,
    budget: int,
    variant: Variant,
    x0: np.ndarray,
    spent: int,
    warm: Warm | None,
    use_model: bool,
    cma_options: dict[str, object],
    sep: bool,
) -> EsResult:
    """The ES prefix (``es_phase="prefix"``) with the local search of
    :func:`_prefix_kind`; ``spent``: the evaluations the earlier routes used."""
    dim = len(x0)
    init_points = max(4, int(np.ceil(ES_INIT_FACTOR * dim)))
    # the anytime hand-over: the design is the best points evaluated so far
    # (answered from memory) instead of fresh uniform draws
    warm_design: dict[str, Any] = (
        {} if warm is None else {"design": warm.best(init_points)}
    )
    # ``prefix_full_per_dim``: no ATM tail at small budgets
    whole = (
        variant.prefix_full_min_per_dim * dim
        < budget
        <= variant.prefix_full_per_dim * dim
    )
    return run_es_phase(
        objective,
        rng,
        x0,
        box,
        (budget if whole else max(1, int(variant.es_fraction * budget))) - spent,
        init_points,
        _prefix_kind(variant, dim, budget),
        use_model,
        cma_options,
        1.0,
        SEP_SHARE if sep else 0.0,
        variant.es_lq_nonsep,
        ("lq2" if budget >= variant.es_lq_min_budget_per_dim * dim else "cma"),
        fd_step=QN_FD_STEP
        if variant.es_kind in ("slsqplq10h", "slsqplq10hd")
        else None,
        **_pair_when_stuck_kw(variant, budget, dim, "step_pair_when_stuck"),
        **({"qn_fallback": True} if variant.qn_fallback == "lbfgsb" else {}),
        **({"sep_smooth_probe": True} if variant.prefix_sep_probe else {}),
        # ``prefix_plateau_pop``: above 30 x dim only (its 8 probe evaluations cost
        # the smooth functions at 30 x dim)
        **(
            {"plateau_pop": True}
            if variant.prefix_plateau_pop
            and budget > QN_MIN_BUDGET_PER_DIM * dim
            and dim >= variant.prefix_plateau_min_dim
            else {}
        ),
        **_small_switch_kw(variant, dim, budget),
        nan_reading=variant.prefix_nan_reading,
        **_prefix_ridge_kw(variant, dim, budget),
        **_retry_low_beta_kw(variant, dim),
        **warm_design,
    )


def _pair_when_stuck_kw(
    variant: Variant, budget: int, dim: int, name: str
) -> dict[str, Any]:
    """``sep_check="stall"`` from ``sep_stall_min_per_dim`` x dim, as STEP's option
    ``name`` (passed only when on, so the other calls stay exactly as recorded)."""
    if variant.sep_check == "stall" and budget >= variant.sep_stall_min_per_dim * dim:
        return {name: True}
    return {}


def _handoff_limit(variant: Variant, budget: int, dim: int, early: _EarlyRoutes) -> int:
    """The evaluations ATM gets before the hand-over: ``cma_handoff`` of the budget,
    at most ``cma_handoff_cap_per_dim`` x dim from ``CMA_HANDOFF_CAP_MIN_DIM``; the
    basin descents do not shorten ATM's sample (``basin_keep_atm``)."""
    atm_share = int(variant.cma_handoff * budget)
    cap = variant.cma_handoff_cap_per_dim
    if cap > 0 and dim >= CMA_HANDOFF_CAP_MIN_DIM:
        atm_share = min(atm_share, int(cap * dim))
    return max(1, atm_share + early.descent_calls)


def _hand_over(
    objective: CountedObjective,
    rng: np.random.RandomState,
    box: Box,
    budget: int,
    variant: Variant,
    best: tuple[np.ndarray, float],
    cma: tuple[bool, dict[str, object], bool],
    record: _Recording,
    last_widths: tuple[float, ...],
    warm: Warm | None,
) -> tuple[np.ndarray, float]:
    """After ATM's share (``cma_handoff``): the separable search (``sep_probe``),
    then :func:`_local_phase`; returns the best point and its value."""
    best_x, best_cost = best
    use_model, cma_options, sep = cma
    dim = len(best_x)
    objective.set_limit(budget)
    nonseparable = False  # the separable search's check failed
    if sep:
        from .step_phase import run_step_phase

        probe = run_step_phase(
            objective,
            best_x,
            best_cost,
            box,
            int(SEP_SHARE * (budget - objective.calls)),
            **_pair_when_stuck_kw(variant, budget, dim, "pair_when_stuck"),
        )
        if probe.cost <= best_cost:
            best_cost, best_x = probe.cost, probe.x
        nonseparable = probe.stalled
    objective.set_limit(budget)
    record.on = False
    handover = _Handover(
        variant=variant,
        rng=rng,
        box=box,
        budget=budget,
        dim=dim,
        use_model=use_model,
        cma_options=cma_options,
        pop_options=_pop_options(variant),
        nonseparable=nonseparable,
        recorded_x=record.x,
        recorded_f=record.f,
        best_x=best_x,
        best_cost=best_cost,
        a0=_warm_shape(variant, dim, record.x, record.f),
        last_widths=last_widths,
        warm_samples=None if warm is None else (warm.x, warm.f),
    )
    _local_phase(handover, objective)
    return handover.best_x, handover.best_cost


def _additive_route(
    objective: CountedObjective,
    rng: np.random.RandomState,
    box: Box,
    budget: int,
    variant: Variant,
    x0: np.ndarray,
) -> tuple[bool, np.ndarray, float]:
    """``chain_route`` / ``sep_route``: (routed, best point, value).

    The pair test at ``x0``. A chain reading with ``chain_route`` (and enough budget)
    takes the band check, and on a linear chain ``run_coord_hess`` from ``x0`` spends
    the rest. A separable reading with ``sep_route`` (and enough budget) gets
    restarted STEP (parabolic steps, the separability check off), the first from the
    pair test's best point, then from uniform points. Otherwise the caller's path
    continues; the evaluations spent here are counted, their best point kept.
    """
    dim = len(x0)
    read = chain_phase.pair_test(objective, rng, x0, box)
    if read is None:
        return True, np.array(x0, dtype=float), float("inf")
    reading, best_x, best_cost = read

    def keep(x: np.ndarray, cost: float) -> None:
        nonlocal best_x, best_cost
        if cost < best_cost:
            best_x, best_cost = np.array(x, dtype=float), float(cost)

    if reading == "chain" and variant.chain_route and budget >= CHAIN_MIN_PER_DIM * dim:
        band = chain_phase.band_consistency(objective, rng, x0, box)
        if band is None:
            return True, best_x, best_cost
        spread, order, x_b, cost_b = band
        keep(x_b, cost_b)
        if spread < chain_phase.BAND_TOL:
            run = chain_phase.run_coord_hess(
                objective, rng, x0, box, budget - objective.calls, order
            )
            keep(run.x, run.cost)
            return True, best_x, best_cost
    if (
        reading == "separable"
        and variant.sep_route
        and budget >= SEP_ROUTE_MIN_PER_DIM * dim
    ):
        from .step_phase import run_step_phase

        x, cost = best_x, best_cost
        while budget - objective.calls > 0:
            step = run_step_phase(
                objective,
                x,
                cost,
                box,
                budget - objective.calls,
                parabolic=True,
                check_tol=float("inf"),
            )
            keep(step.x, step.cost)
            x = box.uniform(rng)
            value = objective.evaluate_point(x)
            if value is None:
                break
            cost = float(value)
            keep(x, cost)
        return True, best_x, best_cost
    return False, best_x, best_cost


# ------------------------------------------------------------ the options ----
def _cma_options(variant: Variant, dim: int) -> dict[str, object]:
    """Keyword options of every ``run_cma`` call (beyond ``model`` and ``a0``)."""
    options: dict[str, object] = {}
    if variant.cma_restart_rule != "legacy":
        options["restart_rule"] = variant.cma_restart_rule
    if variant.cma_active:
        options["active"] = True
    if variant.cma_model_high == "diagonal" and dim > CMA_MODEL_MAX_DIM:
        options["model_kind"] = "diagonal"
    return options


def _pop_options(variant: Variant) -> dict[str, object]:
    """Keyword options of the local phase's ``run_population_restarts`` calls."""
    options: dict[str, object] = {}
    if variant.pop_lambda != "default":
        options["lambda_rule"] = variant.pop_lambda
    if variant.pop_active:
        options["active"] = True
    return options


# ---------------------------------------------------------------- prefix ----
def _small_switch_kw(variant: Variant, dim: int, budget: int) -> dict[str, Any]:
    """``run_es_phase``'s ``small_switch`` argument, passed only
    inside its envelope so every other call stays exactly as recorded."""
    if not (
        variant.small_switch
        and dim <= SMALL_SWITCH_MAX_DIM
        and budget <= SMALL_SWITCH_MAX_PER_DIM * dim
    ):
        return {}
    rule = qn_phase.SmallSwitch(
        trials=variant.small_switch_trials,
        parallel_cos=SMALL_SWITCH_COS,
        keep_unit=variant.small_switch_keep_unit,
        radius=SMALL_SWITCH_RADIUS,
        rest=SMALL_SWITCH_REST,
    )
    return {"small_switch": rule}


def _prefix_ridge_kw(variant: Variant, dim: int, budget: int) -> dict[str, Any]:
    """``run_es_phase``'s ridge reading (``prefix_ridge_after``), passed
    only when on and inside its ``prefix_ridge_max_per_dim`` envelope, so every other
    call stays exactly as recorded."""
    if not variant.prefix_ridge_after or (
        variant.prefix_ridge_max_per_dim
        and budget > variant.prefix_ridge_max_per_dim * dim
    ):
        return {}
    return {
        "ridge_after": variant.prefix_ridge_after,
        "ridge_radius": PREFIX_RIDGE_RADIUS,
    }


def _retry_low_beta_kw(variant: Variant, dim: int) -> dict[str, Any]:
    """``run_es_phase``'s ``descent_retries`` and ``low_beta``, passed only when on so
    every other call stays exactly as recorded."""
    on: dict[str, Any] = {}
    if variant.prefix_descent_retries:
        on["descent_retries"] = variant.prefix_descent_retries
    if variant.prefix_low_beta and dim >= variant.prefix_low_beta_min_dim:
        on["low_beta"] = variant.prefix_low_beta
    return on


def _prefix_runs(variant: Variant, dim: int, budget: int) -> bool:
    """Whether the ES prefix runs (``es_phase`` and its budget limits; needs a box)."""
    return variant.es_phase != "none" and budget <= variant.es_max_budget_per_dim * dim


def _prefix_kind(variant: Variant, dim: int, budget: int) -> str:
    """The local search the ES prefix runs after its design (``run_es_phase``'s kind).

    In order:

    * at ``dim <= nm_max_dim``: one SLSQP descent then Nelder-Mead (``"slsqpnm"``:
      ``nm_stage="qnprefix30"`` up to 30 x dim); Nelder-Mead alone (``"nm"``);
    * the (1+1)-CMA-ES (``"cma"``) where the configured kind needs more budget:
      ``lq2`` below ``es_lq_min_budget_per_dim``, the curvature-gated
      ``slsqplq`` below ``QN_MIN_BUDGET_PER_DIM`` x dim, and ``slsqplq10c`` and
      ``slsqplq10h`` there unless the budget reaches ``QN_MIN_BUDGET``;
    * ``"slsqplqc"`` (the two-direction probe) for ``slsqplq10c``, ``slsqplq10h``
      and ``slsqplq10hd`` below 30 x dim; ``"slsqplq"`` for the three ``slsqplq10``
      kinds otherwise; else ``es_kind`` itself.
    * ``es_kind="slsqplq10hd"`` is ``slsqplq10h`` with ``"slsqpdes"`` (one SLSQP
      descent from the design's best, then the (1+1)) below ``des_max_per_dim`` x dim
      and ``QN_MIN_BUDGET``, where ``slsqplq10h`` skips the probe.
    """
    small = budget < QN_MIN_BUDGET_PER_DIM * dim
    if (
        dim <= variant.nm_max_dim
        and variant.nm_stage == "qnprefix30"
        and budget <= QN_MIN_BUDGET_PER_DIM * dim
    ):
        return "slsqpnm"
    if dim <= variant.nm_max_dim:
        return "nm"
    kind = variant.es_kind
    if (
        kind == "slsqplq10hd"
        and budget < variant.des_max_per_dim * dim
        and budget < QN_MIN_BUDGET
    ):
        return "slsqpdes"
    if (
        (kind == "lq2" and budget < variant.es_lq_min_budget_per_dim * dim)
        or (kind == "slsqplq" and small)
        or (kind in ("slsqplq10c", "slsqplq10h") and small and budget < QN_MIN_BUDGET)
    ):
        return "cma"
    if kind in ("slsqplq10c", "slsqplq10h", "slsqplq10hd") and small:
        return "slsqplqc"
    if kind in ("slsqplq10c", "slsqplq10h", "slsqplq10hd"):
        return "slsqplq"
    return kind


# ------------------------------------------------------------- hand-over ----
@dataclass
class _Handover:
    """The state the phases after ATM share (``cma_handoff``).

    ``best_x`` and ``best_cost`` are the best point so far; every phase offers its
    result to :meth:`keep`. ``a0`` is the warm-start factor of the first CMA run
    (dropped once used), ``last_widths`` ATM's final widths, and
    ``sigma_override`` the hand-over sigma an SLSQP stage leaves for the finisher;
    together they give :meth:`start_sigma`.
    """

    variant: Variant
    rng: np.random.RandomState
    box: Box
    budget: int
    dim: int
    use_model: bool
    cma_options: dict[str, object]
    pop_options: dict[str, object]
    nonseparable: bool  # the separable search's check failed
    recorded_x: list[np.ndarray]
    recorded_f: list[float]
    best_x: np.ndarray
    best_cost: float
    a0: np.ndarray | None
    last_widths: tuple[float, ...]
    sigma_override: list[float] = field(default_factory=list)
    # ``rugged_direct_beta``: the finisher's populations start at once
    direct_populations: bool = False
    # the anytime hand-over: what earlier stages evaluated (points, values), the
    # sampled restarts' first samples; None in every other run
    warm_samples: tuple[np.ndarray, np.ndarray] | None = None

    def keep(self, x: np.ndarray, cost: float) -> None:
        """Take ``x`` as the best point if it is at least as good."""
        if cost <= self.best_cost:
            self.best_cost, self.best_x = cost, x

    def start_sigma(self) -> float:
        """The CMA phase's start sigma.

        An SLSQP stage's hand-over sigma if one ran, else the geometric mean of
        ATM's final widths, else a fifth of the box.
        """
        if self.sigma_override:
            return self.sigma_override[-1]
        if self.last_widths:
            return float(np.exp(np.mean(np.log(self.last_widths))))
        return INITIAL_SIGMA_SHARE * self.width

    @property
    def width(self) -> float:
        return float(np.mean(self.box.upper - self.box.lower))

    def uniform_point(self) -> np.ndarray:
        """A uniform point of the box (one ``rand(dim)`` draw)."""
        box = self.box
        return box.uniform(self.rng)

    def done(self) -> EsResult:
        return EsResult(self.best_x, float(self.best_cost), 0.0, 0)


def _warm_shape(
    variant: Variant,
    dim: int,
    recorded_x: list[np.ndarray],
    recorded_f: list[float],
) -> np.ndarray | None:
    """The first CMA run's Cholesky factor from ATM's archive (``cma_warm_start``)."""
    if variant.cma_warm_start and recorded_x:
        return warm_factor(np.array(recorded_x), np.array(recorded_f))
    return None


def _local_phase(h: _Handover, objective: Any) -> EsResult:
    """The local phase after ATM, from ``h.best_x``.

    Returns the best point found (also left in ``h``).
    """
    variant, dim = h.variant, h.dim
    if dim <= variant.nm_max_dim and variant.nm_stage not in ("prefix", "qnprefix30"):
        from .nm_phase import run_nm_restarts

        simplex = run_nm_restarts(
            objective,
            h.rng,
            h.best_x,
            h.best_cost,
            h.box,
            h.budget - objective.calls,
            h.start_sigma(),
        )
        h.keep(simplex.x, simplex.cost)
    local_kind = (
        variant.cma_local if dim >= variant.cma_local_min_dim else "one_plus_one"
    )
    local_pop_options = h.pop_options
    nonsep = h.nonseparable and dim >= CMA_LOCAL_NONSEP_MIN_DIM
    if nonsep and variant.cma_local_nonsep in qn_phase.QN_ROUTES:
        options = _qn_local_phase(
            qn_phase.QN_ROUTES[variant.cma_local_nonsep], h, objective
        )
        if options is None:
            return h.done()
        local_pop_options = options
    if local_kind == "then_ipop" and variant.pop_gate_per_dim > 0:
        _finish_gated_then_ipop(h, objective, local_pop_options)
    elif local_kind == "then_ipop":
        _finish_then_ipop(h, objective)
    else:
        _finish_one_plus_one(h, objective)
    return h.done()


def _qn_local_phase(
    route: QnRoute, h: _Handover, objective: Any
) -> dict[str, object] | None:
    """The SLSQP routes of ``cma_local_nonsep`` (``QN_ROUTES``).

    Reads the curvature at the best point (after a plateau check, ``plateau_check``)
    and, where it reads smooth, runs restarted SLSQP for a stage (a share of the
    local budget, capped at ``QN_LOCAL_CAP_PER_DIM`` x dim) that the finisher
    continues, starting at the stage's hand-over sigma. With ``extend``, a stage
    whose first descent was still running continues for at most another capped
    stage; with ``basins``, converged descents that end in
    separate basins (hill-valley test) keep restarting SLSQP to the end. Rugged or
    plateau readings get ``steps`` populations (``rugged_steps``), or IPOP
    populations with 4x lambda on plateaus (``plateau_ipop4``).

    Returns the finisher's population options, or None when the local phase is
    over (budget spent, or a route that runs to the end).
    """
    variant, dim, box, rng, budget = h.variant, h.dim, h.box, h.rng, h.budget
    checked = qn_phase.plateau_check(objective, rng, h.best_x, h.best_cost, box)
    if checked is None:
        return None
    plateau = checked[0]
    probed = (
        None
        if plateau
        else qn_phase.curvature_beta(objective, rng, h.best_x, h.best_cost, box)
    )
    beta: float | None = None
    if probed is not None:
        beta, probe_x, probe_cost, _ = probed
        h.keep(probe_x, probe_cost)
    elif not plateau:
        return None
    smooth = beta is not None and beta >= qn_phase.BETA_THRESHOLD
    if (
        smooth
        and route.smooth_bfgs_max_per_dim
        and budget <= route.smooth_bfgs_max_per_dim * dim
        and _smooth_bfgs(route, h, objective)
    ):
        return None
    if smooth:
        return _qn_smooth_stage(route, h, objective)
    if route.plateau_ipop4 and plateau:
        # Plateaus (f7; literature report rank 3, BIPOP's large populations):
        # IPOP runs from a uniform point at sigma0 = box / 5 with 4x the default
        # lambda, doubling per restart.
        flat = run_population_restarts(
            objective,
            rng,
            h.uniform_point(),
            np.inf,
            INITIAL_SIGMA_SHARE * h.width,
            box,
            budget - objective.calls,
            True,
            lambda_rule="ipop4",
            active=variant.pop_active,
        )
        h.keep(flat.x, flat.cost)
        return None
    left = budget - objective.calls
    low = 0 < route.rugged_ipop_low_dim <= dim < route.rugged_ipop_min_dim
    if (
        route.rugged_ipop
        and beta is not None
        and (
            (
                dim >= route.rugged_ipop_min_dim
                and left >= qn_phase.RUGGED_IPOP_MIN_LEFT_PER_DIM * dim
            )
            or (low and left >= qn_phase.RUGGED_IPOP_LOW_LEFT_PER_DIM * dim)
        )
    ):
        _rugged_populations(route, h, objective, left, low)
        return None
    # The rugged branch composes in this order (each only when on):
    # ``rugged_direct_beta`` reads the hand-over probe's beta first; a reading in
    # [``rugged_direct_min_beta``, ``rugged_direct_beta``) skips the capped try and
    # the finisher's first (1+1). Otherwise the try runs and ``rugged_try_rest`` may
    # continue it: kinked tries (f13) read above 1.0, so ``cutkink`` and a
    # ``rugged_direct_beta`` of 1.0 act on different runs.
    direct = (
        variant.rugged_direct_beta > 0
        and beta is not None
        and beta < variant.rugged_direct_beta
        # ``rugged_direct_min_beta`` (0: no lower bound)
        and (
            variant.rugged_direct_min_beta <= 0
            or beta >= variant.rugged_direct_min_beta
        )
    )
    if route.rugged_try_per_dim and not direct and beta is not None:
        _rugged_try(route, h, objective)
    if route.rugged_steps:
        # Rugged or plateau: populations sized to the budget left.
        if direct:
            h.direct_populations = True
        return {**h.pop_options, "lambda_rule": "steps"}
    return h.pop_options


def _qn_options(route: QnRoute, h: _Handover) -> tuple[float | None, dict[str, Any]]:
    """The route's SLSQP step and the options its SLSQP calls pass (only when on,
    so the other routes' calls stay exactly as recorded)."""
    fd_step = QN_FD_STEP if route.fd_step else None
    on: dict[str, Any] = {"fallback": True} if h.variant.qn_fallback == "lbfgsb" else {}
    return fd_step, on


def _smooth_bfgs(route: QnRoute, h: _Handover, objective: Any) -> bool:
    """``smooth_bfgs_max_per_dim``: at small budgets the scaled-FD BFGS
    takes a smooth reading; True when it took the rest of the budget.

    With ``smooth_bfgs_descents`` it runs that many descents, and the
    usual staged route continues from its best point.
    """
    # passed only when on, so the other routes' calls stay exactly as recorded
    bfgs_on: dict[str, Any] = {}
    if route.smooth_bfgs_descents:
        bfgs_on["max_descents"] = route.smooth_bfgs_descents
    if route.smooth_bfgs_box_qp:
        bfgs_on["box_qp"] = True
    polished = qn_phase.run_scaled_bfgs(
        objective,
        h.rng,
        h.best_x,
        h.best_cost,
        h.box,
        h.budget - objective.calls,
        **bfgs_on,
    )
    h.keep(polished.x, polished.cost)
    return not route.smooth_bfgs_descents


def _qn_smooth_stage(
    route: QnRoute, h: _Handover, objective: Any
) -> dict[str, object] | None:
    """A smooth reading: the SLSQP stage, its extension or the multi-basin restarts,
    Rosenbrock's second descent and the kink check; the finisher's options, or None
    when the local phase is over."""
    dim, box, rng, budget = h.dim, h.box, h.rng, h.budget
    fd_step, on = _qn_options(route, h)
    stage = qn_phase.QN_LOCAL_SHARE * (budget - objective.calls)
    stage = min(stage, qn_phase.QN_LOCAL_CAP_PER_DIM * dim)
    ends: list[Descent] = []
    gate: list[bool] = []
    # passed only when on, so the other routes' calls stay exactly as recorded
    gated: dict[str, Any] = (
        {"gate_once": True, "gate_out": gate} if route.cap_gate else {}
    )
    qn = qn_phase.run_slsqp_restarts(
        objective,
        rng,
        h.best_x,
        h.best_cost,
        box,
        int(stage),
        ends,
        fd_step=fd_step,
        extend_to=(
            int(
                budget
                - objective.calls
                - max(
                    qn_phase.QN_RESERVE_SHARE * (budget - objective.calls),
                    qn_phase.QN_RESERVE_PER_DIM * dim,
                )
            )
            if route.extend_in_descent
            else None
        ),
        **on,
        **gated,
    )
    h.keep(qn.x, qn.cost)
    # ``cap_gate``: the extended first descent passed the gate at the cap
    passed = bool(gate) and gate[0]
    if on and qn_phase.failure_majority([d.status for d in ends]):
        # Most of the stage's SLSQP descents failed: on bbob only f12 (bent cigar)
        # does this. L-BFGS-B restarts take the rest of the budget.
        bent = qn_phase.run_lbfgsb_restarts(
            objective, rng, h.best_x, h.best_cost, box, budget - objective.calls
        )
        h.keep(bent.x, bent.cost)
        return None
    if route.extend != "none" and (
        not ends or (passed and ends[0].status != 0)
    ):  # ``cap_gate``: an extended descent that failed (f12) still continues
        # The first descent was still running at the cap (f8, f9, f13 at
        # D 20-40): let SLSQP go on for a share of what is left (at most
        # another capped stage), then the CMA finisher.
        more_budget = qn_phase.QN_EXTEND_SHARE * (budget - objective.calls)
        more_budget = min(more_budget, qn_phase.QN_LOCAL_CAP_PER_DIM * dim)
        more = qn_phase.run_slsqp_restarts(
            objective,
            rng,
            h.best_x,
            h.best_cost,
            box,
            int(more_budget),
            fd_step=fd_step,
            **on,
        )
        h.keep(more.x, more.cost)
        if more.used:  # else keep the stage's hand-over sigma
            qn = more
    elif route.basins:
        basins = qn_phase.multi_basin(objective, ends, box)
        if basins is None:
            return None
        if basins:
            _basin_restarts(route, h, objective, fd_step, on)
            return None
    left = budget - objective.calls
    if (
        route.second_descent_min_left_per_dim
        and passed
        and len(ends) == 1
        and ends[0].status == 0
        and left >= route.second_descent_min_left_per_dim * dim
    ):
        # Rosenbrock's second minimum (f8, f9 at D 20-40): one more
        # descent from a uniform point, leaving the finisher its reserve.
        start = h.uniform_point()
        reserve = max(
            qn_phase.QN_RESERVE_SHARE * left, qn_phase.QN_RESERVE_PER_DIM * dim
        )
        second = qn_phase.run_slsqp_restarts(
            objective,
            rng,
            start,
            np.inf,
            box,
            int(left - reserve),
            max_descents=1,
            fd_step=fd_step,
            **on,
        )
        h.keep(second.x, second.cost)
    if route.kink_small_step and dim >= route.kink_min_dim and gate and not gate[0]:
        # f13: the gate read the first descent at its cap as not
        # progressing; on a kink at the hand-over point the scaled-FD BFGS takes
        # the rest instead of the finisher.
        kinked = qn_phase.kink_check(objective, h.best_x, h.best_cost, box)
        if kinked is None:
            return None
        kink, _d2, x_k, cost_k = kinked
        h.keep(x_k, cost_k)
        left = budget - objective.calls
        if kink and left >= qn_phase.KINK_MIN_LEFT_PER_DIM * dim:
            fine = qn_phase.run_scaled_bfgs(
                objective, rng, h.best_x, h.best_cost, box, left
            )
            h.keep(fine.x, fine.cost)
            return None
    h.sigma_override.append(qn.sigma)
    return h.pop_options


def _basin_restarts(
    route: QnRoute,
    h: _Handover,
    objective: Any,
    fd_step: float | None,
    on: dict[str, Any],
) -> None:
    """Smooth with several basins (Gallagher-like): restarted SLSQP for the rest of
    the budget (``basins_sampled``: from the best unclaimed sample point, with the
    gap-scaled stop; ``basins_scipy_step``: at SciPy's step, more and cheaper
    descents for Gallagher's lottery)."""
    step = None if route.basins_scipy_step else fd_step
    left = h.budget - objective.calls
    if route.basins_sampled:
        warm_samples: dict[str, Any] = (
            {} if h.warm_samples is None else {"samples": h.warm_samples}
        )
        more = qn_phase.run_sampled_restarts(
            objective,
            h.rng,
            h.best_x,
            h.best_cost,
            h.box,
            left,
            fd_step=step,
            **warm_samples,
        )
    else:
        more = qn_phase.run_slsqp_restarts(
            objective, h.rng, h.best_x, h.best_cost, h.box, left, fd_step=step, **on
        )
    h.keep(more.x, more.cost)


def _rugged_populations(
    route: QnRoute, h: _Handover, objective: Any, left: int, low: bool
) -> None:
    """Rugged with a large budget left (f15-f18, f23): the plateau route's
    schedule, no (1+1) first and no protection.

    ``fractal_exempt``: a fractal reading (f23-like) is no case for
    populations; restarted SLSQP leads f23 at D 20-40 on the field. Otherwise the
    populations keep the schedule they had before the probe (its evaluations come
    off their end).
    """
    dim, box, rng = h.dim, h.box, h.rng
    if route.fractal_exempt and not low:
        micro = qn_phase.micro_slopes(
            objective, h.best_x, h.best_cost, box, centered=True
        )
        if micro is None:
            return
        fractal, _slopes, x_m, cost_m = micro
        h.keep(x_m, cost_m)
        if fractal:
            rest = qn_phase.run_slsqp_restarts(
                objective, rng, h.best_x, h.best_cost, box, h.budget - objective.calls
            )
            h.keep(rest.x, rest.cost)
            return
    wide = (
        dim >= qn_phase.RUGGED_IPOP8_MIN_DIM
        and left >= qn_phase.RUGGED_IPOP8_MIN_LEFT_PER_DIM * dim
    )
    rough = run_population_restarts(
        objective,
        rng,
        h.uniform_point(),
        np.inf,
        INITIAL_SIGMA_SHARE * h.width,
        box,
        left,
        True,
        lambda_rule="ipop8" if wide else "ipop4",
        active=h.variant.pop_active,
    )
    h.keep(rough.x, rough.cost)


def _rugged_try(route: QnRoute, h: _Handover, objective: Any) -> None:
    """``rugged_try_per_dim``: f13 reads rugged at ATM's best point below
    D 20; one capped scaled-FD BFGS descent first, then the populations, within the
    route's dimension and budget envelope.

    ``rugged_try_rest="cutkink"``: a try still descending at its cap
    goes on (the same descent, its inverse Hessian kept) to the end of the budget
    where its best point then reads a kink (f13's ridge).
    """
    dim, box, budget = h.dim, h.box, h.budget
    try_per_dim = route.rugged_try_per_dim
    try_max_budget_per_dim = route.rugged_try_max_budget_per_dim
    if not (
        budget - objective.calls >= 2 * try_per_dim * dim
        and (not route.rugged_try_max_dim or dim <= route.rugged_try_max_dim)
        and (not try_max_budget_per_dim or budget <= try_max_budget_per_dim * dim)
    ):
        return
    cap = int(try_per_dim * dim)
    # passed only when on, so the recorded calls stay exactly as recorded
    extend: dict[str, Any] = {}
    if h.variant.rugged_try_rest == "cutkink":
        extend["extend_to"] = budget - objective.calls
        extend["extend_gate"] = lambda x_e, cost_e: _kink_reads(
            objective, x_e, cost_e, box
        )
    tried = qn_phase.run_scaled_bfgs(
        objective, h.rng, h.best_x, h.best_cost, box, cap, max_descents=1, **extend
    )
    h.keep(tried.x, tried.cost)


def _kink_reads(objective: Any, x: np.ndarray, cost: float, box: Box) -> bool:
    """``qn_phase.kink_check`` at ``x`` as a yes / no (no when the budget ran out)."""
    kinked = qn_phase.kink_check(objective, x, cost, box)
    return kinked is not None and bool(kinked[0])


# ------------------------------------------------------------- finishers ----
def _finish_gated_then_ipop(
    h: _Handover, objective: Any, pop_options: dict[str, object]
) -> None:
    """The composites' finisher (``cma_local="then_ipop"`` with ``pop_gate_per_dim``).

    The restarted (1+1)-CMA-ES from the best point, except that a stall with at
    least ``gate`` evaluations left, a run reaching ``pop_switch_per_dim`` x dim
    (``run_cap``) or a rugged curvature reading (``cma_probe_beta``) launches
    population runs instead: from a uniform point, or (``pop_switch_start``) from
    the best point at the run's width. Then the (1+1) resumes at the best point,
    and a last uniform restart fills what is left.
    """
    variant, dim, box, rng, budget = h.variant, h.dim, h.box, h.rng, h.budget
    gate = int(np.ceil(variant.pop_gate_per_dim * dim))
    width = h.width
    if h.direct_populations and budget - objective.calls >= gate:
        # ``rugged_direct_beta``: the populations start at once,
        # as after the (1+1)'s own rugged probe (uniform point, a fifth of the box)
        first = EsResult(h.best_x, float(h.best_cost), h.start_sigma(), 0, True)
    else:
        first = run_cma(
            objective,
            rng,
            h.best_x,
            h.best_cost,
            h.start_sigma(),
            box,
            budget - objective.calls,
            restarts=True,
            model=h.use_model,
            a0=h.a0,
            stop_when_left=gate,
            probe_at=(
                int(np.ceil(variant.cma_probe_at_per_dim * dim))
                if dim >= variant.cma_probe_min_dim
                else 0
            ),
            probe_beta=variant.cma_probe_beta,
            run_cap=(
                int(
                    np.ceil(
                        min(variant.pop_switch_per_dim * dim, budget - objective.calls)
                    )
                )
                if dim >= variant.pop_switch_min_dim
                else 0
            ),
            **h.cma_options,  # type: ignore[arg-type]
        )
        h.keep(first.x, first.cost)
    if first.stalled and objective.calls < budget:
        start = h.uniform_point()
        start_cost, pop_sigma = float("inf"), INITIAL_SIGMA_SHARE * width
        if first.capped and variant.pop_switch_start != "uniform":
            start, start_cost = h.best_x, h.best_cost
            pop_sigma = float(np.clip(2 * first.sigma, 1e-8 * width, pop_sigma))
        later = run_population_restarts(
            objective,
            rng,
            start,
            start_cost,
            pop_sigma,
            box,
            budget - objective.calls,
            True,
            min_left=gate,
            must_beat=h.best_cost if variant.pop_protect else None,
            **pop_options,  # type: ignore[arg-type]
        )
        h.keep(later.x, later.cost)
    if (
        objective.calls < budget
        and variant.pop_switch_per_dim > 0
        and dim >= variant.pop_switch_min_dim
    ):
        # Resume the (1+1) at the best point with its last width scale.
        fill = run_cma(
            objective,
            rng,
            h.best_x,
            h.best_cost,
            first.sigma,
            box,
            budget - objective.calls,
            restarts=True,
            model=h.use_model,
            **h.cma_options,  # type: ignore[arg-type]
        )
        h.keep(fill.x, fill.cost)
    if objective.calls < budget:
        x_fill = h.uniform_point()
        cost_fill = objective.evaluate_point(x_fill)
        if cost_fill is not None:
            h.keep(x_fill, cost_fill)
            fill = run_cma(
                objective,
                rng,
                x_fill,
                cost_fill,
                INITIAL_SIGMA_SHARE * width,
                box,
                budget - objective.calls,
                restarts=True,
                model=h.use_model,
                **h.cma_options,  # type: ignore[arg-type]
            )
            h.keep(fill.x, fill.cost)


def _finish_then_ipop(h: _Handover, objective: Any) -> None:
    """``cma_local="then_ipop"`` without the gate: one (1+1) run, then IPOP runs."""
    box, rng, budget = h.box, h.rng, h.budget
    first = run_cma(
        objective,
        rng,
        h.best_x,
        h.best_cost,
        h.start_sigma(),
        box,
        budget - objective.calls,
        restarts=False,
        model=h.use_model,
        a0=h.a0,
        **h.cma_options,  # type: ignore[arg-type]
    )
    h.keep(first.x, first.cost)
    if objective.calls < budget:
        later = run_population_restarts(
            objective,
            rng,
            h.uniform_point(),
            float("inf"),
            INITIAL_SIGMA_SHARE * h.width,
            box,
            budget - objective.calls,
            True,
            **h.pop_options,  # type: ignore[arg-type]
        )
        h.keep(later.x, later.cost)


def _finish_one_plus_one(h: _Handover, objective: Any) -> None:
    """The restarted (1+1)-CMA-ES to the end of the budget."""
    if objective.calls < h.budget:
        es = run_cma(
            objective,
            h.rng,
            h.best_x,
            h.best_cost,
            h.start_sigma(),
            h.box,
            h.budget - objective.calls,
            restarts=True,
            model=h.use_model,
            a0=h.a0,
            **h.cma_options,  # type: ignore[arg-type]
        )
        h.keep(es.x, es.cost)
