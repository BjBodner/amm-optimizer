"""Public entry point, and the routing between the phases of the research composites.

:func:`minimize` runs, in this order (each step only when its ``Variant`` options
ask for it; the default ``ATMConfig()`` runs ATM alone, as the original):

1. **The prefix** (``es_phase="prefix"``, budgets up to ``es_max_budget_per_dim``):
   ``es_phase.run_es_phase`` with the local search of :func:`_prefix_kind`
   (the (1+1)-CMA-ES, SLSQP where the curvature probe reads smooth, lq2
   populations, Nelder-Mead at very low D, ...). ATM then continues from its best.
   With ``qn_fallback="lbfgsb"`` (``c32``) an SLSQP call whose failed descents
   outnumber its converged ones hands the rest to restarted L-BFGS-B.
2. **ATM** (``search.Search``), restarted by ``restart_mode``; with
   ``cma_handoff`` only for its share of the budget.
3. **The hand-over** (``cma_handoff``, when the prefix did not run): the separable
   search (``sep_probe``), then :func:`_local_phase`.
4. **The local phase** (:func:`_local_phase`): Nelder-Mead at very low D, the
   SLSQP routes of ``cma_local_nonsep`` (:func:`_qn_local_phase`, configured by
   ``qn_phase.QN_ROUTES``) on non-separable functions: a smooth reading runs an
   SLSQP stage (L-BFGS-B for the rest when most descents failed, ``c32``), a plateau
   IPOP populations (``qn8`` on), a rugged reading the finisher's ``steps``
   populations, or from D 20 with enough budget unprotected IPOP from a uniform
   point (``qn17``, ``c33``). Then one finisher:
   :func:`_finish_gated_then_ipop` (the composites ``c23``-``c27``),
   :func:`_finish_then_ipop`, :func:`_finish_population`, :func:`_finish_regimes`
   or :func:`_finish_one_plus_one`.

The hand-over state that these functions share is a :class:`_Handover`. The phase
functions are looked up as this module's globals at call time (``run_cma``,
``run_population_restarts``, ``run_es_phase``) and ``qn_phase``'s functions and
constants as attributes of that module, so tests can monkeypatch either.
"""

from __future__ import annotations

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
    run_es,
    run_es_phase,
)
from .objective import CountedObjective
from .plan import in_box_units, ipop_pool, iteration_limit, resolve_config, with_pool
from .pop_cma import run_population_restarts
from .profiling import Profiler, resolve_profiler
from .qn_phase import (
    QN_FD_STEP,
    QN_MIN_BUDGET,
    QN_MIN_BUDGET_PER_DIM,
    Descent,
    QnRoute,
)
from .regimes import run_regimes
from .result import OptimizeResult
from .search import Search
from .simplex import run_nelder_mead


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    budget: int,
    *,
    seed: int | None = None,
    rng: np.random.RandomState | None = None,
    config: ATMConfig | None = None,
    profile: bool | Profiler = False,
    bounds: tuple[object, object] | None = None,
) -> OptimizeResult:
    """Minimise ``fun`` starting from ``x0`` with about ``budget`` evaluations.

    Pass ``seed`` for a reproducible run, or an existing ``RandomState`` as
    ``rng`` to continue its stream (this is how results are compared against
    the original code, which used NumPy's global state).

    ``profile=True`` times the named blocks of the algorithm and returns them as
    ``result.timings``; pass a :class:`~amm_optimizer.profiling.TimingProfiler` to
    accumulate over several runs. Profiling does not change the result.

    ``bounds=(lower, upper)`` (scalars or one value per coordinate) tells the
    optimizer the search box. It is not enforced and changes nothing by itself;
    only variants that use the box (``initial_design``, ``bounds``) read it, and
    they raise ``ValueError`` without it. This engine does not check ``x0`` against
    the box; :func:`amm_optimizer.minimize` clips an ``x0`` outside it onto the box,
    with a ``UserWarning``.

    ``config`` defaults to the original algorithm; ``presets.get("c26")`` (and
    the other names of ``presets.names()``) are the research composites. The
    module docstring lists the phases a composite runs.

    With ``config.variant.anytime`` set, the run never learns ``budget``, which then
    only caps the calls (:mod:`.anytime`).
    """
    return _minimize(
        fun,
        x0,
        budget,
        seed=seed,
        rng=rng,
        config=config,
        profile=profile,
        bounds=bounds,
        warm=None,
    )


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
    if (
        config is not None
        and config.variant.box_normalize
        and bounds is not None
        and budget >= 1
    ):
        # ``box_normalize`` (the novel-ideas toy): run on the box mapped onto bbob's
        # [-5, 5]^D, coordinate by coordinate, so every share of the box width and
        # every absolute step means what it means on bbob; bbob's own box is not
        # mapped (the run is exactly the unnormalised one)
        norm = Box.make(bounds, len(x0))
        if not (np.all(norm.lower == -5.0) and np.all(norm.upper == 5.0)):
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
    config = resolve_config(config or ATMConfig(), budget, len(x0))
    total_iterations = iteration_limit(config, budget)
    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")
    if bounds is None and config.variant.needs_box:
        raise ValueError("this variant needs bounds=(lower, upper)")
    box = None if bounds is None else Box.make(bounds, len(x0))
    # ``box_steps`` (the novel-ideas toy): absolute x steps in units of width / 10;
    # set on every call, so a previous call's unit never leaks into this one
    qn_phase.set_x_unit(
        float(np.mean(box.upper - box.lower)) / 10.0
        if box is not None and config.variant.box_steps
        else 1.0
    )
    if box is not None and config.variant.width_units == "box":
        config = in_box_units(config, box)
    profiler = resolve_profiler(profile)
    # Variant ``cma_warm_start``: ATM's evaluations, kept while ``recording[0]``.
    recorded_x: list[np.ndarray] = []
    recorded_f: list[float] = []
    if warm is not None and config.variant.cma_warm_start:
        # the anytime hand-over: the first CMA run's shape learns from every point
        # the earlier stages evaluated, not only from this stage's ATM sample
        recorded_x.extend(warm.x)
        recorded_f.extend(warm.f.tolist())
    recording = [False]
    evaluated = fun
    if config.variant.cma_warm_start or config.variant.cma_placement == "mlsl":

        def evaluated(x: np.ndarray) -> float:
            kept = np.array(x, dtype=float)  # before ``fun``, which may write into x
            value = fun(x)
            if recording[0]:
                recorded_x.append(kept)
                recorded_f.append(float(value))
            return value

    objective = CountedObjective(
        evaluated,
        budget,
        strict=config.strict_budget,
        profiler=profiler,
        # the novel-ideas toy: passed only when on
        **({"memo": True} if config.variant.eval_memo else {}),
    )

    start = ResourceConductor(
        ConductorConfig(total_resources=config.initial_population)
    )
    allocation = start.step(start.rest, np.zeros(N_SAMPLERS))

    search = Search(objective, rng, config, total_iterations, profiler, box)
    refine = _tail_budget(config, budget, len(x0), box)
    best_cost: float = 10**100
    best_x = x0.copy()
    x_start = x0
    nrestarts = 0
    known_cost: float | None = None
    width_scale: float | None = None
    explored = False
    out_of_budget = False
    with np.errstate(invalid="ignore"), profiler.block("total"):
        variant = config.variant
        dim = len(x0)
        use_model = (
            variant.cma_model == "quadratic" and dim <= variant.cma_model_max_dim
        )
        cma_options = _cma_options(variant, dim)
        pop_options = _pop_options(variant)
        lean = variant.es_lean_start and dim >= 10
        mlsl = variant.cma_placement == "mlsl" and dim <= variant.cma_placement_max_dim
        sep = (
            variant.sep_probe
            and box is not None
            and budget >= variant.sep_min_budget_per_dim * dim
        )
        spent = 0
        descent_calls = 0  # ``basin_keep_atm``: the descents' evaluations
        centre: tuple[np.ndarray, float] | None = None
        if (
            box is not None
            and variant.centre_point != "off"
            and dim >= variant.centre_point_min_dim
            and budget >= 2
        ):
            # ``centre_point`` (research, gaps4): the first evaluation is the box
            # centre; ``"record"`` keeps it only as a candidate for the best point,
            # ``"start"`` also starts the run there when it beats ``x0``
            c_x = 0.5 * (box.lower + box.upper)
            c_cost = objective.evaluate_point(c_x)
            if c_cost is not None:
                centre = (c_x, float(c_cost))
                if variant.centre_point == "start":
                    x0_cost = objective.evaluate_point(x0)
                    if x0_cost is not None and c_cost < x0_cost:
                        x0 = x_start = c_x.copy()
        if (
            box is not None
            and variant.small_engine == "cobyqa"
            and dim <= variant.small_engine_max_dim
            and budget <= variant.small_engine_max_per_dim * dim
        ):
            # ``small_engine`` (cycle 61): restarted COBYQA spends the whole budget
            out = cobyqa_phase.run_cobyqa_restarts(
                objective, rng, x0, box, budget, variant.small_engine_radius
            )
            best_x, best_cost = out.x, out.cost
            spent = objective.calls
            explored = out_of_budget = True
        # ``sep_route_low`` (research, gaps3): the pair test below ``additive_min_dim``.
        # With the D 2-3 routes it composes in this order: ``small_engine`` (up to
        # ``small_engine_max_per_dim`` x dim) spends the whole budget first, so it never
        # meets this test (from ``sep_route_min_per_dim`` x dim); a test that does not
        # route keeps its evaluations spent and its best point, and ``basin_route``
        # runs after it, as in the gaps3 toy's runs on ``c44``
        low_sep = (
            variant.sep_route_low != "off"
            and variant.sep_route_low_dim <= dim < variant.additive_min_dim
        )
        if (
            box is not None
            and not out_of_budget
            and dim >= variant.additive_min_dim
            and (
                (variant.chain_route and budget >= variant.chain_min_per_dim * dim)
                or (variant.sep_route and budget >= variant.sep_route_min_per_dim * dim)
            )
        ):
            routed, best_x, best_cost = _additive_route(
                objective, rng, box, budget, variant, x0
            )
            spent = objective.calls
            if routed:  # ``chain_route`` / ``sep_route``: the budget is spent
                explored = out_of_budget = True
        elif (
            box is not None
            and not out_of_budget
            and low_sep
            and variant.sep_route
            and budget >= variant.sep_route_min_per_dim * dim
        ):
            routed, best_x, best_cost = _additive_route(
                objective,
                rng,
                box,
                budget,
                variant,
                x0,
                chain=False,
                route=variant.sep_route_low in ("route", "strict"),
                **({"strict": True} if variant.sep_route_low == "strict" else {}),
            )
            spent = objective.calls
            if routed:
                explored = out_of_budget = True
        if (
            box is not None
            and not out_of_budget
            and variant.basin_route
            and 2 <= dim <= variant.basin_max_dim
            and budget >= variant.basin_min_per_dim * dim
        ):
            # ``basin_route`` (cycle 53, the D 2 toy): a design reading; on a smooth
            # multi-basin reading MLSL spends the budget, otherwise the usual path
            # continues from the design's best point
            # ``basin_descents`` (cycle 60): passed only when on, so the other
            # configurations' calls stay exactly as recorded
            basin_start: list[tuple[np.ndarray, float]] = []
            descents: dict[str, Any] = (
                {"descents": variant.basin_descents, "start_out": basin_start}
                if variant.basin_descents
                and budget <= variant.basin_descents_max_per_dim * dim
                else {}
            )
            routed, best_x, best_cost, basin_readings = basin_phase.multi_basin_route(
                objective,
                rng,
                box,
                budget - objective.calls,
                **descents,
            )
            spent = objective.calls
            if variant.basin_keep_atm:
                descent_calls = int(basin_readings.get("descent_calls", 0.0))
            if routed or objective.calls >= budget:
                explored = out_of_budget = True
            else:
                # after descents the usual path starts from the design's best point,
                # while their best stays ``best_cost`` unless something beats it
                x0 = x_start = np.array(
                    basin_start[0][0] if basin_start else best_x, dtype=float
                )
        if box is not None and not out_of_budget and _prefix_runs(variant, dim, budget):
            prefix_kind = _prefix_kind(variant, dim, budget)
            init_points = (
                4 if lean else max(4, int(np.ceil(variant.es_init_factor * dim)))
            )
            warm_design: dict[str, Any] = (
                {} if warm is None else {"design": warm.best(init_points)}
            )
            # ``prefix_full_per_dim`` (cycle 38): no ATM tail at small budgets
            whole = (
                variant.prefix_full_min_per_dim * dim
                < budget
                <= variant.prefix_full_per_dim * dim
            )
            es = run_es_phase(
                objective,
                rng,
                x0,
                box,
                (budget if whole else max(1, int(variant.es_fraction * budget)))
                - spent,
                init_points,
                prefix_kind,
                use_model,
                cma_options,
                float(np.sqrt(10 / dim)) if lean else 1.0,
                variant.sep_share if sep else 0.0,
                variant.es_lq_nonsep,
                ("lq2" if budget >= variant.es_lq_min_budget_per_dim * dim else "cma"),
                variant.sep_parabolic,
                variant.sep_parabolic_late,
                fd_step=QN_FD_STEP
                if variant.es_kind in ("slsqplq10h", "slsqplq10hk", "slsqplq10hd")
                else None,
                step_pair_check=variant.sep_check == "pair",
                **(
                    {"step_pair_when_stuck": True}
                    if variant.sep_check == "stall"
                    and budget >= variant.sep_stall_min_per_dim * dim
                    else {}
                ),
                simplex=variant.local_method == "nelder_mead"
                and dim <= variant.simplex_max_dim,
                **({"qn_fallback": True} if variant.qn_fallback == "lbfgsb" else {}),
                **({"qn_false_success": True} if variant.qn_false_success else {}),
                **({"sep_smooth_probe": True} if variant.prefix_sep_probe else {}),
                # ``preplat``: above 30 x dim only (its 8 probe evaluations cost the
                # smooth functions at 30 x dim in scratch)
                **(
                    {"plateau_pop": True}
                    if variant.prefix_plateau_pop
                    and budget > QN_MIN_BUDGET_PER_DIM * dim
                    and dim >= variant.prefix_plateau_min_dim
                    else {}
                ),
                **({"plateau_rugged": True} if variant.prefix_plateau_rugged else {}),
                **_small_switch_kw(variant, dim, budget),
                # cycle 64; ``prefix_nan_min_per_dim`` (parked2): below that budget
                # an undefined reading stays rugged
                nan_reading=variant.prefix_nan_reading
                if budget >= variant.prefix_nan_min_per_dim * dim
                else "rugged",
                **_prefix_ridge_kw(variant, dim, budget),  # the f13 toy
                qn_floor=variant.qn_floor,  # research, ideas #24 / #32 / #55
                # the novel-ideas toy: passed only when on
                **({"probe_faces": True} if variant.probe_faces == "tangent" else {}),
                **({"step_box_check": True} if variant.step_box_check else {}),
                **_parked2_kw(variant, dim, budget),
                # the anytime hand-over: the design is the best points evaluated so
                # far (answered from memory) instead of fresh uniform draws
                **warm_design,
            )
            x_start, known_cost, width_scale = es.x, es.cost, es.sigma
            explored = True
        if refine:  # ``refine_fraction``: keep the tail's evaluations back
            objective.hold_back(refine)
        handoff = box is not None and variant.cma_handoff > 0 and not explored
        if handoff:
            atm_share = int(variant.cma_handoff * budget)
            cap = variant.cma_handoff_cap_per_dim
            low = variant.cma_handoff_cap_low_per_dim  # research, idea #46
            if low > 0 and budget <= variant.cma_handoff_cap_low_max_per_dim * dim:
                cap = min(cap, low) if cap > 0 else low
            if cap > 0 and dim >= variant.cma_handoff_cap_min_dim:
                atm_share = min(atm_share, int(cap * dim))
            # ``basin_keep_atm`` (cycle 60): the descents do not shorten ATM's sample
            objective.set_limit(max(1, atm_share + descent_calls))
            recording[0] = variant.cma_warm_start or variant.cma_placement == "mlsl"
        last_widths: tuple[float, ...] = ()
        last_width = float("nan")
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
            last_width = outcome.width
            nrestarts += int(outcome.restarted)
            x_start = outcome.x
            if config.strict_budget:
                known_cost = outcome.cost
            if outcome.cost <= best_cost:
                best_cost = outcome.cost
                best_x = outcome.x
            if (
                outcome.restarted
                and not out_of_budget
                and variant.restart_mode in ("uniform", "ipop")
            ):
                assert box is not None  # ``needs_box`` holds for these modes
                x_start = box.lower + (box.upper - box.lower) * rng.rand(len(x0))
                cost = objective.evaluate_point(x_start)
                if cost is None:
                    break
                known_cost = cost
                if cost <= best_cost:
                    best_cost, best_x = cost, x_start
                if variant.restart_mode == "ipop":
                    pool = ipop_pool(config, variant, nrestarts)
                    restart_config = with_pool(config, pool)
                    search = Search(objective, rng, restart_config, None, profiler, box)
                    conductor = ResourceConductor(ConductorConfig(total_resources=pool))
                    allocation = conductor.step(conductor.rest, np.zeros(N_SAMPLERS))
        if refine and box is not None:
            objective.release()
            tail = _local_tail(
                config, objective, rng, best_x, best_cost, last_width, box
            )
            # the local search starts from the best point and never gets worse
            best_x, best_cost = tail.x, tail.cost
        if handoff and objective.calls < budget:
            assert box is not None
            objective.set_limit(budget)
            nonseparable = False  # the separable search's check failed
            if sep:
                from .step_phase import run_step_phase

                stuck: dict[str, Any] = (
                    {"pair_when_stuck": True}
                    if variant.sep_check == "stall"
                    and budget >= variant.sep_stall_min_per_dim * dim
                    else {}
                )
                if variant.step_box_check:  # the novel-ideas toy: only when on
                    stuck["box_check"] = True
                probe = run_step_phase(
                    objective,
                    best_x,
                    best_cost,
                    box,
                    int(variant.sep_share * (budget - objective.calls)),
                    parabolic=variant.sep_parabolic
                    and variant.sep_parabolic_scope == "both",
                    parabolic_late=variant.sep_parabolic_late,
                    pair_check=variant.sep_check == "pair",
                    **stuck,
                )
                if probe.cost <= best_cost:
                    best_cost, best_x = probe.cost, probe.x
                nonseparable = probe.stalled
            objective.set_limit(budget)
            recording[0] = False
            handover = _Handover(
                variant=variant,
                # ``fork_seed`` (research): an independent continuation from here
                rng=rng
                if variant.fork_seed < 0
                else np.random.RandomState(variant.fork_seed),
                box=box,
                budget=budget,
                dim=dim,
                use_model=use_model,
                cma_options=cma_options,
                pop_options=pop_options,
                nonseparable=nonseparable,
                mlsl=mlsl,
                recorded_x=recorded_x,
                recorded_f=recorded_f,
                search=search,
                allocation=allocation,
                best_x=best_x,
                best_cost=best_cost,
                a0=_warm_shape(variant, dim, recorded_x, recorded_f),
                last_widths=last_widths,
                warm_samples=None if warm is None else (warm.x, warm.f),
            )
            _local_phase(handover, objective)
            best_x, best_cost = handover.best_x, handover.best_cost
        if centre is not None and centre[1] < best_cost:  # ``centre_point``
            best_x, best_cost = centre

    return OptimizeResult(
        x=np.array(best_x),
        fun=float(best_cost),
        nfev=objective.calls,
        nrestarts=nrestarts,
        timings=profiler.report(),
    )


def _additive_route(
    objective: CountedObjective,
    rng: np.random.RandomState,
    box: Box,
    budget: int,
    variant: Variant,
    x0: np.ndarray,
    chain: bool = True,
    route: bool = True,
    strict: bool = False,
) -> tuple[bool, np.ndarray, float]:
    """``chain_route`` / ``sep_route`` (cycle 50): (routed, best point, value).

    ``chain=False`` skips the chain branch and ``route=False`` the separable one
    (``sep_route_low`` below ``additive_min_dim``; research, gaps3).
    The pair test at ``x0``. A chain reading with ``chain_route`` (and enough budget)
    takes the band check, and on a linear chain ``run_coord_hess`` from ``x0`` spends
    the rest. A separable reading with ``sep_route`` (and enough budget) gets
    restarted STEP (parabolic steps, the separability check off), the first from the
    pair test's best point, then from uniform points. Otherwise the caller's path
    continues; the evaluations spent here are counted, their best point kept.
    """
    dim = len(x0)
    read = (
        chain_phase.pair_test(objective, rng, x0, box, strict=True)
        if strict
        else chain_phase.pair_test(objective, rng, x0, box)
    )
    if read is None:
        return True, np.array(x0, dtype=float), float("inf")
    reading, best_x, best_cost = read

    def keep(x: np.ndarray, cost: float) -> None:
        nonlocal best_x, best_cost
        if cost < best_cost:
            best_x, best_cost = np.array(x, dtype=float), float(cost)

    if strict and reading == "separable":
        # ``"strict"`` (gaps3): a separable reading is read again at a second random
        # point t (at D 2 the test has one pair, and f7's plateaus can be additive
        # there); only two separable readings route
        again = chain_phase.pair_test(objective, rng, x0, box, strict=True)
        if again is None:
            return True, best_x, best_cost
        reading = again[0]
        keep(again[1], again[2])
    if (
        reading == "chain"
        and chain
        and variant.chain_route
        and budget >= variant.chain_min_per_dim * dim
    ):
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
        and route
        and variant.sep_route
        and budget >= variant.sep_route_min_per_dim * dim
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
            x = box.lower + (box.upper - box.lower) * rng.rand(dim)
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
    if variant.cma_model == "lq" and dim <= variant.cma_lq_max_dim:
        options["model_kind"] = "lq"
        options["model_full"] = dim <= variant.cma_model_max_dim
        options["model_sigma"] = variant.cma_model_sigma
    if variant.cma_restart_rule != "legacy":
        options["restart_rule"] = variant.cma_restart_rule
    if variant.cma_active:
        options["active"] = True
    if variant.cma_model_high == "diagonal" and dim > variant.cma_model_max_dim:
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
    """``run_es_phase``'s ``small_switch`` argument (the small10 toy), passed only
    inside its envelope so every other call stays exactly as recorded."""
    if not (
        variant.small_switch
        and dim <= variant.small_switch_max_dim
        and budget <= variant.small_switch_max_per_dim * dim
    ):
        return {}
    rule = qn_phase.SmallSwitch(
        trials=variant.small_switch_trials,
        parallel_cos=variant.small_switch_cos,
        keep_unit=variant.small_switch_keep_unit,
        radius=variant.small_switch_radius,
        rest=variant.small_switch_rest,
    )
    return {"small_switch": rule}


def _prefix_ridge_kw(variant: Variant, dim: int, budget: int) -> dict[str, Any]:
    """``run_es_phase``'s ridge reading (the f13 toy, ``prefix_ridge_after``), passed
    only when on and inside its ``prefix_ridge_max_per_dim`` envelope, so every other
    call stays exactly as recorded."""
    if not variant.prefix_ridge_after or (
        variant.prefix_ridge_max_per_dim
        and budget > variant.prefix_ridge_max_per_dim * dim
    ):
        return {}
    return {
        "ridge_after": variant.prefix_ridge_after,
        "ridge_radius": variant.prefix_ridge_radius,
    }


def _parked2_kw(variant: Variant, dim: int, budget: int) -> dict[str, Any]:
    """``run_es_phase``'s research options (off in every preset), passed only when on so
    every other call stays exactly as recorded."""
    on: dict[str, Any] = {}
    if variant.prefix_nan_clip:
        on["nan_clip"] = True
    if variant.prefix_nan_reprobe:
        on["nan_reprobe"] = True
    if variant.prefix_plateau_ties:
        on["plateau_ties"] = variant.prefix_plateau_ties
    if (
        0 < variant.prefix_fractal_min_dim <= dim
        and budget >= variant.prefix_fractal_min_per_dim * dim
    ):
        on["fractal_route"] = True
    if variant.prefix_descent_retries:
        on["descent_retries"] = variant.prefix_descent_retries
    if variant.prefix_low_beta and dim >= variant.prefix_low_beta_min_dim:  # gaps4
        on["low_beta"] = variant.prefix_low_beta
    return on


def _prefix_runs(variant: Variant, dim: int, budget: int) -> bool:
    """Whether the ES prefix runs (``es_phase`` and its budget limits; needs a box)."""
    limit = (
        variant.es_small_max_budget_per_dim
        if dim <= variant.es_small_dim
        else variant.es_max_budget_per_dim
    )
    return variant.es_phase != "none" and budget <= limit * dim


def _prefix_kind(variant: Variant, dim: int, budget: int) -> str:
    """The local search the ES prefix runs after its design (``run_es_phase``'s kind).

    In order:

    * at ``dim <= nm_max_dim``: one SLSQP descent then Nelder-Mead (``"slsqpnm"``:
      ``nm_stage="qnprefix"``, or ``"qnprefix30"`` / ``"qnprefix30g"`` up to 30 x
      dim); the gated prefix then Nelder-Mead from ``QN_MIN_BUDGET`` evaluations
      (``"slsqpnmg"``, ``"qnprefix30g"``); Nelder-Mead alone (``"nm"``);
    * the (1+1)-CMA-ES (``"cma"``) where the configured kind needs more budget:
      ``lq2`` below ``es_lq_min_budget_per_dim``, the curvature-gated
      ``slsqplq`` below ``QN_MIN_BUDGET_PER_DIM`` x dim, and ``slsqplq10``,
      ``slsqplq10c`` and ``slsqplq10h`` there unless the budget reaches
      ``QN_MIN_BUDGET``; ``"ctrk"`` (the start-point probe of ``slsqplq10hk``) there
      instead;
    * ``"slsqplqc"`` (the two-direction probe) for ``slsqplq10c``, ``slsqplq10h``
      and ``slsqplq10hk`` below 30 x dim; ``"slsqplq"`` for the four ``slsqplq10``
      kinds otherwise; else ``es_kind`` itself.
    * cycle 33: ``es_kind="slsqplq10hd"`` is ``slsqplq10h`` with ``"slsqpdes"`` (one
      SLSQP descent from the design's best, then the (1+1)) where ``slsqplq10h`` skips
      the probe (below 30 x dim and ``QN_MIN_BUDGET``); cycle 58: below
      ``des_max_per_dim`` x dim instead.
    """
    small = budget < QN_MIN_BUDGET_PER_DIM * dim
    if dim <= variant.nm_max_dim and (
        variant.nm_stage == "qnprefix"
        or (
            variant.nm_stage in ("qnprefix30", "qnprefix30g")
            and budget <= QN_MIN_BUDGET_PER_DIM * dim
        )
    ):
        return "slsqpnm"
    if (
        dim <= variant.nm_max_dim
        and variant.nm_stage == "qnprefix30g"
        and budget >= QN_MIN_BUDGET
    ):
        return "slsqpnmg"
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
        or (
            kind in ("slsqplq10", "slsqplq10c", "slsqplq10h")
            and small
            and budget < QN_MIN_BUDGET
        )
    ):
        return "cma"
    if kind == "slsqplq10hk" and small and budget < QN_MIN_BUDGET:
        return "ctrk"
    if kind in ("slsqplq10c", "slsqplq10h", "slsqplq10hk", "slsqplq10hd") and small:
        return "slsqplqc"
    if kind in ("slsqplq10", "slsqplq10c", "slsqplq10h", "slsqplq10hk", "slsqplq10hd"):
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
    together they give :meth:`start_sigma`. ``search`` and ``allocation`` are ATM's,
    for ``cma_restart="atm"``.
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
    mlsl: bool
    recorded_x: list[np.ndarray]
    recorded_f: list[float]
    search: Search
    allocation: np.ndarray
    best_x: np.ndarray
    best_cost: float
    a0: np.ndarray | None
    last_widths: tuple[float, ...]
    sigma_override: list[float] = field(default_factory=list)
    # ``rugged_direct_beta`` (research, gaps3): the finisher's populations start at once
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
        return np.asarray(box.lower + (box.upper - box.lower) * self.rng.rand(self.dim))

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
    if dim <= variant.nm_max_dim and variant.nm_stage not in (
        "prefix",
        "qnprefix",
        "qnprefix30",
        "qnprefix30g",
    ):
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
    nonsep = (
        h.nonseparable and dim >= variant.cma_local_nonsep_min_dim
    ) or variant.local_force != "none"
    if nonsep and variant.cma_local_nonsep in qn_phase.QN_ROUTES:
        options = _qn_local_phase(
            qn_phase.QN_ROUTES[variant.cma_local_nonsep], h, objective
        )
        if options is None:
            return h.done()
        local_pop_options = options
    elif nonsep and variant.cma_local_nonsep in ("ipop", "population"):
        # IPOP population runs with the default lambda (and the population
        # surrogate, if set) for non-separable functions.
        local_kind = variant.cma_local_nonsep
        local_pop_options = {
            k: v for k, v in h.pop_options.items() if k != "lambda_rule"
        }
    if local_kind == "then_ipop" and variant.pop_gate_per_dim > 0:
        _finish_gated_then_ipop(h, objective, local_pop_options)
    elif local_kind == "then_ipop":
        _finish_then_ipop(h, objective)
    elif local_kind in ("population", "ipop"):
        _finish_population(h, objective, local_kind, local_pop_options)
    elif variant.cma_regimes != "none":
        _finish_regimes(h, objective)
    else:
        _finish_one_plus_one(h, objective)
    return h.done()


def _qn_local_phase(
    route: QnRoute, h: _Handover, objective: Any
) -> dict[str, object] | None:
    """The SLSQP routes of ``cma_local_nonsep`` (cycles 13-18; ``QN_ROUTES``).

    Reads the curvature at the best point (after a plateau check, ``plateau_check``)
    and, where it reads smooth, runs restarted SLSQP: for the whole local phase
    (``staged=False``), or for a stage (a share of the local budget, ``capped`` at
    ``QN_LOCAL_CAP_PER_DIM`` x dim, ``seeded`` from ATM's archive) that the finisher
    continues, starting at the stage's hand-over sigma. With ``extend``, a stage
    whose first descent was still running continues (``"capped"``: for at most
    another capped stage); with ``basins``, converged descents that end in
    separate basins (hill-valley test) keep restarting SLSQP to the end. Rugged or
    plateau readings get ``steps`` populations (``rugged_steps``), or IPOP
    populations with 4x lambda on plateaus (``plateau_ipop4``).

    Returns the finisher's population options, or None when the local phase is
    over (budget spent, or a route that runs to the end).
    """
    variant, dim, box, rng, budget = h.variant, h.dim, h.box, h.rng, h.budget
    plateau = False
    if route.plateau_check:
        checked = qn_phase.plateau_check(objective, rng, h.best_x, h.best_cost, box)
        if checked is None:
            return None
        plateau = checked[0]
    # the novel-ideas toy: passed only when on
    faces_on: dict[str, Any] = (
        {"faces": True} if variant.probe_faces == "tangent" else {}
    )
    probed = (
        None
        if plateau
        else qn_phase.curvature_beta(
            objective, rng, h.best_x, h.best_cost, box, **faces_on
        )
    )
    if probed is not None:
        beta, probe_x, probe_cost, _ = probed
        h.keep(probe_x, probe_cost)
    elif not plateau:
        return None
    smooth = probed is not None and beta >= qn_phase.BETA_THRESHOLD
    if probed is not None and h.variant.local_force != "none":  # research only
        smooth = h.variant.local_force == "smooth"
    fd_step = QN_FD_STEP if route.fd_step else None
    fallback = variant.qn_fallback == "lbfgsb"
    # passed only when on, so the other routes' calls stay exactly as recorded
    on: dict[str, Any] = {"fallback": True} if fallback else {}
    if variant.qn_false_success:  # research, idea #23
        on["false_success"] = True
    if variant.qn_floor != "none":  # research, ideas #24 / #32 / #55
        on["floor_bfgs"] = variant.qn_floor
    if (
        smooth
        and route.smooth_bfgs_max_per_dim
        and budget <= route.smooth_bfgs_max_per_dim * dim
    ):
        # ``smooth_bfgs_max_per_dim`` (cycle 56): at small budgets the scaled-FD BFGS
        # takes the rest of a smooth reading, instead of the SLSQP stage and finisher
        # passed only when on, so the recorded arms stay exactly as recorded
        bfgs_on: dict[str, Any] = {}
        if route.smooth_bfgs_descents:
            bfgs_on["max_descents"] = route.smooth_bfgs_descents
        if route.smooth_bfgs_box_qp:
            bfgs_on["box_qp"] = True
        given = budget - objective.calls
        polished = qn_phase.run_scaled_bfgs(
            objective,
            rng,
            h.best_x,
            h.best_cost,
            box,
            given,
            **bfgs_on,
        )
        h.keep(polished.x, polished.cost)
        again_budget = int(variant.smooth_bfgs_second * polished.used)
        if (
            variant.smooth_bfgs_second > 0
            and route.smooth_bfgs_descents == 1
            and 0 < polished.used < given
            and budget - objective.calls >= again_budget > 0
        ):
            # ``smooth_bfgs_second`` (research, gaps4): the descent ended on its own
            # (f9's second minimum at D 10, 300 x dim); one more from a uniform point
            again = qn_phase.run_scaled_bfgs(
                objective, rng, h.uniform_point(), np.inf, box, again_budget, **bfgs_on
            )
            h.keep(again.x, again.cost)
        # ``smooth_bfgs_descents`` (cycle 59): after that many descents the usual staged
        # route continues from the BFGS's best point
        if not route.smooth_bfgs_descents:
            return None
    if smooth and route.staged:
        stage = qn_phase.QN_LOCAL_SHARE * (budget - objective.calls)
        if route.capped:
            stage = min(stage, qn_phase.QN_LOCAL_CAP_PER_DIM * dim)
        ends: list[Descent] = []
        gate: list[bool] = []
        # passed only when on, so the other routes' calls stay exactly as recorded
        gated: dict[str, Any] = (
            {"gate_once": True, "gate_out": gate} if route.cap_gate else {}
        )
        # ``stage_extend_restart`` (parked2, the SLSQP-cells toy's ``stagerx``)
        if (
            variant.stage_extend_restart
            and route.extend_in_descent
            and (
                variant.stage_extend_max_per_dim <= 0
                or budget <= variant.stage_extend_max_per_dim * dim
            )
        ):
            gated["extend_restart"] = True
        seeds = None
        if route.seeded and h.recorded_x:
            order = np.argsort(np.array(h.recorded_f), kind="stable")
            count = max(3, int(qn_phase.SEED_SHARE * len(order)))
            seeds = np.array(h.recorded_x)[order[:count]]
        qn = qn_phase.run_slsqp_restarts(
            objective,
            rng,
            h.best_x,
            h.best_cost,
            box,
            int(stage),
            seeds,
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
        if fallback and qn_phase.failure_majority([d.status for d in ends]):
            # Most of the stage's SLSQP descents failed: in c31's own runs only f12
            # (bent cigar) does this (cycle 27 trace). L-BFGS-B restarts take the
            # rest of the budget (cycle 28, COCO leaders report rank 1).
            bent = qn_phase.run_lbfgsb_restarts(
                objective, rng, h.best_x, h.best_cost, box, budget - objective.calls
            )
            h.keep(bent.x, bent.cost)
            return None
        if route.extend != "none" and (
            not ends or (passed and ends[0].status != 0)
        ):  # ``cap_gate``: an extended descent that failed (f12) still continues
            # The first descent was still running at the cap (f8, f9, f13 at
            # D 20-40): let SLSQP go on for a share of what is left (``"capped"``:
            # at most another capped stage), then the CMA finisher.
            more_budget = qn_phase.QN_EXTEND_SHARE * (budget - objective.calls)
            if route.extend == "capped":
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
            failed = sum(d.status != 0 for d in ends)
            if (
                route.failures_to_population
                and len(ends) >= 2
                and failed > len(ends) - failed
            ):
                # Most descents failed: in c30's own runs only f12 (bent cigar)
                # does this at D 10-40 (cycle 25). Population CMA-ES restarts take
                # the rest ("qn13": ranked by the lq2 surrogate, HCMA's saACM idea;
                # "qn13n": plain IPOP).
                bent = run_population_restarts(
                    objective,
                    rng,
                    h.best_x,
                    h.best_cost,
                    INITIAL_SIGMA_SHARE * h.width,
                    box,
                    budget - objective.calls,
                    True,
                    active=variant.pop_active,
                    surrogate=route.failures_surrogate,
                )
                h.keep(bent.x, bent.cost)
                return None
            basins = qn_phase.multi_basin(objective, ends, box)
            if basins is None:
                return None
            if basins:
                # Smooth with several basins (Gallagher-like): keep restarting
                # SLSQP for the rest of the budget (``basins_sampled``: from the
                # best unclaimed sample point, with the gap-scaled stop).
                warm_samples: dict[str, Any] = (
                    {} if h.warm_samples is None else {"samples": h.warm_samples}
                )
                more = (
                    qn_phase.run_sampled_restarts(
                        objective,
                        rng,
                        h.best_x,
                        h.best_cost,
                        box,
                        budget - objective.calls,
                        fd_step=None if route.basins_scipy_step else fd_step,
                        # research (idea #33): passed only when on
                        **(
                            {"polish_per_dim": variant.sampled_polish_per_dim}
                            if variant.sampled_polish_per_dim > 0
                            else {}
                        ),
                        **warm_samples,
                    )
                    if route.basins_sampled
                    else qn_phase.run_slsqp_restarts(
                        objective,
                        rng,
                        h.best_x,
                        h.best_cost,
                        box,
                        budget - objective.calls,
                        # ``basins_scipy_step``: SciPy's step here (more, cheaper
                        # descents for Gallagher's lottery)
                        fd_step=None if route.basins_scipy_step else fd_step,
                        **on,
                    )
                )
                h.keep(more.x, more.cost)
                return None
        left = budget - objective.calls
        if (
            route.second_descent_min_left_per_dim
            and passed
            and len(ends) == 1
            and ends[0].status == 0
            and left >= route.second_descent_min_left_per_dim * dim
        ):
            # Rosenbrock's second minimum (f8, f9 at D 20-40; cycle 33): one more
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
            # f13 (cycles 36, 44): the gate read the first descent at its cap as not
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
        if not route.keep_warm_shape:
            h.a0 = None
        return h.pop_options
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
        and probed is not None
        and not smooth
        and (
            (
                dim >= route.rugged_ipop_min_dim
                and left >= qn_phase.RUGGED_IPOP_MIN_LEFT_PER_DIM * dim
            )
            or (low and left >= qn_phase.RUGGED_IPOP_LOW_LEFT_PER_DIM * dim)
        )
    ):
        # Rugged with a large budget left (f15-f18, f23; cycle 29): the plateau
        # route's schedule, no (1+1) first and no protection.
        if route.fractal_exempt and not low:
            # ``fract`` (cycle 35): a fractal reading (f23-like) is no case for
            # populations; restarted SLSQP leads f23 at D 20-40 on the field.
            micro = qn_phase.micro_slopes(
                objective,
                h.best_x,
                h.best_cost,
                box,
                centered=True,
            )
            if micro is None:
                return None
            fractal, _slopes, x_m, cost_m = micro
            h.keep(x_m, cost_m)
            if fractal:
                rest = qn_phase.run_slsqp_restarts(
                    objective, rng, h.best_x, h.best_cost, box, budget - objective.calls
                )
                h.keep(rest.x, rest.cost)
                return None
            # Otherwise the populations keep the schedule they had before the probe
            # (its evaluations come off their end), so the run stays ``qn19``'s.
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
            active=variant.pop_active,
        )
        h.keep(rough.x, rough.cost)
        return None
    # The c45 candidate's rugged branch composes in this order (each only when on):
    # ``rugged_direct_beta`` (gaps3) reads the hand-over probe's beta first; a reading
    # in [``rugged_direct_min_beta``, ``rugged_direct_beta``) skips the capped try and
    # the finisher's first (1+1). Otherwise the try runs (with the f13 toy's cap and
    # envelope overrides) and ``rugged_try_rest`` may continue it: kinked tries (f13)
    # read above 1.0, so ``cutkink`` and a ``rugged_direct_beta`` of 1.0 act on
    # different runs. ``probe_faces`` (novel ideas) changes the beta both read.
    direct = (
        variant.rugged_direct_beta > 0
        and probed is not None
        and not smooth
        and beta < variant.rugged_direct_beta
        # ``rugged_direct_min_beta`` (0: no lower bound)
        and (
            variant.rugged_direct_min_beta <= 0
            or beta >= variant.rugged_direct_min_beta
        )
    )
    # the f13 toy's overrides of the try's cap and budget envelope (0: the route's)
    try_per_dim = variant.rugged_try_cap_per_dim or route.rugged_try_per_dim
    try_max_budget_per_dim = (
        variant.rugged_try_max_per_dim or route.rugged_try_max_budget_per_dim
    )
    if (
        route.rugged_try_per_dim
        and not direct
        and probed is not None
        and not smooth
        and budget - objective.calls >= 2 * try_per_dim * dim
        and (not route.rugged_try_max_dim or dim <= route.rugged_try_max_dim)
        and (not try_max_budget_per_dim or budget <= try_max_budget_per_dim * dim)
    ):
        # ``rugged_try_per_dim`` (cycle 44): f13 reads rugged at ATM's best point
        # below D 20; one capped scaled-FD BFGS descent first, then the populations
        cap = int(try_per_dim * dim)
        # ``rugged_try_rest`` (the f13 toy): a try still descending at its cap goes on
        # (the same descent, its inverse Hessian kept) to the end of the budget;
        # ``"cutkink"`` only where its best point then reads a kink (f13's ridge).
        # Passed only when on, so the recorded calls stay exactly as recorded
        extend: dict[str, Any] = {}
        if variant.rugged_try_rest != "none":
            extend["extend_to"] = budget - objective.calls
            if variant.rugged_try_rest == "cutkink":
                extend["extend_gate"] = lambda x_e, cost_e: _kink_reads(
                    objective, x_e, cost_e, box
                )
        tried = qn_phase.run_scaled_bfgs(
            objective, rng, h.best_x, h.best_cost, box, cap, max_descents=1, **extend
        )
        h.keep(tried.x, tried.cost)
    if route.rugged_steps:
        # Rugged or plateau: populations sized to the budget left (``rugged_grow``:
        # growing while the budget is large).
        if direct:
            h.direct_populations = True
        return {
            **h.pop_options,
            "lambda_rule": "stepsgrow" if route.rugged_grow else "steps",
        }
    if smooth:  # ``staged=False``: SLSQP for the whole local phase
        qn = qn_phase.run_slsqp_restarts(
            objective,
            rng,
            h.best_x,
            h.best_cost,
            box,
            budget - objective.calls,
            **on,
        )
        h.keep(qn.x, qn.cost)
        return None
    return h.pop_options


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
        # ``rugged_direct_beta`` (research, gaps3): the populations start at once,
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
            placement="uniform",
            archive=None,
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
        cov0 = None
        if first.capped and variant.pop_switch_start != "uniform":
            start, start_cost = h.best_x, h.best_cost
            factor = 1 if variant.pop_switch_start == "carry" else 2
            pop_sigma = float(np.clip(factor * first.sigma, 1e-8 * width, pop_sigma))
            if variant.pop_switch_start == "carry":
                cov0 = first.shape
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
            cov0=cov0,
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


def _finish_population(
    h: _Handover, objective: Any, kind: str, pop_options: dict[str, object]
) -> None:
    """``cma_local`` ``"population"`` or ``"ipop"``: population runs from the best."""
    local = run_population_restarts(
        objective,
        h.rng,
        h.best_x,
        h.best_cost,
        h.start_sigma(),
        h.box,
        h.budget - objective.calls,
        kind == "ipop",
        **pop_options,  # type: ignore[arg-type]
    )
    h.keep(local.x, local.cost)


def _finish_regimes(h: _Handover, objective: Any) -> None:
    """``cma_regimes``: BIPOP-style small and large restarts (``regimes``)."""
    kwargs: dict[str, object] = {"model": h.use_model, "a0": h.a0}
    kwargs.update(h.cma_options)
    h.best_x, h.best_cost, _runs = run_regimes(
        objective,
        h.rng,
        h.best_x,
        h.best_cost,
        h.start_sigma(),
        h.box,
        h.budget,
        h.variant.cma_regimes == "nbipop",
        kwargs,
    )


def _finish_one_plus_one(h: _Handover, objective: Any) -> None:
    """The restarted (1+1)-CMA-ES to the end of the budget.

    With ``cma_restart="atm"`` a stalled run hands back to ATM from the best point
    for as long as the run took (at least 50 dim evaluations), then the CMA phase
    restarts from ATM's best.
    """
    alternate = h.variant.cma_restart == "atm"
    budget = h.budget
    while objective.calls < budget:
        es = run_cma(
            objective,
            h.rng,
            h.best_x,
            h.best_cost,
            h.start_sigma(),
            h.box,
            budget - objective.calls,
            restarts=not alternate,
            model=h.use_model,
            a0=h.a0,
            placement="mlsl" if h.mlsl else "uniform",
            archive=(h.recorded_x, h.recorded_f) if h.mlsl else None,
            **h.cma_options,  # type: ignore[arg-type]
        )
        h.a0 = None
        h.keep(es.x, es.cost)
        if not es.stalled or objective.calls >= budget:
            break
        # ATM resumes from the best point for as long as the CMA run took.
        segment = max(es.used, 50 * h.dim)
        objective.set_limit(min(budget, objective.calls + segment))
        segment_done = False
        while not segment_done:
            outcome = h.search.run(h.best_x, h.allocation, h.best_cost, None, True)
            segment_done = outcome.out_of_budget
            h.last_widths = outcome.widths
            h.keep(outcome.x, outcome.cost)
        objective.set_limit(budget)


# ------------------------------------------------------------ late refinement ----
def _tail_budget(config: ATMConfig, budget: int, dim: int, box: Box | None) -> int:
    """Evaluations kept back for the local search at the end (``refine_fraction``)."""
    variant = config.variant
    if (
        box is not None
        and budget >= variant.refine_min_budget_per_dim * dim
        and dim <= variant.refine_max_dim
    ):
        return int(variant.refine_fraction * budget)
    return 0


def _local_tail(
    config: ATMConfig,
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    width: float,
    box: Box,
) -> EsResult:
    """The simplex (or (1+1)-ES) search from ATM's best point on the last budget.

    ``width`` is the geometric mean of ATM's final widths (``SearchOutcome.width``).
    """
    variant = config.variant
    local = (
        run_nelder_mead
        if variant.local_method == "nelder_mead" and len(x) <= variant.nm_refine_max_dim
        else run_es
    )
    return local(
        objective,
        rng,
        x,
        float(cost),
        max(width, 1e-300) / np.sqrt(3.0),
        box,
        objective.remaining,
    )
