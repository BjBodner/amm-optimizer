"""A (1+1)-ES prefix for the search (variant ``es_phase``).

On smooth unimodal landscapes a (1+1)-ES with the 1/5 success rule reaches a
given precision in a few times fewer evaluations than ATM's pool-based
iterations, which pay for a whole pool before the incumbent moves. This module
spends the first share of the budget on such an ES (after a small uniform design
to pick the start) and hands the incumbent and the step size to ATM, which then
takes over for the harder part of the landscape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .bounds import Box, clip
from .objective import CountedObjective

# The 1/5 rule in its usual exponential form: grow by exp(0.8 / dim) on success
# and shrink by exp(-0.2 / dim) on failure, which is stationary at 20% success.
SUCCESS_GROWTH = 0.8
FAILURE_SHRINK = 0.2
INITIAL_SIGMA_SHARE = 0.2  # of the mean box width
PLATEAU_SEED = 6_271  # ``plateau_pop``'s fixed probe directions (per dim)
MIN_SIGMA_SHARE = 1e-12  # of the mean box width: below this the ES has converged
# A coordinate sitting on a wall is pushed back inside by half of the noise, which
# makes a step that leaves all of k pinned coordinates untouched as rare as 2**-k
# and stalls the ES at a corner optimum (bbob f5) in more than a few dimensions.
# With this probability a noise component that would pull a pinned coordinate
# inside is mirrored to point outward (the candidate is then clipped back onto the
# wall), so pinned coordinates mostly stay put and the free ones keep improving.
PIN_STICK = 0.75


@dataclass(frozen=True)
class EsResult:
    x: np.ndarray
    cost: float
    sigma: float
    # Evaluations spent, and whether the run ended by stagnation (``run_cma``
    # with ``restarts=False``) rather than by its budget.
    used: int = 0
    stalled: bool = False
    # ``run_cma(run_cap=...)``: the run handed over at its cap, not at a stall.
    capped: bool = False
    # With ``capped``: the run's covariance shape A A^T scaled to unit determinant.
    shape: np.ndarray | None = None
    # With ``capped``: the share of the run's last 10 dim samples that tied.
    ties: float = 0.0


def run_es_phase(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x0: np.ndarray,
    box: Box,
    evaluations: int,
    init_points: int,
    kind: str = "isotropic",
    model: bool = False,
    cma_options: dict[str, object] | None = None,
    sigma_scale: float = 1.0,
    step_share: float = 0.0,
    lq_nonsep_only: bool = False,
    rugged_kind: str = "cma",
    step_parabolic: bool = False,
    step_parabolic_late: bool = False,
    fd_step: float | None = None,
    step_pair_check: bool = False,
    simplex: bool = False,
    qn_fallback: bool = False,
    step_pair_when_stuck: bool = False,
    sep_smooth_probe: bool = False,
    plateau_pop: bool = False,
    plateau_rugged: bool = False,
    small_switch: Any = None,
    nan_reading: str = "rugged",
    ridge_after: int = 0,
    ridge_radius: float = 0.1,
    qn_false_success: bool = False,
    qn_floor: str = "none",
    probe_faces: bool = False,
    step_box_check: bool = False,
    nan_clip: bool = False,
    nan_reprobe: bool = False,
    plateau_ties: int = 0,
    fractal_route: bool = False,
    descent_retries: int = 0,
    low_beta: float = 0.0,
    design: Any = None,  # np.ndarray | None (Any for the callers' **options)
) -> EsResult:
    """Spend up to ``evaluations`` on a uniform design and a local search from its best.

    The local search is the (1+1)-ES, or Nelder-Mead if ``simplex``.

    Candidates are clipped onto the box, so a corner optimum is reached exactly.
    A failed or NaN evaluation never replaces the incumbent. Returns the best
    point, its cost and the final step size. ``kind="cma"`` runs the restarted
    (1+1)-Cholesky-CMA-ES of ``cma_phase`` after the design instead;
    ``kind="lq2"`` IPOP population runs ranked by ``surrogate``;
    ``kind="slsqp"`` (internal: where a probe read smooth): restarted SLSQP
    (``qn_phase``); ``"slsqpnm"``: one SLSQP descent, then the Nelder-Mead
    restarts of ``"nm"`` (cycle 17, D <= 2);
    ``"slsqpnmg"``: the two-direction curvature probe, then restarted SLSQP where it
    reads smooth, else ``"nm"`` (cycle 19, D <= 2); ``"slsqplq"``: restarted SLSQP
    where ``qn_phase.curvature_beta`` reads the function as smooth, else
    ``rugged_kind`` (cycle 12, ``c25``); ``"slsqplqc"``: ``"slsqplq"`` with the
    two-direction probe (cycle 17).
    ``"ctrk"``: the median two-scale curvature probe (``curvature_kappa``, 8
    evaluations) at the start point, then restarted SLSQP from there where it reads
    smooth, else ``"cma"`` (cycle 26).
    ``fd_step``: SLSQP's forward-difference step (cycle 20; SciPy's default if None).
    ``step_pair_check``: the separable search's two-axis additivity check (cycle 23).
    ``qn_fallback``: SLSQP hands a failing call to L-BFGS-B (cycle 28,
    ``run_slsqp_restarts``'s ``fallback``).
    ``"slsqpdes"`` (cycle 33, ``q7ed``): one SLSQP descent from the design's best, then
    the (1+1) on the rest.
    ``lq_nonsep_only``: the (1+1) instead of ``lq2`` or ``slsqplq`` when the
    separable search ran and its separability check passed (``sep_smooth_probe``,
    cycle 38: for ``"slsqplq"`` and ``"slsqplqc"`` the curvature probe decides there
    too, SLSQP where smooth and the (1+1) where rugged; f2 at D 5, 30 x D trailed
    restarted SLSQP by .65).
    ``small_switch`` (a ``qn_phase.SmallSwitch``; the small10 toy): the first SLSQP
    descent of ``"slsqpdes"`` and ``"slsqp"`` is read online and may hand the rest to
    restarted COBYQA (``cobyqa_phase.run_cobyqa_from``); with its ``rest``, COBYQA
    also replaces the (1+1) after ``"slsqpdes"``'s descent and after a rugged probe.
    ``nan_reading`` (cycle 64, the SLSQP-cells toy): where an undefined probe slope
    of ``"slsqplq"`` / ``"slsqplqc"`` goes (``curvature_beta``'s NaN: fewer than two
    positive mirrored losses, mostly probe pairs clipped at a box face): ``"rugged"``
    (as before), ``"smooth"``, or ``"plateau"`` (smooth unless one ``plateau_check``
    then reads a plateau, which keeps f7's plateaus on the rugged kind).
    ``ridge_after`` (the f13 toy): where the probe sends the prefix to SLSQP, its
    first descent is read once after that many iterations (``qn_phase.ridge_reading``);
    a ridge hands the rest to restarted COBYQA (``cobyqa_phase``, initial trust
    radius ``ridge_radius`` x box width). With ``small_switch`` too, the switch's
    checkpoints (after the first two line searches) come first; the ridge is read
    only on a descent the switch kept.
    ``qn_false_success`` (research, idea #23): the SLSQP calls count a status-0 exit
    that never left its start as failed (``run_slsqp_restarts``'s ``false_success``).
    ``qn_floor`` (research, ideas #24 / #32 / #55): ``run_slsqp_restarts``'s
    ``floor_bfgs`` (``"bfgs"`` or ``"cd"``).
    ``probe_faces`` (``Variant.probe_faces="tangent"``, the novel-ideas toy): the
    curvature probes keep every mirrored pair inside the box
    (``curvature_beta(faces=True)``). ``step_box_check``
    (``Variant.step_box_check``): the separable search's early check
    (``run_step_phase(box_check=True)``).

    Research options from a second screen of parked ideas, each off by default:

    * ``nan_clip``: with ``nan_reading`` not ``"rugged"``, an undefined reading goes
      that way only when clipping explains it (no probe pair whose mirror stayed inside
      the box has a negative loss); a negative unclipped loss is concavity, so the
      reading stays rugged.
    * ``nan_reprobe``: with ``nan_reading`` not ``"rugged"``, an undefined reading is
      taken again from the point moved inward by the probe's largest scale (no pair
      clipped there); only a smooth second reading goes that way.
    * ``plateau_ties``: the plateau checks (``plateau_pop``'s, and the NaN reading's
      with ``nan_reading="plateau"``) run only when the separable search's own probes
      repeated the incumbent's value at least that often (``run_step_phase``'s
      ``ties_out``); without the search they run as before.
    * ``fractal_route``: on a rugged reading of ``"slsqplq"`` / ``"slsqplqc"``, the
      local phase's centred ``micro_slopes`` test; a fractal reading (f23) gets
      restarted SLSQP at SciPy's step for the rest (the SLSQP-cells toy's ``fract``).
    * ``low_beta`` (the gaps4 toy): a finite rugged reading below it takes the
      (1+1)-CMA-ES instead of the ``"lq2"`` populations (f23 at 100 x dim).
    * ``design`` (the anytime hand-over): these points (the best evaluated so far)
      instead of ``init_points`` uniform draws.
    * ``descent_retries``: the single descent of ``"slsqpdes"`` (outside
      ``small_switch``) and ``"slsqpnm"`` is retried from a uniform point, up to that
      many times, after a descent that did not move, unless its first gradient was
      exactly zero (a plateau; ``qn_phase.FlatGradientReader``).
    """
    dim = len(x0)
    width = float(np.mean(box.upper - box.lower))
    # passed only when on, so the other prefixes' calls stay exactly as recorded
    qn_on: dict[str, Any] = {"fallback": True} if qn_fallback else {}
    if qn_false_success:
        qn_on["false_success"] = True
    if qn_floor != "none":
        qn_on["floor_bfgs"] = qn_floor
    x = np.array(x0, dtype=float)
    cost = objective.evaluate_point(x)
    assert cost is not None  # the first evaluation of a run always fits
    used = 1
    if kind == "ctrk":  # cycle 26: route on the curvature at the start point (x0)
        from .qn_phase import BETA_THRESHOLD, curvature_kappa, run_slsqp_restarts

        probed = curvature_kappa(objective, rng, x, cost, box)
        if probed is None:
            return EsResult(x, cost, INITIAL_SIGMA_SHARE * width * sigma_scale, used)
        kappa, probe_x, probe_cost, spent = probed
        used += spent
        if probe_cost < cost:
            x, cost = probe_x, probe_cost
        if kappa >= BETA_THRESHOLD:
            return run_slsqp_restarts(
                objective,
                rng,
                x,
                cost,
                box,
                evaluations - used,
                fd_step=fd_step,
                **qn_on,
            )
        kind = "cma"
    if design is None:
        design = box.lower + (box.upper - box.lower) * rng.rand(
            min(init_points, evaluations - used), dim
        )
    else:  # the anytime hand-over: the best points earlier stages evaluated
        design = design[: min(init_points, evaluations - used)]
    for candidate in design:
        value = objective.evaluate_point(candidate)
        used += 1
        if value is not None and value < cost:
            x, cost = candidate, value
    separable = False
    step_ties: list[int] = []  # ``plateau_ties``: the search's exact ties
    step_ran = False
    if step_share > 0:  # ``sep_probe``: the separable coordinate search
        from .step_phase import run_step_phase

        # passed only when on, so the other calls stay exactly as recorded
        step_on: dict[str, Any] = {}
        if step_pair_when_stuck:
            step_on["pair_when_stuck"] = True
        if step_box_check:
            step_on["box_check"] = True
        if plateau_ties:
            step_on["ties_out"] = step_ties
        probe = run_step_phase(
            objective,
            x,
            cost,
            box,
            int(step_share * (evaluations - used)),
            parabolic=step_parabolic,
            parabolic_late=step_parabolic_late,
            pair_check=step_pair_check,
            **step_on,
        )
        step_ran = True
        x, cost, used = probe.x, probe.cost, used + probe.used
        separable = not probe.stalled
        if (
            lq_nonsep_only
            and kind in ("lq2", "slsqplq")
            and separable
            and not (sep_smooth_probe and kind == "slsqplq")
        ):
            kind = "cma"  # the separability check passed: keep the (1+1)
        if sep_smooth_probe and lq_nonsep_only and separable:
            rugged_kind = "cma"  # separable but rugged (f3, f4): the (1+1) as before

    def plateau_populations(
        x: np.ndarray, cost: float, used: int
    ) -> tuple[bool, EsResult]:
        """``plateau_pop``: (handled, result) after ``plateau_check`` at ``x``."""
        from .pop_cma import run_population_restarts
        from .qn_phase import plateau_check

        flat = plateau_check(
            objective, np.random.RandomState(PLATEAU_SEED + dim), x, cost, box
        )
        if flat is None:
            return True, EsResult(
                x, cost, INITIAL_SIGMA_SHARE * width * sigma_scale, used
            )
        used += flat[1]
        if not flat[0]:
            return False, EsResult(x, cost, 0.0, used)
        pop = run_population_restarts(
            objective,
            rng,
            x,
            cost,
            INITIAL_SIGMA_SHARE * width * sigma_scale,
            box,
            evaluations - used,
            True,
            active=True,
        )
        if pop.cost <= cost:
            x, cost = pop.x, pop.cost
        return True, EsResult(x, cost, pop.sigma, used + pop.used)

    # ``plateau_ties`` (parked2): no check where the search's probes never tied
    tie_skip = bool(plateau_ties) and step_ran and len(step_ties) < plateau_ties
    # ``plateau_rugged`` (cycle 43): on the probe's kinds the check runs only after a
    # rugged reading, so smooth readings (f10-f14) keep every evaluation
    if (
        plateau_pop
        and not tie_skip
        and (
            kind in ("nm", "slsqpnm")
            or (kind in ("slsqplq", "slsqplqc") and not plateau_rugged)
        )
    ):
        handled, checked = plateau_populations(x, cost, used)
        if handled:
            return checked
        used = checked.used
    if kind in ("slsqplq", "slsqplqc"):  # SLSQP where the probe reads smooth
        from .qn_phase import (
            BETA_DIRECTIONS,
            BETA_SCALES,
            BETA_THRESHOLD,
            CHEAP_BETA_DIRECTIONS,
            curvature_beta,
        )

        # The c45 candidate's prefix reading composes in this order (each step only
        # when on): ``probe_faces`` makes the probe's directions face-tangent before
        # it spends anything (fewer clipped pairs, so fewer NaN slopes; ``nan_clip``'s
        # ``pairs`` see the directions as probed); a slope that is still NaN is then
        # read by ``nan_reading`` (with ``nan_clip``, then the plateau check unless
        # ``plateau_ties`` skips it, then ``nan_reprobe``, whose second probe takes
        # ``probe_faces`` too); a rugged reading may take ``fractal_route``; on SLSQP,
        # ``small_switch``'s checkpoints come first and the ridge reading
        # (``ridge_after``) after them. ``descent_retries`` acts only on the
        # one-descent kinds (``"slsqpdes"``, ``"slsqpnm"``) further down.
        pairs: list[tuple[float, bool]] = []
        probe_dirs = CHEAP_BETA_DIRECTIONS if kind == "slsqplqc" else BETA_DIRECTIONS
        faces_on: dict[str, Any] = {"faces": True} if probe_faces else {}
        pairs_on: dict[str, Any] = {"pairs_out": pairs} if nan_clip else {}
        probed = curvature_beta(
            objective,
            rng,
            x,
            cost,
            box,
            probe_dirs,
            **faces_on,
            **pairs_on,
        )
        if probed is None:
            return EsResult(x, cost, INITIAL_SIGMA_SHARE * width * sigma_scale, used)
        beta, x, cost, spent = probed
        used += spent
        if plateau_pop and plateau_rugged and beta < BETA_THRESHOLD:
            handled, checked = plateau_populations(x, cost, used)
            if handled:
                return checked
            used = checked.used
        kind = "slsqp" if beta >= BETA_THRESHOLD else rugged_kind
        if low_beta and kind == "lq2" and beta < low_beta:
            # ``low_beta`` (gaps4): very rough readings (Katsuura reads -0.3 to 0.3
            # at 100 x dim) keep the elitist (1+1), which beats the populations there
            kind = "cma"
        if (
            nan_reading != "rugged"
            and np.isnan(beta)
            and not (nan_clip and any(v < 0 and not c for v, c in pairs))
        ):
            flat_nan = False
            if nan_reading == "plateau" and not tie_skip:
                from .qn_phase import plateau_check

                flat = plateau_check(objective, rng, x, cost, box)
                if flat is None:
                    return EsResult(
                        x, cost, INITIAL_SIGMA_SHARE * width * sigma_scale, used
                    )
                used += flat[1]
                flat_nan = flat[0]
            if not flat_nan and nan_reprobe:
                # ``nan_reprobe`` (parked2): the same probe again from the point moved
                # inward by its largest scale, where no pair is clipped; smooth there
                # reads smooth, anything else stays rugged
                margin = BETA_SCALES[-1] * width
                inner = np.clip(x, box.lower + margin, box.upper - margin)
                inner_cost = objective.evaluate_point(inner)
                if inner_cost is None:
                    return EsResult(
                        x, cost, INITIAL_SIGMA_SHARE * width * sigma_scale, used
                    )
                used += 1
                again = curvature_beta(
                    objective,
                    rng,
                    inner,
                    float(inner_cost),
                    box,
                    probe_dirs,
                    **faces_on,
                )
                if again is None:
                    return EsResult(
                        x, cost, INITIAL_SIGMA_SHARE * width * sigma_scale, used
                    )
                used += again[3]
                if again[2] < cost:
                    x, cost = again[1], again[2]
                flat_nan = not again[0] >= BETA_THRESHOLD
            if not flat_nan:
                kind = "slsqp"
        if fractal_route and beta < BETA_THRESHOLD:
            # ``fractal_route`` (parked2; the SLSQP-cells toy's ``fract``): the local
            # phase's fractal test on the rugged reading; a fractal one (f23) gets
            # restarted SLSQP at SciPy's step instead of the rugged kind
            from .qn_phase import micro_slopes, run_slsqp_restarts

            before = objective.calls
            micro = micro_slopes(objective, x, cost, box, centered=True)
            if micro is None:
                return EsResult(
                    x, cost, INITIAL_SIGMA_SHARE * width * sigma_scale, used
                )
            used += objective.calls - before
            fractal, _slopes, x_m, cost_m = micro
            if cost_m < cost:
                x, cost = x_m, cost_m
            if fractal:  # as the toy measured it: no L-BFGS-B fallback
                return run_slsqp_restarts(
                    objective, rng, x, cost, box, evaluations - used
                )
    if (
        kind == "slsqpnmg"
    ):  # D <= 2, >= 200 evaluations: SLSQP where smooth, else simplex
        from .qn_phase import BETA_THRESHOLD, CHEAP_BETA_DIRECTIONS, curvature_beta

        nmg_on: dict[str, Any] = {"faces": True} if probe_faces else {}
        probed = curvature_beta(
            objective, rng, x, cost, box, CHEAP_BETA_DIRECTIONS, **nmg_on
        )
        if probed is None:
            return EsResult(x, cost, INITIAL_SIGMA_SHARE * width * sigma_scale, used)
        beta, x, cost, spent = probed
        used += spent
        kind = "slsqp" if beta >= BETA_THRESHOLD else "nm"

    def switch_reader() -> Any:
        from . import qn_phase
        from .qn_phase import SCIPY_FD_STEP, SwitchReader

        step = (fd_step if fd_step is not None else SCIPY_FD_STEP) * qn_phase.x_unit()
        return SwitchReader(step, dim, small_switch)

    def cobyqa_rest(x: np.ndarray, cost: float, used: int, sigma: float) -> EsResult:
        from .cobyqa_phase import run_cobyqa_from

        rest = run_cobyqa_from(
            objective, rng, x, cost, box, evaluations - used, small_switch.radius, sigma
        )
        return EsResult(rest.x, rest.cost, sigma, used + rest.used)

    def ridge_rest(first: EsResult, x: np.ndarray, cost: float, used: int) -> EsResult:
        """``ridge_after`` read a ridge: restarted COBYQA from the descent's best."""
        from .cobyqa_phase import run_cobyqa_restarts

        used += first.used
        if first.cost <= cost:
            x, cost = first.x, first.cost
        rest = run_cobyqa_restarts(
            objective, rng, x, box, evaluations - used, ridge_radius
        )
        if rest.cost <= cost:
            x, cost = rest.x, rest.cost
        return EsResult(x, cost, first.sigma, first.used + rest.used)

    if kind == "slsqp":
        from .qn_phase import run_slsqp_restarts

        if small_switch is None and ridge_after <= 0:
            return run_slsqp_restarts(
                objective,
                rng,
                x,
                cost,
                box,
                evaluations - used,
                fd_step=fd_step,
                **qn_on,
            )
        # ``ridge_after`` (the f13 toy) and ``small_switch`` read the same first
        # descent; the switch's checkpoints come before the ridge's 7 iterations
        reader = switch_reader() if small_switch is not None else None
        ridge: list[bool] = []
        read_on: dict[str, Any] = {"reader": reader} if reader is not None else {}
        if ridge_after > 0:
            read_on |= {"ridge_after": ridge_after, "ridge_out": ridge}
        out = run_slsqp_restarts(
            objective,
            rng,
            x,
            cost,
            box,
            evaluations - used,
            fd_step=fd_step,
            **read_on,
            **qn_on,
        )
        if ridge and ridge[0]:
            return ridge_rest(out, x, cost, used)
        if (
            reader is None
            or not reader.stopped_at
            or evaluations - used - out.used <= 0
        ):
            return out
        if out.cost <= cost:
            x, cost = out.x, out.cost
        return cobyqa_rest(x, cost, used + out.used, out.sigma)
    if kind in ("slsqpdes", "slsqpnm") and descent_retries and small_switch is None:
        # ``descent_retries`` (parked2): the guarded descent, then the (1+1) / simplex
        x, cost, spent = _guarded_descents(
            objective,
            rng,
            x,
            cost,
            box,
            evaluations - used,
            fd_step if kind == "slsqpdes" else None,
            qn_on,
            descent_retries,
        )
        used += spent
        kind = "cma" if kind == "slsqpdes" else "nm"
    if kind == "slsqpdes":  # ``q7ed``: one SLSQP descent from the design's best,
        # then the (1+1) on the rest (cycle 33; 10 x dim at D 5-19)
        from .qn_phase import run_slsqp_restarts

        reader = switch_reader() if small_switch is not None else None
        first = run_slsqp_restarts(
            objective,
            rng,
            x,
            cost,
            box,
            evaluations - used,
            max_descents=1,
            fd_step=fd_step,
            **qn_on,
            **({"reader": reader} if reader is not None else {}),
        )
        used += first.used
        if first.cost <= cost:
            x, cost = first.x, first.cost
        if (
            reader is not None
            and evaluations - used > 0
            and (reader.stopped_at or small_switch.rest)
        ):
            return cobyqa_rest(x, cost, used, first.sigma)
        kind = "cma"
    if kind == "slsqpnm":  # one SLSQP descent, then the simplex from its best
        from .qn_phase import run_slsqp_restarts

        first = run_slsqp_restarts(
            objective,
            rng,
            x,
            cost,
            box,
            evaluations - used,
            max_descents=1,
            **qn_on,
        )
        used += first.used
        if first.cost <= cost:
            x, cost = first.x, first.cost
        kind = "nm"
    if kind == "nm":
        from .nm_phase import run_nm_restarts

        return run_nm_restarts(
            objective,
            rng,
            x,
            cost,
            box,
            evaluations - used,
            INITIAL_SIGMA_SHARE * width * sigma_scale,
        )
    if kind == "lq2":
        from .pop_cma import run_population_restarts

        return run_population_restarts(
            objective,
            rng,
            x,
            cost,
            INITIAL_SIGMA_SHARE * width * sigma_scale,
            box,
            evaluations - used,
            True,
            active=True,
            surrogate=kind,
        )
    if kind == "cma":
        from .cma_phase import run_cma

        if small_switch is not None and small_switch.rest and evaluations - used > 0:
            return cobyqa_rest(x, cost, used, INITIAL_SIGMA_SHARE * width * sigma_scale)
        return run_cma(
            objective,
            rng,
            x,
            cost,
            INITIAL_SIGMA_SHARE * width * sigma_scale,
            box,
            evaluations - used,
            model=model,
            **(cma_options or {}),  # type: ignore[arg-type]
        )
    if simplex:
        from .simplex import INITIAL_EDGE, run_nelder_mead

        return run_nelder_mead(
            objective, rng, x, cost, INITIAL_EDGE * width, box, evaluations - used
        )
    return run_es(
        objective, rng, x, cost, INITIAL_SIGMA_SHARE * width, box, evaluations - used
    )


def _guarded_descents(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    box: Box,
    evaluations: int,
    fd_step: float | None,
    qn_on: dict[str, Any],
    retries: int,
) -> tuple[np.ndarray, float, int]:
    """``descent_retries``: SLSQP descents, (best point, its value, evaluations used).

    A descent from ``x`` with ``max_descents=1``; when it ends without converging away
    from its start (SciPy's false status 0 at the start, or a failure), a uniform
    restart, at most ``retries`` times, unless the first descent's first gradient was
    exactly zero (``FlatGradientReader``: a plateau, f7, keeps its budget for the
    finisher). The SLSQP-cells toy's ``retry5`` checked with ``plateau_check`` (4
    evaluations) instead; that code went with ``lean_gd`` (dead code 6).
    """
    from .qn_phase import Descent, FlatGradientReader, run_slsqp_restarts

    dim = len(x)
    used = 0
    start, start_cost = np.array(x, dtype=float), float(cost)
    for attempt in range(retries + 1):
        if evaluations - used <= dim + 1:
            break
        ends: list[Descent] = []
        reader = FlatGradientReader(dim) if attempt == 0 else None
        run = run_slsqp_restarts(
            objective,
            rng,
            start,
            start_cost,
            box,
            evaluations - used,
            None,
            ends,
            max_descents=1,
            fd_step=fd_step,
            reader=reader,
            **qn_on,
        )
        used += run.used
        if run.cost < cost:
            x, cost = run.x, run.cost
        moved = bool(ends) and ends[0].status == 0 and not np.allclose(ends[0].x, start)
        if not ends or moved or attempt == retries:
            break
        if reader is not None and reader.flat:  # a plateau: the finisher, no retries
            break
        start = box.lower + (box.upper - box.lower) * rng.rand(dim)
        value = objective.evaluate_point(start)
        if value is None:
            break
        used += 1
        start_cost = float(value)
        if start_cost < cost:
            x, cost = start.copy(), start_cost
    return x, cost, used


def run_es(
    objective: CountedObjective,
    rng: np.random.RandomState,
    x: np.ndarray,
    cost: float,
    sigma: float,
    box: Box,
    evaluations: int,
) -> EsResult:
    """A (1+1)-ES with the 1/5 rule from ``x`` (cost known), up to ``evaluations``."""
    dim = len(x)
    width = float(np.mean(box.upper - box.lower))
    floor = MIN_SIGMA_SHARE * width
    used = 0
    while used < evaluations and sigma > floor:
        step = sigma * rng.randn(dim)
        inward = ((x <= box.lower) & (step > 0)) | ((x >= box.upper) & (step < 0))
        if inward.any():
            step = np.where(inward & (rng.rand(dim) < PIN_STICK), -step, step)
        candidate = clip(x + step, box.lower, box.upper)
        value = objective.evaluate_point(candidate)
        if value is None:
            break
        used += 1
        if value <= cost:
            x, cost = candidate, value
            sigma *= np.exp(SUCCESS_GROWTH / dim)
        else:
            sigma *= np.exp(-FAILURE_SHRINK / dim)
    return EsResult(x, float(cost), float(sigma))
