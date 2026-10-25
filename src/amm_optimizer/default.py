"""The public ``minimize``: AMM by default, with sensible defaults for the rest.

``amm_optimizer.minimize`` is a thin entry point over the engine in :mod:`api`
(``amm_optimizer.api.minimize``). The engine takes every argument explicitly and its
own default is the original ATM sampler, which the golden, parity and
characterisation tests pin. This wrapper changes which configuration a call without
``config=`` gets, fills in the arguments a caller can leave out (the starting point
and the budget), and reads the usual shapes of ``bounds``.
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Callable
from typing import Any

import numpy as np

from . import presets
from .api import minimize as _engine_minimize
from .config import ATMConfig
from .profiling import Profiler
from .result import OptimizeResult

#: The preset a call without ``config=`` runs (the configuration frozen for the paper).
DEFAULT_PRESET = "c46"
#: A call without ``budget`` spends this many evaluations per dimension ...
DEFAULT_BUDGET_PER_DIM = 100
#: ... and at least this many in total.
DEFAULT_BUDGET_MIN = 100


class _AmbiguousBoundsError(ValueError):
    """A tuple of two length-2 sequences that reads as pairs and as (lower, upper)."""


def _pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``a`` and ``b`` as the two rows of ``[(low, high), (low, high)]``."""
    both = np.array([a, b], dtype=float)
    return both[:, 0], both[:, 1]


def _is_box(lower: np.ndarray, upper: np.ndarray) -> bool:
    with np.errstate(invalid="ignore"):
        return bool(np.all(lower < upper))


def _as_box(bounds: object) -> tuple[np.ndarray, np.ndarray]:
    """``bounds`` as ``(lower, upper)`` float arrays (0-d or one value per coordinate).

    Accepted: ``scipy.optimize.Bounds``; a tuple ``(lower, upper)`` of scalars or
    arrays (the engine's own form); a list, array or other sequence of ``(low, high)``
    pairs, one per coordinate (SciPy's form). Only a tuple of two length-2 sequences
    can be read both ways; it is taken as the one that describes a box, and an error
    names both readings when both do.
    """
    try:
        if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
            # Bounds(-5, 5) stores length-1 arrays that stand for every coordinate
            lb, ub = (np.asarray(b, dtype=float) for b in (bounds.lb, bounds.ub))
            return (
                lb.reshape(()) if lb.size == 1 else lb,
                ub.reshape(()) if ub.size == 1 else ub,
            )
        if isinstance(bounds, tuple) and len(bounds) == 2:
            lower, upper = (np.asarray(b, dtype=float) for b in bounds)
            if lower.shape == (2,) and upper.shape == (2,):
                p_lower, p_upper = _pairs(lower, upper)
                as_pairs, as_arrays = _is_box(p_lower, p_upper), _is_box(lower, upper)
                if as_pairs and as_arrays:
                    raise _AmbiguousBoundsError(
                        f"bounds={bounds!r} for two coordinates can mean two "
                        "(low, high) pairs or (lower, upper) arrays; pass a list of "
                        "pairs, [(low0, high0), (low1, high1)], or "
                        "scipy.optimize.Bounds(lower, upper)"
                    )
                if as_pairs:
                    return p_lower, p_upper
            return lower, upper
        table = np.asarray(bounds, dtype=float)
    except _AmbiguousBoundsError:
        raise
    except (TypeError, ValueError) as error:
        raise ValueError(
            "bounds must be finite numbers: (lower, upper), a list of (low, high) "
            "pairs, or scipy.optimize.Bounds. AMM needs a finite box"
        ) from error
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(
            "bounds must be (lower, upper), a list of (low, high) pairs with one "
            f"pair per coordinate, or scipy.optimize.Bounds; got an array of shape "
            f"{table.shape}"
        )
    return table[:, 0], table[:, 1]


def _resolve(
    x0: object, budget: object, bounds: object
) -> tuple[np.ndarray, int, tuple[np.ndarray, np.ndarray] | None]:
    """The starting point, the budget and the box, with the omitted ones filled in."""
    box = None if bounds is None else _as_box(bounds)
    if x0 is None:
        if box is None:
            raise ValueError("pass x0 (the starting point) or bounds (the search box)")
        lower, upper = box
        width = max(lower.size, upper.size)
        if lower.ndim == 0 and upper.ndim == 0:
            raise ValueError(
                "cannot tell the number of variables from bounds=(low, high) with "
                "scalars: pass x0, or one (low, high) pair per coordinate"
            )
        start = (
            np.broadcast_to(lower, (width,)) + np.broadcast_to(upper, (width,))
        ) / 2.0
    else:
        start = np.asarray(x0, dtype=float)
        if start.ndim != 1 or start.size == 0:
            raise ValueError(f"x0 must be a 1-d array, got shape {start.shape}")
        if box is not None:
            for side in box:
                if side.ndim == 1 and side.size != start.size:
                    raise ValueError(
                        f"bounds describe {side.size} coordinates but x0 has "
                        f"{start.size}"
                    )
    if budget is None:
        count = max(DEFAULT_BUDGET_MIN, DEFAULT_BUDGET_PER_DIM * start.size)
    else:
        try:
            count = operator.index(budget)  # type: ignore[arg-type]
        except TypeError:
            if isinstance(budget, float | np.floating) and float(budget).is_integer():
                count = int(budget)  # 1e4
            else:
                raise ValueError(
                    f"budget must be a whole number of evaluations, got {budget!r}"
                ) from None
    return start, count, box


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray | None = None,
    budget: int | None = None,
    *,
    bounds: object | None = None,
    seed: int | None = None,
    anytime: bool = False,
    config: ATMConfig | None = None,
    rng: np.random.RandomState | None = None,
    profile: bool | Profiler = False,
) -> OptimizeResult:
    """Minimise ``fun`` over a box with at most ``budget`` evaluations.

    The shortest call is ``minimize(fun, bounds=[(-5, 5)] * 10)``: it searches the
    10-dimensional box, starts from its centre, spends ``100 x 10`` evaluations and
    returns the best point found.

    Without ``config`` this runs AMM (Adaptive Multi Mode), the configuration
    ``presets.get("c46")`` frozen for the paper. AMM does not run one algorithm: it
    spends a small, counted part of the budget on cheap online probes of the
    objective near the incumbent (curvature, plateau, separability, kink, basin) and
    routes the rest of the budget among classical engines (quasi-Newton, trust
    region, coordinate search, multistart, CMA-ES family). The original ATM sampler
    is one of those engines.

    Parameters
    ----------
    fun : callable
        The objective, ``fun(x) -> float`` for an array ``x`` of shape ``(D,)``. It
        must be deterministic: the probes compare function values, so noise
        misleads them. Non-finite values count as failed evaluations and never win;
        a non-finite value at the starting point raises ``ValueError``.
    x0 : array_like, optional
        The starting point, shape ``(D,)``. Default: the centre of the box.
    budget : int, optional
        The maximum number of calls of ``fun``. All of them are used; the best point
        evaluated is returned even if the budget ends mid-iteration. A whole float
        such as ``1e4`` is accepted. Default: ``100 x D`` evaluations, at least 100.
        AMM was tested with 10 to 1000 x D evaluations and plans its work from the
        budget, so pass the number you can afford.
    bounds : tuple, list or scipy.optimize.Bounds, optional
        The search box. Any of: a list of ``(low, high)`` pairs, one per coordinate
        (SciPy's form); ``(lower, upper)`` with scalars or one value per coordinate;
        ``scipy.optimize.Bounds``. Required when ``config`` is not given, because AMM
        reads the box; the limits must be finite. The box is used, not enforced as a
        constraint: samples are folded into it before ``fun`` is called, so ``fun``
        only sees points inside it. With ``x0`` omitted, give one ``(low, high)``
        pair per coordinate so the number of variables is known.
    seed : int, optional
        Seed for a reproducible run. A run is deterministic given the seed, the
        platform and the BLAS thread count. Default: a fresh seed.
    anytime : bool, optional
        ``True`` turns on the anytime (replay) mode, for runs whose budget is not
        known in advance or that you will cut short (for example by raising an
        exception from ``fun`` when time runs out, keeping the best point seen).
        Up to 10 000 evaluations it never learns ``budget``; see ``docs/ANYTIME.md``
        for what it costs. It trails the default by about .05 to .12 of the
        target share between 30 x D and 1000 x D evaluations when the budget is
        known, so leave it off if you know your budget. Equivalent to
        ``config=presets.with_changes("c46", anytime="replay")``.
    config : ATMConfig, optional
        The configuration to run. ``None`` (the default) means ``presets.get("c46")``
        (AMM). Any other value, such as another entry of ``presets.names()`` or
        ``ATMConfig()``, is passed to the engine unchanged. ``ATMConfig()`` runs the
        original ATM sampler alone and does not need ``bounds``.
    rng : numpy.random.RandomState, optional
        An existing random state to continue instead of ``seed``.
    profile : bool or Profiler, optional
        ``True`` times the named blocks of the algorithm and returns them as
        ``result.timings``; pass a :class:`~amm_optimizer.TimingProfiler` to
        accumulate over several runs. Profiling does not change the result.

    Returns
    -------
    OptimizeResult
        The best point ``x``, its value ``fun`` and the number of evaluations
        ``nfev``.

    Raises
    ------
    ValueError
        If ``config`` is not given and ``bounds`` is missing, if neither ``x0`` nor
        per-coordinate ``bounds`` say how many variables there are, if ``x0`` and
        ``bounds`` disagree on that number, if ``budget`` is not a whole number or is
        below 1, or if ``fun`` is not finite at the starting point.
    TypeError
        If ``fun`` is not callable.

    Notes
    -----
    Version 0.5.0 made ``x0`` and ``budget`` optional and added the SciPy forms of
    ``bounds`` and ``anytime``; every call that worked before runs unchanged.
    Version 0.4.0 changed the default configuration: earlier releases ran the
    original ATM sampler when ``config`` was omitted. To get that behaviour pass
    ``config=ATMConfig()`` (``bounds`` is then optional), or call
    ``amm_optimizer.api.minimize``, whose default is the original sampler. Tested
    on noiseless, continuous, box-constrained problems with budgets of 10 to 1000
    times the dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import amm_optimizer as amm
    >>> def f(x):
    ...     return float(np.sum((x - 1.0) ** 2))
    >>> result = amm.minimize(f, bounds=[(-5, 5)] * 3, seed=0)
    >>> result.nfev
    300
    >>> result = amm.minimize(f, np.zeros(3), 500, seed=0, bounds=(-5, 5))
    >>> result.nfev
    500
    """
    if not callable(fun):
        raise TypeError(f"fun must be callable, got {type(fun).__name__}")
    start, count, box = _resolve(x0, budget, bounds)
    if config is None:
        if box is None:
            raise ValueError(
                "AMM needs a search box: pass bounds=[(low, high), ...] or "
                "bounds=(lower, upper), or pass config=ATMConfig() to run the "
                "original ATM sampler, which does not need one"
            )
        config = presets.get(DEFAULT_PRESET)
    if anytime and config.variant.anytime == "off":
        config = dataclasses.replace(
            config, variant=dataclasses.replace(config.variant, anytime="replay")
        )
    engine_bounds: Any = None if box is None else box
    return _engine_minimize(
        fun,
        start,
        count,
        seed=seed,
        rng=rng,
        config=config,
        profile=profile,
        bounds=engine_bounds,
    )
