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
import numbers
import operator
import warnings
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
    """Read ``a`` and ``b`` as the two rows of ``[(low, high), (low, high)]``."""
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
    """Return the start point, the budget and the box, filling in omitted ones."""
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
        if not np.all(np.isfinite(start)):
            # checked before any evaluation is spent (review B6, decision D2 a)
            raise ValueError(f"x0 must be finite, got {start}")
        if box is not None:
            for side in box:
                if side.ndim == 1 and side.size != start.size:
                    raise ValueError(
                        f"bounds describe {side.size} coordinates but x0 has "
                        f"{start.size}"
                    )
            lower, upper = box
            if np.any(start < lower) or np.any(start > upper):
                # one policy on every box (review B2): ``fun`` only sees points in it
                warnings.warn(
                    "x0 is outside bounds; clipped to the box",
                    UserWarning,
                    stacklevel=3,
                )
                start = np.clip(start, lower, upper)
    if budget is None:
        count = max(DEFAULT_BUDGET_MIN, DEFAULT_BUDGET_PER_DIM * start.size)
    elif isinstance(budget, bool | np.bool_):
        raise TypeError(
            f"budget must be a whole number of evaluations, got the bool {budget!r}"
        )
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


def _check_options(seed: object, rng: object, config: object) -> ATMConfig | None:
    """Check ``seed``, ``rng`` and ``config`` before the run starts (review B6).

    Raises plain errors for values of the wrong kind and returns ``config`` with a
    preset name resolved.
    """
    if isinstance(config, str):
        config = presets.get(config)  # ValueError for an unknown name
    elif config is not None and not isinstance(config, ATMConfig):
        raise TypeError(
            f"config must be an ATMConfig or a preset name such as 'c46', got "
            f"{type(config).__name__}"
        )
    if rng is not None and not isinstance(
        rng, np.random.RandomState | np.random.Generator
    ):
        raise TypeError(
            "rng must be a numpy.random.Generator or RandomState, got "
            f"{type(rng).__name__}; to seed the run pass seed=<int> instead"
        )
    if seed is None or isinstance(seed, np.random.SeedSequence):
        return config
    allowed = "seed must be an integer in [0, 2**32) or a numpy.random.SeedSequence"
    if isinstance(seed, bool | np.bool_):
        raise TypeError(f"{allowed}, got the bool {seed!r}")
    try:
        value = operator.index(seed)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"{allowed}, got {type(seed).__name__} {seed!r}") from None
    if not 0 <= value < 2**32:
        raise ValueError(f"{allowed}, got {value}")
    return config


def _target(ftarget: object) -> float | None:
    """Return ``ftarget`` as a float, or raise a plain ``TypeError``."""
    if ftarget is None:
        return None
    if isinstance(ftarget, bool | np.bool_) or not isinstance(ftarget, numbers.Real):
        raise TypeError(f"ftarget must be a real number, got {ftarget!r}")
    return float(ftarget)


def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray | None = None,
    budget: int | None = None,
    *,
    bounds: object | None = None,
    seed: int | np.random.SeedSequence | None = None,
    anytime: bool = False,
    config: ATMConfig | str | None = None,
    rng: np.random.Generator | np.random.RandomState | None = None,
    profile: bool | Profiler = False,
    callback: Callable[[OptimizeResult], object] | None = None,
    ftarget: float | None = None,
    catch_interrupt: bool = False,
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
        misleads them. Non-finite values (NaN, +-inf) count as failed evaluations
        and never win, at the starting point too. ``fun`` gets its own copy of
        ``x`` and must return a real scalar (``float``, a NumPy float, ``int``);
        anything else raises ``TypeError``.
    x0 : array_like, optional
        The starting point, shape ``(D,)``. Default: the centre of the box. An ``x0``
        outside ``bounds`` is clipped onto the box, with a ``UserWarning``.
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
    seed : int or numpy.random.SeedSequence, optional
        Seed for a reproducible run, an integer in ``[0, 2**32)`` or a
        ``SeedSequence`` (its first 32-bit word of state is the seed). A run is
        deterministic given the seed, the platform and the BLAS thread count.
        Default: a fresh seed from the operating system; ``result.seed`` records
        the seed used either way, so ``seed=result.seed`` repeats the run.
    anytime : bool, optional
        ``True`` turns on the anytime (replay) mode, for runs whose budget is not
        known in advance or that you will cut short (with ``callback`` or
        ``ftarget``, keeping the best point seen).
        Up to 10 000 evaluations it never learns ``budget``; see ``docs/ANYTIME.md``
        for what it costs. It trails the default by about .05 to .12 of the
        target share between 30 x D and 1000 x D evaluations when the budget is
        known, so leave it off if you know your budget. Equivalent to
        ``config=presets.with_changes("c46", anytime="replay")``.
    config : ATMConfig or str, optional
        The configuration to run. ``None`` (the default) means the default preset,
        AMM (frozen as ``c46``). A string is a preset name, read by
        ``presets.get``: ``"default"`` (the same as ``None``), ``"c46"``,
        ``"atm"`` (``ATMConfig()``, the original ATM sampler alone, which does not
        need ``bounds``), ``"composite"`` (``ATMConfig.composite()``), or a
        research preset of ``presets.names(research=True)``. An ``ATMConfig`` is
        passed to the engine unchanged.
    rng : numpy.random.Generator or numpy.random.RandomState, optional
        Randomness to use instead of ``seed``. A ``Generator`` gives the run one
        draw, ``rng.integers(2**32)``, as its seed (recorded in ``result.seed``);
        a ``RandomState`` is continued as it is (``result.seed`` is then
        ``None``).
    profile : bool or Profiler, optional
        ``True`` times the named blocks of the algorithm and returns them as
        ``result.timings``; pass a :class:`~amm_optimizer.TimingProfiler` to
        accumulate over several runs. Profiling does not change the result.
    callback : callable, optional
        ``callback(intermediate_result)``, called after every evaluation that
        improves the best value found so far, with an :class:`OptimizeResult`
        holding the best point ``x``, its value ``fun``, ``nfev`` and ``nit``.
        Raising ``StopIteration`` ends the run: ``minimize`` returns that best
        point with ``status`` 2. Its return value is ignored, and it changes
        nothing else about the run. Any other exception propagates.
    ftarget : float, optional
        Stop as soon as the best value is at or below ``ftarget`` (``status``
        1). Default: run the whole budget.
    catch_interrupt : bool, optional
        ``True`` turns a ``KeyboardInterrupt`` (Ctrl-C) during the run into a
        result: the best point so far, with ``status`` 3, ``success=False`` and
        a ``RuntimeWarning``. Default ``False``: the interrupt propagates. An
        interrupt before any finite value was found always propagates.

    Returns
    -------
    OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` (fields as attributes or keys):
        the best point ``x``, its value ``fun``, the number of evaluations
        ``nfev``, ``nit``, ``status``, ``success``, ``message`` and the ``seed``
        used; see :class:`~amm_optimizer.OptimizeResult`.

    Raises
    ------
    ValueError
        If ``config`` is not given and ``bounds`` is missing, if neither ``x0`` nor
        per-coordinate ``bounds`` say how many variables there are, if ``x0`` and
        ``bounds`` disagree on that number, if ``x0`` is not finite, if ``budget`` is
        not a whole number or is below 1, if ``seed`` is outside ``[0, 2**32)``,
        or if ``config`` names no preset. All are raised before ``fun`` is called.
    TypeError
        If ``fun`` or ``callback`` is not callable, if ``config`` is neither an
        ``ATMConfig`` nor a preset name, if ``rng`` is not a NumPy
        ``Generator`` or ``RandomState``, if ``seed`` is not an integer or a
        ``SeedSequence``, if ``ftarget`` is not a number, if ``budget`` is a
        ``bool``, or if ``fun`` returns something other than a real scalar.

    Notes
    -----
    Thread safety: runs in separate threads are independent; a run keeps its
    state in its own call, not in module globals. Warnings that
    ``fun`` raises reach the caller, except inside SciPy's quasi-Newton calls,
    where floating-point overflow and divide-by-zero warnings are silenced.

    Version 0.6.0 made the result a :class:`scipy.optimize.OptimizeResult`,
    added ``callback``, ``ftarget``, ``catch_interrupt``, preset names as
    ``config``, ``Generator`` and ``SeedSequence`` seeds and ``result.seed``, and
    made a non-finite value at the starting point a failed evaluation (it raised
    ``ValueError`` before).
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
    >>> result = amm.minimize(f, bounds=[(-5, 5)] * 3, budget=3000, ftarget=1e-8)
    >>> result.status, result.fun <= 1e-8, result.nfev < 3000
    (1, True, True)
    """
    if not callable(fun):
        raise TypeError(f"fun must be callable, got {type(fun).__name__}")
    if callback is not None and not callable(callback):
        raise TypeError(f"callback must be callable, got {type(callback).__name__}")
    target = _target(ftarget)
    config = _check_options(seed, rng, config)
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
        callback=callback,
        ftarget=target,
        catch_interrupt=catch_interrupt,
    )
