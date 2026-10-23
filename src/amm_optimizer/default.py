"""The public ``minimize``: AMM by default.

``amm_optimizer.minimize`` is a thin entry point over the engine in :mod:`api`
(``amm_optimizer.api.minimize``). The engine's own default is the original ATM
sampler, which the golden, parity and characterisation tests pin; this wrapper
only changes which configuration a call without ``config=`` gets.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import presets
from .api import minimize as _engine_minimize
from .config import ATMConfig
from .profiling import Profiler
from .result import OptimizeResult

#: The preset a call without ``config=`` runs (the configuration frozen for the paper).
DEFAULT_PRESET = "c46"


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
    """Minimise ``fun`` starting from ``x0`` with at most ``budget`` evaluations.

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
        a non-finite value at ``x0`` raises ``ValueError``.
    x0 : array_like
        The starting point, shape ``(D,)``.
    budget : int
        The maximum number of calls of ``fun``. All of them are used; the best
        point evaluated is returned even if the budget ends mid-iteration.
    seed : int, optional
        Seed for a reproducible run. A run is deterministic given the seed.
    rng : numpy.random.RandomState, optional
        An existing random state to continue instead of ``seed``.
    config : ATMConfig, optional
        The configuration to run. ``None`` (the default) means ``presets.get("c46")``
        (AMM). Any other value, such as another entry of ``presets.names()`` or
        ``ATMConfig()``, is passed to the engine unchanged. ``ATMConfig()`` runs the
        original ATM sampler alone and does not need ``bounds``.
    profile : bool or Profiler, optional
        ``True`` times the named blocks of the algorithm and returns them as
        ``result.timings``; pass a :class:`~amm_optimizer.TimingProfiler` to
        accumulate over several runs. Profiling does not change the result.
    bounds : tuple, optional
        ``(lower, upper)``, the search box (scalars or one value per coordinate).
        **Required when ``config`` is not given**, because AMM reads the box. The
        box is used, not enforced as a constraint: samples are folded into it
        before ``fun`` is called, so ``fun`` only sees points inside it.

    Returns
    -------
    OptimizeResult
        The best point ``x``, its value ``fun`` and the number of evaluations
        ``nfev``.

    Raises
    ------
    ValueError
        If ``config`` is not given and ``bounds`` is missing, if ``budget < 1``, or
        if ``fun`` is not finite at ``x0``.

    Notes
    -----
    Version 0.4.0 changed the default. Earlier releases ran the original ATM sampler
    when ``config`` was omitted. To get that behaviour pass ``config=ATMConfig()``
    (``bounds`` is then optional), or call ``amm_optimizer.api.minimize``, whose
    default is the original sampler. Tested on noiseless, continuous, box-constrained
    problems with budgets of 10 to 1000 times the dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import amm_optimizer as amm
    >>> def f(x):
    ...     return float(np.sum((x - 1.0) ** 2))
    >>> result = amm.minimize(f, np.zeros(3), 300, seed=0, bounds=(-5, 5))
    >>> result.nfev
    300
    """
    if config is None:
        if bounds is None:
            raise ValueError(
                "AMM needs a search box: pass bounds=(lower, upper), or pass "
                "config=ATMConfig() to run the original ATM sampler, which does "
                "not need one"
            )
        config = presets.get(DEFAULT_PRESET)
    return _engine_minimize(
        fun,
        x0,
        budget,
        seed=seed,
        rng=rng,
        config=config,
        profile=profile,
        bounds=bounds,
    )
