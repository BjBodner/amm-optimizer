"""Return type of :func:`amm_optimizer.minimize`."""

from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult as _ScipyResult  # type: ignore[import-untyped]

from .profiling import Timings

#: ``status``: the run ended normally (it spent its budget, or its configuration
#: finished before that).
BUDGET = 0
#: ``status``: the best value reached ``ftarget``.
TARGET = 1
#: ``status``: the callback raised ``StopIteration``.
CALLBACK = 2
#: ``status``: a ``KeyboardInterrupt`` ended the run (``catch_interrupt=True``).
INTERRUPTED = 3

MESSAGES = {
    BUDGET: "The run ended normally: it used its budget or finished its plan.",
    TARGET: "The best value reached ftarget.",
    CALLBACK: "The callback raised StopIteration.",
    INTERRUPTED: "Interrupted by KeyboardInterrupt; returned the best point so far.",
}


class OptimizeResult(_ScipyResult):  # type: ignore[misc]
    """The result of :func:`amm_optimizer.minimize`, as SciPy returns it.

    A subclass of :class:`scipy.optimize.OptimizeResult`, so it is a ``dict``
    whose keys are also attributes: ``result.x`` and ``result["x"]`` are the same
    object, ``result.keys()`` lists the fields, and code written for
    ``scipy.optimize.minimize`` reads it unchanged.

    Attributes
    ----------
    x : numpy.ndarray
        The best point evaluated, shape ``(D,)``.
    fun : float
        ``fun(x)``, the best value found.
    nfev : int
        The number of calls of ``fun``.
    nit : int
        The number of evaluations that improved the best value found so far,
        that is, the number of times ``callback`` was (or would have been)
        called. AMM switches between several engines, so it has no single
        iteration count; this is the one count all of them share.
    nrestarts : int
        The number of restarts of the ATM sampler (in the anytime mode, the
        number of stages after the first).
    success : bool
        ``True`` unless a ``KeyboardInterrupt`` ended the run (``status`` 3).
    status : int
        Why the run ended: ``0``, it used its budget or its configuration
        finished; ``1``, the best value reached ``ftarget``; ``2``, ``callback``
        raised ``StopIteration``; ``3``, a ``KeyboardInterrupt`` ended it
        (``catch_interrupt=True``).
    message : str
        ``status`` in words.
    seed : int or None
        The seed the run used, an integer in ``[0, 2**32)``: passing it back as
        ``seed=`` repeats the run. ``None`` when the run continued a
        ``numpy.random.RandomState`` given as ``rng``.
    timings : Timings or None
        The time per named block, when ``minimize(..., profile=...)`` asked for
        it; else ``None``.

    Notes
    -----
    Before version 0.6.0 this was a dataclass; ``dataclasses.replace`` and
    ``dataclasses.asdict`` no longer apply. Use ``dict(result)`` for a plain
    dictionary.

    Examples
    --------
    >>> import numpy as np
    >>> import amm_optimizer as amm
    >>> r = amm.minimize(lambda x: float(np.sum(x**2)), bounds=[(-1, 1)] * 2, seed=0)
    >>> r["x"] is r.x
    True
    >>> r.status, r.success, r.nfev
    (0, True, 200)
    """

    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    nrestarts: int
    success: bool
    status: int
    message: str
    seed: int | None
    timings: Timings | None
