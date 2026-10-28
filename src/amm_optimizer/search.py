"""One search: four samplers alternating with hyperparameter updates.

A search runs until the budget is spent or the restart policy fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import numpy as np

from . import hyperparams, legacy_stream
from .bounds import Box
from .config import N_SAMPLERS, ATMConfig
from .control import start_log_widths
from .meta import HyperparameterSearch
from .objective import CountedObjective
from .profiling import NULL_PROFILER, Profiler
from .section import SectionRunner
from .state import SamplerState, SearchState


@dataclass
class SearchOutcome:
    """What one :meth:`Search.run` ended with.

    Attributes
    ----------
    x : numpy.ndarray
        The best point of the search.
    cost : float
        Its value (``inf`` when every evaluation failed).
    restarted : bool
        Whether the restart policy ended the search.
    out_of_budget : bool
        Whether the budget ended it.
    widths : tuple of float
        The samplers' final widths (success-rate control only; else empty).
    width : float
        The geometric mean of the samplers' final widths (``nan`` for the
        original sampler, whose widths are a schedule).
    """

    x: np.ndarray
    cost: float
    restarted: bool
    out_of_budget: bool
    widths: tuple[float, ...] = ()
    width: float = float("nan")


class Search:
    """The ATM sampler: one search, from a start point, until a restart or the budget.

    Four samplers draw candidates around the incumbent; their hyperparameters are
    updated between sections of iterations (:class:`~.meta.HyperparameterSearch`).

    Parameters
    ----------
    objective : CountedObjective
        The objective, with the budget bookkeeping.
    rng : numpy.random.RandomState
        The run's random stream (shared with the other phases).
    config : ATMConfig
        The configuration.
    total_iterations : int or None
        The iteration limit, or ``None`` to run until the budget or a restart.
    profiler : Profiler, optional
        Where the named blocks are timed.
    box : Box, optional
        The search box, for the variants that read it.
    """

    def __init__(
        self,
        objective: CountedObjective,
        rng: np.random.RandomState,
        config: ATMConfig,
        total_iterations: int | None,
        profiler: Profiler = NULL_PROFILER,
        box: Box | None = None,
    ) -> None:
        self._profiler = profiler
        self._objective = objective
        self._rng = rng
        self._config = config
        self._total_iterations = total_iterations
        self._section = SectionRunner(rng, objective, config, profiler, box)

    def run(
        self,
        x_start: np.ndarray,
        allocation: np.ndarray,
        known_cost: float | None = None,
        width_scale: float | None = None,
        initial_design: bool = True,
    ) -> SearchOutcome:
        """Search from ``x_start`` until a restart or the end of the budget.

        Parameters
        ----------
        x_start : numpy.ndarray
            The start point.
        allocation : numpy.ndarray
            The samples per iteration of each of the four samplers.
        known_cost : float, optional
            The value at ``x_start``, if known (it is then not evaluated again).
            A non-finite value, given or evaluated, counts as a failed
            evaluation: the search goes on from ``x_start`` with cost ``inf``.
        width_scale : float, optional
            Multiplies the starting widths of the success-rate controller (a step
            size learned before this search).
        initial_design : bool, optional
            ``False`` skips the variant's uniform first pool (something already
            explored).

        Returns
        -------
        SearchOutcome
            The best point, its value and why the search ended.
        """
        self._section.initial_design = initial_design
        rng = self._rng
        n_params = len(x_start)

        prof = self._profiler
        with prof.block("init"):
            samplers = [SamplerState(n_params, rng) for _ in range(N_SAMPLERS)]
            for sampler, n in zip(samplers, allocation, strict=True):
                sampler.n_samples = n

            initial = hyperparams.clip_to_bounds(
                legacy_stream.draw_initial_hyperparameters(rng, N_SAMPLERS),
                hyperparams.bounds(),
            )
            for sampler, column in zip(samplers, initial.T, strict=True):
                sampler.set_hyperparameters(column)
            if self._config.variant.amplitude_control != "legacy":
                widths = start_log_widths(self._config.variant, width_scale)
                for sampler, log_width in zip(samplers, widths, strict=True):
                    sampler.random_growth = log_width

            start_cost = known_cost
            if start_cost is None:
                start_cost = self._objective.evaluate_start(x_start)
            if not np.isfinite(start_cost):
                # a failed evaluation, as anywhere else (decision D2 b): the
                # search goes on, and any finite value beats the start
                start_cost = float("inf")
            variant = self._config.variant
            learn_covariance = (
                variant.covariance != "none" and n_params <= variant.cov_max_dim
            )
            state = SearchState(
                best=x_start,
                second_best=x_start,
                previous_cost=start_cost,
                change_vector=np.zeros(n_params),
                current_cost=start_cost,
                scale_moment=(
                    np.ones(n_params)
                    if self._config.variant.scale_learning != "none"
                    else None
                ),
                cov=np.eye(n_params) if learn_covariance else None,
                cov_factor=np.eye(n_params) if learn_covariance else None,
                cov_path=(
                    np.zeros(n_params)
                    if learn_covariance and variant.cov_path != "none"
                    else None
                ),
            )
            legacy_stream.burn_directional_pool(
                rng, n_params, self._config.initial_population
            )

        # Legacy: the original also scored this first section with a meta cost,
        # but only to form a cost change that no sampler reads. Omitted.
        self._section.run(state, samplers)

        meta = HyperparameterSearch(initial)
        # ``None``: no limit, the search ends by budget or restart only.
        rounds = (
            count()
            if self._total_iterations is None
            else range(
                int(
                    np.ceil(
                        self._total_iterations
                        / self._config.iterations_per_hyperparameter_set
                    )
                )
            )
        )
        for _ in rounds:
            # Further rounds after these would do no observable work.
            if state.restart or state.force_stop:
                break
            with prof.block("meta_propose"):
                candidates = meta.propose(rng)
            for sampler, column in zip(samplers, candidates.T, strict=True):
                sampler.set_hyperparameters(column)
            legacy_stream.burn_directional_pool(
                rng, n_params, self._config.initial_population
            )
            gains = self._section.run(state, samplers)
            with prof.block("meta_update"):
                meta.update(gains)

        return SearchOutcome(
            x=state.best,
            cost=state.current_cost,
            restarted=state.restart,
            out_of_budget=state.force_stop,
            widths=(
                tuple(float(np.exp(s.random_growth)) for s in samplers)
                if self._config.variant.amplitude_control != "legacy"
                else ()
            ),
            width=(
                float(np.exp(np.mean([s.random_growth for s in samplers])))
                if self._config.variant.amplitude_control != "legacy"
                else float("nan")
            ),
        )
