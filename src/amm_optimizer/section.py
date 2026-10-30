"""One block of iterations with fixed sampler hyperparameters.

An iteration draws candidates from every sampler, evaluates them one by one,
moves the incumbent to the best candidate, and lets the conductor shift samples
toward the samplers that improved the cost the most.
"""

from __future__ import annotations

import numpy as np

from .bounds import Box, repair, uniform_columns
from .conductor import ResourceConductor
from .config import N_SAMPLERS, ATMConfig
from .control import update_covariance, update_log_widths
from .objective import CountedObjective
from .profiling import NULL_PROFILER, Profiler
from .restart import check_restart
from .sampler import sample_options, two_mode_samples
from .state import SamplerState, SearchState, best_eighth, runner_up


def penalize_failures(cost: np.ndarray, previous_cost: float) -> np.ndarray:
    """Give failed evaluations (NaN, +-inf) the worst finite cost.

    A failed point can then never win, never distorts the normalised cost
    changes, and never empties the best-eighth selection. Batches without
    failures are returned unchanged.
    """
    failed = ~np.isfinite(cost)
    if not failed.any():
        return cost
    finite = cost[~failed]
    worst = previous_cost if finite.size == 0 else max(finite.max(), previous_cost)
    return np.where(failed, worst, cost)


class SectionRunner:
    def __init__(
        self,
        rng: np.random.RandomState,
        objective: CountedObjective,
        config: ATMConfig,
        profiler: Profiler = NULL_PROFILER,
        box: Box | None = None,
    ) -> None:
        self._profiler = profiler
        self._box = box
        self._rng = rng
        self._objective = objective
        self._config = config
        self._conductor = ResourceConductor(config.conductor)
        self._options = sample_options(config.variant)
        self.initial_design = True  # set per search by ``Search.run``

    def run(self, state: SearchState, samplers: list[SamplerState]) -> np.ndarray:
        """Run ``iterations_per_hyperparameter_set`` iterations.

        Returns the best normalised cost change of every sampler per iteration,
        shape ``(N_SAMPLERS, iterations)``. Stops early (and sets a flag on
        ``state``) when the budget runs out or the restart policy fires.
        """
        prof = self._profiler
        n_iterations = self._config.iterations_per_hyperparameter_set
        best_changes = np.zeros((N_SAMPLERS, n_iterations))
        with prof.block("section"):
            for i in range(n_iterations):
                with prof.block("draw"):
                    samples, boundaries = self._draw(state, samplers)
                with prof.block("evaluate"):
                    cost, complete = self._objective.evaluate_columns(
                        samples, self._known_costs(state, boundaries)
                    )
                if not complete:
                    state.force_stop = True
                    if self._config.strict_budget and len(cost):
                        self._keep_best_of_partial(state, samples, cost)
                    break
                cost = penalize_failures(cost, state.previous_cost)
                with prof.block("advance"):
                    best_changes[:, i] = self._advance(
                        state, samplers, samples, boundaries, cost
                    )
                if state.restart:
                    break
        return best_changes

    def _known_costs(
        self, state: SearchState, boundaries: np.ndarray
    ) -> dict[int, float] | None:
        """Costs that need no evaluation: the incumbent in each sampler's column 0."""
        if self._config.variant.reevaluate_incumbent:
            return None
        return {int(b): state.previous_cost for b in boundaries[:-1]}

    @staticmethod
    def _keep_best_of_partial(
        state: SearchState, samples: np.ndarray, cost: np.ndarray
    ) -> None:
        """The budget ran out mid-batch: keep the best point evaluated so far."""
        cost = penalize_failures(cost, state.previous_cost)
        winner = int(np.argmin(cost))
        state.best = samples[:, winner]
        state.current_cost = cost[winner]

    def _draw(
        self, state: SearchState, samplers: list[SamplerState]
    ) -> tuple[np.ndarray, np.ndarray]:
        n_params = len(state.best)
        parents = state.parents
        parts = []
        prof = self._profiler
        for sampler in samplers:
            adaptive = np.broadcast_to(
                sampler.adaptive_vector[:, np.newaxis], (n_params, sampler.n_samples)
            )
            with prof.block("sample"):
                samples, sampler.random_growth, sampler.targeted_growth = (
                    two_mode_samples(
                        state.best,
                        parents,
                        sampler.n_samples,
                        state.change_vector,
                        sampler.random_growth,
                        sampler.targeted_growth,
                        sampler.params,
                        adaptive,
                        self._rng,
                        prof,
                        self._options,
                        state.cov_factor,
                    )
                )
            if self._box is not None:
                self._shape_to_box(samples, state)
            parts.append(samples)
        boundaries = np.cumsum([0] + [p.shape[1] for p in parts])
        return np.concatenate(parts, axis=1), boundaries

    def _shape_to_box(self, samples: np.ndarray, state: SearchState) -> None:
        """Variants that know the box: a uniform first pool, folding."""
        assert self._box is not None
        variant = self._config.variant
        if (
            variant.initial_design == "uniform"
            and self.initial_design
            and not state.cost_history
        ):
            samples[:, 1:] = uniform_columns(samples.shape[1] - 1, self._box, self._rng)
        if variant.bounds == "fold":
            repair(samples, self._box, variant.corner_clip, self._rng)

    def _learn_covariance(
        self,
        state: SearchState,
        samplers: list[SamplerState],
        samples: np.ndarray,
        boundaries: np.ndarray,
        cost: np.ndarray,
    ) -> None:
        """Rank-mu update of the search covariance from this iteration's samples."""
        assert state.cov is not None
        widths = np.concatenate(
            [
                np.full(boundaries[k + 1] - boundaries[k], np.exp(s.random_growth))
                for k, s in enumerate(samplers)
            ]
        )
        real = np.ones(len(cost), dtype=bool)
        real[boundaries[:-1]] = False  # the incumbent's own columns carry no step
        if not real.any():
            return
        steps = ((samples[:, real] - state.best[:, np.newaxis]) / widths[real]).T
        state.cov, state.cov_factor = update_covariance(
            state.cov, steps, cost[real], self._config.variant
        )

    def _advance(
        self,
        state: SearchState,
        samplers: list[SamplerState],
        samples: np.ndarray,
        boundaries: np.ndarray,
        cost: np.ndarray,
    ) -> np.ndarray:
        """Update the incumbent and the allocation; returns per-sampler gains."""
        prof = self._profiler
        with prof.block("select"):
            winner = int(np.argmin(cost))
            state.current_cost = cost[winner]
            new_change = samples[:, winner] - state.best
            if state.cov is not None:
                self._learn_covariance(state, samplers, samples, boundaries, cost)

        # Legacy (audit B1): the original also updated an "adaptive amplitude"
        # here, but never read the result. It is omitted.
        probe = cost - state.previous_cost
        if self._config.variant.amplitude_control != "legacy":
            update_log_widths(samplers, probe, boundaries, self._config.variant)
        with prof.block("reallocate"):
            gains = self._reallocate(probe, boundaries, samplers)

        state.previous_cost = state.current_cost
        state.change_vector = new_change
        state.cost_history.append(state.current_cost)

        with prof.block("select"):
            state.best = samples[:, winner]
            # The next draw reads one kind of parent only (see ``multi_parent``),
            # so only that one is selected.
            if state.multi_parent:
                state.best_eighth = samples[:, best_eighth(cost)]
            else:
                state.second_best = samples[
                    :, runner_up(cost, winner, self._config.second_best)
                ]

        with prof.block("restart_check"):
            state.restart, _reason = check_restart(
                [s.random_growth for s in samplers],
                state.cost_history,
                self._config.restart,
            )
        return gains

    def _reallocate(
        self,
        probe: np.ndarray,
        boundaries: np.ndarray,
        samplers: list[SamplerState],
    ) -> np.ndarray:
        if np.abs(probe).sum() != 0:
            probe = probe / np.sqrt(np.matmul(probe, probe))
        gains = np.zeros(N_SAMPLERS)
        for k in range(N_SAMPLERS):
            segment = probe[boundaries[k] : boundaries[k + 1]]
            gains[k] = segment[np.argmin(segment)]
        current = np.array([s.n_samples for s in samplers])
        allocation = self._conductor.step(current, -gains)
        for sampler, n in zip(samplers, allocation, strict=True):
            sampler.n_samples = n
        return gains
