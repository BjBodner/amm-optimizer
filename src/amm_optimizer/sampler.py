"""Two-mode sample generation: a random search plus a targeted search.

Every function here is a pure translation of the original ``py`` helpers.
The order of random draws and the order of floating-point operations are kept
identical so that a given ``RandomState`` reproduces the original numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .config import Variant
from .profiling import NULL_PROFILER, Profiler

# Per-sample amplitude is capped at this value (original: ``np.minimum(.., 10)``).
MAX_RANDOM_AMPLITUDE = 10.0
# Targeted amplitudes above this (or NaN) are replaced by ``100 * U(0, 1)``.
TARGETED_AMPLITUDE_RESET = 1000.0
TARGETED_AMPLITUDE_FALLBACK = 100.0
# Range of the log-uniform targeted multiplier (``success_rate_targeted``).
TARGETED_MULTIPLIER_RANGE = (0.1, 10.0)
# ``exp(term)`` is only evaluated below this exponent to avoid overflow.
MAX_EXPONENT = 100.0
PHASE_EPSILON = 0.01


class ModeParams(NamedTuple):
    """The eight hyperparameters of one two-mode sampler."""

    random_growth_rate: float
    random_period: float
    max_random_search: float
    targeted_growth_rate: float
    targeted_decay_rate: float
    significant_change: float
    alpha: float
    beta: float

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> ModeParams:
        # The legacy vector has 11 rows; only the first eight are used.
        return cls(*vector[:8])


@dataclass(frozen=True)
class SampleOptions:
    """Departures from the original sampler, passed by the sample level only.

    ``None`` selects the original statements, and an instance with every field at
    its default must behave identically. The meta level, which reuses
    :func:`two_mode_samples`, never passes options.
    """

    # Take the targeted step for any non-zero move, not only a move with a
    # positive coordinate.
    targeted_gate_any: bool = False
    # Targeted step length: a log-uniform multiple (0.1 to 10) of the last move.
    targeted_loguniform: bool = False
    # ``random_growth`` is the logarithm of a scalar width, adapted outside
    # (``control``): no sawtooth, no random vector, no growth factors.
    scalar_width: bool = False
    # Gaussian noise with the covariance factor passed as ``noise_factor``.
    use_covariance: bool = False


def sample_options(variant: Variant) -> SampleOptions | None:
    """The sampler options a variant asks for; ``None`` if it asks for none."""
    rate_controlled = variant.amplitude_control != "legacy"
    with_targeted = variant.amplitude_control == "success_rate_targeted"
    options = SampleOptions(
        targeted_gate_any=with_targeted,
        targeted_loguniform=with_targeted,
        scalar_width=rate_controlled,
        use_covariance=variant.covariance != "none",
    )
    return None if options == SampleOptions() else options


def periodic_decay(growth_factor: float, period: float) -> float:
    """``sin(phase + eps) - 1`` where phase wraps every quarter period."""
    phase = np.mod(growth_factor * np.pi / (2 * period), np.pi / 2)
    return float(np.sin(phase + PHASE_EPSILON) - 1)


def random_search_amplitude(
    max_search: float,
    growth_rate: float,
    growth_factor: float,
    period: float,
    adaptive_amplitude: np.ndarray,
) -> np.ndarray:
    scale = np.exp(growth_rate * periodic_decay(growth_factor, period))
    amplitude = max_search * scale * adaptive_amplitude
    return np.asarray(np.minimum(amplitude, MAX_RANDOM_AMPLITUDE))


def update_growth_factors(
    change_vector: np.ndarray,
    significant_change: float,
    random_growth: float,
    targeted_growth: float,
    reset_random_growth: bool = True,
) -> tuple[float, float]:
    """Reset random growth on a significant move, reset targeted growth on none."""
    total_change = np.abs(change_vector).sum()
    if total_change > significant_change:
        if reset_random_growth:
            random_growth = 0
        targeted_growth += 1
    if total_change < significant_change:
        targeted_growth = 0
    return random_growth, targeted_growth


def _tile_columns(vector: np.ndarray, n_columns: int) -> np.ndarray:
    """Read-only ``(len(vector), n_columns)`` view repeating ``vector``."""
    return np.broadcast_to(vector[:, np.newaxis], (len(vector), n_columns))


def two_parent_crossover(
    seed: np.ndarray, second: np.ndarray, n_samples: int, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    seed_tiled = _tile_columns(seed, n_samples)
    second_tiled = _tile_columns(second, n_samples)
    genes = rng.rand(len(seed), n_samples)
    child = (genes > 0.5) * seed_tiled + (genes <= 0.5) * second_tiled
    return seed_tiled, child


def multi_parent_crossover(
    seed: np.ndarray, parents: np.ndarray, n_samples: int, rng: np.random.RandomState
) -> tuple[np.ndarray, np.ndarray]:
    n_params = len(seed)
    seed_tiled = _tile_columns(seed, n_samples)
    picks = rng.randint(0, parents.shape[1], (n_params, n_samples))
    child = parents[np.arange(n_params)[:, None], picks]
    return seed_tiled, child


def targeted_search(
    change_vector: np.ndarray,
    n_samples: int,
    params: ModeParams,
    random_growth: float,
    previous_random_growth: float,
    targeted_growth: float,
    rng: np.random.RandomState,
    loguniform: bool = False,
) -> np.ndarray:
    """Step along the last successful direction with a random, decaying length.

    ``loguniform`` replaces the length by a log-uniform multiple of the move.
    """
    random_numbers = rng.rand(n_samples, 1)
    if loguniform:
        low, high = TARGETED_MULTIPLIER_RANGE
        multiplier = np.exp(np.log(low) + random_numbers * np.log(high / low))
        return np.asarray((change_vector * multiplier).T)
    base = params.max_random_search * np.exp(
        params.random_growth_rate
        * periodic_decay(previous_random_growth, params.random_period)
    )
    exponent = (
        targeted_growth * params.targeted_growth_rate
        - random_growth * params.targeted_decay_rate
    )
    if exponent < MAX_EXPONENT:
        amplitude = base * random_numbers * np.exp(exponent)
    else:
        amplitude = TARGETED_AMPLITUDE_FALLBACK * random_numbers
    if np.abs(amplitude).max() > TARGETED_AMPLITUDE_RESET or np.isnan(amplitude.max()):
        amplitude = TARGETED_AMPLITUDE_FALLBACK * random_numbers
    return np.asarray((change_vector * amplitude).T)


def two_mode_samples(
    seed: np.ndarray,
    parents: np.ndarray,
    n_samples: int,
    change_vector: np.ndarray,
    random_growth: float,
    targeted_growth: float,
    params: ModeParams,
    adaptive_amplitude: np.ndarray,
    rng: np.random.RandomState,
    profiler: Profiler = NULL_PROFILER,
    options: SampleOptions | None = None,
    noise_factor: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float]:
    """Generate ``n_samples`` candidate columns around ``seed``.

    ``parents`` is either a single second-best vector (two-parent crossover)
    or a matrix of good individuals (multi-parent crossover). Column 0 of the
    result is always ``seed``. Returns the samples and the updated random and
    targeted growth factors. ``profiler`` times the four stages.
    """
    n_params = len(seed)
    previous_random_growth = random_growth
    scalar_width = options is not None and options.scalar_width
    if not scalar_width:
        random_growth, targeted_growth = update_growth_factors(
            change_vector,
            params.significant_change,
            random_growth,
            targeted_growth,
        )

    crossover = two_parent_crossover if parents.ndim == 1 else multi_parent_crossover
    with profiler.block("sample.crossover"):
        seed_tiled, random_part = crossover(seed, parents, n_samples, rng)

    with profiler.block("sample.amplitude"):
        if scalar_width:
            amplitude = np.exp(random_growth)
        else:
            amplitude = random_search_amplitude(
                params.max_random_search,
                params.random_growth_rate,
                random_growth,
                params.random_period,
                adaptive_amplitude,
            )
    with profiler.block("sample.noise"):
        if noise_factor is not None and options is not None and options.use_covariance:
            # Gaussian with the uniform noise's standard deviation, 1 / sqrt(3)
            noise = (noise_factor @ rng.randn(n_params, n_samples)) / np.sqrt(3.0)
        else:
            noise = 2 * (rng.rand(n_params, n_samples) - 0.5)
        random_part += noise * amplitude - seed_tiled

    targeted = np.zeros((n_params, n_samples))
    if options is not None and options.targeted_gate_any:
        gate_open = bool(change_vector.any())
    else:
        gate_open = bool(change_vector.max() > 0)
    if gate_open:
        with profiler.block("sample.targeted"):
            targeted = targeted_search(
                change_vector,
                n_samples,
                params,
                random_growth,
                previous_random_growth,
                targeted_growth,
                rng,
                options is not None and options.targeted_loguniform,
            )

    samples = seed[:, np.newaxis] + targeted + random_part
    samples[:, 0] = seed
    return (
        samples,
        random_growth if scalar_width else random_growth + 1,
        targeted_growth,
    )
