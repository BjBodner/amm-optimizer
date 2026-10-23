"""The search box, and how samples that leave it are brought back.

The original does not know the box: the caller clips and penalises. A variant
that knows it can keep every sample inside, and store the repaired point as the
incumbent so that the search cannot drift out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The ``clip`` ufunc itself, for the hot loops. ``np.clip`` reaches the same loop (so
# the same bits) through three Python wrappers, which cost more than clipping a short
# vector. The ufunc lives in ``numpy._core`` from NumPy 2.0 and in ``numpy.core``
# before; ``np.clip`` is the public fallback, so the import never fails.
try:
    clip: np.ufunc = np._core.umath.clip  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover  (NumPy 1.x)
    try:
        clip = np.core.umath.clip  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        clip = np.clip  # type: ignore[assignment]


@dataclass(frozen=True)
class Box:
    """Per-coordinate lower and upper bounds, both of length ``dim``."""

    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def make(cls, bounds: tuple[object, object], dim: int) -> Box:
        """From ``(lower, upper)``, each a scalar or a sequence of length ``dim``."""
        lower, upper = (
            np.array(np.broadcast_to(np.asarray(b, dtype=float), (dim,)))
            for b in bounds
        )
        if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise ValueError("bounds must be finite")
        if not (lower < upper).all():
            raise ValueError("every lower bound must be below its upper bound")
        return cls(lower, upper)


def reflect(x: np.ndarray, box: Box) -> np.ndarray:
    """Fold ``x`` into the box by repeated reflection at the walls."""
    lower, upper = box.lower[:, np.newaxis], box.upper[:, np.newaxis]
    width = upper - lower
    y = np.mod(x - lower, 2 * width)
    return np.asarray(lower + np.where(y > width, 2 * width - y, y))


def repair(
    samples: np.ndarray,
    box: Box,
    corner_clip: float,
    rng: np.random.RandomState,
) -> None:
    """Bring every coordinate outside the box back in, in place.

    Column 0 (the incumbent) is left alone. Out-of-box coordinates are reflected
    at the walls. With probability ``corner_clip`` a sample that has any
    coordinate outside is instead clipped onto the box *as a whole* (all its
    out-of-box coordinates onto the walls they crossed), so that optima in a
    corner (bbob f5, solved by clipping in the original) stay reachable: a far
    sample then lands in the corner with probability ``corner_clip``. Infinite
    coordinates are always clipped. A random draw is made only when
    ``corner_clip > 0`` and some sample has a coordinate outside.
    """
    body = samples[:, 1:]
    lower, upper = box.lower[:, np.newaxis], box.upper[:, np.newaxis]
    outside = (body < lower) | (body > upper)
    if not outside.any():
        return
    use_clip = ~np.isfinite(body)
    if corner_clip > 0:
        leaving = outside.any(axis=0)
        clip_sample = np.zeros(body.shape[1], dtype=bool)
        clip_sample[leaving] = rng.rand(int(leaving.sum())) < corner_clip
        use_clip = use_clip | clip_sample[np.newaxis, :]
    repaired = np.where(use_clip, np.clip(body, lower, upper), reflect(body, box))
    samples[:, 1:] = np.where(outside, repaired, body)


def uniform_columns(n_columns: int, box: Box, rng: np.random.RandomState) -> np.ndarray:
    """``n_columns`` points drawn uniformly from the box, shape ``(dim, n)``."""
    dim = len(box.lower)
    return np.asarray(
        box.lower[:, np.newaxis]
        + (box.upper - box.lower)[:, np.newaxis] * rng.rand(dim, n_columns)
    )
