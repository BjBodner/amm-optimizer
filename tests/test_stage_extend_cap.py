"""``stage_extend_max_per_dim``: ``stage_extend_restart`` only at budgets up to that
many x dim (Batch 89 rejected the uncapped option for its 1000 x D losses)."""

import numpy as np
import pytest

from amm_optimizer import presets, qn_phase
from amm_optimizer.api import minimize
from amm_optimizer.config import Variant

ORIGINAL = qn_phase.run_slsqp_restarts


def rotated_ellipsoid(x):
    """Smooth and ill-conditioned: the hand-over reads it smooth (the staged route)."""
    x = np.asarray(x, dtype=float)
    q, _ = np.linalg.qr(np.random.RandomState(7).normal(size=(x.size, x.size)))
    z = q @ (x - 1.0)
    return float(np.sum(10.0 ** (3.0 * np.arange(x.size) / (x.size - 1)) * z * z))


def stage_calls(monkeypatch, config, budget, dim=5):
    """The ``extend_restart`` argument of every staged SLSQP call of one run."""
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs.get("extend_restart", False))
        return ORIGINAL(*args, **kwargs)

    monkeypatch.setattr(qn_phase, "run_slsqp_restarts", spy)
    x0 = np.random.RandomState(11).uniform(-5, 5, dim)
    minimize(rotated_ellipsoid, x0, budget, seed=3, config=config, bounds=(-5.0, 5.0))
    return seen


def test_default_is_no_limit():
    assert Variant().stage_extend_max_per_dim == 0.0
    assert presets.get("c46") == presets.with_changes(
        "c46", stage_extend_max_per_dim=0.0
    )
    with pytest.raises(ValueError):
        presets.with_changes("c46", stage_extend_max_per_dim=-1.0)


def test_the_cap_gates_the_extension_by_budget(monkeypatch):
    dim = 5
    sx = presets.with_changes("c46", stage_extend_restart=True)
    capped = presets.with_changes(
        "c46", stage_extend_restart=True, stage_extend_max_per_dim=300.0
    )
    at_300 = stage_calls(monkeypatch, sx, 300 * dim)
    assert at_300 and any(at_300), "the staged route must run on this smooth function"
    assert stage_calls(monkeypatch, capped, 300 * dim) == at_300
    assert not any(stage_calls(monkeypatch, capped, 1000 * dim))
    assert any(stage_calls(monkeypatch, sx, 1000 * dim))
