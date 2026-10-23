"""Variants ``cov_steps`` and ``cov_path`` of the rank-mu covariance."""

import numpy as np
import pytest

import amm_optimizer as atm
from amm_optimizer.config import ATMConfig, Variant
from amm_optimizer.control import update_covariance

COV = dict(
    amplitude_control="success_rate_targeted",
    covariance="rank_mu",
    cov_rate=3.0,
    cov_select=0.5,
    pool_rule="budget",
    pool_min=12,
)


def test_rank_one_term_pulls_the_shape_toward_the_path():
    variant = Variant(**COV)
    rng = np.random.RandomState(0)
    steps, cost = rng.randn(20, 4), rng.rand(20)
    path = np.array([3.0, 0.0, 0.0, 0.0])
    plain, _ = update_covariance(np.eye(4), steps, cost, variant)
    pulled, _ = update_covariance(np.eye(4), steps, cost, variant, path)
    assert pulled[0, 0] / np.trace(pulled) > plain[0, 0] / np.trace(plain)


def test_needs_the_covariance_variant():
    with pytest.raises(ValueError, match="rank_mu"):
        Variant(amplitude_control="success_rate_targeted", cov_steps="noise")
    with pytest.raises(ValueError, match="rank_mu"):
        Variant(amplitude_control="success_rate_targeted", cov_path="move")


@pytest.mark.parametrize(
    "changes",
    [
        dict(cov_steps="noise"),
        dict(cov_steps="noise", cov_path="noise"),
        dict(cov_steps="noise", cov_path="move"),
    ],
)
def test_runs_within_the_budget(changes):
    q, _ = np.linalg.qr(np.random.RandomState(1).randn(5, 5))
    scales = 1e4 ** (np.arange(5) / 4)

    def fun(x):
        return float(np.sum(scales * (q @ x) ** 2))

    config = ATMConfig(variant=Variant(**COV, **changes))
    result = atm.minimize(fun, np.full(5, 2.0), 3000, seed=4, config=config)
    assert result.nfev == 3000
    assert np.isfinite(result.fun)
