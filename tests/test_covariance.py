"""``covariance="rank_mu"`` (E15): a covariance learned from selected samples."""

import numpy as np
import pytest

import amm_optimizer as atm
import amm_optimizer.section as section
from amm_optimizer.config import Variant
from amm_optimizer.control import update_covariance
from amm_optimizer.sampler import SampleOptions, sample_options
from test_sampler_options import draw

COV = Variant(
    amplitude_control="success_rate_targeted",
    covariance="rank_mu",
    cov_rate=3.0,
    cov_select=0.5,
    pool_rule="budget",
    pool_min=12,
)


def shape_of(variant, steps, cost, cov=None):
    dim = steps.shape[1]
    return update_covariance(np.eye(dim) if cov is None else cov, steps, cost, variant)


def test_the_result_is_symmetric_with_unit_geometric_mean_and_a_matching_factor():
    rng = np.random.RandomState(0)
    steps = rng.randn(12, 4) * np.array([3.0, 1.0, 0.3, 0.1])
    shape, factor = shape_of(COV, steps, rng.rand(12))
    np.testing.assert_allclose(shape, shape.T, atol=1e-12)
    values = np.linalg.eigvalsh(shape)
    assert np.exp(np.mean(np.log(values))) == pytest.approx(1.0)
    np.testing.assert_allclose(factor @ factor.T, shape, atol=1e-10)


def test_repeated_selection_of_correlated_steps_learns_the_direction():
    rng = np.random.RandomState(1)
    direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    cov = np.eye(3)
    for _ in range(80):
        steps = rng.randn(8, 3) * 0.3 + rng.randn(8, 1) * 2.0 * direction
        cost = -np.abs(steps @ direction)  # the long steps along the ridge win
        cov, _ = update_covariance(cov, steps, cost, COV)
    values, vectors = np.linalg.eigh(cov)
    assert abs(vectors[:, -1] @ direction) > 0.95
    assert values[-1] / values[0] > 10


def test_the_eigenvalue_ratio_is_capped(monkeypatch):
    import amm_optimizer.control as control

    monkeypatch.setattr(control, "COV_CAP", 50.0)
    capped = Variant(
        amplitude_control="success_rate_targeted",
        covariance="rank_mu",
        cov_rate=100.0,
    )
    steps = np.array([[100.0, 0.0], [90.0, 0.1], [80.0, 0.0], [95.0, 0.2]])
    cov = np.eye(2)
    for _ in range(20):
        cov, _ = update_covariance(cov, steps, np.arange(4.0), capped)
    values = np.linalg.eigvalsh(cov)
    assert values[-1] / values[0] <= 50.0 * (1 + 1e-9)


def test_a_single_selected_step_barely_moves_the_covariance():
    cov, _ = shape_of(
        Variant(
            amplitude_control="success_rate_targeted",
            covariance="rank_mu",
            cov_select=0.01,
        ),
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        np.array([0.0, 1.0, 2.0]),
    )
    np.testing.assert_allclose(cov, np.eye(2), atol=1e-12)  # mu_eff = 1: rate 0


def test_samples_follow_the_covariance_factor_and_the_width():
    options = SampleOptions(scalar_width=True, use_covariance=True)
    factor = np.array([[2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.1]])
    (samples, _, _), _ = draw(
        [0, 0, 0], options, n=20000, random_growth=np.log(0.6), noise_factor=factor
    )
    empirical = np.cov(samples[:, 1:])
    expected = 0.6**2 / 3.0 * (factor @ factor.T)
    np.testing.assert_allclose(empirical, expected, rtol=0.1, atol=0.003)


def test_the_variant_asks_for_covariance_noise():
    assert sample_options(COV).use_covariance


def test_covariance_parameters_are_validated():
    base = {"amplitude_control": "success_rate_targeted"}
    for kwargs, match in [
        ({"covariance": "diag"}, "covariance"),
        ({"covariance": "rank_mu", "cov_rate": 0.0}, "cov_rate"),
        ({"covariance": "rank_mu", "cov_select": 0.0}, "cov_select"),
        ({"covariance": "rank_mu", "cov_select": 1.5}, "cov_select"),
    ]:
        with pytest.raises(ValueError, match=match):
            Variant(**base, **kwargs)
    with pytest.raises(ValueError, match="amplitude_control"):
        Variant(covariance="rank_mu")


def test_a_run_learns_an_elongated_shape_on_a_rotated_ellipsoid(monkeypatch):
    dim = 4
    rng = np.random.RandomState(0)
    rotation = np.linalg.qr(rng.randn(dim, dim))[0]
    weights = 10.0 ** (4 * np.arange(dim) / (dim - 1))
    seen = []
    original = section.update_covariance

    def spy(*args, **kwargs):
        out = original(*args, **kwargs)
        seen.append(out[0])
        return out

    monkeypatch.setattr(section, "update_covariance", spy)

    def ellipsoid(x):
        z = rotation @ (np.asarray(x) - 1.0)
        return float(np.sum(weights * z * z))

    config = atm.ATMConfig(variant=COV)
    result = atm.api.minimize(
        ellipsoid, np.zeros(dim), 3000, seed=0, config=config, bounds=(-5, 5)
    )
    assert result.nfev == 3000
    values = np.linalg.eigvalsh(seen[-1])
    assert values[-1] / values[0] > 30  # far from the unit sphere it started as


def test_the_covariance_is_not_learned_above_the_dimension_limit(monkeypatch):
    calls = []
    original = section.update_covariance
    monkeypatch.setattr(
        section,
        "update_covariance",
        lambda *args: calls.append(1) or original(*args),
    )
    import amm_optimizer.search as search

    monkeypatch.setattr(search, "COV_MAX_DIM", 3)
    limited = COV
    objective = lambda x: float(np.sum(np.asarray(x) ** 2))  # noqa: E731
    atm.api.minimize(
        objective,
        np.ones(4),
        300,
        seed=0,
        config=atm.ATMConfig(variant=limited),
        bounds=(-5, 5),
    )
    assert not calls
    atm.api.minimize(
        objective,
        np.ones(3),
        300,
        seed=0,
        config=atm.ATMConfig(variant=limited),
        bounds=(-5, 5),
    )
    assert calls


def test_an_iteration_with_only_incumbent_columns_leaves_the_covariance_alone():
    from amm_optimizer.objective import CountedObjective
    from amm_optimizer.state import SamplerState, SearchState

    runner = section.SectionRunner(
        np.random.RandomState(0),
        CountedObjective(lambda x: 0.0, 10),
        atm.ATMConfig(variant=COV),
    )
    samplers = [SamplerState(2, np.random.RandomState(k)) for k in range(4)]
    state = SearchState(
        best=np.zeros(2),
        second_best=np.zeros(2),
        previous_cost=1.0,
        change_vector=np.zeros(2),
        current_cost=1.0,
        cov=np.eye(2),
        cov_factor=np.eye(2),
    )
    runner._learn_covariance(
        state, samplers, np.zeros((2, 4)), np.arange(5), np.ones(4)
    )
    np.testing.assert_array_equal(state.cov, np.eye(2))
