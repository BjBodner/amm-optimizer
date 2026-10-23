import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from amm_optimizer.conductor import ConductorConfig, ResourceConductor


@pytest.fixture
def conductor():
    return ResourceConductor(ConductorConfig(total_resources=60))


@pytest.mark.parametrize(
    "change_vector, min_allocation, max_allocation",
    [
        ([-1, 0, 0, 0], 3, 19),
        ([1, 0, 0, 0], 11, 27),
    ],
)
def test_conductor_settles_within_expected_range(
    conductor, change_vector, min_allocation, max_allocation
):
    """Carried over from the original suite: 50 steps under a constant push."""
    allocation = np.array([15, 15, 15, 15])
    for _ in range(50):
        allocation = conductor.step(allocation, np.array(change_vector))

    assert allocation.min() >= min_allocation
    assert allocation.max() <= max_allocation
    assert allocation.sum() == 60


def test_zero_change_keeps_resting_allocation(conductor):
    allocation = conductor.step(conductor.rest, np.zeros(4))
    np.testing.assert_array_equal(allocation, [15, 15, 15, 15])


def test_total_is_conserved_for_random_pushes():
    rng = np.random.RandomState(0)
    conductor = ResourceConductor(ConductorConfig(total_resources=64))
    allocation = np.array([16, 16, 16, 16])
    for _ in range(200):
        allocation = conductor.step(allocation, rng.randn(4))
        assert allocation.sum() == 64


def test_nan_change_keeps_the_current_allocation():
    """The original reset everything to 16 each, which broke conservation."""
    conductor = ResourceConductor(ConductorConfig(total_resources=60))
    current = np.array([20, 10, 15, 15])
    allocation = conductor.step(current, np.full(4, np.nan))
    np.testing.assert_array_equal(allocation, current)
    assert allocation.sum() == 60


def test_partially_nan_change_keeps_the_current_allocation():
    conductor = ResourceConductor(ConductorConfig(total_resources=60))
    current = np.array([12, 18, 15, 15])
    allocation = conductor.step(current, np.array([1.0, np.nan, 0.0, 0.0]))
    np.testing.assert_array_equal(allocation, current)


STIFF = ConductorConfig(
    total_resources=60, self_spring=1.0, interaction_spring=400.0, mass=10.0
)


def test_custom_config_never_starves_a_sampler():
    """A strong coupling used to drive allocations to zero or below."""
    conductor = ResourceConductor(STIFF)
    allocation = np.array([15, 15, 15, 15])
    for push in ([1, 0, 0, 0], [0, -1, 0, 0], [-1, 1, 1, 1], [1, -1, -1, -1]):
        for _ in range(5):
            allocation = conductor.step(allocation, np.array(push, dtype=float))
            assert allocation.min() >= 1
            assert allocation.sum() == 60


@settings(max_examples=200, deadline=None)
@given(
    pushes=st.lists(
        st.lists(st.floats(-1e6, 1e6), min_size=4, max_size=4),
        min_size=1,
        max_size=15,
    ),
    stiff=st.booleans(),
)
def test_allocation_invariants_for_any_finite_pushes(pushes, stiff):
    conductor = ResourceConductor(STIFF if stiff else ConductorConfig(60))
    allocation = np.array([15, 15, 15, 15])
    for push in pushes:
        allocation = conductor.step(allocation, np.array(push))
        assert allocation.dtype == np.int64
        assert allocation.min() >= 1
        assert allocation.sum() == 60
