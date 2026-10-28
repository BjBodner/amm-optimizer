"""Variants ``crossover``, ``noise``, ``skip_starved`` and ``width_mode``."""

import numpy as np
import pytest

from amm_optimizer.config import Variant
from amm_optimizer.control import LADDER, start_log_widths, update_log_widths
from amm_optimizer.sampler import ModeParams, sample_options, two_mode_samples
from amm_optimizer.state import SamplerState

# research-only options, deprecated in 0.6.0 (they warn when set)
pytestmark = pytest.mark.filterwarnings("ignore:Variant:FutureWarning")

SR = dict(amplitude_control="success_rate_targeted")


def draw(variant: Variant, parents: np.ndarray, n: int = 4000, width: float = 0.0):
    rng = np.random.RandomState(0)
    samples, _, _ = two_mode_samples(
        np.zeros(3),
        parents,
        n,
        np.zeros(3),
        float(np.log(width)) if width else -50.0,
        0.0,
        ModeParams.from_vector(np.zeros(11)),
        np.ones((3, n)),
        rng,
        options=sample_options(variant),
    )
    return samples


def test_no_crossover_centres_on_the_seed():
    parents = np.full(3, 4.0)
    mixed = draw(Variant(**SR), parents)
    centred = draw(Variant(**SR, crossover="none"), parents)
    assert np.abs(mixed[:, 1:]).max() == pytest.approx(4.0)
    assert np.abs(centred[:, 1:]).max() < 1e-12


def test_gaussian_noise_keeps_the_standard_deviation():
    uniform = draw(Variant(**SR, crossover="none"), np.zeros(3), width=1.0)
    gauss = draw(
        Variant(**SR, crossover="none", noise="gaussian"), np.zeros(3), width=1.0
    )
    assert uniform.std() == pytest.approx(gauss.std(), rel=0.05)
    assert np.abs(gauss).max() > 1.0 >= np.abs(uniform).max()


def samplers_with(widths, sizes):
    out = []
    for w, n in zip(widths, sizes, strict=True):
        s = SamplerState(2, np.random.RandomState(0))
        s.random_growth, s.n_samples = float(np.log(w)), n
        out.append(s)
    return out


def boundaries(sizes):
    return np.concatenate([[0], np.cumsum(sizes)])


def test_skip_starved_leaves_a_sampler_without_samples_alone():
    sizes = [13, 1, 1, 1]
    probe = np.zeros(16)
    for skip in (False, True):
        samplers = samplers_with([1, 1, 1, 1], sizes)
        update_log_widths(
            samplers, probe, boundaries(sizes), Variant(**SR, skip_starved=skip)
        )
        moved = [s.random_growth != 0.0 for s in samplers]
        assert moved == ([True] * 4 if not skip else [True, False, False, False])


def test_ladder_starts_at_the_original_widths_and_moves_together():
    variant = Variant(**SR, width_mode="ladder")
    widths = np.exp(start_log_widths(variant))
    assert widths == pytest.approx([0.5, 1.0, 2.0, 4.0])
    sizes = [13, 1, 1, 1]
    samplers = samplers_with(widths, sizes)
    probe = np.ones(16)
    probe[1:7] = -1.0  # 6 of the 12 real samples improved: rate .5 > .15
    update_log_widths(samplers, probe, boundaries(sizes), variant)
    after = np.exp([s.random_growth for s in samplers])
    assert after / after[2] == pytest.approx(np.array(LADDER) / LADDER[2])
    # sigma 2 * exp(1.5 * .35) = 3.4 exceeds the cap: the widest rung (2 sigma)
    # stops at sr_max_width = 5
    assert after[2] == pytest.approx(min(2.0 * np.exp(1.5 * 0.35), 2.5))
    assert after[3] == pytest.approx(5.0)


def test_shared_width_is_one_width():
    variant = Variant(**SR, width_mode="shared")
    widths = start_log_widths(variant)
    assert len(set(widths)) == 1
    assert np.exp(widths[0]) == pytest.approx(np.sqrt(2.0))


def test_these_flags_need_a_success_rate_controller():
    with pytest.raises(ValueError, match="success-rate"):
        Variant(skip_starved=True)
    with pytest.raises(ValueError, match="success-rate"):
        Variant(width_mode="ladder")
