"""The anytime mode (``Variant.anytime``, ``amm_optimizer.anytime``).

Off (the default) nothing changes, hashes included. ``"replay"``: up to
``anytime_switch`` evaluations each stage is exactly the configuration's run at its
planned budget, and a point an earlier stage evaluated costs no call; ``budget``
only caps the calls. Then one ordinary run gets what is left, fed the archive.
"""

import hashlib

import numpy as np
import pytest

from amm_optimizer import ATMConfig, anytime, minimize, presets
from amm_optimizer.config import Variant

BOX = (-5.0, 5.0)


def rosenbrock(x):
    x = np.asarray(x)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


class Logged:
    """``fun`` that records every point it is called at."""

    def __init__(self, fun):
        self.fun, self.points = fun, []

    def __call__(self, x):
        self.points.append(np.array(x, dtype=float))
        return self.fun(x)


def digest(points):
    sha = hashlib.sha256()
    for p in points:
        sha.update(np.asarray(p, dtype=float).tobytes())
    return sha.hexdigest()


# research-only options (deprecated in 0.6.0)
@pytest.mark.filterwarnings("ignore:Variant:FutureWarning")
def test_the_new_fields_leave_every_recorded_hash_as_it_was():
    assert presets.get("c46").config_hash() == "2153948ebed51150"
    assert ATMConfig().config_hash() == "f8fc4662f473b680"
    on = presets.with_changes("c46", anytime="replay")
    assert on.config_hash() != "2153948ebed51150"
    factor = presets.with_changes("c46", anytime="replay", anytime_factor=3.0)
    assert factor.config_hash() != on.config_hash()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"anytime": "continue"},  # only off and replay are supported
        {"anytime_factor": 1.0},
        {"anytime_start_per_dim": 0.0},
        {"anytime_switch": 0},
    ],
)
def test_the_anytime_options_are_validated(kwargs):
    with pytest.raises(ValueError, match="anytime"):
        Variant(**kwargs)


def test_off_is_the_plain_run():
    """The default dispatch reads one attribute: the same calls as before."""
    a, b = Logged(rosenbrock), Logged(rosenbrock)
    config = presets.get("c46")
    minimize(a, np.full(5, 3.0), 600, seed=4, config=config, bounds=BOX)
    off = presets.with_changes("c46", anytime="off")
    minimize(b, np.full(5, 3.0), 600, seed=4, config=off, bounds=BOX)
    assert digest(a.points) == digest(b.points)


@pytest.mark.parametrize("dim", [2, 5])
def test_each_replay_stage_is_the_run_at_its_plan(dim):
    """Stage k asks for exactly the points of c46's run at 10 * 2^k * dim with the
    same seed; only those no earlier stage asked for are called."""
    x0 = np.full(dim, 3.0)
    config = presets.with_changes("c46", anytime="replay")
    anytime.STAGE_LOG.clear()
    asked: list[list[np.ndarray]] = []
    real = anytime._Store.replaying

    def spy(self, x):
        if self.requested == 0:
            asked.append([])
        asked[-1].append(np.array(x, dtype=float))
        return real(self, x)

    called = Logged(rosenbrock)
    try:
        anytime._Store.replaying = spy
        result = minimize(called, x0, 70 * dim, seed=3, config=config, bounds=BOX)
    finally:
        anytime._Store.replaying = real
    stage_seed = int(np.random.RandomState(3).randint(2**31 - 1))
    plans = [10 * dim, 20 * dim, 40 * dim]
    assert [e["plan"] for e in anytime.STAGE_LOG] == [*plans, 80 * dim]
    for plan, points in zip(plans, asked[:3], strict=True):
        alone = Logged(rosenbrock)
        minimize(
            alone,
            x0,
            plan,
            rng=np.random.RandomState(stage_seed),
            config=presets.get("c46"),
            bounds=BOX,
        )
        assert digest(points) == digest(alone.points)
    assert result.nfev == len(called.points) == 70 * dim
    assert sum(e["calls"] for e in anytime.STAGE_LOG) == 70 * dim
    assert result.fun == min(rosenbrock(p) for p in called.points)
    for entry in anytime.STAGE_LOG[1:3]:
        assert entry["calls"] == entry["requested"] - entry["replayed"]


def test_the_budget_caps_the_calls_and_never_plans():
    """Two caps give the same calls up to the smaller one."""
    config = presets.with_changes("c46", anytime="replay")
    short, long = Logged(rosenbrock), Logged(rosenbrock)
    minimize(short, np.full(3, 2.0), 137, seed=8, config=config, bounds=BOX)
    minimize(long, np.full(3, 2.0), 400, seed=8, config=config, bounds=BOX)
    assert len(short.points) == 137
    assert digest(short.points) == digest(long.points[:137])


def test_a_stop_from_outside_ends_the_run():
    class StopError(Exception):
        pass

    count = [0]

    def fun(x):
        count[0] += 1
        if count[0] > 75:
            raise StopError
        return rosenbrock(x)

    config = presets.with_changes("c46", anytime="replay")
    with pytest.raises(StopError):
        minimize(fun, np.full(5, 3.0), 10_000, seed=1, config=config, bounds=BOX)


def test_a_given_design_replaces_the_uniform_draws():
    from amm_optimizer import es_phase
    from amm_optimizer.bounds import Box
    from amm_optimizer.objective import CountedObjective

    box = Box.make(BOX, 3)
    design = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [-1.0, 2.0, 0.0]])
    seen = Logged(rosenbrock)
    es_phase.run_es_phase(
        CountedObjective(seen, 40),
        np.random.RandomState(0),
        np.full(3, 4.0),
        box,
        40,
        3,
        design=design,
    )
    assert np.array_equal(np.array(seen.points[1:4]), design)


def test_earlier_samples_seed_the_sampled_restarts():
    """The first descent starts from the best given sample that is far enough from
    the incumbent, before any new sample is drawn."""
    from amm_optimizer import qn_phase
    from amm_optimizer.bounds import Box
    from amm_optimizer.objective import CountedObjective

    box = Box.make(BOX, 2)
    pts = np.array([[4.0, 4.0], [1.0, 1.0], [3.9, 4.0]])  # [1, 1] is the minimum
    vals = np.array([rosenbrock(p) for p in pts])
    seen = Logged(rosenbrock)
    qn_phase.run_sampled_restarts(
        CountedObjective(seen, 200),
        np.random.RandomState(1),
        pts[2].copy(),
        float(vals[2]),
        box,
        60,
        samples=(pts, vals),
    )
    plain = Logged(rosenbrock)
    qn_phase.run_sampled_restarts(
        CountedObjective(plain, 200),
        np.random.RandomState(1),
        pts[2].copy(),
        float(vals[2]),
        box,
        60,
    )
    first = 10 * 2  # SAMPLE_PER_DIM x dim fresh samples come first in both
    assert digest(seen.points[:first]) == digest(plain.points[:first])
    assert digest(seen.points) != digest(plain.points)
    assert min(rosenbrock(p) for p in seen.points) <= min(
        rosenbrock(p) for p in plain.points
    )


# research-only options (deprecated in 0.6.0)
@pytest.mark.filterwarnings("ignore:Variant:FutureWarning")
def test_the_handover_gets_what_is_left_and_the_archive():
    """Stages up to the switch are replayed; then one run gets the rest of the
    budget, its repeats answered from the replay part's memory."""
    anytime.STAGE_LOG.clear()
    config = presets.with_changes("c46", anytime="replay", anytime_switch=200)
    called = Logged(rosenbrock)
    result = minimize(called, np.full(5, 3.0), 1500, seed=6, config=config, bounds=BOX)
    log = anytime.STAGE_LOG
    assert [e.get("plan") for e in log[:-1]] == [50, 100, 200]
    handover = log[-1]
    assert handover["handover"] == 1500 - handover["before"]
    assert handover["remembered"] > 0  # at least the start point is not called again
    # a remembered point costs the hand-over's plan but no call, so the run may end
    # that many calls short of the cap (the harness restarts on the rest)
    assert sum(e["calls"] for e in log) == result.nfev == len(called.points)
    assert 1500 - handover["remembered"] <= result.nfev <= 1500
    assert len({p.tobytes() for p in called.points[: handover["before"]]}) <= 200 * 2


def test_a_run_within_the_switch_never_hands_over():
    anytime.STAGE_LOG.clear()
    config = presets.with_changes("c46", anytime="replay")
    minimize(rosenbrock, np.full(4, 1.5), 3000, seed=1, config=config, bounds=BOX)
    assert all("handover" not in e for e in anytime.STAGE_LOG)


# research-only options (deprecated in 0.6.0)
@pytest.mark.filterwarnings("ignore:Variant:FutureWarning")
def test_the_handover_maps_its_archive_onto_a_normalised_box():
    """``box_normalize`` runs on [-5, 5]^D; the archive is mapped there too."""
    seen = Logged(lambda x: float(np.sum((np.asarray(x) - 0.3) ** 2)))
    config = presets.with_changes("c46", anytime="replay", anytime_switch=60)
    result = minimize(seen, np.full(3, 0.9), 600, seed=2, config=config, bounds=(0, 1))
    pts = np.array(seen.points)
    assert 550 <= len(pts) <= 600 and pts.min() >= 0.0 and pts.max() <= 1.0
    assert result.fun < 1e-6


def test_the_stage_log_holds_the_stages_of_the_last_run_only():
    """Review B4: the log was process-global and never cleared."""
    config = presets.with_changes("c46", anytime="replay")

    def stages(seed):
        minimize(rosenbrock, np.full(2, 3.0), 200, seed=seed, config=config, bounds=BOX)
        return [dict(entry) for entry in anytime.LAST_STAGES]

    alone = stages(2)
    for seed in range(3):
        last = stages(seed)
    assert anytime.STAGE_LOG is anytime.LAST_STAGES
    assert last == alone
    assert [entry["stage"] for entry in last] == list(range(len(last)))
