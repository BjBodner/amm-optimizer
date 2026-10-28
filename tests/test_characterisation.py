"""The shipped configurations and the recorded arms reproduce their bbob runs exactly.

Refactors of ``atm`` must keep these snapshots bit for bit (``docs/DEVIATIONS.md``);
a deliberate behaviour change re-records them with
``tools/record_characterisation.py`` in its own commit. See ``characterisation_cases``
for what is hashed and for the tiers (``tools/ci.py --tier``):

* default (``-m "not slow and not merge and not research"``): the whole grid for
  ``PINNED`` (the default configuration and the current composite);
* merge (``-m "not slow"``): also the whole grid for ``RECENT``, the ``SEPARATING``
  cases for every named config, and the arm-snapshot completeness check;
* full (``-m slow`` included): every named config on the grid, every recorded arm.

The arm tests are ``research``: they need ``benchmarks/arms``,
``benchmarks/quick_scan.py`` and ``tests/characterisation_arms.json``, and skip
without them.

Each test item checks one config (or arm) on all of its cases and lists the cases
that differ. The fixtures record only the configs of the selected items, so ``-k``
narrows the work too.

The runs happen in a fresh process (the recorder with ``--emit``), with the BLAS
libraries single-threaded from the start, as the benchmark harness runs them: in
this process other tests may already have loaded SciPy's OpenBLAS with several
threads, and its reductions then round SLSQP's steps differently.
"""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("cocoex")

# bit-exact snapshots of one environment (tests/conftest.py skips them elsewhere)
pytestmark = pytest.mark.snapshot

import characterisation_cases as cc  # noqa: E402

NAMED = json.loads(cc.NAMED_SNAPSHOTS.read_text(encoding="utf-8"))
ARMS = (
    json.loads(cc.ARM_SNAPSHOTS.read_text(encoding="utf-8"))
    if cc.ARM_SNAPSHOTS.exists()
    else {"_meta": {}, "configs": {}}
)
RECORDER = cc.ROOT / "tools" / "record_characterisation.py"
JOBS = os.environ.get("ATM_CHARACTERISATION_JOBS", "2")
ARM_KEYS = sorted(set(ARMS["configs"]) - set(cc.REMOVED_BY_MERGE))
research = [
    pytest.mark.research,
    pytest.mark.skipif(
        cc.RESEARCH_DATA_MISSING is not None,
        reason=f"research data not in this checkout: {cc.RESEARCH_DATA_MISSING}",
    ),
]


def _research(test):
    for mark in research:
        test = mark(test)
    return test


def _same_environment(meta):
    here = cc.environment()
    keys = ("numpy", "scipy", "system")
    if any(meta.get(k) != here[k] for k in keys):
        pytest.skip(f"recorded on {[meta.get(k) for k in keys]}, running on {here}")


def _record(tmp_path_factory, *args):
    """Run the recorder in a fresh process; returns its snapshots by set name."""
    out = tmp_path_factory.mktemp("characterisation") / "run.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(cc.ROOT / "src"), *filter(None, [env.get("PYTHONPATH")])]
    )
    subprocess.run(
        [sys.executable, str(RECORDER), *args, "--jobs", JOBS, "--emit", str(out)],
        env=env,
        check=True,
        capture_output=True,
    )
    return {
        which: json.loads(path.read_text(encoding="utf-8"))
        for which in ("named", "arms")
        if (path := out.with_name(f"run_{which}.json")).exists()
    }


def _selected(request, test_name, param, default):
    """The ``param`` values of the selected ``test_name`` items (``default`` if none),
    so that a fixture records only what this run checks."""
    values = {
        item.callspec.params[param]
        for item in request.session.items
        if getattr(item, "originalname", None) == test_name
        and hasattr(item, "callspec")
    }
    return sorted(values) or list(default)


def _differing(runs, stored, key):
    got = runs["configs"][key]["cases"]
    want = stored["configs"][key]["cases"]
    return [case for case in got if got[case] != want.get(case)]


@pytest.fixture(scope="module")
def pinned_runs(request, tmp_path_factory):
    _same_environment(NAMED["_meta"])
    names = _selected(request, "test_pinned_grid", "name", cc.PINNED)
    return _record(tmp_path_factory, "--set", "named", "--only", ",".join(names))[
        "named"
    ]


@pytest.fixture(scope="module")
def recent_runs(request, tmp_path_factory):
    _same_environment(NAMED["_meta"])
    names = _selected(request, "test_recent_grid", "name", RECENT_ONLY)
    return _record(tmp_path_factory, "--set", "named", "--only", ",".join(names))[
        "named"
    ]


@pytest.fixture(scope="module")
def separating_runs(request, tmp_path_factory):
    _same_environment(NAMED["_meta"])
    names = _selected(request, "test_separating_set", "name", cc.NAMED)
    args = ("--set", "named", "--cases", "sep", "--only", ",".join(names))
    return _record(tmp_path_factory, *args)["named"]


@pytest.fixture(scope="module")
def grid_runs(request, tmp_path_factory):
    _same_environment(NAMED["_meta"])
    names = _selected(request, "test_named_grid", "name", cc.NAMED)
    return _record(tmp_path_factory, "--set", "named", "--only", ",".join(names))[
        "named"
    ]


@pytest.fixture(scope="module")
def arm_runs(request, tmp_path_factory):
    _same_environment(ARMS["_meta"])
    keys = _selected(request, "test_recorded_arm", "key", ARM_KEYS)
    return _record(tmp_path_factory, "--set", "arms", "--only", ",".join(keys))["arms"]


def test_the_snapshots_cover_every_named_config_and_case():
    stored = NAMED["configs"]
    assert set(stored) == set(cc.NAMED)
    for name, text in cc.NAMED.items():
        assert stored[name]["config"] == text
        assert set(stored[name]["cases"]) == {c.name for c in cc.GRID}, name


@pytest.mark.merge
@_research
def test_every_recorded_arm_has_a_snapshot():
    stored = ARMS["configs"]
    current = set(cc.arm_configs())
    assert not current & set(cc.REMOVED_BY_MERGE)
    missing = current - set(stored)
    assert not missing, (
        f"{len(missing)} arm configs lack a snapshot: run "
        "`python tools/record_characterisation.py --set arms --missing`"
    )
    assert set(stored) == current | set(cc.REMOVED_BY_MERGE)
    assert all(
        set(e["cases"]) == {c.name for c in cc.ARM_PROBES} for e in stored.values()
    )


def test_every_named_config_changes_some_case_of_its_base():
    """Guards the grid itself: a route that changes no case is not pinned by it."""
    cases = {name: NAMED["configs"][name]["cases"] for name in cc.NAMED}
    for name in cc.NAMED:
        if name == "c23" or name in cc.SAME_AS_BASE_ON_THE_GRID:
            continue
        base = cc.BASES.get(name, "c26")
        changed = [c for c in cases[name] if cases[name][c] != cases[base][c]]
        assert changed, f"{name} equals {base} on every case of the grid"


def test_the_separating_set_tells_every_named_config_from_its_base():
    """The merge tier checks every named config on ``SEPARATING`` only; on those
    cases each must differ from its base, or a regression could pass unseen."""
    cases = {name: NAMED["configs"][name]["cases"] for name in cc.NAMED}
    for name in cc.NAMED:
        if name == "c23" or name in cc.SAME_AS_BASE_ON_THE_GRID:
            continue
        base = cc.BASES.get(name, "c26")
        changed = [
            c.name for c in cc.SEPARATING if cases[name][c.name] != cases[base][c.name]
        ]
        assert changed, f"{name} equals {base} on every separating case"


def test_the_in_process_runner_matches_the_recorder(pinned_runs):
    """``characterisation_cases.run`` here and in the recorder agree on a case
    without SLSQP (so without the BLAS-threading caveat)."""
    case = cc.FAST[0]  # f1 at D 2, 100 x D: the prefix's Nelder-Mead and the (1+1)
    want = pinned_runs["configs"]["default"]["cases"][case.name]
    assert cc.run(cc.named_config("default"), case) == want


@pytest.mark.parametrize("name", cc.PINNED)
def test_pinned_grid(pinned_runs, name):
    differing = _differing(pinned_runs, NAMED, name)
    assert not differing, f"{name}: {len(differing)} cases differ: {differing[:12]}"


RECENT_ONLY = [name for name in cc.RECENT if name not in cc.PINNED]


@pytest.mark.merge
@pytest.mark.parametrize("name", RECENT_ONLY)
def test_recent_grid(recent_runs, name):
    differing = _differing(recent_runs, NAMED, name)
    assert not differing, f"{name}: {len(differing)} cases differ: {differing[:12]}"


@pytest.mark.merge
@pytest.mark.parametrize("name", list(cc.NAMED))
def test_separating_set(separating_runs, name):
    differing = _differing(separating_runs, NAMED, name)
    assert not differing, f"{name}: {len(differing)} cases differ: {differing}"


@pytest.mark.slow
@pytest.mark.parametrize("name", list(cc.NAMED))
def test_named_grid(grid_runs, name):
    differing = _differing(grid_runs, NAMED, name)
    assert not differing, f"{name}: {len(differing)} cases differ: {differing[:12]}"


@pytest.mark.slow
@_research
@pytest.mark.parametrize("key", ARM_KEYS)
def test_recorded_arm(arm_runs, key):
    differing = _differing(arm_runs, ARMS, key)
    assert not differing, f"arm {key}: probes differ: {differing}"
