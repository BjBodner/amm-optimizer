"""The local CI runner (tools/ci.py): its steps and its pass/fail behaviour."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_ci():
    spec = importlib.util.spec_from_file_location("ci_script", ROOT / "tools" / "ci.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


ci = load_ci()


def names(full):
    return [step.name for step in ci.steps(full)]


def test_default_tier_is_lint_format_types_and_quick_tests():
    steps = ci.steps("default")
    assert len(steps) == 4
    assert steps[0].command[1:4] == ["-m", "ruff", "check"]
    assert "--check" in steps[1].command
    assert steps[2].command[-1] == "mypy"
    assert steps[3].command[-2:] == ["-m", "not slow and not merge and not research"]
    assert ci.steps(False) == steps  # the old fast flag


def test_research_tests_run_in_the_merge_and_full_tiers():
    assert "research" not in " ".join(ci.steps("merge")[-1].command)
    assert "research" not in " ".join(ci.steps("full")[-1].command)


def test_merge_tier_deselects_only_the_slow_sweep():
    assert ci.steps("merge")[-1].command[-2:] == ["-m", "not slow"]


def test_an_unknown_tier_is_refused():
    with pytest.raises(ValueError):
        ci.steps("nightly")


def test_full_gate_runs_all_tests_with_the_coverage_gate():
    last = ci.steps(True)[-1].command
    assert "--cov-fail-under=100" in last and "--cov-branch" in last
    assert "not slow" not in last


def test_every_step_uses_the_interpreter_that_runs_it():
    assert all(step.command[0] == sys.executable for step in ci.steps(True))


def fake(code):
    return ci.Step(f"exit {code}", [sys.executable, "-c", f"raise SystemExit({code})"])


def test_run_returns_zero_only_when_everything_passes(capsys):
    assert ci.run([fake(0), fake(0)], keep_going=False) == 0
    assert "ALL CHECKS PASSED" in capsys.readouterr().out


def test_run_stops_at_the_first_failure(capsys):
    assert ci.run([fake(0), fake(3), fake(0)], keep_going=False) == 1
    out = capsys.readouterr().out
    assert "not run after the first failure: 1 step" in out
    assert "CHECKS FAILED" in out


def test_keep_going_runs_every_step_but_still_fails(capsys):
    assert ci.run([fake(1), fake(0)], keep_going=True) == 1
    out = capsys.readouterr().out
    assert out.count("PASS") >= 1 and out.count("FAIL") >= 1
    assert "not run after" not in out


def test_list_prints_the_steps_without_running_them():
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "ci.py"), "--list"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert done.returncode == 0
    assert "lint (ruff check)" in done.stdout and "tests, default tier" in done.stdout
    assert "summary" not in done.stdout  # nothing was executed


def test_the_pre_push_hook_runs_the_runner_and_is_not_enabled_by_default():
    hook = (ROOT / ".githooks" / "pre-push").read_text(encoding="utf-8")
    assert "tools/ci.py" in hook
    config = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert ".githooks" not in config.stdout  # opt-in only


def test_every_step_tests_this_checkouts_src_first():
    first = ci.step_env()["PYTHONPATH"].split(os.pathsep)[0]
    assert Path(first) == ci.ROOT / "src"


def test_a_bare_pytest_tests_this_checkouts_atm_too():
    # tests/conftest.py: in a worktree the venv's editable install would give
    # the main checkout's package, in this process and in the subprocesses tests start
    import amm_optimizer as atm

    src = (ROOT / "src").resolve()
    assert Path(atm.__file__).resolve().is_relative_to(src)
    done = subprocess.run(
        [sys.executable, "-c", "import amm_optimizer; print(amm_optimizer.__file__)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(done.stdout.strip()).resolve().is_relative_to(src)
