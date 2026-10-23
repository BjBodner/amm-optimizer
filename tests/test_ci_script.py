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


def test_research_tier_is_lint_format_types_research_check_and_quick_tests():
    steps = ci.steps("research")
    assert len(steps) == 5
    assert steps[0].command[1:4] == ["-m", "ruff", "check"]
    assert "--check" in steps[1].command
    assert steps[2].command[-1] == "mypy"
    assert steps[3].command[1:] == ["tools/ci.py", "--research-check"]
    assert steps[4].command[-2:] == ["-m", "not slow and not merge"]
    assert ci.steps(False) == steps  # the old fast flag


def test_every_tier_runs_the_research_check():
    for tier in ci.TIERS:
        assert any("--research-check" in s.command for s in ci.steps(tier))


def test_the_audit_scripts_are_all_reachable_today():
    assert ci.unreachable_audit_scripts() == []


def write(path, text=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_an_orphan_audit_script_and_an_unlisted_archived_one_are_reported(tmp_path):
    audit = tmp_path / "research" / "audit"
    write(audit / "entry.py", "import helper\n")
    write(audit / "helper.py", "def f():\n    import deep\n")
    write(audit / "deep.py")
    write(audit / "used_by_parked.py")
    write(audit / "old.py")
    write(audit / "orphan.py")
    write(tmp_path / "research" / "parked" / "study.py", "import used_by_parked\n")
    write(
        audit / "archive" / "README.md",
        "## Scripts\n\n| Script | What |\n|---|---|\n"
        "| `old.py` | x |\n| `gone.py` | y |\n"
        "\n## Code removed\n\n| `orphan.py` | not a script row |\n",
    )
    write(audit / "archive" / "gone.py")
    write(audit / "archive" / "unlisted.py")
    problems = ci.unreachable_audit_scripts(audit, ("entry", "missing"), ("parked",))
    assert len(problems) == 3
    assert "entry point missing.py does not exist" in problems[0]
    assert problems[1].startswith("audit/orphan.py: not an entry point")
    assert problems[2].startswith("audit/archive/unlisted.py: no line")


def test_a_worktree_admin_folder_without_gitdir_is_reported(tmp_path):
    write(tmp_path / "worktrees" / "ok" / "gitdir", "C:/work/ok/.git\n")
    (tmp_path / "worktrees" / "exp").mkdir()
    problems = ci.stale_worktree_admin_dirs(tmp_path)
    assert len(problems) == 1 and "exp has no gitdir file" in problems[0]
    assert ci.stale_worktree_admin_dirs(tmp_path / "nowhere") == []
    assert ci.stale_worktree_admin_dirs(None) == []


def test_the_research_check_runs_on_its_own():
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "ci.py"), "--research-check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stdout
    assert "every audit script is reachable" in done.stdout


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
    assert "lint (ruff check)" in done.stdout and "tests, research tier" in done.stdout
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
