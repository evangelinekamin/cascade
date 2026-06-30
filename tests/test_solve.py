"""Tests for the run_solve assembly (the runnable verified worker)."""

from unittest.mock import MagicMock

import cascade.swarm.solve as solve_mod
from cascade.swarm.solve import (
    DEFAULT_TEST_CMD,
    SolveResult,
    _run_tests_in,
    _test_command,
    run_solve,
)


def _fake_app(test_cmd=None):
    app = MagicMock()
    app.providers = {"openai": MagicMock()}
    app.config.get_default_provider.return_value = "openai"
    app.config.data = (
        {"workflows": {"verify": {"test": test_cmd}}} if test_cmd else {}
    )
    return app


def test_test_command_prefers_config():
    assert _test_command(_fake_app("ruff check && pytest")) == "ruff check && pytest"


def test_test_command_falls_back_to_default():
    assert _test_command(_fake_app()) == DEFAULT_TEST_CMD


def test_run_tests_in_reports_pass_and_fail(tmp_path):
    _out, rc = _run_tests_in("true", str(tmp_path), 10)
    assert rc == 0
    _out, rc = _run_tests_in("false", str(tmp_path), 10)
    assert rc != 0


def test_run_solve_missing_provider_returns_error():
    app = _fake_app()
    app.providers = {}
    result = run_solve(app, "do x", provider_name="ghost")
    assert result.passed is False
    assert "not available" in result.error


def test_run_solve_wires_worker_and_passes(monkeypatch):
    app = _fake_app("pytest")

    fake_prepared = MagicMock()
    fake_prepared.path = "/tmp/wt-solve"
    fake_manager = MagicMock()
    fake_manager.prepare.return_value = fake_prepared
    snap = MagicMock()
    snap.diff_stat = "1 file changed"
    snap.diff_excerpt = "+ added line"
    snap.changed_files = ("foo.py",)
    fake_manager.capture_snapshot.return_value = snap

    monkeypatch.setattr(solve_mod, "WorktreeManager", lambda *a, **k: fake_manager)
    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", lambda *a, **k: "edited foo.py")
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda cmd, cwd, timeout: ("ok", 0))

    stages = []
    result = run_solve(app, "add foo", on_progress=lambda s, d: stages.append(s))

    assert isinstance(result, SolveResult)
    assert result.passed is True
    assert result.iterations == 1
    assert result.provider == "openai"
    assert result.worktree_path == "/tmp/wt-solve"
    assert result.diff_stat == "1 file changed"
    assert result.changed_files == ("foo.py",)
    # progress was reported through the lifecycle
    assert "workspace" in stages
    assert "verifying" in stages
    assert "verified" in stages


def test_run_solve_retries_until_tests_pass(monkeypatch):
    app = _fake_app("pytest")
    fake_prepared = MagicMock()
    fake_prepared.path = "/tmp/wt-solve"
    fake_manager = MagicMock()
    fake_manager.prepare.return_value = fake_prepared
    fake_manager.capture_snapshot.return_value = MagicMock(
        diff_stat="", diff_excerpt="", changed_files=()
    )
    monkeypatch.setattr(solve_mod, "WorktreeManager", lambda *a, **k: fake_manager)
    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", lambda *a, **k: "edited")

    results = iter([("FAILED", 1), ("ok", 0)])
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda cmd, cwd, timeout: next(results))

    result = run_solve(app, "fix it", max_iterations=3)

    assert result.passed is True
    assert result.iterations == 2
