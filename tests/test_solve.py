"""Tests for the run_solve assembly (the runnable verified worker)."""

from types import SimpleNamespace
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


def test_is_infra_failure_flags_commands_that_did_not_run():
    # command missing, pytest missing, or nothing collected -- none of which an
    # agent can fix by editing code.
    assert solve_mod._is_infra_failure("/bin/sh: 1: python: not found", 127) is True
    assert solve_mod._is_infra_failure("/usr/bin/python: No module named pytest", 1) is True
    assert solve_mod._is_infra_failure("no tests ran in 0.01s", 5) is True


def test_is_infra_failure_false_for_genuine_test_failures():
    assert solve_mod._is_infra_failure("1 failed, 3 passed in 0.2s", 1) is False
    assert solve_mod._is_infra_failure("2 passed in 0.1s", 0) is False


def test_run_solve_aborts_immediately_on_a_broken_gate(monkeypatch):
    # A verify command that cannot execute must abort at once -- no agent
    # iteration, no escalation -- with a clear "did not run" error.
    app = _fake_app("python -m pytest -x -q")
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt-broken")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=()
    )
    monkeypatch.setattr(solve_mod, "WorktreeManager", lambda *a, **k: fm)

    agent_calls: list[int] = []
    monkeypatch.setattr(
        solve_mod,
        "run_agent_in_worktree",
        lambda *a, **k: agent_calls.append(1) or "edited",
    )
    monkeypatch.setattr(
        solve_mod,
        "_run_tests_in",
        lambda cmd, cwd, timeout: ("/bin/sh: python: command not found", 127),
    )

    result = run_solve(app, "fix the validator")

    assert result.passed is False
    assert result.iterations == 0
    assert agent_calls == []  # no agent iteration wasted on a broken gate
    assert "did not run" in result.error.lower()


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
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)

    results = iter([("FAILED", 1), ("ok", 0)])
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda cmd, cwd, timeout: next(results))

    result = run_solve(app, "fix it", max_iterations=3)

    assert result.passed is True
    assert result.iterations == 2


def _tiered_app(bulk="bulk-x", frontier="frontier-x"):
    app = MagicMock()
    prov = MagicMock()
    prov.config = SimpleNamespace(model=frontier)
    app.providers = {"openai": prov}
    app.config.get_default_provider.return_value = "openai"
    app.config.get_model_for = MagicMock(
        side_effect=lambda name, mode_name=None, fast=False: bulk if fast else frontier
    )
    app.config.data = {"workflows": {"verify": {"test": "pytest"}}}
    return app, prov


def _patch_solve(monkeypatch, observed, test_results):
    def fake_agent(provider, prompt, path, system=None):
        observed.append(provider.config.model)
        return "edited"

    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=()
    )
    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", fake_agent)
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: next(test_results))
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)
    monkeypatch.setattr(solve_mod, "WorktreeManager", lambda *a, **k: fm)


def test_escalates_to_frontier_after_first_failure(monkeypatch):
    app, prov = _tiered_app()
    observed: list[str] = []
    _patch_solve(monkeypatch, observed, iter([("fail", 1), ("ok", 0)]))

    result = run_solve(app, "x", escalate=True, escalate_after=1, max_iterations=3)

    # iteration 1 ran the bulk model; iteration 2 escalated to the frontier model
    assert observed == ["bulk-x", "frontier-x"]
    assert result.models_used == ("bulk-x", "frontier-x")
    # the provider's model is restored to its original value afterward
    assert prov.config.model == "frontier-x"


def test_run_solve_accumulates_and_reports_token_usage(monkeypatch):
    app = MagicMock()
    prov = MagicMock()
    prov.config = SimpleNamespace(model="frontier")
    prov.last_usage = (100, 40)  # each agent iteration reports this usage
    app.providers = {"openai": prov}
    app.config.get_default_provider.return_value = "openai"
    app.config.get_model_for = MagicMock(
        side_effect=lambda name, mode_name=None, fast=False: "bulk" if fast else "frontier"
    )
    app.config.data = {"workflows": {"verify": {"test": "pytest"}}}

    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=()
    )
    monkeypatch.setattr(solve_mod, "WorktreeManager", lambda *a, **k: fm)
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)
    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", lambda *a, **k: "edited")
    results = iter([("FAIL", 1), ("ok", 0)])  # fail then pass -> 2 agent iterations
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: next(results))

    reported: list[tuple[int, int]] = []
    result = run_solve(app, "x", on_tokens=lambda i, o: reported.append((i, o)))

    assert result.input_tokens == 200  # 2 iterations x 100
    assert result.output_tokens == 80  # 2 iterations x 40
    assert reported == [(100, 40), (100, 40)]  # reported live, once per iteration


def test_no_escalation_uses_frontier_throughout(monkeypatch):
    app, prov = _tiered_app()
    observed: list[str] = []
    _patch_solve(monkeypatch, observed, iter([("fail", 1), ("ok", 0)]))

    result = run_solve(app, "x", escalate=False, max_iterations=3)

    assert observed == ["frontier-x", "frontier-x"]
    assert set(result.models_used) == {"frontier-x"}
