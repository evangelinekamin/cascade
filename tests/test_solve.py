"""Tests for the run_solve assembly (the runnable verified worker)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cascade.swarm.solve as solve_mod
from cascade.swarm.solve import (
    DEFAULT_TEST_CMD,
    SolveResult,
    _run_tests_in,
    _test_command,
    run_solve,
    run_verified_task,
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


def test_test_command_prefers_project_local_cascade_yml(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".cascade.yml").write_text(
        "workflows:\n  verify:\n    test: uv run pytest tests/ -q\n"
    )
    monkeypatch.chdir(tmp_path)
    # the global config says one thing; the project-local file must win
    assert _test_command(_fake_app("python -m pytest -x -q")) == "uv run pytest tests/ -q"


def test_test_command_supports_toplevel_verify_shape(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".cascade.yml").write_text("verify:\n  test: uv run pytest -q\n")
    monkeypatch.chdir(tmp_path)
    assert _test_command(_fake_app("global")) == "uv run pytest -q"


def test_project_config_found_from_subdirectory(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".cascade.yml").write_text("verify:\n  test: pytest-from-root\n")
    sub = tmp_path / "pkg" / "mod"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    assert _test_command(_fake_app("global")) == "pytest-from-root"


def test_test_command_falls_back_to_global_without_project_file(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)  # no .cascade.yml present
    assert _test_command(_fake_app("ruff && pytest")) == "ruff && pytest"


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
    def fake_agent(provider, prompt, path, system=None, max_rounds=None):
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


# --- Fix 1: the agentic build loop must raise the tool-call budget --------------


def _capture_max_rounds(monkeypatch):
    """Patch the solve internals and return a list capturing forwarded max_rounds."""
    captured: list[int] = []

    def fake_agent(provider, prompt, path, system=None, max_rounds=None):
        captured.append(max_rounds)
        return "edited"

    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", fake_agent)
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: ("ok", 0))
    return captured


def _prov():
    prov = MagicMock()
    prov.config = SimpleNamespace(model="m")
    return prov


def test_run_verified_task_defaults_max_rounds_to_15(monkeypatch):
    captured = _capture_max_rounds(monkeypatch)
    run_verified_task(
        _prov(), "/tmp/wt", "task", "pytest", bulk_model="b", frontier_model="f"
    )
    assert captured == [15]


def test_run_verified_task_threads_max_rounds_override(monkeypatch):
    captured = _capture_max_rounds(monkeypatch)
    run_verified_task(
        _prov(),
        "/tmp/wt",
        "task",
        "pytest",
        bulk_model="b",
        frontier_model="f",
        max_rounds=42,
    )
    assert captured == [42]


def test_run_solve_defaults_max_rounds_to_15(monkeypatch):
    app = _fake_app("pytest")
    captured = _capture_max_rounds(monkeypatch)
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=()
    )
    monkeypatch.setattr(solve_mod, "WorktreeManager", lambda *a, **k: fm)

    run_solve(app, "x")
    assert captured == [15]


def test_run_solve_threads_max_rounds_override(monkeypatch):
    app = _fake_app("pytest")
    captured = _capture_max_rounds(monkeypatch)
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=()
    )
    monkeypatch.setattr(solve_mod, "WorktreeManager", lambda *a, **k: fm)

    run_solve(app, "x", max_rounds=7)
    assert captured == [7]


# --- Fix 2: the scaffolded gating tests must be immutable to the worker ----------


def test_snapshot_test_files_detects_by_all_rules(tmp_path):
    # by basename
    (tmp_path / "test_foo.py").write_text("a")
    (tmp_path / "bar_test.py").write_text("b")
    (tmp_path / "conftest.py").write_text("c")
    # by directory segment (non-test basenames still count under tests/ or test/)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "helpers.py").write_text("d")
    (tmp_path / "pkg" / "test").mkdir(parents=True)
    (tmp_path / "pkg" / "test" / "data.txt").write_text("e")
    # non-test files that must be excluded
    (tmp_path / "feature.py").write_text("impl")
    (tmp_path / "README.md").write_text("doc")
    (tmp_path / "pkg" / "module.py").write_text("mod")

    snap = solve_mod._snapshot_test_files(str(tmp_path))

    got = {Path(p).relative_to(tmp_path).as_posix() for p in snap}
    assert got == {
        "test_foo.py",
        "bar_test.py",
        "conftest.py",
        "tests/helpers.py",
        "pkg/test/data.txt",
    }
    # content is captured verbatim
    assert snap[str(tmp_path / "test_foo.py")] == "a"


def test_restore_files_overwrites_only_mapped_paths(tmp_path):
    protected = tmp_path / "test_spec.py"
    protected.write_text("ORIGINAL")
    other = tmp_path / "impl.py"
    other.write_text("IMPL")

    snapshot = {str(protected): "ORIGINAL"}
    protected.write_text("TAMPERED")  # worker weakens the gate
    other.write_text("CHANGED")  # legitimate worker edit

    solve_mod._restore_files(snapshot)

    assert protected.read_text() == "ORIGINAL"  # restored to the contract
    assert other.read_text() == "CHANGED"  # not in mapping -> left alone


def test_verified_task_restores_tampered_tests_but_keeps_impl_and_new_files(
    tmp_path, monkeypatch
):
    # A scaffolded gating test exists before the worker runs.
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_feature.py").write_text("SPEC = 'original'\n")

    def fake_agent(provider, prompt, path, system=None, max_rounds=None):
        root = Path(path)
        # worker tampers with the gating test (reward hacking)...
        (root / "tests" / "test_feature.py").write_text("SPEC = 'HACKED'\n")
        # ...writes a legitimate implementation (a non-test file)...
        (root / "feature.py").write_text("def feature():\n    return 1\n")
        # ...and creates brand-new files, including a new test file.
        (root / "tests" / "test_extra.py").write_text("EXTRA = 1\n")
        (root / "notes.txt").write_text("scratch\n")
        return "edited"

    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", fake_agent)
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: ("ok", 0))

    result, _models = run_verified_task(
        _prov(), str(tmp_path), "add feature", "pytest", bulk_model="b", frontier_model="f"
    )

    # The gating test is restored to its scaffolded content...
    assert (tmp_path / "tests" / "test_feature.py").read_text() == "SPEC = 'original'\n"
    # ...while the worker's implementation and any new files are left intact.
    assert (tmp_path / "feature.py").read_text() == "def feature():\n    return 1\n"
    assert (tmp_path / "tests" / "test_extra.py").read_text() == "EXTRA = 1\n"
    assert (tmp_path / "notes.txt").read_text() == "scratch\n"
    assert result.passed is True


# --- Light iteration memory: the worker is told what it already changed ----------


def _init_repo_with_baseline(tmp_path):
    import subprocess

    def git(*args):
        subprocess.run(
            ["git", "-c", "user.name=T", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )

    git("init")
    git("commit", "--allow-empty", "-m", "baseline")


def test_worktree_change_summary_lists_new_and_modified_files(tmp_path):
    _init_repo_with_baseline(tmp_path)
    (tmp_path / "impl.py").write_text("def f():\n    return 1\n")  # brand-new file

    summary = solve_mod._worktree_change_summary(str(tmp_path))

    assert "impl.py" in summary  # new files show up (via intent-to-add)


def test_worktree_change_summary_is_capped(tmp_path):
    _init_repo_with_baseline(tmp_path)
    for i in range(20):
        (tmp_path / f"file_{i}.py").write_text("x = 1\n")

    summary = solve_mod._worktree_change_summary(str(tmp_path), cap=60)
    assert len(summary) <= 60


def test_worktree_change_summary_empty_outside_a_git_repo(tmp_path):
    assert solve_mod._worktree_change_summary(str(tmp_path)) == ""


def test_run_verified_task_feeds_prior_changes_into_the_retry(monkeypatch):
    prompts: list[str] = []

    def fake_agent(provider, prompt, path, system=None, max_rounds=None):
        prompts.append(prompt)
        return "edited"

    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", fake_agent)
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)
    monkeypatch.setattr(
        solve_mod, "_worktree_change_summary", lambda path, **k: "MEMO_DIFFSTAT impl.py | 3 +++"
    )
    results = iter([("FAILED test_x", 1), ("ok", 0)])
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: next(results))

    run_verified_task(
        _prov(), "/tmp/wt", "task", "pytest", bulk_model="b", frontier_model="f"
    )

    assert "MEMO_DIFFSTAT" not in prompts[0]  # first pass has no prior work
    assert "MEMO_DIFFSTAT" in prompts[1]  # retry builds on what already changed
