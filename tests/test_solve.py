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
    worker_guidance_for,
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
    app.config.get_bulk_model = MagicMock(side_effect=lambda name, mode_name=None: bulk)
    app.config.data = {"workflows": {"verify": {"test": "pytest"}}}
    return app, prov


def _patch_solve(monkeypatch, observed, test_results):
    def fake_agent(provider, prompt, path, system=None, max_rounds=None, on_tool_event=None):
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


def test_solve_bulk_tier_resolves_via_get_bulk_model(monkeypatch):
    """The bulk iteration comes from get_bulk_model, decoupled from fast_model."""
    app, prov = _tiered_app()
    app.config.get_bulk_model = MagicMock(
        side_effect=lambda name, mode_name=None: "dedicated-bulk"
    )
    observed: list[str] = []
    _patch_solve(monkeypatch, observed, iter([("fail", 1), ("ok", 0)]))

    run_solve(app, "x", escalate=True, escalate_after=1, max_iterations=3)

    assert observed[0] == "dedicated-bulk"


def test_run_agent_in_worktree_forwards_tool_events(tmp_path):
    """API-provider agents forward tool events so /solve can show live activity."""
    from contextlib import nullcontext
    from cascade.swarm.workspace import run_agent_in_worktree

    prov = MagicMock()
    prov._use_cli_proxy = False
    prov.working_directory = MagicMock(return_value=nullcontext())
    prov.ask_with_tools = MagicMock(return_value=("ok", []))

    def cb(event):
        return None

    result = run_agent_in_worktree(prov, "task", str(tmp_path), on_tool_event=cb)

    assert result == "ok"
    assert prov.ask_with_tools.call_args.kwargs["on_tool_event"] is cb


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
    app.config.get_bulk_model = MagicMock(side_effect=lambda name, mode_name=None: "bulk")
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

    def fake_agent(provider, prompt, path, system=None, max_rounds=None, on_tool_event=None):
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

    def fake_agent(provider, prompt, path, system=None, max_rounds=None, on_tool_event=None):
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

    result, _models, _providers = run_verified_task(
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

    def fake_agent(provider, prompt, path, system=None, max_rounds=None, on_tool_event=None):
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


# --- Fix 3: blast-radius guardrail (drop a careless model's needless edits) -------


def _git_repo(tmp_path, files: dict[str, str]) -> None:
    """Init a git repo at *tmp_path* and commit *files* (rel path -> content)."""
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
    for rel, content in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    git("add", "-A")
    git("commit", "-m", "baseline")


def _git_diff(tmp_path, ref: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "diff", ref], cwd=tmp_path, capture_output=True, text=True
    ).stdout


def _pytest_rc(tmp_path) -> int:
    import subprocess

    return subprocess.run(
        ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    ).returncode


# A shared helper file with f() at the top and enough padding that a rewrite of
# f() and an appended g() land in two separate diff hunks (one non-additive, one
# additive) -- the exact shape of the dogfooded regression.
_SHARED_BASELINE = (
    "def f():\n    return 1\n\n\n"
    "PAD_A = 1\nPAD_B = 2\nPAD_C = 3\nPAD_D = 4\nPAD_E = 5\n"
    "PAD_F = 6\nPAD_G = 7\nPAD_H = 8\nPAD_I = 9\nPAD_J = 10\n"
)
_SHARED_WORKER = (
    "def f():\n    return 999\n\n\n"  # needless rewrite of a pre-existing line
    "PAD_A = 1\nPAD_B = 2\nPAD_C = 3\nPAD_D = 4\nPAD_E = 5\n"
    "PAD_F = 6\nPAD_G = 7\nPAD_H = 8\nPAD_I = 9\nPAD_J = 10\n\n\n"
    "def g():\n    return 2\n"  # the feature: a purely additive new function
)


_ADDITIVE_DIFF = (
    "diff --git a/new.py b/new.py\n"
    "new file mode 100644\n"
    "index 0000000..1111111\n"
    "--- /dev/null\n"
    "+++ b/new.py\n"
    "@@ -0,0 +1,2 @@\n"
    "+def g():\n"
    "+    return 2\n"
)

_NONADDITIVE_DIFF = (
    "diff --git a/shared.py b/shared.py\n"
    "index a465610..0a510f3 100644\n"
    "--- a/shared.py\n"
    "+++ b/shared.py\n"
    "@@ -1,3 +1,3 @@\n"
    " def f():\n"
    "-    return 1\n"
    "+    return 999\n"
    " END\n"
)

# The real two-hunk diff git emits for _SHARED_BASELINE -> _SHARED_WORKER.
_MIXED_DIFF = (
    "diff --git a/shared.py b/shared.py\n"
    "index a465610..0a510f3 100644\n"
    "--- a/shared.py\n"
    "+++ b/shared.py\n"
    "@@ -1,5 +1,5 @@\n"
    " def f():\n"
    "-    return 1\n"
    "+    return 999\n"
    " \n"
    " \n"
    " PAD_A = 1\n"
    "@@ -12,3 +12,7 @@ PAD_G = 7\n"
    " PAD_H = 8\n"
    " PAD_I = 9\n"
    " PAD_J = 10\n"
    "+\n"
    "+\n"
    "+def g():\n"
    "+    return 2\n"
)


def test_hunk_is_additive_for_pure_additions():
    files = solve_mod._parse_file_diffs(_ADDITIVE_DIFF)
    assert len(files) == 1
    assert len(files[0].hunks) == 1
    assert files[0].hunks[0].is_additive is True


def test_hunk_is_nonadditive_when_a_line_is_removed_or_changed():
    files = solve_mod._parse_file_diffs(_NONADDITIVE_DIFF)
    assert files[0].hunks[0].is_additive is False


def test_parse_classifies_each_hunk_of_a_mixed_file():
    files = solve_mod._parse_file_diffs(_MIXED_DIFF)
    assert len(files) == 1
    hunks = files[0].hunks
    assert len(hunks) == 2
    assert hunks[0].is_additive is False  # the f() rewrite
    assert hunks[1].is_additive is True  # the appended g()


def test_hunk_body_respects_declared_counts_when_content_looks_like_a_marker():
    # A content line that itself begins with "@@" / "diff --git" must not split
    # the hunk -- the declared line counts keep the parse honest.
    tricky = (
        "diff --git a/doc.py b/doc.py\n"
        "--- a/doc.py\n"
        "+++ b/doc.py\n"
        "@@ -1,2 +1,2 @@\n"
        " keep\n"
        "-old = '@@ not a header'\n"
        "+new = 'diff --git also not a header'\n"
    )
    files = solve_mod._parse_file_diffs(tricky)
    assert len(files) == 1
    assert len(files[0].hunks) == 1  # not fooled into starting a second hunk
    assert files[0].hunks[0].is_additive is False


def test_nonadditive_patch_keeps_only_the_changed_hunk():
    files = solve_mod._parse_file_diffs(_MIXED_DIFF)
    patch, affected = solve_mod._nonadditive_patch(files)
    assert affected == ("shared.py",)
    assert "-    return 1" in patch and "+    return 999" in patch  # the change hunk
    assert "def g()" not in patch  # the additive hunk is excluded


def test_nonadditive_patch_is_empty_for_a_pure_additive_diff():
    files = solve_mod._parse_file_diffs(_ADDITIVE_DIFF)
    patch, affected = solve_mod._nonadditive_patch(files)
    assert patch == ""
    assert affected == ()


def test_nonadditive_patch_skips_whole_file_deletions():
    deletion = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "index a465610..0000000\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-def f():\n"
        "-    return 1\n"
    )
    patch, affected = solve_mod._nonadditive_patch(solve_mod._parse_file_diffs(deletion))
    assert patch == ""  # no current file to restore into -> left for the loop
    assert affected == ()


def test_revert_nonadditive_hunks_keeps_addition_and_restores_line(tmp_path):
    # The revert mechanism in isolation: a single file with both an added function
    # and a changed pre-existing line -- reverting must leave the addition and
    # restore the original line.
    import subprocess

    _git_repo(tmp_path, {"shared.py": _SHARED_BASELINE})
    baseline = solve_mod._establish_guardrail_baseline(str(tmp_path))
    assert baseline
    (tmp_path / "shared.py").write_text(_SHARED_WORKER)

    patch, affected = solve_mod._nonadditive_patch(
        solve_mod._parse_file_diffs(_git_diff(tmp_path, baseline))
    )
    assert affected == ("shared.py",)
    rc = subprocess.run(
        ["git", "apply", "--reverse", "--whitespace=nowarn", "-"],
        cwd=tmp_path,
        input=patch,
        capture_output=True,
        text=True,
    ).returncode
    assert rc == 0

    result = (tmp_path / "shared.py").read_text()
    assert "return 1" in result  # pre-existing line restored
    assert "return 999" not in result  # needless change dropped
    assert "def g():" in result  # the addition is kept


def test_establish_baseline_returns_none_outside_a_git_repo(tmp_path):
    assert solve_mod._establish_guardrail_baseline(str(tmp_path)) is None


def test_minimize_returns_false_outside_a_git_repo(tmp_path):
    # git diff fails -> graceful fallback, no crash.
    assert solve_mod._minimize_blast_radius(str(tmp_path), "HEAD", "true", 10) is False


def test_minimize_is_a_noop_when_the_worker_only_added(tmp_path):
    _git_repo(tmp_path, {"shared.py": "def f():\n    return 1\n"})
    baseline = solve_mod._establish_guardrail_baseline(str(tmp_path))
    # a purely additive change to a tracked file (appended function)
    (tmp_path / "shared.py").write_text(
        "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
    )
    # even against a failing command, there is nothing non-additive to revert
    assert solve_mod._minimize_blast_radius(str(tmp_path), baseline, "false", 5) is False
    assert "def g():" in (tmp_path / "shared.py").read_text()  # append left intact


def test_guardrail_greens_the_suite_by_dropping_a_needless_modification(
    tmp_path, monkeypatch
):
    # The money test: a worker builds the feature (adds g) but needlessly rewrites
    # a shared helper (f), reddening an unrelated test. The guardrail must drop the
    # needless rewrite, keep the feature, and end green.
    _git_repo(
        tmp_path,
        {
            "shared.py": _SHARED_BASELINE,
            "test_keep.py": "import shared\n\n\ndef test_keep():\n    assert shared.f() == 1\n",
            "test_new.py": "import shared\n\n\ndef test_new():\n    assert shared.g() == 2\n",
        },
    )

    def fake_agent(provider, prompt, path, system=None, max_rounds=None, on_tool_event=None):
        (Path(path) / "shared.py").write_text(_SHARED_WORKER)
        return "edited shared.py"

    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", fake_agent)

    result, _models, _providers = run_verified_task(
        _prov(),
        str(tmp_path),
        "add g()",
        "python3 -m pytest -q -p no:cacheprovider",
        bulk_model="b",
        frontier_model="f",
        max_iterations=1,
    )

    assert result.passed is True
    assert result.guardrail_fired is True
    shared = (tmp_path / "shared.py").read_text()
    assert "def g():" in shared  # the feature is kept
    assert "return 1" in shared and "return 999" not in shared  # f() restored
    assert _pytest_rc(tmp_path) == 0  # the suite is genuinely green


def test_guardrail_reports_progress_when_it_fires(tmp_path, monkeypatch):
    _git_repo(
        tmp_path,
        {
            "shared.py": _SHARED_BASELINE,
            "test_keep.py": "import shared\n\n\ndef test_keep():\n    assert shared.f() == 1\n",
            "test_new.py": "import shared\n\n\ndef test_new():\n    assert shared.g() == 2\n",
        },
    )
    monkeypatch.setattr(
        solve_mod,
        "run_agent_in_worktree",
        lambda *a, **k: (Path(a[2]) / "shared.py").write_text(_SHARED_WORKER) or "edited",
    )
    stages: list[str] = []
    run_verified_task(
        _prov(),
        str(tmp_path),
        "add g()",
        "python3 -m pytest -q -p no:cacheprovider",
        bulk_model="b",
        frontier_model="f",
        max_iterations=1,
        on_progress=lambda stage, detail: stages.append(stage),
    )
    assert "guardrail" in stages


def test_guardrail_restores_worker_changes_when_the_revert_breaks_the_target(
    tmp_path, monkeypatch
):
    # A worker whose modification of existing code is NECESSARY for the target:
    # reverting it breaks the target, so the guardrail cannot reach green. It must
    # restore the worker's changes and must NOT claim green.
    _git_repo(
        tmp_path,
        {
            "shared.py": "def f():\n    return 1\n",
            "test_keep.py": "import shared\n\n\ndef test_keep():\n    assert shared.f() == 1\n",
            "test_new.py": "import shared\n\n\ndef test_new():\n    assert shared.f() == 2\n",
        },
    )

    def fake_agent(provider, prompt, path, system=None, max_rounds=None, on_tool_event=None):
        # the only way to satisfy test_new is to change f() -- which breaks test_keep
        (Path(path) / "shared.py").write_text("def f():\n    return 2\n")
        return "edited"

    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", fake_agent)

    result, _models, _providers = run_verified_task(
        _prov(),
        str(tmp_path),
        "make f return 2",
        "python3 -m pytest -q -p no:cacheprovider",
        bulk_model="b",
        frontier_model="f",
        max_iterations=1,
    )

    assert result.passed is False
    assert result.guardrail_fired is False
    # the worker's (load-bearing) change is restored, not left reverted
    assert "return 2" in (tmp_path / "shared.py").read_text()


# --- Cross-provider escalation: a stronger provider takes over on stall ----------
#
# The frontier-directs-bulk payoff: a cheap bulk model does the volume, and only
# when it keeps failing does a stronger *provider* (not merely a stronger model on
# the same provider) take over the remaining iterations.


def _run_agent_recorder(seen):
    """A run_agent_in_worktree stub recording (provider, active model) per call."""

    def fake_agent(provider, prompt, path, system=None, max_rounds=None, on_tool_event=None):
        seen.append((provider, provider.config.model))
        return "edited"

    return fake_agent


def test_run_verified_task_escalates_to_a_different_provider(monkeypatch):
    # With an escalation_provider set, iterations past escalate_after run on that
    # second provider and its model -- not the bulk provider.
    primary = _prov()  # config.model == "m"
    escalation = MagicMock()
    escalation.config = SimpleNamespace(model="glm-default")
    seen: list = []
    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", _run_agent_recorder(seen))
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)
    results = iter([("fail", 1), ("fail", 1), ("ok", 0)])
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: next(results))

    result, models_used, providers_used = run_verified_task(
        primary,
        "/tmp/wt",
        "task",
        "pytest",
        bulk_model="bulk",
        frontier_model="frontier",
        escalate=True,
        escalate_after=1,
        escalation_provider=escalation,
        escalation_model="glm-4.6",
        provider_label="deepseek",
        escalation_label="glm",
        max_iterations=3,
    )

    assert seen[0] == (primary, "bulk")  # iteration 1: primary + bulk
    assert seen[1] == (escalation, "glm-4.6")  # iteration 2: hand off to glm
    assert seen[2] == (escalation, "glm-4.6")  # iteration 3: stays on glm
    assert models_used == ["bulk", "glm-4.6", "glm-4.6"]
    assert providers_used == ["deepseek", "glm", "glm"]
    assert result.passed is True
    # both providers' models are restored to their originals afterward
    assert primary.config.model == "m"
    assert escalation.config.model == "glm-default"


def test_escalation_provider_falls_back_to_its_own_model_when_none(monkeypatch):
    # escalation_model omitted -> use the escalation provider's own configured model.
    primary = _prov()
    escalation = MagicMock()
    escalation.config = SimpleNamespace(model="glm-default")
    seen: list = []
    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", _run_agent_recorder(seen))
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)
    results = iter([("fail", 1), ("ok", 0)])
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: next(results))

    _result, models_used, _providers = run_verified_task(
        primary,
        "/tmp/wt",
        "task",
        "pytest",
        bulk_model="bulk",
        frontier_model="frontier",
        escalate_after=1,
        escalation_provider=escalation,
        escalation_model=None,
        max_iterations=2,
    )

    assert seen[1] == (escalation, "glm-default")
    assert models_used == ["bulk", "glm-default"]


def test_run_verified_task_without_escalation_provider_is_unchanged(monkeypatch):
    # No escalation_provider: escalation stays within the same provider, lifting
    # bulk -> frontier exactly as before (full backward compat).
    prov = _prov()
    seen: list = []
    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", _run_agent_recorder(seen))
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)
    results = iter([("fail", 1), ("ok", 0)])
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: next(results))

    _result, models_used, providers_used = run_verified_task(
        prov,
        "/tmp/wt",
        "task",
        "pytest",
        bulk_model="bulk",
        frontier_model="frontier",
        escalate=True,
        escalate_after=1,
        provider_label="openai",
        max_iterations=3,
    )

    assert seen == [(prov, "bulk"), (prov, "frontier")]
    assert models_used == ["bulk", "frontier"]
    assert providers_used == ["openai", "openai"]  # one provider throughout


def _dual_app(bulk="bulk-x", frontier="frontier-x", glm_model="glm-4.6"):
    """An app with a cheap primary ('deepseek') and a stronger escalation ('glm')."""
    app = MagicMock()
    primary = MagicMock()
    primary.config = SimpleNamespace(model=frontier)
    glm = MagicMock()
    glm.config = SimpleNamespace(model=glm_model)
    app.providers = {"deepseek": primary, "glm": glm}
    app.config.get_default_provider.return_value = "deepseek"

    def _model_for(name, mode_name=None, fast=False):
        if name == "glm":
            return glm_model
        return bulk if fast else frontier

    app.config.get_model_for = MagicMock(side_effect=_model_for)
    app.config.get_bulk_model = MagicMock(side_effect=lambda name, mode_name=None: bulk)
    app.config.data = {"workflows": {"verify": {"test": "pytest"}}}
    return app, primary, glm


def _patch_dual_solve(monkeypatch, seen, test_results):
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=()
    )
    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", _run_agent_recorder(seen))
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: next(test_results))
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)
    monkeypatch.setattr(solve_mod, "WorktreeManager", lambda *a, **k: fm)


def test_run_solve_escalate_to_hands_off_to_another_provider(monkeypatch):
    app, primary, glm = _dual_app()
    seen: list = []
    _patch_dual_solve(monkeypatch, seen, iter([("fail", 1), ("ok", 0)]))

    result = run_solve(
        app, "x", escalate=True, escalate_after=1, escalate_to="glm", max_iterations=3
    )

    assert seen[0] == (primary, "bulk-x")  # bulk builds first
    assert seen[1] == (glm, "glm-4.6")  # glm finishes on stall
    assert result.models_used == ("bulk-x", "glm-4.6")
    assert result.providers_used == ("deepseek", "glm")


def test_run_solve_escalate_to_accepts_provider_model_tuple(monkeypatch):
    app, _primary, glm = _dual_app()
    seen: list = []
    _patch_dual_solve(monkeypatch, seen, iter([("fail", 1), ("ok", 0)]))

    result = run_solve(
        app, "x", escalate_after=1, escalate_to=("glm", "glm-air"), max_iterations=3
    )

    assert seen[1] == (glm, "glm-air")  # the explicit model in the tuple wins
    assert result.models_used == ("bulk-x", "glm-air")
    assert result.providers_used == ("deepseek", "glm")


def test_run_solve_escalate_to_unknown_provider_stays_within_primary(monkeypatch):
    # An escalate_to that names no available provider degrades gracefully to the
    # original within-provider frontier escalation.
    app, primary, _glm = _dual_app()
    seen: list = []
    _patch_dual_solve(monkeypatch, seen, iter([("fail", 1), ("ok", 0)]))

    result = run_solve(app, "x", escalate_after=1, escalate_to="ghost", max_iterations=3)

    assert seen == [(primary, "bulk-x"), (primary, "frontier-x")]
    assert result.models_used == ("bulk-x", "frontier-x")
    assert result.providers_used == ("deepseek", "deepseek")


def test_run_solve_without_escalate_to_is_unchanged(monkeypatch):
    app, primary, _glm = _dual_app()
    seen: list = []
    _patch_dual_solve(monkeypatch, seen, iter([("fail", 1), ("ok", 0)]))

    result = run_solve(app, "x", escalate_after=1, max_iterations=3)

    assert seen == [(primary, "bulk-x"), (primary, "frontier-x")]
    assert result.providers_used == ("deepseek", "deepseek")


# --- Per-model behavioral guidance: steer each cheap model past its failure modes --


def test_worker_guidance_for_deepseek_includes_dogfooded_steers():
    guidance = worker_guidance_for("deepseek/deepseek-v4-flash")
    # targeted-edit steer: prefer replace_in_file over whole-file write_file rewrites
    assert "replace_in_file" in guidance
    assert "write_file" in guidance
    # blast-radius steer: leave shared helpers alone unless the task needs them
    assert "shared" in guidance
    # ngspice convergence steer: model a regulator as a behavioral CURRENT source
    assert "CURRENT source" in guidance


def test_worker_guidance_for_unknown_model_is_empty():
    # a model with no registered failure modes gets no appended steering
    assert worker_guidance_for("claude-opus-4-8") == ""


def test_worker_guidance_for_matches_case_insensitively():
    assert worker_guidance_for("DeepSeek-V4") == worker_guidance_for("deepseek-v4")
    assert worker_guidance_for("DeepSeek-V4") != ""


def _capture_worker_systems(monkeypatch):
    """Patch the solve internals and return a list capturing each worker system."""
    systems: list[str] = []

    def fake_agent(provider, prompt, path, system=None, max_rounds=None, on_tool_event=None):
        systems.append(system)
        return "edited"

    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", fake_agent)
    monkeypatch.setattr(solve_mod, "_preflight_gate", lambda *a, **k: None)
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: ("ok", 0))
    return systems


def test_run_verified_task_appends_deepseek_guidance_to_the_worker_system(monkeypatch):
    systems = _capture_worker_systems(monkeypatch)

    run_verified_task(
        _prov(),
        "/tmp/wt",
        "task",
        "pytest",
        bulk_model="deepseek/deepseek-v4-flash",
        frontier_model="f",
    )

    assert len(systems) == 1
    # the base worker prompt is preserved, with the DeepSeek steering appended
    assert solve_mod._WORKER_SYSTEM in systems[0]
    assert "replace_in_file" in systems[0]
    assert "CURRENT source" in systems[0]


def test_run_verified_task_omits_guidance_for_a_non_deepseek_model(monkeypatch):
    systems = _capture_worker_systems(monkeypatch)

    run_verified_task(
        _prov(),
        "/tmp/wt",
        "task",
        "pytest",
        bulk_model="claude-opus-4-8",
        frontier_model="f",
    )

    # no matching failure modes -> the worker system is exactly the base prompt
    assert systems == [solve_mod._WORKER_SYSTEM]


# --- Anti-proxy verification classifier: annotate what the gate exercised --------
#
# A cheap model can green the gate without real work by pointing it at a grep, a
# `test -f`, or a bare `true`. classify_verification labels what the resolved
# verify command actually exercises so run_solve can warn on a passing proxy or
# sentinel check -- an advisory note that never changes result.passed.


def test_classify_verification_detects_a_real_test_runner():
    assert solve_mod.classify_verification("uv run pytest tests/ -q") == "test"
    assert solve_mod.classify_verification("python -m pytest -x") == "test"
    assert solve_mod.classify_verification("python -m unittest discover") == "test"
    assert solve_mod.classify_verification("go test ./...") == "test"
    assert solve_mod.classify_verification("npx jest") == "test"
    assert solve_mod.classify_verification("vitest run") == "test"
    assert solve_mod.classify_verification("cargo test --all") == "test"


def test_classify_verification_detects_syntactic_only_checks():
    assert solve_mod.classify_verification("ruff check .") == "syntactic"
    assert solve_mod.classify_verification("flake8 src/") == "syntactic"
    assert solve_mod.classify_verification("mypy cascade") == "syntactic"
    assert solve_mod.classify_verification("tsc --noEmit") == "syntactic"
    assert solve_mod.classify_verification("python -m py_compile foo.py") == "syntactic"
    assert solve_mod.classify_verification("eslint .") == "syntactic"


def test_classify_verification_detects_proxy_checks():
    # existence/grep checks prove the check ran, nothing about behavior
    assert solve_mod.classify_verification("test -f out.txt") in {"proxy", "sentinel"}
    assert solve_mod.classify_verification("grep -q def foo.py") == "proxy"
    assert solve_mod.classify_verification("rg TODO cascade/") == "proxy"
    assert solve_mod.classify_verification("ls dist/") == "proxy"
    assert solve_mod.classify_verification("cat result.json") == "proxy"
    assert solve_mod.classify_verification("[ -f out.txt ]") == "proxy"


def test_classify_verification_detects_sentinels():
    assert solve_mod.classify_verification("touch done") == "sentinel"
    assert solve_mod.classify_verification("true") == "sentinel"
    assert solve_mod.classify_verification(":") == "sentinel"


def test_classify_verification_prefers_a_test_runner_anywhere():
    # a real runner outranks a lint or a proxy sharing the same command line
    assert solve_mod.classify_verification("ruff check . && pytest -q") == "test"
    assert solve_mod.classify_verification("grep -q foo bar.py && pytest") == "test"
    assert solve_mod.classify_verification("touch done && cargo test") == "test"


def test_classify_verification_is_unknown_for_unrecognized_commands():
    assert solve_mod.classify_verification("make build") == "unknown"
    assert solve_mod.classify_verification("./run_ci.sh") == "unknown"
    assert solve_mod.classify_verification("") == "unknown"


def _solve_env(monkeypatch, rc=0):
    """Wire a minimal passing run_solve: fake worktree + agent + test result."""
    fm = MagicMock()
    fm.prepare.return_value = SimpleNamespace(path="/tmp/wt")
    fm.capture_snapshot.return_value = SimpleNamespace(
        diff_stat="", diff_excerpt="", changed_files=()
    )
    monkeypatch.setattr(solve_mod, "WorktreeManager", lambda *a, **k: fm)
    monkeypatch.setattr(solve_mod, "run_agent_in_worktree", lambda *a, **k: "edited")
    monkeypatch.setattr(solve_mod, "_run_tests_in", lambda c, w, t: ("ok", rc))


def test_run_solve_records_verification_kind_on_a_passing_run(monkeypatch):
    app = _fake_app("pytest")
    _solve_env(monkeypatch)

    result = run_solve(app, "add foo")

    assert result.passed is True
    assert result.verification_kind == "test"


def test_run_solve_warns_when_a_passing_check_is_a_proxy(monkeypatch):
    app = _fake_app("grep -q def cascade/foo.py")
    _solve_env(monkeypatch)

    result = run_solve(app, "add foo")

    assert result.passed is True  # the warning is advisory, not a gate
    assert result.verification_kind == "proxy"
    assert "proxy" in result.error
    assert "may not exercise real behavior" in result.error


def test_run_solve_does_not_warn_for_a_real_test_runner(monkeypatch):
    app = _fake_app("pytest")
    _solve_env(monkeypatch)

    result = run_solve(app, "add foo")

    assert result.verification_kind == "test"
    assert "may not exercise real behavior" not in result.error
