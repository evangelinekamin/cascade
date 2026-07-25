"""Tests for competitive multi-provider execution."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

from cascade.commands import CommandHandler
from cascade.swarm import (
    CompetitionEntry,
    CompetitionJudgment,
    CompetitionOrchestrator,
    CompetitionResult,
)
from cascade.swarm.worktree import WorktreeManager
from cascade.providers.usage import Usage


def _mock_provider(name, response=None, model="test-model", judge_payload=None):
    """Create a provider stub for competition tests."""
    prov = MagicMock()
    prov.name = name
    prov.config = MagicMock()
    prov.config.model = model
    prov.last_usage = Usage(input=100, output=50)

    def _ask_single(prompt, system=None):
        if system is not None:
            if judge_payload is not None:
                return judge_payload
            return json.dumps({
                "winner_provider": "gemini",
                "rationale": "Gemini had the strongest answer.",
                "summary": "Use the Gemini result.",
            })
        return response or f"{name} answer"

    prov.ask_single.side_effect = _ask_single
    return prov


def _mock_app(providers=None):
    app = MagicMock()
    if providers is None:
        providers = {
            "claude": _mock_provider("claude", response="claude answer"),
            "gemini": _mock_provider("gemini", response="gemini answer"),
            "openai": _mock_provider("openai", response="openai answer"),
        }
    app.providers = providers
    return app


def _init_git_repo(tmpdir: str) -> None:
    subprocess.run(["git", "init"], cwd=tmpdir, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "cascade@example.com"],
        cwd=tmpdir,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Cascade Test"],
        cwd=tmpdir,
        check=True,
        capture_output=True,
        text=True,
    )
    Path(tmpdir, "app.txt").write_text("base\n")
    subprocess.run(["git", "add", "app.txt"], cwd=tmpdir, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmpdir,
        check=True,
        capture_output=True,
        text=True,
    )


class _CodeProvider:
    def __init__(self, name: str, judge_payload: str | None = None):
        self.name = name
        self.config = MagicMock()
        self.config.model = f"{name}-model"
        self.last_usage = Usage(input=80, output=20)
        self._use_cli_proxy = True
        self._workdir = None
        self._judge_payload = judge_payload

    @contextmanager
    def working_directory(self, path: str):
        previous = self._workdir
        self._workdir = path
        try:
            yield
        finally:
            self._workdir = previous

    def ask_single(self, prompt: str, system: str | None = None) -> str:
        if system and "Respond with JSON only" in system:
            return self._judge_payload or json.dumps({
                "winner_provider": "openai",
                "rationale": "OpenAI made the strongest code change.",
                "summary": "Use the OpenAI workspace.",
            })
        assert self._workdir is not None
        Path(self._workdir, "app.txt").write_text(f"{self.name}\n")
        return f"{self.name} updated app.txt"

    def ask_with_tools(self, messages, tools, system=None):
        raise AssertionError("CLI-style code provider should not use tool calling here")


class _NoChangeCodeProvider(_CodeProvider):
    def ask_single(self, prompt: str, system: str | None = None) -> str:
        if system and "Respond with JSON only" in system:
            return super().ask_single(prompt, system)
        assert self._workdir is not None
        return f"{self.name} inspected the workspace but changed nothing"


class TestCompetitionOrchestrator:
    def test_picks_best_judge_provider(self):
        app = _mock_app({"gemini": _mock_provider("gemini"), "openai": _mock_provider("openai")})
        compete = CompetitionOrchestrator(app)
        assert compete._judge_provider == "gemini"

    def test_execute_selects_judged_winner(self):
        providers = {
            "claude": _mock_provider("claude", response="claude answer"),
            "gemini": _mock_provider("gemini", response="gemini answer"),
            "openai": _mock_provider("openai", response="openai answer"),
        }
        app = _mock_app(providers)
        compete = CompetitionOrchestrator(app)

        result = compete.execute("Solve the task")

        assert result.winner_provider == "gemini"
        assert result.winner_response == "gemini answer"
        assert result.judgment == CompetitionJudgment(
            winner_provider="gemini",
            rationale="Gemini had the strongest answer.",
            summary="Use the Gemini result.",
        )
        assert result.total_tokens == 450

    def test_execute_falls_back_when_judge_output_invalid(self):
        providers = {
            "claude": _mock_provider("claude", response="claude answer", judge_payload="not json"),
            "gemini": _mock_provider("gemini", response="gemini answer"),
            "openai": _mock_provider("openai", response="openai answer"),
        }
        app = _mock_app(providers)
        compete = CompetitionOrchestrator(app)

        result = compete.execute("Solve the task")

        assert result.winner_provider == "claude"
        assert "Judge result unavailable" in result.judgment.rationale

    def test_execute_handles_provider_failure(self):
        broken = _mock_provider("openai")
        broken.ask_single.side_effect = RuntimeError("boom")
        providers = {
            "claude": _mock_provider("claude", response="claude answer"),
            "gemini": _mock_provider("gemini", response="gemini answer"),
            "openai": broken,
        }
        app = _mock_app(providers)
        compete = CompetitionOrchestrator(app)

        result = compete.execute("Solve the task")

        failures = [entry for entry in result.entries if not entry.success]
        assert len(failures) == 1
        assert failures[0].provider == "openai"
        assert failures[0].error == "boom"

    def test_execute_code_keeps_only_the_winner_worktree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_git_repo(tmpdir)
            providers = {
                "claude": _CodeProvider("claude"),
                "openai": _CodeProvider("openai"),
                "gemini": _CodeProvider(
                    "gemini",
                    judge_payload=json.dumps({
                        "winner_provider": "openai",
                        "rationale": "OpenAI changed the target file correctly.",
                        "summary": "Use the OpenAI worktree.",
                    }),
                ),
            }
            app = _mock_app(providers)
            compete = CompetitionOrchestrator(app, judge_provider="gemini")

            previous_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                result = compete.execute_code("Update app.txt", providers=["claude", "openai"])
            finally:
                os.chdir(previous_cwd)

            assert result.winner_provider == "openai"
            winner_entry = next(entry for entry in result.entries if entry.provider == "openai")
            loser_entry = next(entry for entry in result.entries if entry.provider == "claude")
            assert winner_entry.retained is True
            assert loser_entry.retained is False
            assert winner_entry.worktree_path
            assert loser_entry.worktree_path == ""
            assert Path(winner_entry.worktree_path, "app.txt").read_text() == "openai\n"
            assert "app.txt" in winner_entry.changed_files
            subprocess.run(
                ["git", "worktree", "remove", "--force", winner_entry.worktree_path],
                cwd=tmpdir,
                check=False,
                capture_output=True,
                text=True,
            )
            shutil.rmtree(Path(winner_entry.worktree_path).parent, ignore_errors=True)

    def test_execute_code_marks_no_diff_runs_as_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_git_repo(tmpdir)
            providers = {
                "claude": _NoChangeCodeProvider("claude"),
                "openai": _CodeProvider("openai"),
                "gemini": _CodeProvider("gemini"),
            }
            app = _mock_app(providers)
            compete = CompetitionOrchestrator(app, judge_provider="gemini")

            previous_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                result = compete.execute_code("Update app.txt", providers=["claude", "openai"])
            finally:
                os.chdir(previous_cwd)

            no_change_entry = next(entry for entry in result.entries if entry.provider == "claude")
            winner_entry = next(entry for entry in result.entries if entry.provider == "openai")
            assert no_change_entry.success is False
            assert no_change_entry.error == "no changes produced"
            assert no_change_entry.changed_files == []
            assert result.winner_provider == "openai"
            subprocess.run(
                ["git", "worktree", "remove", "--force", winner_entry.worktree_path],
                cwd=tmpdir,
                check=False,
                capture_output=True,
                text=True,
            )
            shutil.rmtree(Path(winner_entry.worktree_path).parent, ignore_errors=True)

    def test_execute_code_list_mode_skips_judge_and_keeps_all_worktrees(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_git_repo(tmpdir)
            providers = {
                "claude": _CodeProvider("claude"),
                "openai": _CodeProvider("openai"),
                # A judge is configured but must not be consulted in list mode.
                "gemini": _CodeProvider("gemini"),
            }
            app = _mock_app(providers)
            compete = CompetitionOrchestrator(app, judge_provider="gemini")

            previous_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                result = compete.execute_code(
                    "Update app.txt", providers=["claude", "openai"], judge=False
                )
            finally:
                os.chdir(previous_cwd)

            # No judge, no winner -- every run retained for external review.
            assert result.judgment is None
            assert result.winner_provider == ""
            assert result.judge_provider == ""
            assert len(result.entries) == 2
            for entry in result.entries:
                assert entry.retained is True
                assert entry.worktree_path

            # A machine-readable manifest is written into the run root for
            # downstream (subagent) processing.
            assert result.run_root and Path(result.run_root).is_dir()
            assert result.manifest_path and Path(result.manifest_path).is_file()
            manifest = json.loads(Path(result.manifest_path).read_text())
            assert {c["label"] for c in manifest["competitors"]} == {"claude", "openai"}
            assert all("worktree_path" in c and "model" in c for c in manifest["competitors"])
            for entry in result.entries:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", entry.worktree_path],
                    cwd=tmpdir,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                shutil.rmtree(Path(entry.worktree_path).parent, ignore_errors=True)

    def test_code_competition_injects_per_model_guidance(self):
        # The competition is single-shot with no verify-loop retry, so a model
        # that narrates instead of acting silently produces "no changes" and
        # loses. DeepSeek (documented for exactly that) must get its steering;
        # a model without a guidance entry must not have anything appended.
        from cascade.swarm import competition as comp

        captured = {}

        def fake_run(provider, prompt, worktree_path, system=None, **kw):
            captured["system"] = system or ""
            return "narrated a plan"

        app = _mock_app({
            "openrouter": _mock_provider("openrouter", model="deepseek/deepseek-v4-flash"),
            "openai": _mock_provider("openai", model="gpt-5.6-terra"),
        })
        compete = CompetitionOrchestrator(app, judge_provider="openai")
        manager = MagicMock()
        manager.capture_snapshot.side_effect = Exception("no worktree")  # -> empty snapshot

        with patch.object(comp, "run_agent_in_worktree", fake_run):
            compete._execute_code_provider("openrouter", "task", "/wt", manager)
            deepseek_system = captured["system"]
            compete._execute_code_provider("openai", "task", "/wt", manager)
            openai_system = captured["system"]

        assert "do not merely narrate" in deepseek_system.lower()
        assert "do not merely narrate" not in openai_system.lower()

    def test_winner_patch_capture_applies_to_the_tree(self):
        # /compete-code captures the winner's patch with a FRESH WorktreeManager
        # (baseline defaults to HEAD = the competition base), so /apply -- which
        # uses apply_to_tree -- can land the winner's changes on the real tree.
        from cascade.swarm.worktree import WorktreeManager

        with tempfile.TemporaryDirectory() as tmpdir:
            _init_git_repo(tmpdir)
            mgr = WorktreeManager(cwd=tmpdir)
            wt = mgr.prepare("winner").path
            (Path(wt) / "new_file.py").write_text("print('winner')\n")

            patch = WorktreeManager().diff_patch(wt)  # fresh manager, HEAD baseline
            assert "new_file.py" in patch

            applied, msg = WorktreeManager.apply_to_tree(tmpdir, patch)
            assert applied, msg
            assert (Path(tmpdir) / "new_file.py").exists()

            subprocess.run(
                ["git", "-C", tmpdir, "worktree", "remove", "--force", wt],
                check=False, capture_output=True, text=True,
            )


class TestWorktreeManager:
    def test_prepare_copies_dirty_and_untracked_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_git_repo(tmpdir)
            Path(tmpdir, "app.txt").write_text("dirty\n")
            Path(tmpdir, "scratch.txt").write_text("hello\n")

            manager = WorktreeManager(cwd=tmpdir)
            prepared = manager.prepare("claude")

            assert Path(prepared.path, "app.txt").read_text() == "dirty\n"
            assert Path(prepared.path, "scratch.txt").read_text() == "hello\n"

            manager.cleanup()

    def test_capture_snapshot_ignores_source_tree_dirty_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _init_git_repo(tmpdir)
            Path(tmpdir, "app.txt").write_text("dirty\n")
            Path(tmpdir, "scratch.txt").write_text("hello\n")

            manager = WorktreeManager(cwd=tmpdir)
            prepared = manager.prepare("claude")

            baseline = manager.capture_snapshot(prepared.path)
            assert baseline.changed_files == ()
            assert baseline.diff_stat == ""
            assert baseline.diff_excerpt == ""

            Path(prepared.path, "app.txt").write_text("agent\n")
            snapshot = manager.capture_snapshot(prepared.path)

            assert snapshot.changed_files == ("app.txt",)
            assert "app.txt" in snapshot.diff_stat
            assert "scratch.txt" not in snapshot.diff_excerpt

            manager.cleanup()


class TestCompeteCommand:
    def test_compete_command_dispatches_and_formats_result(self):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.providers = {
            "claude": MagicMock(),
            "gemini": MagicMock(),
        }
        app.cli_app = cli_app
        app.state = MagicMock()
        app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
        app.screen = MagicMock()
        app.screen.run_worker.side_effect = lambda fn, **kwargs: fn()

        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)
        progress = MagicMock()
        progress.docked = False  # inline-fallback handle: cleared via remove()
        handler._mount_progress_indicator = MagicMock(return_value=progress)

        fake_result = CompetitionResult(
            objective="build thing",
            entries=[
                CompetitionEntry(provider="claude", response="claude answer", tokens=150, duration_seconds=1.1),
                CompetitionEntry(provider="gemini", response="gemini answer", tokens=140, duration_seconds=0.9),
            ],
            judgment=CompetitionJudgment(
                winner_provider="gemini",
                rationale="Gemini is clearer.",
                summary="Use Gemini.",
            ),
            winner_provider="gemini",
            winner_response="gemini answer",
            total_tokens=290,
            judge_provider="claude",
        )

        with patch("cascade.swarm.CompetitionOrchestrator") as mock_orchestrator:
            instance = mock_orchestrator.return_value

            def _execute(objective, providers=None, on_progress=None):
                if on_progress:
                    on_progress("competing", "[claude] running")
                return fake_result

            instance.execute.side_effect = _execute
            handler._cmd_compete(["build", "thing"])

        instance.execute.assert_called_once_with(
            "build thing",
            providers=["claude", "gemini"],
            on_progress=instance.execute.call_args.kwargs["on_progress"],
        )
        assert posted[0] == (
            "Competition dispatching: build thing\n"
            "Providers: claude, gemini\n"
            "Judge: auto"
        )
        progress.set_label.assert_called_with("claude running | gemini queued")
        progress.remove.assert_called_once()
        app.record_message.assert_any_call("user", "/compete build thing", token_count=0)
        app.record_message.assert_any_call("system", posted[-1], token_count=0)
        assert "Competition complete. Judge: claude" in posted[-1]
        assert "Winner: gemini" in posted[-1]
        assert "--- Winner ---" in posted[-1]
        assert "gemini answer" in posted[-1]

    def test_compete_command_accepts_provider_subset_and_judge(self):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.providers = {
            "claude": MagicMock(),
            "gemini": MagicMock(),
            "openai": MagicMock(),
        }
        app.cli_app = cli_app
        app.state = MagicMock()
        app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
        app.screen = MagicMock()
        app.screen.run_worker.side_effect = lambda fn, **kwargs: fn()

        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)

        fake_result = CompetitionResult(
            objective="build thing",
            entries=[
                CompetitionEntry(provider="claude", response="claude answer", tokens=150, duration_seconds=1.1),
                CompetitionEntry(provider="openai", response="openai answer", tokens=140, duration_seconds=0.9),
            ],
            judgment=CompetitionJudgment(
                winner_provider="openai",
                rationale="OpenAI is clearer.",
                summary="Use OpenAI.",
            ),
            winner_provider="openai",
            winner_response="openai answer",
            total_tokens=290,
            judge_provider="gemini",
        )

        with patch("cascade.swarm.CompetitionOrchestrator") as mock_orchestrator:
            instance = mock_orchestrator.return_value
            instance.execute.return_value = fake_result

            handler._cmd_compete(
                ["--providers", "claude,openai", "--judge", "gemini", "build", "thing"]
            )

        mock_orchestrator.assert_called_once_with(
            cli_app, judge_provider="gemini", provider_overrides={}
        )
        instance.execute.assert_called_once_with(
            "build thing",
            providers=["claude", "openai"],
            on_progress=instance.execute.call_args.kwargs["on_progress"],
        )
        assert posted[0] == (
            "Competition dispatching: build thing\n"
            "Providers: claude, openai\n"
            "Judge: gemini"
        )

    def test_compete_code_command_formats_winner_worktree(self):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.providers = {
            "claude": MagicMock(),
            "openai": MagicMock(),
            "gemini": MagicMock(),
        }
        app.cli_app = cli_app
        app.state = MagicMock()
        app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
        app.screen = MagicMock()
        app.screen.run_worker.side_effect = lambda fn, **kwargs: fn()

        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)
        progress = MagicMock()
        progress.docked = False  # inline-fallback handle: cleared via remove()
        handler._mount_progress_indicator = MagicMock(return_value=progress)

        fake_result = CompetitionResult(
            objective="fix bug",
            entries=[
                CompetitionEntry(
                    provider="claude",
                    response="claude summary",
                    tokens=150,
                    duration_seconds=1.2,
                    changed_files=["app.txt"],
                ),
                CompetitionEntry(
                    provider="openai",
                    response="openai summary",
                    tokens=140,
                    duration_seconds=1.0,
                    changed_files=["app.txt", "tests.txt"],
                    diff_stat=" app.txt | 2 +-\n 1 file changed, 1 insertion(+), 1 deletion(-)",
                    worktree_path="/tmp/cascade-compete/openai",
                    retained=True,
                ),
            ],
            judgment=CompetitionJudgment(
                winner_provider="openai",
                rationale="OpenAI changed the right file.",
                summary="Use the retained worktree.",
            ),
            winner_provider="openai",
            winner_response="openai summary",
            total_tokens=290,
            judge_provider="gemini",
        )

        with patch("cascade.swarm.CompetitionOrchestrator") as mock_orchestrator:
            instance = mock_orchestrator.return_value
            instance.execute_code.return_value = fake_result

            handler._cmd_compete_code(
                ["--providers", "claude,openai", "--judge", "gemini", "fix", "bug"]
            )

        mock_orchestrator.assert_called_once_with(
            cli_app, judge_provider="gemini", provider_overrides={}
        )
        instance.execute_code.assert_called_once_with(
            "fix bug",
            providers=["claude", "openai"],
            on_progress=instance.execute_code.call_args.kwargs["on_progress"],
            judge=True,
        )
        assert posted[0] == (
            "Code competition dispatching: fix bug\n"
            "Providers: claude, openai\n"
            "Judge: gemini"
        )
        progress.remove.assert_called_once()
        app.record_message.assert_any_call("user", "/compete-code --providers claude,openai --judge gemini fix bug", token_count=0)
        app.record_message.assert_any_call("system", posted[-1], token_count=0)
        assert "Winner worktree: /tmp/cascade-compete/openai" in posted[-1]
        assert "[openai] OK (1.0s, 140 tok, 140 tok/s) | 1 file changed, 1 insertion(+), 1 deletion(-)" in posted[-1]
        assert "Files: app.txt, tests.txt" in posted[-1]

    def test_compete_code_command_shows_no_diff_failures(self):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.providers = {
            "claude": MagicMock(),
            "openai": MagicMock(),
        }
        app.cli_app = cli_app
        app.state = MagicMock()
        app.call_from_thread.side_effect = lambda fn, *args: fn(*args)
        app.screen = MagicMock()
        app.screen.run_worker.side_effect = lambda fn, **kwargs: fn()

        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)

        fake_result = CompetitionResult(
            objective="fix bug",
            entries=[
                CompetitionEntry(
                    provider="claude",
                    response="claude summary",
                    tokens=150,
                    duration_seconds=1.2,
                    success=False,
                    error="no changes produced",
                ),
                CompetitionEntry(
                    provider="openai",
                    response="openai summary",
                    tokens=140,
                    duration_seconds=1.0,
                    changed_files=["app.txt"],
                    worktree_path="/tmp/cascade-compete/openai",
                    retained=True,
                ),
            ],
            judgment=CompetitionJudgment(
                winner_provider="openai",
                rationale="OpenAI changed the right file.",
                summary="Use the retained worktree.",
            ),
            winner_provider="openai",
            winner_response="openai summary",
            total_tokens=290,
            judge_provider="claude",
        )

        with patch("cascade.swarm.CompetitionOrchestrator") as mock_orchestrator:
            instance = mock_orchestrator.return_value
            instance.execute_code.return_value = fake_result
            handler._cmd_compete_code(["fix", "bug"])

        assert "[claude] FAIL: no changes produced (1.2s, 150 tok, 125 tok/s) | no diff" in posted[-1]

    def test_compete_command_rejects_unknown_provider(self):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.providers = {
            "claude": MagicMock(),
            "gemini": MagicMock(),
        }
        app.cli_app = cli_app
        app.state = MagicMock()
        app.screen = MagicMock()

        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)

        handler._cmd_compete(["--providers=claude,unknown", "build", "thing"])

        assert posted == [
            "Competition provider(s) not found: unknown. Available: claude, gemini"
        ]

    def test_compete_command_rejects_unknown_judge(self):
        app = MagicMock()
        cli_app = MagicMock()
        cli_app.providers = {
            "claude": MagicMock(),
            "gemini": MagicMock(),
        }
        app.cli_app = cli_app
        app.state = MagicMock()
        app.screen = MagicMock()

        handler = CommandHandler(app)
        posted = []
        handler._post_system = lambda text: posted.append(text)

        handler._cmd_compete(["--judge=openai", "build", "thing"])

        assert posted == [
            "Competition judge 'openai' not found. Available: claude, gemini"
        ]


def test_parse_compete_args_supports_models_flag():
    p, m, j, o = CommandHandler._parse_compete_args(
        ["--models", "deepseek/deepseek-v4-flash,moonshotai/kimi-k3",
         "--judge", "claude", "build", "the", "thing"]
    )
    assert p is None
    assert m == ["deepseek/deepseek-v4-flash", "moonshotai/kimi-k3"]
    assert j == "claude"
    assert o == "build the thing"


def test_models_flag_clones_openrouter_per_model():
    from types import SimpleNamespace
    from cascade.providers.base import ProviderConfig
    from cascade.providers.openrouter import OpenRouterProvider

    base = OpenRouterProvider(ProviderConfig(api_key="k", model="base-model"))
    cli_app = SimpleNamespace(providers={"openrouter": base})
    handler = CommandHandler(MagicMock())
    handler._post_system = lambda *a, **k: None

    built = handler._build_model_overrides(
        cli_app, ["deepseek/deepseek-v4-flash", "moonshotai/kimi-k3"], "Code competition"
    )
    assert built is not None
    labels, overrides = built
    # Filesystem-safe labels, one clone each, running the requested model, distinct
    # instances from the base (so config.model swaps don't collide).
    assert labels == ["deepseek-deepseek-v4-flash", "moonshotai-kimi-k3"]
    assert overrides[labels[0]].config.model == "deepseek/deepseek-v4-flash"
    assert overrides[labels[1]].config.model == "moonshotai/kimi-k3"
    assert overrides[labels[0]] is not base


def test_models_flag_clears_pinned_order_for_per_model_routing():
    from types import SimpleNamespace
    from cascade.providers.base import ProviderConfig
    from cascade.providers.openrouter import OpenRouterProvider

    base = OpenRouterProvider(
        ProviderConfig(
            api_key="k",
            model="base-model",
            context_window=200000,
            provider_preferences={"order": ["Baidu"], "require_parameters": True},
        )
    )
    cli_app = SimpleNamespace(providers={"openrouter": base})
    handler = CommandHandler(MagicMock())
    handler._post_system = lambda *a, **k: None

    _, overrides = handler._build_model_overrides(
        cli_app, ["deepseek/deepseek-v4-flash", "moonshotai/kimi-k3"], "Code competition"
    )
    clone = overrides["deepseek-deepseek-v4-flash"]
    # The inherited pin and window are dropped so each model's own metadata fetch
    # picks its best endpoints; unrelated preferences survive.
    assert clone.config.context_window is None
    assert "order" not in (clone.config.provider_preferences or {})
    assert clone.config.provider_preferences["require_parameters"] is True
    # The base provider is untouched.
    assert base.config.provider_preferences["order"] == ["Baidu"]


def test_models_flag_disables_fallback_so_a_model_fails_as_itself():
    from types import SimpleNamespace
    from cascade.providers.base import ProviderConfig
    from cascade.providers.openrouter import OpenRouterProvider

    base = OpenRouterProvider(ProviderConfig(api_key="k", model="base-model"))
    cli_app = SimpleNamespace(providers={"openrouter": base})
    handler = CommandHandler(MagicMock())
    handler._post_system = lambda *a, **k: None

    _, overrides = handler._build_model_overrides(
        cli_app, ["z-ai/glm-5.2", "minimax/minimax-m3"], "Code competition"
    )
    clone = overrides["z-ai-glm-5.2"]
    # A rate-limited competitor must FAIL as itself, never silently retry on the
    # default fallback model and be scored under the requested name.
    assert clone.config.fallback_model == "z-ai/glm-5.2"
    assert clone.get_fallback_model() is None


def test_models_flag_disambiguates_label_collisions_instead_of_dropping():
    from types import SimpleNamespace
    from cascade.providers.base import ProviderConfig
    from cascade.providers.openrouter import OpenRouterProvider

    base = OpenRouterProvider(ProviderConfig(api_key="k", model="base-model"))
    cli_app = SimpleNamespace(providers={"openrouter": base})
    handler = CommandHandler(MagicMock())
    handler._post_system = lambda *a, **k: None

    # Two distinct ids that normalize to the same filesystem-safe label.
    labels, overrides = handler._build_model_overrides(
        cli_app, ["a/b:c", "a/b-c"], "Code competition"
    )
    assert len(overrides) == 2  # both kept as competitors, neither silently merged
    assert sorted(o.config.model for o in overrides.values()) == ["a/b-c", "a/b:c"]


def test_models_flag_needs_openrouter_and_two_models():
    from types import SimpleNamespace

    handler = CommandHandler(MagicMock())
    handler._post_system = lambda *a, **k: None
    # No openrouter provider -> None.
    assert handler._build_model_overrides(
        SimpleNamespace(providers={}), ["a/b", "c/d"], "x"
    ) is None
