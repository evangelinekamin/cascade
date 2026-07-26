"""Tests for model-selected workflow routing and constrained execution."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import cascade.swarm.auto as auto
from textual.message_pump import active_app

from cascade.providers.base import ProviderConfig
from cascade.screens.main import MainScreen
from cascade.swarm.auto import RouteDecision, WorkflowKind
from cascade.swarm.outcome import RunOutcome
from cascade.providers.usage import Usage


class _Config:
    def __init__(self, orchestration=None):
        self.orchestration = orchestration or {
            "enabled": True,
            "modes": ["design", "plan", "build", "test"],
            "router_provider": "openrouter",
            "router_model": "openai/gpt-oss-120b",
            "recon_provider": "openrouter",
            "recon_model": "openai/gpt-oss-120b",
            "fast_provider": "openrouter",
            "fast_model": "stepfun/step-3.7-flash",
            "fast_provider_preferences": {
                "allow_fallbacks": True,
                "require_parameters": True,
            },
            "provider_preferences": {
                "order": ["cerebras"],
                "allow_fallbacks": True,
                "require_parameters": True,
            },
            "recon_max_rounds": 10,
        }

    def get_orchestration_config(self):
        return self.orchestration

    def get_prompt_config(self):
        return {}

    def get_escalation_target(self, _provider):
        return None

    def get_model_for(self, provider, mode_name=None, fast=False):
        if provider == "openrouter" and fast:
            return "stepfun/step-3.7-flash"
        return f"{provider}-frontier"


class _FakeOpenRouter:
    route_payload = {
        "workflow": "recon",
        "reason": "repository evidence is required",
        "confidence": 0.94,
    }
    last_instance = None

    def __init__(self, config):
        self.config = config
        self.last_usage = Usage(input=11, output=3)
        self.tools = None
        type(self).last_instance = self

    def ask_structured(self, prompt, schema, **kwargs):
        self.structured_call = (prompt, schema, kwargs)
        return dict(self.route_payload)

    def ask_with_tools(self, messages, tools, **kwargs):
        self.tools = tools
        self.tool_call = (messages, kwargs)
        self.last_usage = Usage(input=20, output=8)
        return "Found it in cascade/config.py.", []


def _app():
    original = _FakeOpenRouter(
        ProviderConfig(api_key="k", model="normal-chat-model")
    )
    return SimpleNamespace(
        config=_Config(),
        providers={"openrouter": original, "openai": MagicMock()},
    ), original


class _FakeWorktreeManager:
    """A WorktreeManager stand-in that hands back a fixed path, doing no git."""

    def __init__(self, path):
        self._path = path
        self.cleaned = False
        self.prepared = []

    def __call__(self, *args, **kwargs):
        return self

    def prepare(self, provider):
        self.prepared.append(provider)
        return SimpleNamespace(provider=provider, path=self._path)

    def cleanup(self, keep_provider=""):
        self.cleaned = True


def _fake_worktree_manager(path):
    """Return (manager, roots) -- the fake manager and a dict the spy fills."""
    return _FakeWorktreeManager(path), {}


def _spy_workspace_tools(roots):
    """A WorkspaceTools factory that records the root it was constructed with."""
    real = auto.WorkspaceTools

    def _factory(root, **kwargs):
        roots["ws"] = root
        return real(root, **kwargs)

    return _factory


def test_auto_routing_is_limited_to_configured_modes(monkeypatch):
    monkeypatch.setattr(auto, "_is_git_worktree", lambda: True)
    app, _original = _app()
    assert auto.should_auto_orchestrate(app, "build") is True
    assert auto.should_auto_orchestrate(app, "design") is True
    app.config.orchestration["modes"] = ["build"]
    assert auto.should_auto_orchestrate(app, "design") is False


def test_auto_routing_requires_a_git_worktree(monkeypatch):
    app, _original = _app()
    # Inside a repository an enabled, configured mode orchestrates.
    monkeypatch.setattr(auto, "_is_git_worktree", lambda: True)
    assert auto.should_auto_orchestrate(app, "build") is True
    # Outside one it falls back to ordinary chat, so a prompt in a non-git
    # directory never reaches the worktree manager and its raw git error.
    monkeypatch.setattr(auto, "_is_git_worktree", lambda: False)
    assert auto.should_auto_orchestrate(app, "build") is False


def test_selector_uses_gpt_oss_on_cerebras_without_mutating_chat_provider():
    app, original = _app()
    app.config.orchestration["local_router"] = False  # exercise the model tier
    decision = auto.select_workflow(app, "Review the config implementation", "build")

    routed = _FakeOpenRouter.last_instance
    assert decision.workflow == WorkflowKind.RECON
    assert decision.router_model == "openai/gpt-oss-120b"
    assert decision.input_tokens == 11 and decision.output_tokens == 3
    assert routed is not original
    assert routed.config.model == "openai/gpt-oss-120b"
    assert routed.config.provider_preferences == {
        "order": ["cerebras"],
        "allow_fallbacks": True,
        "require_parameters": True,
    }
    assert original.config.model == "normal-chat-model"


def test_selector_falls_back_to_chat_when_the_router_is_unavailable():
    # When no routing model can be built, an unrouted prompt is handled as chat
    # (which has tools) -- NOT force-routed to a worktree solve by a lexical guess.
    app, _original = _app()
    app.providers = {}
    app.config.orchestration["local_router"] = False  # skip the local tier too

    decision = auto.select_workflow(app, "Implement a parser fix", "build")

    assert decision.workflow == WorkflowKind.CHAT
    assert decision.router_provider == "fallback"


# --- Local first-tier router: precision-1 allowlist (abstains unless certain) ----


def test_local_router_short_circuits_a_clear_edit_without_calling_the_model():
    app, _original = _app()
    _FakeOpenRouter.last_instance = None  # prove the model router is never built

    decision = auto.select_workflow(app, "fix the login redirect bug", "build")

    assert decision.workflow == WorkflowKind.SOLVE
    assert decision.router_provider == "local"
    assert decision.router_model == ""  # no model was consulted
    assert decision.cost == 0.0 and decision.input_tokens == 0
    assert _FakeOpenRouter.last_instance is None  # the API tier was skipped entirely


def test_local_router_can_be_disabled_by_config():
    app, _original = _app()
    app.config.orchestration["local_router"] = False

    decision = auto.select_workflow(app, "fix the login redirect bug", "build")

    # With the local tier off, even a clear-cut prompt reaches the model router.
    assert decision.router_provider == "openrouter"


def test_local_router_allowlist_routes():
    """The only three shapes the local tier will confidently route."""
    solve = [
        "fix the login redirect bug",
        "add a dark-mode toggle to settings",
        "refactor this one function",
        "please add a health-check endpoint",
        "fix the errors codex found",   # referential object is fine; lane-context resolves it
    ]
    for p in solve:
        assert auto._local_route(p, "build").workflow == WorkflowKind.SOLVE, p
    recon = ["review the auth module", "explain the config file", "read the codebase"]
    for p in recon:
        assert auto._local_route(p, "build").workflow == WorkflowKind.RECON, p
    for p in ["thanks, that worked", "hey", "hi", "thank you"]:
        assert auto._local_route(p, "build").workflow == WorkflowKind.CHAT, p


def test_local_router_abstains_on_everything_ambiguous():
    """The mis-routes the four reviewers flagged -- each must now abstain."""
    for p in [
        "update: the tests now pass on CI",              # a heading, not a command
        "write me a summary of the codebase",            # addressed to the assistant
        "explain how async/await works in JS",           # general knowledge (the slash)
        "explain how the map function works in JavaScript",  # general knowledge (generic noun)
        "explain how OAuth works, e.g. the refresh flow",    # general knowledge (the dot)
        "Should we update this dependency?",             # a question
        "What does `delete` mean here?",                 # a question
        "can you fix the login bug?",                    # a question (trailing ?)
        "hey, continue fixing it",                       # greeting prefix on an action
        "do option a",                                   # vague deixis
        "the parser thing",                              # no imperative
        "review the parser and fix what's broken",       # "parser" is not a repo-object noun
    ]:
        assert auto._local_route(p, "build") is None, f"{p!r} should abstain"


def test_local_router_is_mode_aware():
    # In think-first modes an edit-shaped ask is more likely ideation -> defer.
    assert auto._local_route("fix the login bug", "plan") is None
    assert auto._local_route("create an implementation plan for OAuth", "plan") is None
    # ...but the same command routes in build mode.
    assert auto._local_route("fix the login bug", "build").workflow == WorkflowKind.SOLVE
    # Recon is mode-agnostic (inspection is always safe).
    assert auto._local_route("review the auth module", "plan").workflow == WorkflowKind.RECON


def test_recon_lane_exposes_only_read_only_tools():
    app, _original = _app()
    decision = RouteDecision(WorkflowKind.RECON, "inspect", 0.9)

    result = auto.execute_auto(app, "Find the config", "openai", decision)

    provider = _FakeOpenRouter.last_instance
    assert result.outcome == RunOutcome.SUCCEEDED
    assert result.execution_provider == "openrouter"
    assert set(provider.tools) == {"read_file", "list_files", "search_files"}
    assert "write_file" not in provider.tools and "run_command" not in provider.tools
    assert "Found it" in result.text


def test_recon_in_test_mode_can_run_checks_but_not_write(monkeypatch, tmp_path):
    app, _original = _app()
    decision = RouteDecision(WorkflowKind.RECON, "verify it works", 0.9)

    # Test-mode recon must operate on a throwaway worktree, never the real cwd.
    fake_worktree = tmp_path / "recon_wt"
    fake_worktree.mkdir()
    manager, roots = _fake_worktree_manager(str(fake_worktree))
    monkeypatch.setattr(auto, "WorktreeManager", manager)
    monkeypatch.setattr(auto, "WorkspaceTools", _spy_workspace_tools(roots))

    result = auto.execute_auto(
        app, "does this project work?", "openai", decision, mode="test",
    )

    provider = _FakeOpenRouter.last_instance
    assert result.outcome == RunOutcome.SUCCEEDED
    # Test mode is verification: recon can actually run the project's checks.
    assert "run_command" in provider.tools
    # ...but must not modify source (no write/edit tools).
    assert "write_file" not in provider.tools
    # ...and those checks run in the isolated worktree, not the user's checkout.
    assert roots["ws"] == str(fake_worktree)
    assert manager.cleaned is True
    _messages, kwargs = provider.tool_call
    system = kwargs["system"].lower()
    assert "may run" in system
    assert "not edit source" in system


def test_focused_route_delegates_to_verified_solve(monkeypatch):
    app, _original = _app()
    solve_result = SimpleNamespace(
        outcome=RunOutcome.SUCCEEDED,
        iterations=2,
        provider="openai",
        models_used=("bulk", "frontier"),
        error="",
        diff_stat="1 file changed",
        changed_files=("a.py",),
        worktree_path="/tmp/solve-wt",
        input_tokens=30,
        output_tokens=10,
    )
    called = MagicMock(return_value=solve_result)
    monkeypatch.setattr(auto, "run_solve", called)
    decision = RouteDecision(
        WorkflowKind.SOLVE,
        "focused edit",
        0.8,
        input_tokens=4,
        output_tokens=2,
    )

    result = auto.execute_auto(app, "Fix a.py", "openai", decision)

    called.assert_called_once()
    assert result.outcome == RunOutcome.SUCCEEDED
    assert (result.input_tokens, result.output_tokens) == (34, 12)
    assert "Review + apply: git -C /tmp/solve-wt diff" in result.text


def test_recon_lane_receives_conversation_context():
    app, _original = _app()
    decision = RouteDecision(WorkflowKind.RECON, "inspect", 0.9)

    auto.execute_auto(
        app, "trace what codex flagged", "openai", decision,
        context="[Prior conversation]\ncodex: CTX_REFERENT the bug is in x.py",
    )

    messages, _kwargs = _FakeOpenRouter.last_instance.tool_call
    # The referent rides along in the recon user message, ahead of the task.
    assert "CTX_REFERENT" in messages[0]["content"]
    assert "trace what codex flagged" in messages[0]["content"]


def test_focused_solve_forwards_conversation_context(monkeypatch):
    app, _original = _app()
    solve_result = SimpleNamespace(
        outcome=RunOutcome.SUCCEEDED, iterations=1, provider="openai",
        models_used=("bulk",), error="", diff_stat="", changed_files=(),
        worktree_path="/tmp/wt", input_tokens=0, output_tokens=0,
    )
    called = MagicMock(return_value=solve_result)
    monkeypatch.setattr(auto, "run_solve", called)
    decision = RouteDecision(WorkflowKind.SOLVE, "edit", 0.8)

    auto.execute_auto(
        app, "fix the errors codex found", "openai", decision,
        context="codex: CTX_REFERENT list of errors",
    )

    assert called.call_args.kwargs["context"] == "codex: CTX_REFERENT list of errors"


def test_fast_solve_starts_on_fast_model_then_escalates_to_active_frontier(monkeypatch):
    app, _original = _app()
    solve_result = SimpleNamespace(
        outcome=RunOutcome.SUCCEEDED,
        iterations=1,
        provider="openrouter",
        models_used=("stepfun/step-3.7-flash",),
        error="",
        diff_stat="1 file changed",
        changed_files=("a.py",),
        worktree_path="/tmp/fast-wt",
        input_tokens=10,
        output_tokens=4,
    )
    called = MagicMock(return_value=solve_result)
    monkeypatch.setattr(auto, "run_solve", called)
    decision = RouteDecision(
        WorkflowKind.SOLVE,
        "tiny localized fix",
        0.95,
        worker_tier="fast",
    )

    result = auto.execute_auto(app, "Fix a one-line typo", "openai", decision)

    kwargs = called.call_args.kwargs
    assert kwargs["provider_name"] == "openrouter"
    assert kwargs["bulk_model_override"] == "stepfun/step-3.7-flash"
    assert kwargs["provider_preferences_override"] == {
        "allow_fallbacks": True,
        "require_parameters": True,
    }
    assert kwargs["escalate_to"] == ("openai", "openai-frontier")
    assert result.outcome == RunOutcome.SUCCEEDED
    assert "fast worker" in result.text


def test_frontier_solve_starts_on_configured_escalation_provider(monkeypatch):
    app, _original = _app()
    app.providers["claude"] = MagicMock()
    app.config.get_escalation_target = lambda provider: (
        "claude" if provider == "openai" else None
    )
    solve_result = SimpleNamespace(
        outcome=RunOutcome.SUCCEEDED,
        iterations=1,
        provider="claude",
        models_used=("claude-frontier",),
        error="",
        diff_stat="1 file changed",
        changed_files=("a.py",),
        worktree_path="/tmp/frontier-wt",
        input_tokens=10,
        output_tokens=4,
    )
    called = MagicMock(return_value=solve_result)
    monkeypatch.setattr(auto, "run_solve", called)
    decision = RouteDecision(
        WorkflowKind.SOLVE,
        "hard ambiguous change",
        0.9,
        worker_tier="frontier",
    )

    auto.execute_auto(app, "Implement the hard change", "openai", decision)

    kwargs = called.call_args.kwargs
    assert kwargs["provider_name"] == "claude"
    assert kwargs["bulk_model_override"] == "claude-frontier"
    assert kwargs["escalate"] is False


def test_unsafe_fanout_plan_reroutes_to_pipeline(monkeypatch):
    app, _original = _app()
    monkeypatch.setattr(
        auto,
        "run_fanout",
        MagicMock(return_value=SimpleNamespace(
            outcome=RunOutcome.BLOCKED,
            error="invalid parallel plan: overlapping ownership",
        )),
    )
    pipeline = SimpleNamespace(
        outcome=RunOutcome.SUCCEEDED,
        provider="openai",
        steps=(SimpleNamespace(
            passed=True,
            id="s1",
            iterations=1,
            description="safe sequential work",
            error="",
        ),),
        error="",
        diff_stat="1 file changed",
        worktree_path="/tmp/pipeline-wt",
        input_tokens=40,
        output_tokens=12,
    )
    pipeline_call = MagicMock(return_value=pipeline)
    monkeypatch.setattr(auto, "run_pipeline", pipeline_call)
    decision = RouteDecision(WorkflowKind.FANOUT, "looked independent", 0.7)

    result = auto.execute_auto(app, "Build both pieces", "openai", decision)

    pipeline_call.assert_called_once()
    assert result.outcome == RunOutcome.SUCCEEDED
    assert "fanout -> pipeline" in result.text


def test_normal_design_mode_prompt_dispatches_selected_workflow(monkeypatch):
    """Automatic orchestration is on the normal prompt path, not slash-command only."""
    cli_app, _original = _app()
    cli_app.providers["openai"] = MagicMock()
    cli_app.prompt_pipeline = MagicMock()
    cli_app.prompt_pipeline.build.return_value = ""
    cli_app.context_builder = MagicMock(source_count=0)
    cli_app.hook_runner = MagicMock()
    cli_app.hook_runner.emit.return_value = None

    app = MagicMock()
    app.cli_app = cli_app
    app.state.messages = []
    app.state.episodes = []
    app.state.compaction_summary = ""  # a real string, as in production
    app.call_from_thread.side_effect = lambda fn, *args: fn(*args)

    screen = MainScreen(
        active_provider="openai",
        mode="design",
        providers=cli_app.providers,
    )
    screen._auto_orchestration_worker = MagicMock()
    decision = RouteDecision(WorkflowKind.SOLVE, "code edit", 0.9)
    monkeypatch.setattr(auto, "select_workflow", MagicMock(return_value=decision))

    token = active_app.set(app)
    try:
        screen._provider_worker("Implement the parser", "openai")
    finally:
        active_app.reset(token)

    screen._auto_orchestration_worker.assert_called_once_with(
        cli_app,
        "Implement the parser",
        "openai",
        decision,
        context="",  # empty history -> no digest, but the arg is threaded
    )
